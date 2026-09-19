#!/usr/bin/env python3
# [PORT gfx1030] NVVM → AMDGPU IR 改写器（单一事实源）。
#
# tests/test_gfx1030.py（回归测试）与批量统计脚本（examples 批量重跑）
# 都 import 本模块。改写是单向、幂等的：必须从原始（未经改写的）产物
# 一次性生成目标文件，禁止对改写产物做第二轮补丁（多轮补丁会引入 SSA
# 重名/残留引用——3.5 节 IR 层适配纪律）。
#
# 管线步骤（每步都经 llc -march=amdgcn -mcpu=gfx1030 + GPU 数值验证）：
#   1. target triple → "amdgcn-amd-amdhsa"
#   2. target datalayout → AMDGPU amdgcn 布局
#   3. ptx_kernel → amdgpu_kernel（调用约定即 kernel 标记）
#   4. llvm.nvvm.read.ptx.sreg.ctaid.x → llvm.amdgcn.workgroup.id.x
#      llvm.nvvm.read.ptx.sreg.tid.x   → llvm.amdgcn.workitem.id.x
#   5. ntid.x → kernarg 新参数 i32 %ntid_x（blockDim 由 host 决定）
#   6. ntid/nctaid .y|.z → "or i32 0, 1" 常量化（1D launch 语义）
#   7. 删除已映射 intrinsic 的 nvvm declare（llc 自动声明 amdgcn 目标 intrinsic）
#   8. alloca → addrspace(5) + use 点 addrspacecast（AMDGPU 本地帧契约，
#      "alloca on amdgpu must be in addrspace(5)"，批量实证 module verifier 拒绝）
#  12. isspacep.local/shared/global → llvm.amdgcn.is.private/shared/global
#      （atomic 地址分类；HIP 语境 local=scratch/private）
# 12b. membar.gl/cta/sys → fence（agent/workgroup/system 域，.ll 层直接写）
# 12c. syncscope("device"/"block") → ("agent"/"workgroup")（NV scope 名 → AMD）
#  13. idp4a.s.s/.u.u → llvm.amdgcn.sdot4/udot4（v_dot4_i32_i8 硬件，clamp=false）；
#      idp2a.* → 乘加展开（a 2×i16 × b 低/高 2 字节 i8，符号按后缀 sext/zext）
#  14. barrier.cta.sync.count / arrive.count → 软件 counted barrier
#      （LDS 计数器+世代自旋回退，helper 函数 + 入口 init，见 _rewrite_counted_barriers）
#  15. __nv_<f>f / __nv_<f> → __ocml_<f>_f32 / __ocml_<f>_f64
#      （libdevice → ocml；符号名经 llvm-nm ocml.bc 实证，llvm-link 链接期解析）
#  16. mbarrier → 软件状态机（RDNA2 无 mbarrier 硬件）：
#      init/arrive(.noComplete)/test_wait(PTX asm)/inval → LDS 单字 i64
#      {pending[31:0], expected[47:32], phase[63:48]} 的原子操作，见
#      _rewrite_mbarriers
#
# 未映射 intrinsic 的 declare 保留——其调用点让 llc 报 Cannot select，
# 作为"能力未覆盖"的显式失败信号（不静默放弃）。

import re

AMD_TRIPLE = "amdgcn-amd-amdhsa"
# 与 phase1 golden 产物（vecadd_amdgcn.ll）逐字一致的 AMDGPU datalayout
AMD_DATALAYOUT = (
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32"
    "-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-i128:128-v16:16-v32:32-n16:32:64"
)


def translate_nvvm_to_amdgcn(ll_text: str) -> str:
    """把 cuda-oxide 产的 NVVM 风格 .ll 文本改写为 amdgcn 后端可编译的 .ll。"""
    text = ll_text

    # 1. triple
    text = re.sub(r'target triple = "[^"]*"', f'target triple = "{AMD_TRIPLE}"', text)

    # 2. datalayout
    text = re.sub(
        r'target datalayout = "[^"]*"',
        f'target datalayout = "{AMD_DATALAYOUT}"',
        text,
    )

    # 3. 调用约定：ptx_kernel → amdgpu_kernel（AMD 靠 CC 认 kernel）
    text = text.replace("ptx_kernel", "amdgpu_kernel")

    # 4. sreg 同名替换（只换被调名，SSA 引用不动；declare 随后删除，llc 自动声明目标 intrinsic）
    text = text.replace(
        "@llvm.nvvm.read.ptx.sreg.ctaid.x()", "@llvm.amdgcn.workgroup.id.x()"
    )
    text = text.replace("@llvm.nvvm.read.ptx.sreg.tid.x()", "@llvm.amdgcn.workitem.id.x()")

    # 5. ntid.x → kernarg 参数 %ntid_x（blockDim 由 host 决定）：
    #    调用点替换为恒等 or，签名追加参数（见 _append_ntid_kernarg_param）
    text = re.sub(
        r"(?:tail )?call i32 @llvm\.nvvm\.read\.ptx\.sreg\.ntid\.x\(\)(?:\s*#\d+)?",
        "or i32 %ntid_x, 0",
        text,
    )
    text = _append_ntid_kernarg_param(text)

    # 6. ntid/nctaid .y|.z → 常量 1（1D launch：block/grid 的 y,z 维度为 1）
    text = re.sub(
        r"(?:tail )?call i32 @llvm\.nvvm\.read\.ptx\.sreg\.(?:ntid|nctaid)\.[yz]\(\)(?:\s*#\d+)?",
        "or i32 0, 1",
        text,
    )

    # 7. 删除已映射 intrinsic 的 declare（未映射的保留——其调用点会让 llc
    #    报 Cannot select，作为"能力未覆盖"的显式失败信号）
    mapped = (
        r"@llvm\.(?:"
        r"nvvm\.read\.ptx\.sreg\.(?:ctaid|tid|ntid)\.x"
        r"|nvvm\.read\.ptx\.sreg\.(?:ntid|nctaid)\.[yz]"
        r"|nvvm\.barrier\.cta\.sync\.aligned\.all"
        r"|nvvm\.barrier\.cta\.sync\.count"
        r"|nvvm\.barrier\.cta\.arrive\.count"
        r"|nvvm\.barrier0"
        r"|nvvm\.shfl\.sync\.(?:idx|bfly|up|down)\.(?:f32|i32)"
        r"|nvvm\.read\.ptx\.sreg\.laneid"
        r"|nvvm\.redux\.sync\.add"
        r"|nvvm\.isspacep\.(?:local|shared|global)"
        r"|nvvm\.mbarrier\.(?:init|arrive|arrive\.noComplete|inval)\.shared"
        r"|nvvm\.membar\.(?:gl|sys|cta)"
        r"|nvvm\.idp4a\.[su]\.[su]"
        r"|nvvm\.idp2a\.[su]\.[su]"
        r"|amdgcn\.work(?:group|item)\.id\.x"
        r")\("
    )
    text = re.sub(rf"declare [^\n]*{mapped}[^\n]*\n", "", text)

    # 8. alloca → addrspace(5)（AMDGPU module verifier：本地帧必须 scratch）
    text = _rewrite_allocas(text)

    # 9. CTA barrier（__syncthreads 语义）→ llvm.amdgcn.s.barrier
    #    （编译探针实证：gfx1030 后端存在 convergent 无参 intrinsic）
    text = re.sub(
        r"(?:tail )?call void @llvm\.nvvm\.barrier\.cta\.sync\.aligned\.all\([^)]*\)"
        r"(?:\s*#\d+)?",
        "call void @llvm.amdgcn.s.barrier()",
        text,
    )
    text = re.sub(
        r"(?:tail )?call void @llvm\.nvvm\.barrier0\(\)(?:\s*#\d+)?",
        "call void @llvm.amdgcn.s.barrier()",
        text,
    )

    # 9b. LDS 静态初始化契约：addrspace(3) 全局量不能带 zeroinitializer
    #     （"unsupported initializer for address space"——LDS 无加载时清零
    #     硬件；hipcc 对 __shared__ 实产 undef。cuda-oxide 的 zeroinit 是
    #     LLVM 全局量文法产物，非语义承诺：SharedArray::UNINIT 即未初始化）
    text = re.sub(
        r"(addrspace\(3\) global\s+[^;\n]+?)\s+zeroinitializer",
        r"\1 undef",
        text,
    )

    # 10. warp shuffle → ds_bpermute（段式 32-lane 语义，见 _rewrite_shuffle）
    text = _rewrite_shuffle(text)

    # 10c. cuda-oxide 的 f64 shuffle 内嵌 PTX asm（64 位拆两个 b32 bfly，
    #      aiter warp.rs 同款实现）→ 双 ds_bpermute 拆分，见 _rewrite_shfl_asm
    text = _rewrite_shfl_asm(text)

    # 11. redux.sync.add → 蝶形归约软件回退（gfx1030 无硬件 redux 指令）
    text = _rewrite_redux_add(text)

    # 12. atomics 族：isspacep → amdgcn.is.*、membar → fence、NV scope 名 → AMD
    text = _rewrite_atomics(text)

    # 12d. Rust core atomics 的内嵌 PTX asm（fence/acquire load/release store/
    #      seqcst load）→ LLVM 原子指令（层叠契约：isspacep 清零后才暴露）
    text = _rewrite_ptx_atomic_asm(text)

    # 13. 整数点积：idp4a → sdot4/udot4（硬件 v_dot4），idp2a → 乘加展开
    text = _rewrite_dotprod(text)

    # 14. counted barrier → 软件 LDS 计数器+世代自旋回退
    text = _rewrite_counted_barriers(text)

    # 15. libdevice __nv_* → ocml __ocml_*（链接期经 llvm-link ocml.bc 解析）
    text = _rewrite_ocml(text)

    # 16. mbarrier → 软件状态机（RDNA2 无硬件等价；aiter 管线实证的
    #     "异步等待退化为软件同步" 路线，见 _rewrite_mbarriers）
    text = _rewrite_mbarriers(text)

    return text


# ---------------------------------------------------------------------------
# Step 8: alloca 地址空间（addrspace(5) 契约）
# ---------------------------------------------------------------------------

# 已是 addrspace(5) 的 alloca 不再匹配（幂等性的关键：改写产物重跑不变）
_ALLOCA_DEF_RE = re.compile(r"^(?P<indent>\s*)(?P<name>%[\w.$]+)\s*=\s*alloca\s+(?P<rest>[^->]*\S)\s*$")

# typed-pointer IR（math_tan 类非 opt 形态）不支持：use 点是 `bitcast <ty>*
# %name to ...`，把 alloca 改 addrspace(5) 后 bitcast 跨地址空间非法
# （须 addrspacecast 且元素类型语法不同）——显式跳过，llc 对该 module 仍报
# verifier 错（显式失败信号）。
_TYPED_IR_RE = re.compile(r"bitcast\s+[^;\n]*\*\s*%")


def _split_top_level_commas(s: str):
    """按括号深度切顶层逗号（<>/[]/() 内的逗号不算）。"""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(s):
        if ch in "<[(":
            depth += 1
        elif ch in ">])":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return parts


def _rewrite_allocas(text: str) -> str:
    """generic 地址空间的 alloca → addrspace(5)，use 点经 addrspacecast 降级。

    正确性论证：
    - alloca 在 entry 块且先于任何 terminator → 其值支配函数内全部 use，
      因此每个 alloca 只需一条 cast（插在 alloca 定义行之后）；
    - use 替换是 SSA 名的 token 级替换（load/store/gep/bitcast/phi 等
      操作数位置统一成立），被调名以外的引用不动；
    - llvm.lifetime/dbg 类要求与 alloca 同地址空间的 use 在 cuda-oxide
      产物中不出现（14 例批量 grep 实证）；一旦出现会被留成类型不匹配
      的 verifier 显式失败信号，不静默出错。
    """
    if "= alloca " not in text or _TYPED_IR_RE.search(text):
        return text

    lines = text.split("\n")
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not (line.startswith("define ") and line.rstrip().endswith("{")):
            out.append(line)
            i += 1
            continue

        # 函数整体收集（define 行到收尾 "}"），函数内可能多个 alloca
        func_start = i
        func_end = _find_function_end(lines, i)
        body = lines[func_start : func_end + 1]
        out.extend(_rewrite_allocas_in_function(body))
        i = func_end + 1

    return "\n".join(out)


def _rewrite_allocas_in_function(body: list) -> list:
    """单个函数内的 alloca 改写（body = define 行.."}" 行）。"""
    allocas = []  # (行号, name, 新定义行)
    for k, line in enumerate(body):
        m = _ALLOCA_DEF_RE.match(line)
        if not m or "addrspace(" in m.group("rest"):
            continue
        name = m.group("name")
        rest = m.group("rest")
        # 语法顺序（hipcc gfx1030 实产对照）：alloca <ty>[, <count>][, align N],
        # addrspace(5)
        align = None
        ma = re.search(r",\s*align\s+(\d+)\s*$", rest)
        if ma:
            align = ma.group(1)
            rest = rest[: ma.start()]
        parts = _split_top_level_commas(rest)
        if len(parts) > 2 or parts[0] == "void":
            continue  # 不可解析形态：保留原样（llc verifier 显式失败）
        ty = parts[0]
        count = parts[1] if len(parts) == 2 else None
        indent = m.group("indent")
        new_def = f"{indent}{name} = alloca {ty}"
        if count is not None:
            new_def += f", {count}"
        if align is not None:
            new_def += f", align {align}"
        new_def += ", addrspace(5)"
        allocas.append((k, name, new_def))

    if not allocas:
        return body

    out = []
    cast_lines_by_after = {}  # alloca 行号 -> 紧随其后的 cast 定义行
    replaced_names = {}  # alloca 名 -> cast 名
    new_def_by_line = {}  # alloca 行号 -> 新定义行
    for k, name, new_def in allocas:
        cast_name = f"{name}.ac"
        replaced_names[name] = cast_name
        new_def_by_line[k] = new_def
        cast_lines_by_after[k] = (
            f"{cast_name} = addrspacecast ptr addrspace(5) {name} to ptr"
        )

    # def 行替换 + use 行替换（token 级），cast 插在对应 alloca 行后
    for k, line in enumerate(body):
        if k in cast_lines_by_after:
            out.append(new_def_by_line[k])
            out.append(cast_lines_by_after[k])
            continue
        for name, cast in replaced_names.items():
            line = re.sub(rf"(?<![\w.$]){re.escape(name)}(?![\w.$])", cast, line)
        out.append(line)
    return out


def _find_function_end(lines, alloca_idx: int) -> int:
    """alloca 之后第一个列 0 的 "}" 行（文本 IR：BB 无大括号，函数体的
    "}" 唯一）。兜底返回 len(lines)。"""
    j = alloca_idx + 1
    while j < len(lines) and lines[j] != "}":
        j += 1
    return j


# ---------------------------------------------------------------------------
# Step 10: warp shuffle → ds_bpermute（+ laneid）
# ---------------------------------------------------------------------------

# 语法（warp_reduce 实产对照）：
#   %r = tail call float @llvm.nvvm.shfl.sync.<mode>.f32
#        (i32 -1, float %val, i32 <delta>, i32 31)
# mode ∈ idx(按索引读)/bfly(xor)/up/down；PTX 语义按 width 段内交换，
# clamp=31 即 width=32（CUDA warp 大小）。
_SHFL_RE = re.compile(
    r"^(?P<indent>\s*)(?P<res>%[\w.$]+)\s*=\s*(?:tail )?call\s+"
    r"(?P<ty>float|i32)\s+@llvm\.nvvm\.shfl\.sync\."
    r"(?P<mode>idx|bfly|up|down)\.(?P<st>f32|i32)\((?P<args>[^)]*)\)\s*(?:#\d+)?\s*$"
)

# membermask 非 -1（部分 warp 参与的 shuffle）不支持：exec 语义在 AMD 侧
# 需要独立处理，保守留下 nvvm 调用让 llc 显式失败
_SHFL_MASK_RE = re.compile(r"i32\s*(-?\d+)\s*$")


def _rewrite_shuffle(text: str) -> str:
    """NVVM shuffle → AMD ds_bpermute 展开（每调用点一段直线 IR）。

    语义映射（wave64 机器上保持 CUDA 32-lane warp 语义——按 32 对齐段
    切分，段内交换，段外回自身值）：
      lane  = mbcnt.lo(-1, 0)                ; 0..63
      seg   = lane & -32                     ; 段基址
      idx   : src = seg + (delta & 31)
      bfly  : src = lane ^ delta             ; delta<32 不跨段
      down  : t = (lane&31)+delta; src = t<=31 ? seg+t : lane
      up    : t = (lane&31)-delta; src = 段内 ? seg+t : lane
      off   = src * 4（ds_bpermute 是字节偏移，NV 是 lane 号——本步的
              核心差异点）
      val   : float 经 bitcast 往返，i32 直通
    结果写回原 SSA 名，下游引用零改动。_laneid 同步映射 mbcnt.lo。
    """
    if "shfl.sync." not in text and "sreg.laneid" not in text:
        return text

    lines = text.split("\n")
    out = []
    for line in lines:
        m = _LANEID_RE.match(line)
        if m:
            lhs = f"{m.group('res')} = " if m.group("res") else ""
            out.append(f"{m.group('indent')}{lhs}call i32 "
                       "@llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)")
            continue
        m = _SHFL_RE.match(line)
        if not m:
            out.append(line)
            continue
        expansion = _expand_shfl(m)
        if expansion is None:  # 不支持的形态（mask 非 -1 等）：原样保留
            out.append(line)
            continue
        out.extend(expansion)
    return "\n".join(out)


_LANEID_RE = re.compile(
    r"^(?P<indent>\s*)(?:(?P<res>%[\w.$]+)\s*=\s*)?(?:tail )?call\s+i32\s+"
    r"@llvm\.nvvm\.read\.ptx\.sreg\.laneid\(\)\s*(?:#\d+)?\s*$"
)


def _arg_operand(arg: str) -> str:
    """"i32 16"/"float %v19" → "16"/"%v19"（剥类型前缀）。"""
    return arg.strip().split(None, 1)[1].strip()


def _expand_shfl(m) -> "list[str] | None":
    res, ty, mode, indent = m.group("res"), m.group("ty"), m.group("mode"), m.group("indent")
    st = m.group("st")
    # 类型与 intrinsic 后缀一致性：float ↔ f32 / i32 ↔ i32
    if (ty == "float") != (st == "f32"):
        return None
    args = [a.strip() for a in _split_top_level_commas(m.group("args"))]
    if len(args) != 4:
        return None
    mask_arg, val_arg, delta_arg, clamp_arg = args
    mm = _SHFL_MASK_RE.match(mask_arg)
    if not mm or mm.group(1) != "-1":
        return None
    cm = _SHFL_MASK_RE.match(clamp_arg)
    if not cm or cm.group(1) != "31":
        return None  # width≠32（非标准 warp 尺寸）：显式不支持
    delta = _arg_operand(delta_arg)
    val = _arg_operand(val_arg)
    r = res  # 中间名以 .b_ 标记（NVVM 名不含下划线，避免碰撞）
    L = indent + f"{r}.b_"

    seq = [f"{indent}{r}.b_ln = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)"]
    if mode == "bfly":
        seq.append(f"{L}src = xor i32 {r}.b_ln, {delta}")
    elif mode == "idx":
        seq.append(f"{L}seg = and i32 {r}.b_ln, -32")
        seq.append(f"{L}d = and i32 {delta}, 31")
        seq.append(f"{L}src = add i32 {L}seg, {L}d")
    elif mode == "down":
        seq.append(f"{L}seg = and i32 {r}.b_ln, -32")
        seq.append(f"{L}lo = and i32 {r}.b_ln, 31")
        seq.append(f"{L}t = add i32 {L}lo, {delta}")
        seq.append(f"{L}in = icmp ule i32 {L}t, 31")
        seq.append(f"{L}s2 = add i32 {L}seg, {L}t")
        seq.append(f"{L}src = select i1 {L}in, i32 {L}s2, i32 {r}.b_ln")
    else:  # up
        seq.append(f"{L}seg = and i32 {r}.b_ln, -32")
        seq.append(f"{L}lo = and i32 {r}.b_ln, 31")
        seq.append(f"{L}t = sub i32 {L}lo, {delta}")
        seq.append(f"{L}in = icmp uge i32 {L}lo, {delta}")
        seq.append(f"{L}s2 = add i32 {L}seg, {L}t")
        seq.append(f"{L}src = select i1 {L}in, i32 {L}s2, i32 {r}.b_ln")

    seq.append(f"{L}off = shl i32 {L}src, 2")
    if ty == "float":
        seq.append(f"{L}bits = bitcast float {val} to i32")
        seq.append(f"{L}got = call i32 @llvm.amdgcn.ds.bpermute(i32 {L}off, i32 {L}bits)")
        seq.append(f"{indent}{res} = bitcast i32 {L}got to float")
    else:
        seq.append(f"{indent}{res} = call i32 @llvm.amdgcn.ds.bpermute(i32 {L}off, i32 {val})")
    return seq


# 10c. cuda-oxide 的 f64 shuffle 走内嵌 PTX asm（shfl.sync.bfly.b32 lo/hi 拆分，
# 与 aiter warp.rs 的 64 位拆分实现同源）。该 asm 是 PTX 目标汇编，AMDGPU 后端
# 直接拒绝（"could not allocate output register for constraint 'l'"）。
# 识别其固定调用形态并展开为双 ds_bpermute：
#   %r = tail call i64 asm sideeffect "... shfl.sync.bfly.b32 lo ... $2, 31, $3 ...",
#        "=l,l,r,r"(i64 <val>, i32 <delta>, i32 -1) [attrs]
_SHFL_ASM_RE = re.compile(
    r'^(?P<indent>\s*)(?P<res>%[\w.$]+)\s*=\s*(?:tail )?call\s+i64\s+'
    r'asm sideeffect "[^"]*shfl\.sync\.bfly\.b32 lo[^"]*?",\s*'
    r'"=l,l,r,r"\(i64\s+(?P<val>%[\w.$]+),\s*i32\s+(?P<delta>[^,)]+),\s*'
    r'i32\s+(?P<mask>[^,)]+)\)\s*(?:#\d+)?\s*$'
)


def _rewrite_shfl_asm(text: str) -> str:
    if "shfl.sync.bfly.b32" not in text:
        return text
    lines = text.split("\n")
    out = []
    for line in lines:
        m = _SHFL_ASM_RE.match(line)
        if not m:
            out.append(line)
            continue
        mask = m.group("mask").strip()
        if mask != "-1":
            out.append(line)  # mask 非 -1：显式不支持，留给 llc 报错
            continue
        res, val = m.group("res"), m.group("val")
        delta = m.group("delta").strip()  # 正则已消耗 "i32 " 前缀
        ind = m.group("indent")
        L = ind + f"{res}.b_"
        out.extend(
            [
                f"{ind}{res}.b_ln = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)",
                f"{L}src = xor i32 {res}.b_ln, {delta}",
                f"{L}off = shl i32 {L}src, 2",
                f"{L}lo = trunc i64 {val} to i32",
                f"{L}hi64 = lshr i64 {val}, 32",
                f"{L}hi = trunc i64 {L}hi64 to i32",
                f"{L}glo = call i32 @llvm.amdgcn.ds.bpermute(i32 {L}off, i32 {L}lo)",
                f"{L}ghi = call i32 @llvm.amdgcn.ds.bpermute(i32 {L}off, i32 {L}hi)",
                f"{L}ghi64 = zext i32 {L}ghi to i64",
                f"{L}ghiup = shl i64 {L}ghi64, 32",
                f"{L}glo64 = zext i32 {L}glo to i64",
                f"{ind}{res} = or i64 {L}glo64, {L}ghiup",
            ]
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Step 11: redux.sync.add → 蝶形归约软件回退
# ---------------------------------------------------------------------------
#
# gfx1030（RDNA2）无硬件 redux 指令（N/A 清单实证）。回退实现：5 轮
# xor-butterfly（16/8/4/2/1），每轮 ds_bpermute 取对侧值后 i32 相加——
# 5 轮后每 lane 得到其 32 对齐段（= CUDA warp）的全和，与 redux.sync.add
# 语义一致（全 warp 参与、广播结果）。wave64 机器上 xor<32 不跨段，段内
# 自洽。性能为 O(log32)·wave 次 LDS 往返，正确性优先（port 纪律 5）。
#
# 语法（redux_sum 实产对照）：
#   %r = tail call i32 @llvm.nvvm.redux.sync.add(i32 <val>, i32 -1) [attrs]
_REDUX_RE = re.compile(
    r"^(?P<indent>\s*)(?P<res>%[\w.$]+)\s*=\s*(?:tail )?call\s+i32\s+"
    r"@llvm\.nvvm\.redux\.sync\.add\(\s*i32\s+(?P<val>[^,)]+?)\s*,\s*"
    r"i32\s+(?P<mask>[^,)]+)\)\s*(?:#\d+)?\s*$"
)


def _rewrite_redux_add(text: str) -> str:
    if "redux.sync.add" not in text:
        return text
    lines = text.split("\n")
    out = []
    for line in lines:
        m = _REDUX_RE.match(line)
        if not m:
            out.append(line)
            continue
        if m.group("mask").strip() != "-1":
            out.append(line)  # 部分 warp 参与：不支持，留给 llc 显式失败
            continue
        res, val, ind = m.group("res"), m.group("val"), m.group("indent")
        L = ind + f"{res}.b_"
        seq = [f"{ind}{res}.b_ln = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)"]
        cur = val
        for shift in (16, 8, 4, 2, 1):
            seq.append(f"{L}m{shift} = xor i32 {res}.b_ln, {shift}")
            seq.append(f"{L}o{shift} = shl i32 {L}m{shift}, 2")
            seq.append(f"{L}g{shift} = call i32 @llvm.amdgcn.ds.bpermute("
                       f"i32 {L}o{shift}, i32 {cur})")
            # 末轮直接写回原 SSA 名，下游引用零改动
            dst = res if shift == 1 else f"{L}s{shift}"
            seq.append(f"{ind}{dst} = add i32 {cur}, {L}g{shift}")
            cur = dst
        out.extend(seq)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Step 12: atomics 族（isspacep / membar / NV scope 名）
# ---------------------------------------------------------------------------

# 探针实证（nightly-2026-08-28 llc, gfx1030）：
# - llvm.amdgcn.is.private/is.shared/is.global 均存在，签名 i1(ptr)，
#   对 generic 指针做硬件地址域判定（alloca 派生 → private=true、全局参数=false，
#   GPU 数值探针 ATOMPROBE PASS）
# - fence 语法：fence syncscope("<domain>") seq_cst / fence seq_cst（空=system）
# - atomicrmw/cmpxchg 的 NV scope 名不被接受（"Unsupported atomic synchronization
#   scope"）：device→agent、block→workgroup（层叠契约：isspacep 清零后才暴露）


def _rewrite_atomics(text: str) -> str:
    """地址分类 + 内存栅栏 + 同步域名的跨架构映射（同名替换保 SSA 引用不变）。"""
    if "isspacep" in text:
        # HIP/AMD 地址空间语义：NV local(线程私有 scratch) = AMD private(AS5)
        text = text.replace("@llvm.nvvm.isspacep.local(", "@llvm.amdgcn.is.private(")
        text = text.replace("@llvm.nvvm.isspacep.shared(", "@llvm.amdgcn.is.shared(")
        text = text.replace("@llvm.nvvm.isspacep.global(", "@llvm.amdgcn.is.global(")
    if "membar" in text:
        # membar.gl（设备域）→ agent；membar.cta（CTA 域）→ workgroup；
        # membar.sys（系统域）→ 空 syncscope = system
        text = re.sub(
            r"(?:tail )?call void @llvm\.nvvm\.membar\.gl\(\)(?:\s*#\d+)?",
            'fence syncscope("agent") seq_cst',
            text,
        )
        text = re.sub(
            r"(?:tail )?call void @llvm\.nvvm\.membar\.cta\(\)(?:\s*#\d+)?",
            'fence syncscope("workgroup") seq_cst',
            text,
        )
        text = re.sub(
            r"(?:tail )?call void @llvm\.nvvm\.membar\.sys\(\)(?:\s*#\d+)?",
            "fence seq_cst",
            text,
        )
    # NV 与 AMD 的 scope 命名不同（atomics 实产：device ×22、block ×2）
    text = text.replace('syncscope("device")', 'syncscope("agent")')
    text = text.replace('syncscope("block")', 'syncscope("workgroup")')
    return text


# 实产形态（atomics 实例 .opt.ll，Rust core::sync::atomic 的 NVPTX 降级）：
#   call void asm sideeffect "fence.acq_rel.{cta|gpu|sys};", "~{memory}"()
#   %r = {tail }call i32 asm sideeffect "ld.acquire.{gpu|sys}.b32 $0, [$1];",
#          "=r,l,~{memory}"(ptr %p)                       ; b64 → i64/ptr 结果
#   %r = call {i64|ptr} asm sideeffect
#          "fence.sc.sys; ld.acquire.sys.b64 $0, [$1];", "=l,l,~{memory}"(ptr %p)
#   call void asm sideeffect "st.release.{gpu|sys}.b{32|64} [$0], $1;",
#          "l,{r|l},~{memory}"(ptr %p, {i32|i64|ptr} %v)
# 映射（语法经 llc 探针实证；GPU MP 数值探针 ATOMPROBE/MP PROBE PASS）：
#   fence: cta→workgroup、gpu→agent、sys→空(system)；acq_rel→acq_rel、sc→seq_cst
#   ld.acquire: load atomic acquire（gpu→syncscope("agent")，sys→默认 system 域）；
#   fence.sc 前缀的 seqcst load → 单条 load atomic seq_cst（更强的单指令等价）
#   st.release: store atomic release（scope 同上）
_PTXX_ASM_FENCE_RE = re.compile(
    r'^(?P<indent>\s*)(?:tail )?call\s+void\s+asm\s+sideeffect\s+'
    r'"fence\.(?P<ord>acq_rel|sc)\.(?P<scope>cta|gpu|sys);",\s*'
    r'"~\{memory\}"\(\)\s*(?:#\d+)?\s*$'
)

_PTXX_ASM_LOAD_RE = re.compile(
    r'^(?P<indent>\s*)(?:(?P<res>%[\w.$]+)\s*=\s*)?(?:tail )?call\s+'
    r'(?P<ty>i32|i64|ptr)\s+asm\s+sideeffect\s+'
    r'"(?:(?P<sc>fence\.sc\.sys); )?ld\.acquire\.(?P<scope>gpu|sys)\.'
    r'b(?P<bits>32|64)\s+\$0,\s*\[\$1\];",\s*"=[rl],l,~\{memory\}"\(\s*ptr\s+'
    r'(?P<attrsq>(?:nonnull\s+)?)?(?P<ptr>[^,)]+?)\s*\)\s*(?:#\d+)?\s*$'
)

_PTXX_ASM_STORE_RE = re.compile(
    r'^(?P<indent>\s*)(?:tail )?call\s+void\s+asm\s+sideeffect\s+'
    r'"st\.release\.(?P<scope>gpu|sys)\.b(?P<bits>32|64)\s+\[\$0\], \$1;",\s*'
    r'"l,[rl],~\{memory\}"\(\s*ptr\s+(?P<attrsq>(?:nonnull\s+)?)?'
    r'(?P<ptr>[^,]+?),\s*(?P<vty>i32|i64|ptr)\s+(?P<val>[^,)]+?)\s*\)\s*(?:#\d+)?\s*$'
)

_SCOPE_MAP = {"cta": 'syncscope("workgroup")', "gpu": 'syncscope("agent")', "sys": ""}
_ORD_MAP = {"acq_rel": "acq_rel", "sc": "seq_cst"}


def _strip_param_attrs(operand: str) -> str:
    """store 值操作数剥参数属性（nonnull/noundef——store 目标无需携带）。"""
    return re.sub(r"^(?:nonnull|noundef)\s+", "", operand.strip())


def _rewrite_ptx_atomic_asm(text: str) -> str:
    if 'asm sideeffect "fence.' not in text and "ld.acquire." not in text \
            and "st.release." not in text:
        return text
    lines = text.split("\n")
    out = []
    for line in lines:
        m = _PTXX_ASM_FENCE_RE.match(line)
        if m:
            scope = _SCOPE_MAP[m.group("scope")]
            scope_s = f"{scope} " if scope else ""
            out.append(f"{m.group('indent')}fence {scope_s}"
                       f"{_ORD_MAP[m.group('ord')]}")
            continue
        m = _PTXX_ASM_LOAD_RE.match(line)
        if m:
            if not m.group("res"):
                out.append(line)  # 结果未使用：保守留残
                continue
            scope = _SCOPE_MAP[m.group("scope")]
            scope_s = f" {scope}" if scope else ""
            order = "seq_cst" if m.group("sc") else "acquire"
            align = 4 if m.group("bits") == "32" else 8
            out.append(
                f"{m.group('indent')}{m.group('res')} = load atomic "
                f"{m.group('ty')}, ptr {m.group('ptr')}{scope_s} {order}, "
                f"align {align}"
            )
            continue
        m = _PTXX_ASM_STORE_RE.match(line)
        if m:
            scope = _SCOPE_MAP[m.group("scope")]
            scope_s = f" {scope}" if scope else ""
            align = 4 if m.group("bits") == "32" else 8
            out.append(
                f"{m.group('indent')}store atomic {m.group('vty')} "
                f"{_strip_param_attrs(m.group('val'))}, ptr {m.group('ptr')}"
                f"{scope_s} release, align {align}"
            )
            continue
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Step 13: 整数点积（idp4a / idp2a）
# ---------------------------------------------------------------------------

# 实产形态（dotprod 实例 .opt.ll）：
#   %r = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %a, i32 %b, i32 %c)
#   %r = tail call i32 @llvm.nvvm.idp2a.s.s(i32 %a, i32 %b, i1 false, i32 %c)
# 语义（例源注释 + NVVM 命名，GPU 数值探针 DOTPROBE PASS，期望值取自例源）：
#   idp4a: d = c + Σ a.byte[i]*b.byte[i]（4×i8，小端打包）
#   idp2a: d = c + a.half0*b.byte[k] + a.half1*b.byte[k+1]
#          （a 为 2×i16；isbottom=false 取 b 低 2 字节，true 取高 2 字节）
# 映射：s.s → sdot4（v_dot4c_i32_i8）、u.u → udot4（v_dot4_u32_u8），clamp=false
# （PTX dp4a 无 clamp）；混合符号 sdot4 覆盖不了、idp2a 无对应单指令 → 乘加展开。


_IDP4A_RE = re.compile(
    r"^(?P<indent>\s*)(?:(?P<res>%[\w.$]+)\s*=\s*)?(?:tail )?call\s+i32\s+"
    r"@llvm\.nvvm\.idp4a\.(?P<sa>[su])\.(?P<sb>[su])\((?P<args>[^)]*)\)\s*(?:#\d+)?\s*$"
)

_IDP2A_RE = re.compile(
    r"^(?P<indent>\s*)(?:(?P<res>%[\w.$]+)\s*=\s*)?(?:tail )?call\s+i32\s+"
    r"@llvm\.nvvm\.idp2a\.(?P<sa>[su])\.(?P<sb>[su])\((?P<args>[^)]*)\)\s*(?:#\d+)?\s*$"
)


def _rewrite_dotprod(text: str) -> str:
    if "idp4a" not in text and "idp2a" not in text:
        return text
    lines = text.split("\n")
    out = []
    for line in lines:
        m4 = _IDP4A_RE.match(line)
        if m4:
            args = [a.strip() for a in _split_top_level_commas(m4.group("args"))]
            if len(args) == 3 and m4.group("sa") == m4.group("sb"):
                dot = "sdot4" if m4.group("sa") == "s" else "udot4"
                lhs = f"{m4.group('res')} = " if m4.group("res") else ""
                out.append(
                    f"{m4.group('indent')}{lhs}call i32 @llvm.amdgcn.{dot}"
                    f"({args[0]}, {args[1]}, {args[2]}, i1 false)"
                )
                continue
            # 混合符号 idp4a：保守留残（llc 显式失败），不用展开路径
            out.append(line)
            continue
        m2 = _IDP2A_RE.match(line)
        if m2:
            seq = _expand_idp2a(m2)
            out.extend(seq if seq is not None else [line])
            continue
        out.append(line)
    return "\n".join(out)


def _expand_idp2a(m) -> "list[str] | None":
    """idp2a.<sa>.<sb>(a, b, isbottom, c) → 提取+乘加直线 IR。

    a 的 2×i16 与 b 的 2×i8 均按符号后缀 sext/zext 后在 i32 域做乘加
    （与 v_dot2 语义一致且无溢出歧义）。isbottom 仅接受字面量（实产 immarg）。"""
    res, indent = m.group("res"), m.group("indent")
    args = [a.strip() for a in _split_top_level_commas(m.group("args"))]
    if len(args) != 4:
        return None
    a, b, isbot, c = (_arg_operand(x) for x in args)
    if isbot == "false":
        sh = 0
    elif isbot == "true":
        sh = 16
    else:
        return None  # 非字面量 select：保守不支持
    z = "zext" if m.group("sa") == "u" else "sext"
    bz = "zext" if m.group("sb") == "u" else "sext"
    if not res:
        return None  # 结果未使用的调用：保留残信号
    D = f"{res}.d_"  # SSA 名前缀（不含缩进；L = indent + D 用于定义行）
    L = indent + D
    seq = [
        f"{L}a0t = trunc i32 {a} to i16",
        f"{L}a0 = {z} i16 {D}a0t to i32",
        f"{L}a1s = lshr i32 {a}, 16",
        f"{L}a1t = trunc i32 {D}a1s to i16",
        f"{L}a1 = {z} i16 {D}a1t to i32",
    ]
    if sh:
        seq.append(f"{L}b0s = lshr i32 {b}, {sh}")
        seq.append(f"{L}b0t = trunc i32 {D}b0s to i8")
    else:
        seq.append(f"{L}b0t = trunc i32 {b} to i8")
    seq.append(f"{L}b0 = {bz} i8 {D}b0t to i32")
    seq.append(f"{L}b1s = lshr i32 {b}, {sh + 8}")
    seq.append(f"{L}b1t = trunc i32 {D}b1s to i8")
    seq.append(f"{L}b1 = {bz} i8 {D}b1t to i32")
    seq.append(f"{L}m0 = mul i32 {D}a0, {D}b0")
    seq.append(f"{L}m1 = mul i32 {D}a1, {D}b1")
    seq.append(f"{L}r0 = add i32 {c}, {D}m0")
    seq.append(f"{indent}{res} = add i32 {D}r0, {D}m1")
    return seq


# ---------------------------------------------------------------------------
# Step 14: counted barrier → LDS 计数器 + 世代自旋（软件回退）
# ---------------------------------------------------------------------------
#
# PTX 编号 barrier（bar.sync id, cnt / bar.arrive id, cnt）允许 CTA 子集参与；
# RDNA2 的 S_BARRIER 是无编号全组 barrier，直接映射会让旁观线程死锁。
# 回退实现（GPU 数值探针 CBPROBE PASS：producer/consumer + split arrive/sync，
# 128 线程 wave64）：
#   cnt[id]：到达计数；gen[id]：世代号。arrive = 计数+1，末到者清零计数并
#   世代+1 释放等待者；sync = 捕获世代 → arrive → 自旋等世代变化。
#   全 seq_cst LDS 原子；正确性优先，性能损耗（每 barrier O(1) LDS 原子 +
#   自旋轮次）如实记录。
# 前提（文档化假设）：使用 barrier 的 kernel 入口块被 CTA 全部线程一致执行
#   （CUDA 常规前提）——入口 init 后有一条全组 s.barrier 保证清零可见性。


_CB_CALL_RE = re.compile(
    r"^(?P<indent>\s*)(?:tail )?call\s+void\s+@llvm\.nvvm\.barrier\.cta\."
    r"(?P<kind>sync|arrive)\.count\(\s*i32\s+(?P<id>-?\d+)\s*,\s*"
    r"i32\s+(?P<n>\d+)\s*\)\s*(?:#\d+)?\s*$"
)

_CB_HELPERS = """

; [PORT gfx1030] 软件 counted barrier 状态槽（LDS 不可静态初始化 → undef，
; 由使用 barrier 的 kernel 入口清零，见各 kernel 的 cb-init 注释块）
@__port_cb_cnt = addrspace(3) global [16 x i32] undef, align 4
@__port_cb_gen = addrspace(3) global [16 x i32] undef, align 4

define internal void @__port_cb_arrive(i32 %id, i32 %n) nounwind {
entry:
  %cnt.p = getelementptr inbounds [16 x i32], ptr addrspace(3) @__port_cb_cnt, i32 0, i32 %id
  %old = atomicrmw add ptr addrspace(3) %cnt.p, i32 1 seq_cst
  %lastn = add i32 %n, -1
  %last = icmp eq i32 %old, %lastn
  br i1 %last, label %release, label %out

release:
  store atomic i32 0, ptr addrspace(3) %cnt.p seq_cst, align 4
  %gen.p = getelementptr inbounds [16 x i32], ptr addrspace(3) @__port_cb_gen, i32 0, i32 %id
  %g = load atomic i32, ptr addrspace(3) %gen.p seq_cst, align 4
  %g1 = add i32 %g, 1
  store atomic i32 %g1, ptr addrspace(3) %gen.p seq_cst, align 4
  ret void

out:
  ret void
}

define internal void @__port_cb_sync(i32 %id, i32 %n) convergent nounwind {
entry:
  %gen.p = getelementptr inbounds [16 x i32], ptr addrspace(3) @__port_cb_gen, i32 0, i32 %id
  %myg = load atomic i32, ptr addrspace(3) %gen.p seq_cst, align 4
  call void @__port_cb_arrive(i32 %id, i32 %n)
  br label %spin

spin:
  %g = load atomic volatile i32, ptr addrspace(3) %gen.p seq_cst, align 4
  %done = icmp ne i32 %g, %myg
  br i1 %done, label %out, label %spin

out:
  ret void
}
"""

_CB_INIT_MARKER = "; [PORT gfx1030] counted-barrier slot init"


def _rewrite_counted_barriers(text: str) -> str:
    """barrier.cta.sync/arrive.count(i32 <id>, i32 <n>) → helper 调用 +
    使用者 kernel 入口注入槽位清零（仅支持常量 id/n；非字面量保守留残）。"""
    if "barrier.cta.sync.count" not in text and "barrier.cta.arrive.count" not in text:
        return text

    lines = text.split("\n")
    out = []
    module_uses_cb = False
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i]
        if not (line.startswith("define ") and line.rstrip().endswith("{")):
            out.append(line)
            i += 1
            continue
        func_end = _find_function_end(lines, i)
        body = lines[i : func_end + 1]
        body, ids, used = _rewrite_cb_in_function(body)
        module_uses_cb = module_uses_cb or used
        out.extend(body)
        i = func_end + 1
    if module_uses_cb and "@__port_cb_cnt = addrspace(3) global" not in text:
        out.append(_CB_HELPERS)
    return "\n".join(out)


def _rewrite_cb_in_function(body: "list[str]") -> "tuple[list[str], set, bool]":
    """单函数内：改写 counted-barrier 调用点并注入入口 init。
    返回 (新函数体, 用到的 id 集, 是否有 counted-barrier 调用)。"""
    ids: "set[int]" = set()
    changed = False
    for k, line in enumerate(body):
        m = _CB_CALL_RE.match(line)
        if m:
            ids.add(int(m.group("id")))
            body[k] = (
                f"{m.group('indent')}call void "
                f"@__port_cb_{m.group('kind')}"
                f"(i32 {m.group('id')}, i32 {m.group('n')})"
            )
            changed = True
    if not ids:
        return body, ids, changed

    # 幂等：init 注释块已在 → 不重复注入
    if any(_CB_INIT_MARKER in ln for ln in body):
        return body, ids, changed

    # 注入点：入口 label（define 行后首个 "xxx:" 行）之后的连续 alloca 之后
    ins = 1
    while ins < len(body) and not body[ins].rstrip().endswith(":"):
        ins += 1
    ins += 1  # 越过 label 行
    while ins < len(body) and "= alloca " in body[ins]:
        ins += 1
    init = [_CB_INIT_MARKER + f" (ids: {','.join(map(str, sorted(ids)))})"]
    for cb_id in sorted(ids):
        init.append(
            f"  %__port_cb_i{cb_id}c = getelementptr inbounds [16 x i32], "
            f"ptr addrspace(3) @__port_cb_cnt, i32 0, i32 {cb_id}"
        )
        init.append(
            f"  store atomic i32 0, ptr addrspace(3) %__port_cb_i{cb_id}c seq_cst, align 4"
        )
        init.append(
            f"  %__port_cb_i{cb_id}g = getelementptr inbounds [16 x i32], "
            f"ptr addrspace(3) @__port_cb_gen, i32 0, i32 {cb_id}"
        )
        init.append(
            f"  store atomic i32 0, ptr addrspace(3) %__port_cb_i{cb_id}g seq_cst, align 4"
        )
    init.append("  call void @llvm.amdgcn.s.barrier()")
    return body[:ins] + init + body[ins:], ids, changed


# ---------------------------------------------------------------------------
# Step 15: libdevice __nv_* → ocml __ocml_*
# ---------------------------------------------------------------------------
#
# ROCm 的 ocml.bc（容器 /opt/rocm/amdgcn/bitcode/）在链接期由 llvm-link 并入
# module，符号名经 llvm-nm 实证：__ocml_<stem>_f32 / __ocml_<stem>_f64。
# NV libdevice 命名：__nv_<stem>f（f32）/ __nv_<stem>（f64）。下表只收录
# 双方命名一一对应且 ocml.bc 导出实证的 stem；表外 __nv_* 留残（链接期
# undefined symbol，显式失败信号）。GPU 数值探针 TANPROBE PASS（f32/f64
# tan 对 host libm ≤2 ULP）。

_OCML_STEMS = (
    # 三角/双曲
    "acos acos acosh asin asinh atan atan2 atanh cbrt cos cosh sin sinh tan tanh",
    # 指数/对数
    "exp exp2 exp10 expm1 log log2 log10 log1p",
    # 幂/取整/绝对值等
    "fabs floor ceil round trunc rint nearbyint fmod fmin fmax fdim fma pow "
    "hypot cbrt copysign fabs rsqrt sqrt erf erfc erfcx erfinv erfcinv tgamma "
    "lgamma ilogb ldexp scalbn frexp modf remainder remquo nextafter",
    # Bessel / 判定
    "j0 j1 y0 y1 i0 i1 isfinite isinf isnan signbit",
    # 复数（NV: __nv_cacosf …）
    "cabs cacos cacosh casin casinh catan catanh ccos ccosh cexp clog csin "
    "csinh csqrt ctan ctanh",
    # sincos（指针出参，签名一致）
    "sincos",
)
_OCML_STEM_SET = sorted(set(" ".join(_OCML_STEMS).split()))


def _rewrite_ocml(text: str) -> str:
    if "__nv_" not in text:
        return text
    for stem in _OCML_STEM_SET:
        # f32 变体在前（@__nv_<stem>f( 与 @__nv_<stem>( 因 "(" 终结互不误配）
        text = text.replace(f"@__nv_{stem}f(", f"@__ocml_{stem}_f32(")
        text = text.replace(f"@__nv_{stem}(", f"@__ocml_{stem}_f64(")
    return text


# ---------------------------------------------------------------------------
# Step 16: mbarrier → 软件状态机（RDNA2 无 mbarrier 硬件，N/A 清单实证）
# ---------------------------------------------------------------------------
#
# NV 语义（对象化同步原语）：mbarrier 是 LDS 里一个 64 位状态对象，arrive
# 原子递减计数，计数归零时相位（phase/parity）翻转并重置计数；wait 自旋等
# 相位变化；.noComplete 表示该次到达不得完成当前相位；inval 是生命周期清理。
#
# 软件等价设计（单字 i64 状态机，位域与 cuda-oxide Barrier 的硬件布局注释
# 同构——单字原子性消除双字 count/phase 分离方案在相位边界的竞态窗口）：
#   bit [31:0]  pending   当前未完成到达数
#   bit [47:32] expected  init 设定的期望到达数（≤ 2^16-1；CUDA block ≤ 1024，
#                         对 PTX 规范上限 2^20 的收窄，如实记录）
#   bit [63:48] phase     相位计数（16 位；token 即该字段值——比硬件 1 位
#                         parity 更强的比较，两相位回绕歧义不存在）
# 映射表（每个 intrinsic 一条；调用形态从 barrier 实产 .opt.ll 抓取）：
#   init.shared(ptr, n)   → 整字原子写 (n<<32)|n（phase=0）
#   arrive.shared(ptr)    → workgroup fence（release：先前的普通 LDS 写对
#                           等待者可见，Step 14/12 同款实证模式）+
#                           atomicrmw sub 1；唯一观察到 pending==1 的线程
#                           补一次完成写（重置 pending=expected、phase+1）
#                           ——valid usage 下完成者唯一（硬件单字 RMW 的
#                           语义由 atomicrmw 保住）；token=旧相位字段
#   arrive.noComplete(n)  → atomicrmw sub n（不补完成写；前提 count <
#                           pending，与 PTX valid usage 一致，越界 = UB 同 NV）
#   test_wait(PTX asm)    → 自旋 load atomic volatile i64 比较相位字段 ≠
#                           token，返回 i32 0/1（与 asm 形态结果类型一致，
#                           下游 trunc-to-i1 与 and-1 两种用法都兼容）
#   inval.shared(ptr)     → 删除调用行（no-op：软件 init 整字重写，inval 的
#                           "下轮 init 前失效" 语义由之覆盖）
# 语义边界（如实记录）：不支持 expect_tx 字节级事务计数（TMA 场景需另行
# 设计）；RDNA2 无异步拷贝，vmcnt 部分排空（tma-like.md）是事务等待的
# 正路，mbarrier 在此只承担到达同步。
# 编译探针（gfx1030 llc 实证）：ds_sub_rtn_u64 / ds_write_b64 / ds_read_b64
# / s_barrier / fence→s_waitcnt lgkmcnt(0) 全部接受（/tmp/probe_mb 探针）。
# GPU 数值验证：barrier 例 3 kernel（全组同步 / LDS 邻居读 / noComplete
# 计数语义）E2E PASS，20 次稳定性全过。
#
# 实产调用形态（barrier 例 .opt.ll，逐字抓取）：
#   tail call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @SYM, i32 %n) #4
#   %t = tail call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) @SYM) #4
#   %t = tail call i64 @llvm.nvvm.mbarrier.arrive.noComplete.shared(
#          ptr addrspace(3) @SYM, i32 %n) #4
#   %r = tail call i32 asm sideeffect "{ .reg .pred %p0;
#          mbarrier.test_wait.shared.b64 %p0, [$1], $2;
#          selp.b32 $0, 1, 0, %p0; }", "=r,l,l,~{memory}"(
#          ptr addrspace(3) @SYM, i64 %t) #3
#   tail call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @SYM) #4

_MB_PTR_OP = r"ptr addrspace\(3\)\s+(?:(?:nonnull|noundef)\s+)*(?P<ptr>[@%][\w.$]+)"

_MB_INIT_RE = re.compile(
    r"^(?P<indent>\s*)(?:tail )?call\s+void\s+@llvm\.nvvm\.mbarrier\.init\.shared\("
    r"\s*" + _MB_PTR_OP + r"\s*,\s*i32\s+(?P<count>[^,)]+?)\s*\)\s*(?:#\d+)?\s*$"
)

_MB_ARRIVE_RE = re.compile(
    r"^(?P<indent>\s*)(?:(?P<res>%[\w.$]+)\s*=\s*)?(?:tail )?call\s+i64\s+"
    r"@llvm\.nvvm\.mbarrier\.arrive\.shared\(\s*" + _MB_PTR_OP
    + r"\s*\)\s*(?:#\d+)?\s*$"
)

_MB_ARRIVE_NC_RE = re.compile(
    r"^(?P<indent>\s*)(?:(?P<res>%[\w.$]+)\s*=\s*)?(?:tail )?call\s+i64\s+"
    r"@llvm\.nvvm\.mbarrier\.arrive\.noComplete\.shared\(\s*" + _MB_PTR_OP
    + r"\s*,\s*i32\s+(?P<count>[^,)]+?)\s*\)\s*(?:#\d+)?\s*$"
)

_MB_INVAL_RE = re.compile(
    r"^(?P<indent>\s*)(?:tail )?call\s+void\s+@llvm\.nvvm\.mbarrier\.inval\.shared\("
    r"\s*" + _MB_PTR_OP + r"\s*\)\s*(?:#\d+)?\s*$"
)

# test_wait 不是 intrinsic 而是 PTX 内嵌 asm（cuda-oxide 生成路径实证），
# 结果 i32 0/1 经 "=r,l,l,~{memory}"
_MB_TEST_WAIT_ASM_RE = re.compile(
    r'^(?P<indent>\s*)(?:(?P<res>%[\w.$]+)\s*=\s*)?(?:tail )?call\s+i32\s+asm\s+'
    r'sideeffect\s+"\{ \.reg \.pred %p0; mbarrier\.test_wait\.shared\.b64 %p0, '
    r'\[\$1\], \$2; selp\.b32 \$0, 1, 0, %p0; \}",\s*"=r,l,l,~\{memory\}"\('
    r"\s*" + _MB_PTR_OP + r"\s*,\s*i64\s+(?P<tok>[^,)]+?)\s*\)\s*(?:#\d+)?\s*$"
)

_MB_HELPERS = """

; [PORT gfx1030] 软件 mbarrier 状态机 helper（状态住对象自身 8 字节，无新增
; LDS 全局量；位域注释见各函数头。LDS 不可静态初始化 → 对象初始为 undef，
; 首次 init 整字写入后才有定义——与 NV "init 必须先于任何 arrive/wait 且须
; 有组内同步发布" 契约一致）

define internal void @__port_mb_init(ptr addrspace(3) %bar, i32 %count) nounwind {
; state = {pending: count, expected: count, phase: 0}
entry:
  %c64 = zext i32 %count to i64
  %c64.hi = shl i64 %c64, 32
  %state = or i64 %c64.hi, %c64
  store atomic i64 %state, ptr addrspace(3) %bar seq_cst, align 8
  ret void
}

define internal i64 @__port_mb_arrive(ptr addrspace(3) %bar) nounwind {
; release 栅栏 + 递减；唯一 pending==1 观察者补完成写；token = 旧相位
entry:
  fence syncscope("workgroup") seq_cst
  %old = atomicrmw sub ptr addrspace(3) %bar, i64 1 seq_cst
  %pend = and i64 %old, 4294967295
  %last = icmp eq i64 %pend, 1
  br i1 %last, label %release, label %out

release:
  %exp.sh = lshr i64 %old, 32
  %exp = and i64 %exp.sh, 65535
  %ph.sh = lshr i64 %old, 48
  %ph = and i64 %ph.sh, 65535
  %ph1 = add i64 %ph, 1
  %ph1.sh = shl i64 %ph1, 48
  %exp.sh2 = shl i64 %exp, 32
  %t = or i64 %ph1.sh, %exp.sh2
  %new = or i64 %t, %exp
  store atomic i64 %new, ptr addrspace(3) %bar seq_cst, align 8
  br label %out

out:
  %tok.sh = lshr i64 %old, 48
  %tok = and i64 %tok.sh, 65535
  ret i64 %tok
}

define internal i64 @__port_mb_arrive_nc(ptr addrspace(3) %bar, i32 %count) nounwind {
; .noComplete：只递减、永不补完成写（count < pending 为 PTX valid usage）
entry:
  fence syncscope("workgroup") seq_cst
  %c64 = zext i32 %count to i64
  %old = atomicrmw sub ptr addrspace(3) %bar, i64 %c64 seq_cst
  %tok.sh = lshr i64 %old, 48
  %tok = and i64 %tok.sh, 65535
  ret i64 %tok
}

define internal i32 @__port_mb_test_wait(ptr addrspace(3) %bar, i64 %token) nounwind {
; 非阻塞单次测试（PTX mbarrier.test_wait 语义 = 谓词而非等待；阻塞语义由
; 调用方的 while(!test_wait) 循环承担——Rust mbarrier_wait 的源码形态）。
; volatile + seq_cst：外提禁令（helper 被内联时防自旋读被提出循环）+ acquire。
; GPU 实证教训：首版把 helper 写成内部自旋，noComplete 分裂模式
; （test_wait 期望立即返回 false）死锁——谓词必须非阻塞。
entry:
  %cur = load atomic volatile i64, ptr addrspace(3) %bar seq_cst, align 8
  %cur.sh = lshr i64 %cur, 48
  %cur.ph = and i64 %cur.sh, 65535
  %done = icmp ne i64 %cur.ph, %token
  %done32 = zext i1 %done to i32
  ret i32 %done32
}
"""


def _rewrite_mbarriers(text: str) -> str:
    """mbarrier 五种实产形态 → 软件状态机 helper 调用（映射表见区块注释）。

    inval 调用行直接删除；其余替换为 @__port_mb_* 调用并保持 SSA 结果名
    不变（下游引用零改动）。幂等：改写产物不含任何被匹配形态。"""
    if "mbarrier" not in text:
        return text

    lines = text.split("\n")
    out = []
    changed = False
    for line in lines:
        m = _MB_TEST_WAIT_ASM_RE.match(line)
        if m:
            res = f"{m.group('res')} = " if m.group("res") else ""
            out.append(
                f"{m.group('indent')}{res}call i32 @__port_mb_test_wait("
                f"ptr addrspace(3) {m.group('ptr')}, i64 {m.group('tok')})"
            )
            changed = True
            continue
        m = _MB_INIT_RE.match(line)
        if m:
            out.append(
                f"{m.group('indent')}call void @__port_mb_init("
                f"ptr addrspace(3) {m.group('ptr')}, i32 {m.group('count')})"
            )
            changed = True
            continue
        m = _MB_ARRIVE_NC_RE.match(line)
        if m:
            res = f"{m.group('res')} = " if m.group("res") else ""
            out.append(
                f"{m.group('indent')}{res}call i64 @__port_mb_arrive_nc("
                f"ptr addrspace(3) {m.group('ptr')}, i32 {m.group('count')})"
            )
            changed = True
            continue
        m = _MB_ARRIVE_RE.match(line)
        if m:
            res = f"{m.group('res')} = " if m.group("res") else ""
            out.append(
                f"{m.group('indent')}{res}call i64 @__port_mb_arrive("
                f"ptr addrspace(3) {m.group('ptr')})"
            )
            changed = True
            continue
        if _MB_INVAL_RE.match(line):
            continue  # no-op：删除调用行（语义论证见区块注释）
        out.append(line)

    if changed and "@__port_mb_init(ptr addrspace(3)" not in text:
        out.append(_MB_HELPERS)
    return "\n".join(out)


def _append_ntid_kernarg_param(text: str) -> str:
    """给调用了 ntid.x 的 kernel 签名末尾追加 `i32 %ntid_x` 参数。"""
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("define ") and line.rstrip().endswith("{"):
            block = [line]
            i += 1
            while i < len(lines) and lines[i] != "}":
                block.append(lines[i])
                i += 1
            if i < len(lines):  # 收尾的 "}" 行
                block.append(lines[i])
                i += 1
            body = "\n".join(block)
            # 仅当函数体用到 %ntid_x 且签名尚未含该参数时追加
            if "%ntid_x" in body and "%ntid_x" not in block[0]:
                # 参数表收尾 ")" = "{" 之前最后一个 ")"（")" 与 "{" 之间只允许
                # 空白/flags（alwaysinline 等）/属性引用 #N——参数内的 "())"
                # （如 captures(none)）必然位于收尾 ")" 之前
                brace = block[0].rfind("{")
                paren = block[0].rfind(")", 0, brace)
                between = block[0][paren + 1 : brace]
                if paren >= 0 and re.fullmatch(r"[\s#\w,=]*", between):
                    block[0] = (
                        block[0][:paren] + ", i32 %ntid_x" + block[0][paren:]
                    )
            out.extend(block)
        else:
            out.append(line)
            i += 1
    return "\n".join(out)
