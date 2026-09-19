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
        r"|amdgcn\.work(?:group|item)\.id\.x"
        r")\("
    )
    text = re.sub(rf"declare [^\n]*{mapped}[^\n]*\n", "", text)

    # 8. alloca → addrspace(5)（AMDGPU module verifier：本地帧必须 scratch）
    text = _rewrite_allocas(text)

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
