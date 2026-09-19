#!/usr/bin/env python3
# [PORT gfx1030] Phase 1 验收 3：gfx1030 移植回归测试。
#
# 固化的端到端链路（全部命令均来自 phase1.md 已验证链，勿改动语义）：
#   纯 Rust #[kernel] vecadd
#     → cargo oxide build（外部项目模式，device-only 依赖）→ <crate>.ll / .opt.ll / .ptx
#     → translate_nvvm_to_amdgcn() 7 步 IR 改写
#         1. target triple → "amdgcn-amd-amdhsa"
#         2. target datalayout → AMDGPU amdgcn 布局
#         3. ptx_kernel → amdgpu_kernel（调用约定即 kernel 标记）
#         4. llvm.nvvm.read.ptx.sreg.ctaid.x → llvm.amdgcn.workgroup.id.x
#            llvm.nvvm.read.ptx.sreg.tid.x   → llvm.amdgcn.workitem.id.x
#         5. ntid.x → kernarg 新参数 i32 %ntid_x（blockDim 由 host 决定）
#         6. ntid/nctaid .y|.z → "or i32 0, 1" 常量化（1D launch 语义）
#         7. 删除已映射 intrinsic 的 nvvm declare（llc 自动声明 amdgcn 目标 intrinsic）
#     → llc -march=amdgcn -mcpu=gfx1030 -amdhsa-code-object-version=5 --filetype=obj
#     → ld.lld -shared（hipModuleLoadData 需要 linked shared object，不是 relocatable）
#     → hipModuleLoadData + hipModuleGetFunction("vecadd") + hipModuleLaunchKernel
#       （kparams = 每参数一指针 ×7，不是指向 kernarg 结构体的单指针）
#     → 数值校验 "PASS: all 1024 elements correct"
#
# 运行前提（本仓库 gfx1030 分支的验收环境）：
#   - 宿主 zhuo-ms：ROCm GPU gfx1030（RX 6950 XT）空闲可用（跑前 rocm-smi 自检）
#   - rocminfo（设备识别测试用：解析 gfx1030 agent / RX 6950 XT Marketing Name）
#   - rustup nightly-2026-08-28（含 rustc-dev / llvm-tools；llc 用工具链自带）
#     以及 ~/.cargo/bin/cargo-oxide 二进制（cargo oxide 子命令的真正落点；
#     探针 crate 在 /tmp 下构建时仓库 .cargo/config.toml alias 不生效）
#   - ~/opt/cuda13（CUDA_TOOLKIT_PATH；device-only 构建不触发 toolkit 探测，仅为与
#     phase1 验收命令保持一致）
#   - docker 容器 zhuo：hipcc + ld.lld + ROCm runtime（GPU 验证在其中执行）
#   - pytest：pip3 install --user pytest
#
# 运行：
#   cd /home/zhuo/workspace/src/cuda-oxide
#   python3 -m pytest tests/test_gfx1030.py -v          # 全部（含 GPU 端到端）
#   python3 -m pytest tests/test_gfx1030.py::test_ir_translate_no_nvvm_residual -v
#                                                       # 仅纯文本测试（无 GPU/构建）

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# [PORT gfx1030] 改写器已抽到共享模块（测试与批量脚本单一事实源）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate_amdgcn import (  # noqa: E402
    AMD_DATALAYOUT,
    AMD_TRIPLE,
    translate_nvvm_to_amdgcn,
)

# ---------------------------------------------------------------------------
# 环境常量（phase1.md 已验证链路的落点）
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
NIGHTLY = "nightly-2026-08-28"
CUDA_TOOLKIT_PATH = os.path.expanduser("~/opt/cuda13")

RUSTLIB_BIN = (
    Path.home()
    / ".rustup"
    / f"toolchains/{NIGHTLY}-x86_64-unknown-linux-gnu"
    / "lib/rustlib/x86_64-unknown-linux-gnu/bin"
)
LLC = RUSTLIB_BIN / "llc"

DOCKER_CONTAINER = "zhuo"
CONTAINER_WORKDIR = "/home/ubuntu/tmp-prof/oxtest"  # 容器内工作目录（phase1 目录约定）


# ---------------------------------------------------------------------------
# 探针 crate 模板（复刻 crates/rustc-codegen-cuda/examples/ox-amdgcn-probe：
# 只依赖 cuda-device —— 纯 proc-macro 依赖，构建完全不碰 cuda-bindings/CUDA toolkit）
# ---------------------------------------------------------------------------

PROBE_MAIN_RS = """\
//! [PORT gfx1030] 回归测试探针: vecadd kernel 的 device 产物(.ll)用于
//! amdgcn 后端验证(llc -march=amdgcn -> hsaco -> hipModuleLoadData)。

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

fn main() {
    println!("probe host stub");
}
"""


def _write_probe_crate(crate_dir: Path) -> None:
    crate_dir.mkdir(parents=True, exist_ok=True)
    (crate_dir / "Cargo.toml").write_text(
        f"""\
[package]
name = "ox-gfx1030-test"
version = "0.1.0"
edition = "2024"
publish = false

# [PORT gfx1030] device-only 探针: 刻意不依赖 cuda-core/cuda-host(即
# cuda-bindings), 构建不触发 CUDA toolkit 探测。

[dependencies]
cuda-device = {{ path = "{REPO_ROOT / 'crates' / 'cuda-device'}" }}

[workspace]
"""
    )
    (crate_dir / "src").mkdir(exist_ok=True)
    (crate_dir / "src" / "main.rs").write_text(PROBE_MAIN_RS)


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ---------------------------------------------------------------------------
# fixture：一次性构建探针 crate（module 级共享，构建是最慢的一步）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_probe(tmp_path_factory):
    crate_dir = tmp_path_factory.mktemp("gfx1030-probe")
    _write_probe_crate(crate_dir)

    env = dict(os.environ, CUDA_TOOLKIT_PATH=CUDA_TOOLKIT_PATH)
    proc = _run(["cargo", f"+{NIGHTLY}", "oxide", "build"], cwd=crate_dir, env=env)
    assert proc.returncode == 0, (
        f"[阶段 .ll-产出: cargo oxide build] 失败 (rc={proc.returncode})\n"
        f"--- 关键错误输出（尾20行）---\n{_tail(proc.stdout + proc.stderr)}\n"
        f"--- 环境前提状态 ---\n{_env_diag()}"
    )

    opt_ll = sorted(crate_dir.glob("*.opt.ll"))
    ll = sorted(p for p in crate_dir.glob("*.ll") if not p.name.endswith(".opt.ll"))
    assert ll, f"未找到 .ll 产物，目录内容: {list(crate_dir.iterdir())}"
    ptx = sorted(crate_dir.glob("*.ptx"))

    class Probe:
        pass

    p = Probe()
    p.dir = crate_dir
    p.ll = ll[0]
    p.opt_ll = opt_ll[0] if opt_ll else None
    p.ptx = ptx[0] if ptx else None
    p.build_log = proc.stdout + proc.stderr
    return p


def _gpu_available() -> bool:
    return _run(["rocm-smi"]).returncode == 0


def _docker_available() -> bool:
    return _run(["docker", "info"]).returncode == 0 and _run(
        ["docker", "exec", DOCKER_CONTAINER, "true"]
    ).returncode == 0


# ---------------------------------------------------------------------------
# 失败诊断助手：每个阶段失败时，断言消息 = 阶段名 + 关键错误输出（尾20行）
#                + 已确认的环境前提状态，让人不看代码就知道死在哪层。
# ---------------------------------------------------------------------------


def _tail(text: str, n: int = 20) -> str:
    """关键错误输出的尾 n 行。"""
    return "\n".join((text or "").splitlines()[-n:])


def _env_diag() -> str:
    """已确认的环境前提状态快照（工具链/容器/GPU）。仅在失败分支调用。"""
    llc_ok = LLC.exists()
    cargo = _run(["cargo", f"+{NIGHTLY}", "--version"])
    docker = _run(["docker", "exec", DOCKER_CONTAINER, "true"])
    gpu = _run(["rocm-smi"])
    return "\n".join(
        [
            f"工具链 llc: {'OK' if llc_ok else '缺失'} ({LLC})",
            (
                f"工具链 cargo {NIGHTLY}: "
                + (f"OK ({cargo.stdout.strip()})" if cargo.returncode == 0
                   else f"FAIL: {_tail(cargo.stderr, 5)}")
            ),
            (
                f"容器 {DOCKER_CONTAINER}: "
                + ("OK" if docker.returncode == 0
                   else f"FAIL: {_tail(docker.stderr, 5)}")
            ),
            (
                "rocm-smi: "
                + ("OK" if gpu.returncode == 0
                   else f"FAIL: {_tail(gpu.stderr, 5)}")
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 测试 1：纯文本改写断言（不需要 GPU / 不需要构建，秒级）
# ---------------------------------------------------------------------------

# 真实 .opt.ll 结构快照（截自 ox-amdgcn-probe 产物，覆盖全部 7 步改写点）
SAMPLE_NVVM_IR = """\
; ModuleID = 'sample'
source_filename = "ox_amdgcn_probe"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [1 x ptr] [ptr @vecadd], section "llvm.metadata"

declare void @llvm.trap() #0

define ptx_kernel void @vecadd(ptr nofree nonnull readonly align 4 captures(none) %v0, i64 %v1, ptr nofree nonnull readonly align 4 captures(none) %v2, i64 %v3, ptr nofree writeonly captures(address_is_null) %v4, i64 %v5) #1 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %v9.i = zext nneg i32 %v2.i to i64
  %v10.i = zext nneg i32 %v3.i to i64
  %v17.i = mul nuw nsw i64 %v9.i, %v10.i
  %v11.i = zext nneg i32 %v4.i to i64
  %v18.i = add nuw nsw i64 %v17.i, %v11.i
  %v4.i2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #3
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #3
  %v5.i = icmp eq i32 %v4.i2, 1
  %v7.i = icmp eq i32 %v6.i, 1
  br label %exit

exit:
  ret void
}

declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #2

attributes #0 = { cold noreturn nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nocallback nofree nounwind willreturn memory(none) }
attributes #3 = { convergent }
"""


def test_ir_translate_no_nvvm_residual():
    """改写后文本不得残留任何 llvm.nvvm 引用，且 7 步语义点全部就位。"""
    out = translate_nvvm_to_amdgcn(SAMPLE_NVVM_IR)

    # 核心断言：无 nvvm 残留
    assert "llvm.nvvm" not in out, "改写后仍有 llvm.nvvm 残留"

    # 7 步逐项验证
    assert f'target triple = "{AMD_TRIPLE}"' in out, "triple 未替换"
    assert f'target datalayout = "{AMD_DATALAYOUT}"' in out, "datalayout 未替换"
    assert "amdgpu_kernel void @vecadd" in out, "ptx_kernel 未改为 amdgpu_kernel"
    assert "ptx_kernel" not in out
    assert "@llvm.amdgcn.workgroup.id.x()" in out, "ctaid.x 未映射"
    assert "@llvm.amdgcn.workitem.id.x()" in out, "tid.x 未映射"
    assert "or i32 %ntid_x, 0" in out, "ntid.x 未 kernarg 化"
    assert re.search(r"i64 %v5, i32 %ntid_x\) #1 \{", out), "签名未追加 i32 %ntid_x"
    assert out.count("or i32 0, 1") == 2, "y/z 维未常量化为 1"
    # 已映射 intrinsic 的 declare 已删，且未误删无关 declare
    assert "@llvm.trap" in out

    # 幂等性：对改写产物再跑一遍，文本不变（禁止多轮补丁纪律的机器断言）
    assert translate_nvvm_to_amdgcn(out) == out, "改写函数不幂等"


# Step 8（alloca addrspace(5)）探针 fixture：数组 + count 操作数两种形态
SAMPLE_ALLOCA_IR = """\
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @alloca_probe(ptr %v0, i64 %v1, ptr %v2, i64 %v3) #1 {
entry:
  %buf = alloca [64 x float], align 4
  %slot = alloca float, i32 1, align 4
  %pa = getelementptr inbounds [4 x i8], ptr %v0, i64 %v1
  %t = load float, ptr %pa, align 4
  %pb = getelementptr inbounds [64 x float], ptr %buf, i64 0, i64 %v1
  store float %t, ptr %pb, align 4
  %ps = getelementptr inbounds float, ptr %slot, i64 0
  store float %t, ptr %ps, align 4
  %u = load float, ptr %ps, align 4
  %pc = getelementptr inbounds [4 x i8], ptr %v2, i64 %v1
  store float %u, ptr %pc, align 4
  ret void
}
"""


def test_ir_translate_alloca_addrspace():
    """Step 8：generic alloca → addrspace(5)，use 点插 addrspacecast（GPU 数值
    已验证的探针契约：c[i] = 2*a[i] 经 scratch 槽往返）。"""
    out = translate_nvvm_to_amdgcn(SAMPLE_ALLOCA_IR)

    # 定义行：数组与 count 操作数两种形态，align 前置、addrspace(5) 收尾
    assert "= alloca [64 x float], align 4, addrspace(5)" in out
    assert "= alloca float, i32 1, align 4, addrspace(5)" in out
    # 每个 alloca 一条 cast，插在定义行后
    assert "%buf.ac = addrspacecast ptr addrspace(5) %buf to ptr" in out
    assert "%slot.ac = addrspacecast ptr addrspace(5) %slot to ptr" in out
    # 所有 use 点（gep/load/store）换到 cast 名，不再引用 generic 槽
    assert re.search(r"getelementptr inbounds \[64 x float\], ptr %buf\.ac", out)
    assert "ptr %slot.ac, i64 0" in out
    assert not re.search(r", ptr %buf[,)\s]", out), "alloca 仍有未降级的 generic use"
    # 已在 addrspace(5) 的 alloca 不再被改（幂等前提）
    assert translate_nvvm_to_amdgcn(out) == out, "alloca 改写不幂等"

    # typed-pointer IR（math_tan 类非 opt 形态）显式跳过：不产生半改写产物
    typed_ir = (
        'target triple = "nvptx64-nvidia-cuda"\n'
        "define void @f(i8* %v0) {\n"
        "entry:\n"
        "  %v41 = alloca {  }, align 1\n"
        "  %v10 = bitcast {  }* %v41 to i8*\n"
        "  ret void\n"
        "}\n"
    )
    out3 = translate_nvvm_to_amdgcn(typed_ir)
    assert "= alloca {  }, align 1" in out3, "typed-IR 的 alloca 被半改写"
    assert "addrspacecast" not in out3


# Step 9（barrier → s.barrier + LDS undef 初始化）探针 fixture
SAMPLE_BARRIER_IR = """\
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@tile = addrspace(3) global [256 x float] zeroinitializer, align 4

declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

define ptx_kernel void @shared_test(ptr %v0, i64 %v1, ptr %v2, i64 %v3) #1 {
entry:
  %tid = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %gep = getelementptr inbounds [256 x float], ptr addrspace(3) @tile, i64 0, i64 0
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #4
  tail call void @llvm.nvvm.barrier0() #4
  ret void
}
"""


def test_ir_translate_barrier():
    """Step 9：__syncthreads → s.barrier；LDS 全局量 zeroinit → undef
    （GPU 数值已验证：sharedmem 邻居读 256/256 PASS）。"""
    out = translate_nvvm_to_amdgcn(SAMPLE_BARRIER_IR)

    assert "call void @llvm.amdgcn.s.barrier()" in out, "barrier.cta 未映射"
    assert "llvm.nvvm.barrier" not in out, "barrier intrinsic 残留"
    assert "undef, align 4" in out, "LDS zeroinitializer 未去初始化"
    assert "zeroinitializer" not in out
    # s.barrier 的 declare 须被删（llc 自动声明）；幂等
    assert "declare void @llvm.amdgcn.s.barrier" not in out
    assert translate_nvvm_to_amdgcn(out) == out, "barrier 改写不幂等"


# Step 10（warp shuffle → ds_bpermute）探针 fixture
SAMPLE_SHFL_IR = """\
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32) #3
declare float @llvm.nvvm.shfl.sync.down.f32(i32, float, i32, i32) #3
declare float @llvm.nvvm.shfl.sync.idx.f32(i32, float, i32, i32) #3
declare float @llvm.nvvm.shfl.sync.up.f32(i32, float, i32, i32) #3
declare i32 @llvm.nvvm.read.ptx.sreg.laneid() #2

define ptx_kernel void @warp_probe(ptr %v0, i64 %v1) #1 {
entry:
  %lane = tail call i32 @llvm.nvvm.read.ptx.sreg.laneid() #3
  %val = load float, ptr %v0, align 4
  %bf = tail call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float %val, i32 1, i32 31) #3
  %dn = tail call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %val, i32 2, i32 31) #3
  %ix = tail call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float %val, i32 0, i32 31) #3
  %up = tail call float @llvm.nvvm.shfl.sync.up.f32(i32 -1, float %val, i32 1, i32 31) #3
  %s = fadd float %bf, %dn
  %s2 = fadd float %s, %ix
  %s3 = fadd float %s2, %up
  store float %s3, ptr %v0, align 4
  ret void
}

define ptx_kernel void @f64_probe(ptr %v0, i64 %v1) #1 {
entry:
  %v = load i64, ptr %v0, align 8
  %bf64 = tail call i64 asm sideeffect "{ .reg .b32 lo; .reg .b32 hi; mov.b64 {lo, hi}, $1; shfl.sync.bfly.b32 lo, lo, $2, 31, $3; shfl.sync.bfly.b32 hi, hi, $2, 31, $3; mov.b64 $0, {lo, hi}; }", "=l,l,r,r"(i64 %v, i32 1, i32 -1) #3
  store i64 %bf64, ptr %v0, align 8
  ret void
}
"""


def test_ir_translate_shuffle():
    """Step 10/10c：shfl → ds_bpermute（*4 字节偏移）、laneid → mbcnt.lo、
    f64 PTX-asm 拆分 → 双 ds_bpermute（GPU 数值已验证四模式 + i32 + f64）。"""
    out = translate_nvvm_to_amdgcn(SAMPLE_SHFL_IR)

    assert "llvm.nvvm" not in out, "shfl/laneid intrinsic 残留"
    # laneid → mbcnt.lo；shfl 结果写回原 SSA 名
    assert "%lane = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)" in out
    # NV lane 号 → AMD 字节偏移：shl src, 2（f32×4 + f64 拆分的 off 共 5）
    assert out.count("shl i32") == 5, "字节偏移转换缺失"
    assert out.count("@llvm.amdgcn.ds.bpermute") == 4 + 2, "f32×4 + f64 拆分×2"
    # down 的段外回自身：select + icmp ule
    assert "icmp ule i32" in out
    assert "select i1" in out
    # f64 asm：lo/hi 双通道 + 重组
    assert "trunc i64 %v to i32" in out
    assert "or i64" in out
    assert "shfl.sync.bfly.b32" not in out, "PTX asm 残留"
    assert translate_nvvm_to_amdgcn(out) == out, "shuffle 改写不幂等"


# Step 11（redux.sync.add → 蝶形回退）探针 fixture
SAMPLE_REDUX_IR = """\
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.redux.sync.add(i32, i32) #3

define ptx_kernel void @redux_probe(ptr %v0, i64 %v1) #1 {
entry:
  %v = load i32, ptr %v0, align 4
  %r1 = tail call i32 @llvm.nvvm.redux.sync.add(i32 %v, i32 -1) #3
  %r2 = tail call i32 @llvm.nvvm.redux.sync.add(i32 1, i32 -1) #3
  %s = add i32 %r1, %r2
  store i32 %s, ptr %v0, align 4
  ret void
}
"""


def test_ir_translate_redux_add():
    """Step 11：redux.sync.add → 5 轮 xor-butterfly ds_bpermute 回退
    （gfx1030 无硬件 redux；GPU 数值已验证：全 warp 和 496 + 广播语义）。"""
    out = translate_nvvm_to_amdgcn(SAMPLE_REDUX_IR)

    assert "llvm.nvvm" not in out, "redux intrinsic 残留"
    # 两处调用点 × 5 轮 ds_bpermute
    assert out.count("@llvm.amdgcn.ds.bpermute") == 10
    # xor 蝶形 mask 序列
    assert "xor i32" in out
    # 结果写回原 SSA 名（下游 %s 引用不动）
    assert "%s = add i32 %r1, %r2" in out
    assert translate_nvvm_to_amdgcn(out) == out, "redux 改写不幂等"


# Step 12（isspacep/membar/syncscope → AMD 地址域语义）探针 fixture
SAMPLE_ATOMICS_IR = """\
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i1 @llvm.nvvm.isspacep.local(ptr captures(none)) #0
declare void @llvm.nvvm.membar.gl() #2
declare void @llvm.nvvm.membar.sys() #2

define ptx_kernel void @atom_probe(ptr %buf, ptr %out) #1 {
entry:
  %loc = alloca i32, align 4
  %loc.ac = addrspacecast ptr addrspace(5) %loc to ptr
  %isp = tail call i1 @llvm.nvvm.isspacep.local(ptr %buf) #8
  br i1 %isp, label %priv, label %glob

priv:
  %pp = addrspacecast ptr %buf to ptr addrspace(5)
  br label %done

glob:
  %r = atomicrmw add ptr %buf, i32 1 syncscope("device") monotonic, align 4
  tail call void @llvm.nvvm.membar.gl() #8
  tail call void @llvm.nvvm.membar.sys() #8
  br label %done

done:
  ret void
}

define ptx_kernel void @atom_asm_probe(ptr %buf, ptr %out) #1 {
entry:
  ; Rust core::sync::atomic 的 PTX asm 降级形态（实产抓取）
  tail call void asm sideeffect "fence.acq_rel.cta;", "~{memory}"() #9
  tail call void asm sideeffect "fence.acq_rel.gpu;", "~{memory}"() #9
  tail call void asm sideeffect "fence.acq_rel.sys;", "~{memory}"() #9
  %la = tail call i32 asm sideeffect "ld.acquire.gpu.b32 $0, [$1];", "=r,l,~{memory}"(ptr %buf) #9
  %lb = tail call i64 asm sideeffect "ld.acquire.sys.b64 $0, [$1];", "=l,l,~{memory}"(ptr %buf) #9
  %lc = call ptr asm sideeffect "fence.sc.sys; ld.acquire.sys.b64 $0, [$1];", "=l,l,~{memory}"(ptr %buf) #9
  tail call void asm sideeffect "st.release.gpu.b32 [$0], $1;", "l,r,~{memory}"(ptr %buf, i32 %la) #9
  tail call void asm sideeffect "st.release.sys.b64 [$0], $1;", "l,l,~{memory}"(ptr %buf, ptr %out) #9
  ret void
}
"""


def test_ir_translate_atomics():
    """Step 12：isspacep.local → is.private、membar.gl/sys → agent/system fence、
    NV scope 名（device/block）→ AMD scope 名（agent/workgroup）、PTX asm 原子族
    （fence/acquire load/seqcst load/release store）→ LLVM 原子指令。
    （GPU 数值已验证：is.private 判定、generic fetch_add + fence、MP litmus
    acquire/release 顺序 PASS。）"""
    out = translate_nvvm_to_amdgcn(SAMPLE_ATOMICS_IR)

    assert "llvm.nvvm" not in out, "isspacep/membar intrinsic 残留"
    assert "@llvm.amdgcn.is.private(ptr %buf)" in out, "isspacep.local 未映射"
    assert 'fence syncscope("agent") seq_cst' in out, "membar.gl 未映射"
    assert "\n  fence seq_cst" in out, "membar.sys 未映射"
    assert 'atomicrmw add ptr %buf, i32 1 syncscope("agent") monotonic' in out, (
        "atomic scope 名未换"
    )
    # NV scope 名全部换掉
    assert 'syncscope("device")' not in out
    assert 'syncscope("block")' not in out
    # PTX asm 原子族 → LLVM 原子指令（asm 不残留）
    assert "asm sideeffect" not in out, "PTX asm 原子残留"
    assert 'fence syncscope("workgroup") acq_rel' in out, "fence.acq_rel.cta 未映射"
    assert 'fence syncscope("agent") acq_rel' in out, "fence.acq_rel.gpu 未映射"
    assert "\n  fence acq_rel" in out, "fence.acq_rel.sys 未映射"
    assert "%la = load atomic i32, ptr %buf syncscope(\"agent\") acquire, align 4" in out
    assert "%lb = load atomic i64, ptr %buf acquire, align 8" in out
    # fence.sc 前缀 seqcst load → 单条 seq_cst load
    assert "%lc = load atomic ptr, ptr %buf seq_cst, align 8" in out
    assert 'store atomic i32 %la, ptr %buf syncscope("agent") release, align 4' in out
    assert "store atomic ptr %out, ptr %buf release, align 8" in out
    assert translate_nvvm_to_amdgcn(out) == out, "atomics 改写不幂等"


# Step 13（idp4a/idp2a → sdot4/udot4/乘加展开）探针 fixture
SAMPLE_DOTPROD_IR = """\
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.idp4a.s.s(i32, i32, i32) #3
declare i32 @llvm.nvvm.idp4a.u.u(i32, i32, i32) #3
declare i32 @llvm.nvvm.idp2a.s.s(i32, i32, i1 immarg, i32) #3
declare i32 @llvm.nvvm.idp2a.u.u(i32, i32, i1 immarg, i32) #3

define ptx_kernel void @dot_probe(i32 %a4, i32 %b4, i32 %a2, i32 %b2, ptr %out) #1 {
entry:
  %r0 = tail call i32 @llvm.nvvm.idp4a.s.s(i32 %a4, i32 %b4, i32 100) #3
  %r1 = tail call i32 @llvm.nvvm.idp4a.u.u(i32 %a4, i32 %b4, i32 100) #3
  %r2 = tail call i32 @llvm.nvvm.idp2a.s.s(i32 %a2, i32 %b2, i1 false, i32 100) #3
  %r3 = tail call i32 @llvm.nvvm.idp2a.u.u(i32 %a2, i32 %b2, i1 true, i32 100) #3
  %s = add i32 %r0, %r1
  %s2 = add i32 %s, %r2
  %s3 = add i32 %s2, %r3
  store i32 %s3, ptr %out, align 4
  ret void
}
"""


def test_ir_translate_dotprod():
    """Step 13：idp4a.s.s/.u.u → sdot4/udot4（v_dot4 硬件，clamp=false）；
    idp2a → 乘加展开（a 2×i16 × b 低/高 2 字节 i8，符号按后缀）。
    （GPU 数值已验证：-310/1738/-24/262120 与例源期望精确一致。）"""
    out = translate_nvvm_to_amdgcn(SAMPLE_DOTPROD_IR)

    assert "llvm.nvvm" not in out, "idp intrinsic 残留"
    assert "call i32 @llvm.amdgcn.sdot4(i32 %a4, i32 %b4, i32 100, i1 false)" in out
    assert "call i32 @llvm.amdgcn.udot4(i32 %a4, i32 %b4, i32 100, i1 false)" in out
    # idp2a isbottom=false：b 取低 2 字节（shift 0/8），signed → sext
    assert "%r2.d_a0 = sext i16 %r2.d_a0t to i32" in out
    assert "%r2.d_b0 = sext i8 %r2.d_b0t to i32" in out
    assert "%r2.d_b1s = lshr i32 %b2, 8" in out
    assert "%r2 = add i32 %r2.d_r0, %r2.d_m1" in out, "结果未写回原 SSA 名"
    # idp2a isbottom=true：b 取高 2 字节（shift 16/24），unsigned → zext
    assert "%r3.d_b0s = lshr i32 %b2, 16" in out
    assert "%r3.d_b1s = lshr i32 %b2, 24" in out
    assert "%r3.d_a0 = zext i16 %r3.d_a0t to i32" in out
    assert "%s = add i32 %r0, %r1" in out, "下游引用未保持"
    assert translate_nvvm_to_amdgcn(out) == out, "dotprod 改写不幂等"


# Step 14（counted barrier → LDS 计数器+世代自旋）探针 fixture
SAMPLE_CBAR_IR = """\
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare void @llvm.nvvm.barrier.cta.sync.count(i32, i32) #1
declare void @llvm.nvvm.barrier.cta.arrive.count(i32, i32) #1

define ptx_kernel void @cb_probe(ptr %out, i64 %nout) #1 {
entry:
  %tid = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  tail call void @llvm.nvvm.barrier.cta.sync.count(i32 1, i32 64) #3
  tail call void @llvm.nvvm.barrier.cta.arrive.count(i32 2, i32 64) #3
  ret void
}
"""


def test_ir_translate_counted_barrier():
    """Step 14：counted barrier → 软件 LDS 计数器+世代自旋回退（helper 函数 +
    入口槽位清零 + 全组 s.barrier）。（GPU 数值已验证：producer/consumer 与
    split arrive/sync，128 线程 wave64 PASS。）"""
    out = translate_nvvm_to_amdgcn(SAMPLE_CBAR_IR)

    assert "llvm.nvvm.barrier" not in out, "counted barrier intrinsic 残留"
    assert "call void @__port_cb_sync(i32 1, i32 64)" in out, "sync.count 未映射"
    assert "call void @__port_cb_arrive(i32 2, i32 64)" in out, "arrive.count 未映射"
    # 槽位状态 + helper 定义
    assert "@__port_cb_cnt = addrspace(3) global [16 x i32] undef, align 4" in out
    assert "@__port_cb_gen = addrspace(3) global [16 x i32] undef, align 4" in out
    assert "define internal void @__port_cb_arrive(i32 %id, i32 %n)" in out
    assert "define internal void @__port_cb_sync(i32 %id, i32 %n) convergent" in out
    # 入口 init：两槽清零 + 全组 barrier（保证清零对原子操作可见）
    assert "counted-barrier slot init (ids: 1,2)" in out
    assert out.count("store atomic i32 0, ptr addrspace(3) %__port_cb_i") == 4
    assert "call void @llvm.amdgcn.s.barrier()" in out
    # 世代自旋（volatile 防优化掉）
    assert "load atomic volatile i32" in out
    assert translate_nvvm_to_amdgcn(out) == out, "counted barrier 改写不幂等"


# ---------------------------------------------------------------------------
# 测试 2：cuda-oxide 管线副产物 .ptx 存在性
# ---------------------------------------------------------------------------


def test_ptx_also_generated(built_probe):
    """cargo oxide 除 .ll 外同时产 .ptx（NV 管线副产物，证明双后端产物共存）。"""
    assert built_probe.ptx is not None, (
        f"未找到 .ptx 产物，目录内容: {list(built_probe.dir.iterdir())}"
    )
    text = built_probe.ptx.read_text()
    assert ".visible .entry" in text or ".target" in text, "产物不像 PTX"


# ---------------------------------------------------------------------------
# 测试 2.5：设备识别（GPU 前提：本机确为 gfx1030 / RX 6950 XT）
# ---------------------------------------------------------------------------


def _gfx_agents():
    """rocminfo 解析：返回 [(agent 名, Marketing Name)]，仅保留 gfx* agent。

    每个取值块内首个 Name: 是 agent 名（后续 cache/pool 小节的 Name 不覆盖）。
    """
    proc = _run(["rocminfo"])
    assert proc.returncode == 0, (
        f"[阶段 设备识别: rocminfo] 失败 (rc={proc.returncode})\n"
        f"--- 关键错误输出（尾20行）---\n{_tail(proc.stderr)}\n"
        f"--- 环境前提状态 ---\n{_env_diag()}"
    )
    agents = []
    cur_name = None
    cur_mkt = None
    for line in proc.stdout.splitlines():
        if re.match(r"\s*Agent\s+\d+", line):
            if cur_name and cur_name.startswith("gfx"):
                agents.append((cur_name, cur_mkt or "?"))
            cur_name = cur_mkt = None
            continue
        if cur_name is None:
            m = re.match(r"\s*Name:\s+(\S+)", line)
            if m:
                cur_name = m.group(1)
        if cur_mkt is None:
            m = re.match(r"\s*Marketing Name:\s+(.+?)\s*$", line)
            if m:
                cur_mkt = m.group(1)
    if cur_name and cur_name.startswith("gfx"):
        agents.append((cur_name, cur_mkt or "?"))
    return agents


def test_gfx1030_device_identity():
    """本机 GPU 确为 gfx1030 / RX 6950 XT（防"测在错误的卡上"）。

    选 rocminfo（宿主侧、不初始化 GPU context）为权威源，而非容器内
    `python3 -c "import torch; print(torch.cuda.get_device_name(0))"`；
    后者 2026-09-19 实测同样报告 "AMD Radeon RX 6950 XT"，可作交叉验证。
    """
    agents = _gfx_agents()
    actual = "; ".join(f"{n} ({m})" for n, m in agents) or "<无 gfx agent>"
    assert any(n == "gfx1030" for n, _ in agents), (
        f"本机未报告 gfx1030 agent，实际: {actual}"
    )
    assert any(n == "gfx1030" and "6950" in m for n, m in agents), (
        f"gfx1030 agent 的 Marketing Name 不含 6950，实际: {actual}"
    )


# ---------------------------------------------------------------------------
# 测试 3：端到端（构建 → 改写 → llc → 容器内链接/编译 → GPU 数值校验）
# ---------------------------------------------------------------------------


def test_vecadd_end_to_end(built_probe):
    """核心回归：Rust kernel 一路到 gfx1030 数值校验 PASS。"""
    if not _docker_available():
        pytest.skip(f"docker 容器 {DOCKER_CONTAINER} 不可用")
    assert _gpu_available(), "rocm-smi 报告 GPU 不可用（GPU down 立即停）"

    # -- 改写（用 .opt.ll：与 phase1 golden 同构的 tail-call 形态）----------
    src_ll = built_probe.opt_ll or built_probe.ll
    translated = translate_nvvm_to_amdgcn(src_ll.read_text())
    assert "llvm.nvvm" not in translated, (
        f"[阶段 IR-改写] 真实产物改写后有 nvvm 残留: {src_ll}\n"
        "--- 残留行（尾20行）---\n"
        f"{_tail(chr(10).join(l for l in translated.splitlines() if 'llvm.nvvm' in l))}"
    )
    amdgcn_ll = built_probe.dir / "vecadd_amdgcn.ll"
    amdgcn_ll.write_text(translated)

    # -- llc 编 code object（relocatable ELF）-----------------------------
    co = built_probe.dir / "vecadd_v5llc.co"
    proc = _run(
        [
            str(LLC),
            "-march=amdgcn",
            "-mcpu=gfx1030",
            "-amdhsa-code-object-version=5",
            "--filetype=obj",
            str(amdgcn_ll),
            "-o",
            str(co),
        ]
    )
    assert proc.returncode == 0, (
        f"[阶段 llc→code-object] 失败 (rc={proc.returncode})\n"
        f"--- 关键错误输出（尾20行）---\n{_tail(proc.stderr)}\n"
        f"--- 环境前提状态 ---\n{_env_diag()}"
    )
    assert co.stat().st_size > 0

    # -- 容器内：ld.lld -shared → hipcc 编 host 探针 → 运行 ----------------
    docker_cp(co, "vecadd_v5llc.co")
    host_cpp = built_probe.dir / "host_gfx1030_test.cpp"
    host_cpp.write_text(HOST_PROBE_CPP.replace("@CO_PATH@", f"{CONTAINER_WORKDIR}/vecadd_linked.co"))
    docker_cp(host_cpp, "host_gfx1030_test.cpp")

    cmds = (
        "set -e; cd {wd}; "
        "ld.lld -shared vecadd_v5llc.co -o vecadd_linked.co; "
        "hipcc -O2 --offload-arch=gfx1030 host_gfx1030_test.cpp -o host_gfx1030_test; "
        "./host_gfx1030_test"
    ).format(wd=CONTAINER_WORKDIR)
    proc = _run(["docker", "exec", DOCKER_CONTAINER, "bash", "-c", cmds], timeout=300)
    assert proc.returncode == 0, (
        f"[阶段 容器内链接/编译/运行] 失败 (rc={proc.returncode})\n"
        f"--- 关键错误输出 stdout（尾20行）---\n{_tail(proc.stdout)}\n"
        f"--- 关键错误输出 stderr（尾20行）---\n{_tail(proc.stderr)}\n"
        f"--- 环境前提状态 ---\n{_env_diag()}"
    )
    assert "PASS: all 1024 elements correct" in proc.stdout, (
        f"[阶段 GPU-数值校验] 未 PASS\n"
        f"--- 关键错误输出 stdout（尾20行）---\n{_tail(proc.stdout)}\n"
        f"--- 环境前提状态 ---\n{_env_diag()}"
    )


def _docker_exec(cmd: str, timeout=60):
    return _run(["docker", "exec", DOCKER_CONTAINER, "bash", "-c", cmd], timeout=timeout)


def docker_cp(src: Path, dest_name: str):
    assert _docker_exec(f"mkdir -p {CONTAINER_WORKDIR}").returncode == 0
    proc = _run(
        [
            "docker",
            "cp",
            str(src),
            f"{DOCKER_CONTAINER}:{CONTAINER_WORKDIR}/{dest_name}",
        ]
    )
    assert proc.returncode == 0, (
        f"[阶段 docker-cp（容器编译前置）] 失败 (rc={proc.returncode})\n"
        f"--- 关键错误输出（尾20行）---\n{_tail(proc.stderr)}\n"
        f"--- 环境前提状态 ---\n{_env_diag()}"
    )


# host 探针：docs/research/cuda-oxide/host_v6.cpp 的参数化版本。
# 要点（phase1 两个根因）：加载的是 ld.lld -shared 之后的 linked 产物；
# kparams 是每参数一指针（3×(ptr,i64) + i32 ntid = 7 个）。
HOST_PROBE_CPP = """\
// [PORT gfx1030] generated by tests/test_gfx1030.py (from docs host_v6.cpp)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <fstream>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \\
    printf("HIP error %s at %s:%d\\n", hipGetErrorString(e), __FILE__, __LINE__); return 1; } } while(0)

int main() {
    std::ifstream f("@CO_PATH@", std::ios::binary);
    std::vector<char> img((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    printf("co bytes: %zu\\n", img.size());

    const size_t N = 1024;
    std::vector<float> ha(N), hb(N), hc(N, 0.f);
    for (size_t i = 0; i < N; i++) { ha[i] = (float)i; hb[i] = 2.f * i; }

    float *da, *db, *dc;
    CK(hipMalloc(&da, N * 4)); CK(hipMalloc(&db, N * 4)); CK(hipMalloc(&dc, N * 4));
    CK(hipMemcpy(da, ha.data(), N * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(db, hb.data(), N * 4, hipMemcpyHostToDevice));

    hipModule_t mod; CK(hipModuleLoadData(&mod, img.data()));
    hipFunction_t fn; CK(hipModuleGetFunction(&fn, mod, "vecadd"));

    // 7 参数各一指针: (ptr a, i64 Na, ptr b, i64 Nb, ptr c, i64 Nc, i32 ntid_x)
    long long Na = N, Nb = N, Nc = N;
    int ntid = 256;
    void* kparams[] = { &da, &Na, &db, &Nb, &dc, &Nc, &ntid };
    CK(hipModuleLaunchKernel(fn, 4, 1, 1, 256, 1, 1, 0, nullptr, kparams, nullptr));
    CK(hipDeviceSynchronize());

    CK(hipMemcpy(hc.data(), dc, N * 4, hipMemcpyDeviceToHost));
    int bad = 0;
    for (size_t i = 0; i < N; i++) {
        float want = 3.f * i;
        if (hc[i] != want) { if (bad < 5) printf("MISMATCH @%zu: got %f want %f\\n", i, hc[i], want); bad++; }
    }
    printf(bad == 0 ? "PASS: all %zu elements correct\\n" : "FAIL: %zu mismatches\\n",
           bad == 0 ? N : (size_t)bad);
    return bad != 0;
}
"""
