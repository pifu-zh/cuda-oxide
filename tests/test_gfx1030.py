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
#   - rustup nightly-2026-08-28（含 rustc-dev / llvm-tools；llc 用工具链自带）
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
        f"cargo oxide build 失败:\nstdout:\n{proc.stdout[-4000:]}\n"
        f"stderr:\n{proc.stderr[-4000:]}"
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
        f"真实产物改写后有 nvvm 残留: {src_ll}"
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
    assert proc.returncode == 0, f"llc 失败:\n{proc.stderr[-4000:]}"
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
        f"容器内链接/编译/运行失败 (rc={proc.returncode}):\n"
        f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
    )
    assert "PASS: all 1024 elements correct" in proc.stdout, (
        f"数值校验未 PASS:\n{proc.stdout[-4000:]}"
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
    assert proc.returncode == 0, f"docker cp 失败: {proc.stderr}"


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
