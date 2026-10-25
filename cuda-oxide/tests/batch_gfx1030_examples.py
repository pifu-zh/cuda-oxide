#!/usr/bin/env python3
# [PORT gfx1030] batch: examples 批量重跑脚本（v1 的 /tmp/batch-report 已按
# 纪律清理，本脚本按 examples-report.md 方法节重建，改写函数 import 共享模块
# tests/translate_amdgcn.py —— 单一事实源）。
#
# 用法: python3 tests/batch_gfx1030_examples.py [--skip-build] [--only NAME]
#            [--integrated]
# 产物: /home/zhuo/tmp-prof/batch_gfx1030/<example>/（device-only 变体 crate + .ll）
#       /home/zhuo/tmp-prof/batch_gfx1030/report.json（逐例机器可读结果）
#
# v4 新增 --integrated：在 Python 改写管线（默认测量，产物 .amdgcn.ll 之上跑
# llc）之外，追加测量"集成化管线"——CUDA_OXIDE_TARGET=gfx1030 直接走
# cargo-oxide 的 amdgcn 后端（mir-lower sreg + prep pass + llc + lld →
# hsaco）。该维度量化 B2–B4 Rust 化后集成管线的新增覆盖（v3 之前仅 vecadd/
# sharedmem 级形状可用）。Python 维度的测量与产物不受影响（integrated 构建在
# 全部例子测量完之后运行）。

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # [adapt-rocm] cuda-oxide/
EX = REPO / "crates/rustc-codegen-cuda/examples"
# [adapt-rocm] 上游编译器已迁到 stable 1.99.1（rustc-dev on stable）
NIGHTLY = "stable"
LLC = Path.home() / (
    ".rustup/toolchains/" + NIGHTLY + "-x86_64-unknown-linux-gnu"
    "/lib/rustlib/x86_64-unknown-linux-gnu/bin/llc"
)
sys.path.insert(0, str(REPO / "tests"))
from translate_amdgcn import translate_nvvm_to_amdgcn  # noqa: E402

# v1 同一批 17 例（examples-report.md）
# [goal batch-expansion 2026-10-26] SWEEP_BATCH: 56 个新例经全管扫描（另 2 例
# printf/kernel_reference_validity 亦为扫描幸存者，共 58——见下分组注释）
# （抽取→构建→改写→llc 全绿，/tmp/sweep_full.json 口径 a-p 段）按族并入。
# 扫描中断于字母 p（shell 故障），q-z 段待续扫后追加。
EXAMPLES = [
    "vecadd", "map_sum", "array_for_loop", "unroll_bounds_check",
    "checked_arith", "device_closures", "dotprod", "sharedmem", "barrier",
    "counted_barrier", "atomics", "redux_sum", "warp_reduce", "math_tan",
    # v1 手动排除的 3 例（抽取器局限，非移植边界证据）
    "array_index", "shared_memory_helper", "const_generic",
    # --- SWEEP_BATCH（对齐/常量内存/转换/数学族） ---
    "addressof_sharedarray", "aligned_field_loads", "aligned_field_stores",
    "array_init", "asin_acos_smoke", "assume_init_uninhabited_trap",
    "barrier_smoke", "bool_phi_cmp", "carrying_mul_add", "cast_chunks",
    "cast_tests", "cbrt_smoke", "compiler_features", "const_aggregate",
    "constant_index_from_end", "constant_memory", "constant_memory_coeffs",
    "constant_memory_simple", "constant_memory_table", "const_table_lookup",
    "const_tuple_table_lookup", "copy_nonoverlapping", "debug_pointer_locals",
    "device_panic_trap", "disjoint_slice_len", "dpx_minmax_chains",
    "drop_glue", "dst_metadata", "dynamic_smem", "enum_abi_multi_payload",
    "enum_payload_addr", "enum_state_transitions", "exact_div_chunks",
    # --- SWEEP_BATCH（数值/f16/数学族） ---
    "f16_stress", "fcmp_ne_nan_smoke", "field_array_assign",
    "float_rounding", "fmaxmin_smoke", "fp_special_literal_smoke",
    "libdevice_math", "math_atan", "math_hyperbolic", "math_no_mir_inline",
    "mem_swap", "ne_bytes_transmute", "nested_index_assignment",
    "numeric_stress",
    # --- SWEEP_BATCH（枚举/Option/ABI 族） ---
    "option_ref_transmute", "option_ref_unwrap_or", "option_unwrap_trap",
    "packed_aggregate_abi", "padded_constants", "partial_warp_reduce",
    "primitive_stress",
    # --- SWEEP_BATCH（编译级过、运行时另见矩阵行） ---
    "printf",  # llc 级过（vprintf 未定义符号存活到链接）；运行时 209 见矩阵
    "kernel_reference_validity",  # llc 级过（grid_constant 形态）
    # --- SWEEP_BATCH q-z 段（2026-10-26 续扫 26 例：slice/tuple/enum 语义族、
    # shared/volatile/unaligned 访存族；194/194 扫描收官，总 pass 87） ---
    "reborrow", "ref_operand_mul", "repr_u32_enum_stride", "select_unpredictable",
    "shared_debug", "shared_ptr_transmute", "shared_slice_unsize",
    "shared_stage_select", "slice_from_raw_parts_cast", "slice_get_mut",
    "slice_index_bounds_check", "slice_iterators", "slice_reslice",
    "split_at_mut", "step_by", "struct_constant_provenance",
    "switchint_128bit", "transmute_scalar_struct", "tuple_array_provenance",
    "tuple_constant_provenance", "tuple_field_lookup", "unaligned_memory",
    "uninit_const", "unroll_smoke", "volatile_memory", "warp_sums",
    # tcgen05 = sweep_error：Blackwell tensor-core 例（llc 崩溃路径），
    # 属 HARDWARE_LIMITED 范畴，不入批
]

# v1 已知排除原因（照抄 examples-report.md，保持口径一致）
V1_EXCLUDED = {
    "array_index": "抽取器 brace 计数被字符串内 {} 干扰",
    "shared_memory_helper": "kernel 体依赖跨模块宏 shared_memory_helper_lib::shared_probe_body!",
    "const_generic": "#[kernel] 宏展开产生 cuda_host 引用，device-only 变体无法满足",
    # [sweep 2026-10-26] 抽取阶段 13 败：无 #[cuda_module] mod 块（manual
    # launch / 跨 crate / 嵌入形态）——抽取器边界，非移植能力边界。
    "cpp_consumes_rust_device": "无 cuda_module mod（跨语言形态）",
    "cross_crate_embedded": "无 cuda_module mod（跨 crate 嵌入）",
    "cross_crate_kernel": "无 cuda_module mod（跨 crate）",
    "cutile_inter_kernel": "无 cuda_module mod（cutile 形态）",
    "gemm_sol_final": "无 cuda_module mod（手写 launch）",
    "hashmap_v2": "无 cuda_module mod",
    "hashmap_v3": "无 cuda_module mod",
    "interop_cubin_identity": "无 cuda_module mod（interop）",
    "manual_launch_generic": "无 cuda_module mod（manual launch）",
    "manual_launch_libdevice": "无 cuda_module mod（manual launch）",
    "standalone_device_fn": "无 cuda_module mod",
    "tilelib_relu": "无 cuda_module mod（tile 库形态）",
    "wgmma_mma_bf16": "无 cuda_module mod",
}


def extract_mod_block(src: str):
    """抓取 #[cuda_module] mod ... { ... }（v1 同款 brace 计数，已知局限：
    不处理字符串/注释内花括号）。返回 (完整块文本, None) 或 (None, 原因)。"""
    m = re.search(r"#\[cuda_module\]\s*(pub\s+)?mod\s+(\w+)\s*\{", src)
    if not m:
        return None, "未找到 #[cuda_module] mod 块"
    i = m.end()  # 指向开 '{' 之后
    depth = 1
    while i < len(src) and depth > 0:
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None, "mod 块 brace 不平衡（字符串内花括号干扰）"
    return src[m.start():i], None


def extract_use_lines(src: str):
    """抽取文件顶层 use 语句（含多行），分类：保留 cuda_device/core/std，
    丢弃 cuda_core/cuda_host。"""
    uses, kept, dropped = [], [], []
    for m in re.finditer(r"^\s*(pub\s+)?use\s+([^;]+);", src, re.M):
        path = m.group(2).strip()
        uses.append((m.start(), m.end(), path))
    for _, _, path in uses:
        root = path.split("::")[0]
        if root == "super":
            continue  # mod 块内部的 use super::*，不属于文件顶层
        if root in ("cuda_core", "cuda_host"):
            dropped.append(path)
        else:
            kept.append(f"use {path};")
    return kept, dropped


def extract_top_level_items(src: str):
    """抽取 mod 块之外的顶层 const/static/type 定义与 crate 内属性 #![..]。
    mod 块经 use super::* 引用它们（v1 脚本同款处理）。行级扫描：从定义行
    起累积到括号平衡且以 ';' 收尾（兼容多行 const 表）。"""
    attrs = re.findall(r"^#!\[[^\]]*\]\s*$", src, re.M)
    items = []
    lines = src.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        # [adapt-rocm] struct/enum 纳入：上游 sharedmem 把 `pub struct V4(...)`
        # 提到 crate 顶层、kernels mod 经 use super::* 引用——缺了它抽取变体
        # 编译失败（E0425）。brace 结构以 "}" 收尾（深度归零），元组结构以
        # ";" 收尾，两种收尾都处理。
        if re.match(r"^(pub\s+)?(const|static|type|struct|enum)\b", line):
            # 定义行上方连续的 #[..] 属性行随定义一起带走
            # （derive(Copy) 不带走 → E0507，sharedmem V4 实证）
            start = i
            while start > 0 and re.match(r"^\s*#\[", lines[start - 1]):
                start -= 1
            i = start
            line = lines[i]
            depth = 0
            buf = []
            while i < len(lines):
                buf.append(lines[i])
                depth += sum(lines[i].count(c) for c in "{[(")
                depth -= sum(lines[i].count(c) for c in "}])")
                i += 1
                if depth <= 0 and lines[i - 1].rstrip().endswith((";", "}")):
                    break
            items.append("\n".join(buf))
        else:
            i += 1
    return attrs, items


def build_variant(name: str, out_dir: Path):
    """按 ox-amdgcn-probe 模板产 device-only 变体 crate。"""
    src = (EX / name / "src/main.rs").read_text()
    block, err = extract_mod_block(src)
    if block is None:
        return False, err
    kept_use, _dropped = extract_use_lines(src)
    attrs, top_items = extract_top_level_items(src)
    # 顶层项可能与 mod 块内的项重名（mod 有自己的作用域），保留即可；
    # 未被引用的 const 只产生 warning，不影响构建。

    # 保证 cuda_module / kernel 在 use 里（多数例自带；缺失则补）
    joined = "\n".join(kept_use)
    need = []
    if re.search(r"#\[cuda_module\]", block) and not re.search(
        r"cuda_device::\{[^}]*cuda_module|use cuda_device::cuda_module", joined
    ):
        need.append("cuda_module")
    if re.search(r"#\[kernel\]", block) and not re.search(
        r"cuda_device::\{[^}]*\bkernel\b", joined
    ):
        need.append("kernel")
    if need:
        kept_use.append(f"use cuda_device::{{{','.join(need)}}};")

    (out_dir / "src").mkdir(parents=True, exist_ok=True)
    (out_dir / "Cargo.toml").write_text(
        f"""\
[package]
name = "ox-batch-{name}"
version = "0.1.0"
edition = "2024"
publish = false

# [PORT gfx1030] batch v2 device-only 变体（自 {name} 抽取，方法同 v1）

[dependencies]
cuda-device = {{ path = "{REPO / 'crates' / 'cuda-device'}" }}

[workspace]
"""
    )
    body = "\n".join(
        [
            f"//! [PORT gfx1030] batch v2: {name} 的 device-only 降级变体（自动生成）。",
            *attrs,
            *kept_use,
            "",
            *top_items,
            "",
            block,
            "",
            'fn main() { println!("probe host stub"); }',
        ]
    )
    (out_dir / "src" / "main.rs").write_text(body)
    return True, None


def run(cmd, cwd=None, timeout=600, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def integrated_build(name: str, d: Path, env: dict):
    """[PORT gfx1030 PhaseB.5] 集成化管线测量：CUDA_OXIDE_TARGET=gfx1030 下
    cargo-oxide 直接产出链接后的 hsaco（mir-lower Amdgcn sreg + amdgcn prep
    pass + llc + lld）。成功 = 退出码 0 且 *.hsaco 存在。"""
    # touch_main_rs 语义：确保后端重跑（集成构建会清理并重建产物）
    src = d / "src" / "main.rs"
    if src.exists():
        os.utime(src, None)
    # [PORT gfx1030][review 2026-10-20] 清掉上一轮遗留的 hsaco：成功判据是
    # "本次构建退出 0 且 hsaco 存在"，残留文件会把一次没产物的构建认证成
    # PASS（clean_generated_files 不清理 *.hsaco）。
    for stale in d.glob("*.hsaco"):
        stale.unlink()
    env_int = dict(env, CUDA_OXIDE_TARGET="gfx1030")
    proc = run(["cargo", f"+{NIGHTLY}", "oxide", "build"], cwd=d, env=env_int)
    hsacos = list(d.glob("*.hsaco"))
    return proc, hsacos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--integrated", action="store_true",
                    help="追加集成化管线（CUDA_OXIDE_TARGET=gfx1030 直出 hsaco）测量")
    args = ap.parse_args()

    env = dict(os.environ,
               # [adapt-rocm] pinned-stable toolchain unlocks -Z
               CUDA_TOOLKIT_PATH=os.path.expanduser("~/opt/cuda13"),
               RUSTC_BOOTSTRAP="1")
    results = {}

    for name in EXAMPLES:
        if args.only and name != args.only:
            continue
        d = Path("/home/zhuo/tmp-prof/batch_gfx1030") / name
        rec = {"example": name}
        if name in V1_EXCLUDED and not args.only:
            rec.update(status="excluded", reason=V1_EXCLUDED[name])
            results[name] = rec
            print(f"[excluded] {name}: {V1_EXCLUDED[name]}")
            continue

        # --- 构建阶段 ---
        if not args.skip_build or not (d / "src/main.rs").exists():
            ok, err = build_variant(name, d)
            if not ok:
                rec.update(status="extract_fail", reason=err)
                results[name] = rec
                print(f"[extract-fail] {name}: {err}")
                continue
            (d / ".cargo").mkdir(exist_ok=True)
            (d / ".cargo" / "config.toml").write_text(
                '[env]\nRUSTC_BOOTSTRAP = "1"\n')
            proc = run(["cargo", f"+{NIGHTLY}", "oxide", "build"], cwd=d, env=env)
            if proc.returncode != 0:
                rec.update(status="build_fail",
                           reason=(proc.stdout + proc.stderr)[-800:])
                results[name] = rec
                print(f"[build-fail] {name}")
                continue

        # [PORT gfx1030] 精确产物名选择：目录里可能残留上一轮 --integrated 的
        # *.amdgcn*.ll（集成管线产物）；旧 glob "*.opt.ll" 会把它们一并匹配且
        # 排序靠前（"amdgcn" < "opt"），Python 维度吃错产物形成双重改写——
        # rebase 后回归数字失真的真实根因之一。这里只认 Python 管线自己的
        # 两个产物名。
        opt_ll = d / f"ox_batch_{name}.opt.ll"
        plain_ll = d / f"ox_batch_{name}.ll"
        src_ll = opt_ll if opt_ll.exists() else plain_ll
        if not src_ll.exists():
            rec.update(status="no_ll", reason="cargo oxide 成功但无 .ll 产物")
            results[name] = rec
            print(f"[no-ll] {name}")
            continue
        rec["ll"] = str(src_ll)

        # --- 改写阶段 ---
        translated = translate_nvvm_to_amdgcn(src_ll.read_text())
        residual = "llvm.nvvm" in translated
        rec["residual"] = residual
        out_ll = d / "amdgcn.ll"
        out_ll.write_text(translated)

        # --- llc 阶段 ---
        obj = d / "gfx1030.co"
        proc = run([str(LLC), "-march=amdgcn", "-mcpu=gfx1030",
                    "-amdhsa-code-object-version=5", "--filetype=obj",
                    str(out_ll), "-o", str(obj)])
        rec["llc_rc"] = proc.returncode
        if proc.returncode == 0:
            rec.update(status="pass")
            print(f"[PASS] {name}")
        else:
            err = proc.stderr
            fails = re.findall(r"error: [^\n]+", err)
            rec.update(status="llc_fail", reason=err[-1500:],
                       first_error=fails[0] if fails else None)
            print(f"[llc-fail] {name}: {fails[0] if fails else err[-200:]}")
        results[name] = rec

    # --- 集成化管线维度（Python 管线全部测量完之后，避免产物互相覆盖）---
    if args.integrated:
        print("\n=== integrated pipeline (CUDA_OXIDE_TARGET=gfx1030 -> hsaco) ===")
        for name in EXAMPLES:
            if args.only and name != args.only:
                continue
            if name in V1_EXCLUDED:
                results[name]["integrated"] = "excluded"
                continue
            d = Path("/home/zhuo/tmp-prof/batch_gfx1030") / name
            if not (d / "src/main.rs").exists():
                results[name]["integrated"] = "no_variant"
                continue
            proc, hsacos = integrated_build(name, d, env)
            if proc.returncode == 0 and hsacos:
                results[name]["integrated"] = "pass"
                print(f"[INTEGRATED PASS] {name}: {hsacos[0].name}")
            else:
                fails = re.findall(r"error[^\n]*", proc.stdout + proc.stderr)
                results[name]["integrated"] = "fail"
                results[name]["integrated_error"] = (
                    fails[0] if fails else (proc.stdout + proc.stderr)[-300:]
                )
                print(f"[INTEGRATED FAIL] {name}: {results[name]['integrated_error']}")

    Path("/home/zhuo/tmp-prof/batch_gfx1030/report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    ok = sum(1 for r in results.values() if r.get("status") == "pass")
    ll_ct = sum(1 for r in results.values()
                if r.get("status") in ("pass", "llc_fail") or r.get("residual"))
    n = len(EXAMPLES)
    summary = f"\nSUMMARY: llc-pass {ok} / {n}; ll-produced {ll_ct} / {n}"
    if args.integrated:
        ipass = sum(1 for r in results.values() if r.get("integrated") == "pass")
        summary += f"; integrated hsaco-pass {ipass} / {n}"
    print(summary)

    # [PORT gfx1030][review 2026-10-20] 编排层退出码：excluded 不算失败，
    # 其余任何非 pass（build/llc/residual）都让进程非零——供 CI/脚本门禁
    # 使用，不再"全失败也 exit 0"。
    hard_fail = any(
        r.get("status") not in ("pass", "excluded") for r in results.values()
    )
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
