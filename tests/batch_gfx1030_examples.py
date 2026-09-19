#!/usr/bin/env python3
# [PORT gfx1030] batch v2: examples 批量重跑脚本（v1 的 /tmp/batch-report 已按
# 纪律清理，本脚本按 examples-report.md 方法节重建，改写函数 import 共享模块
# tests/translate_amdgcn.py —— 单一事实源）。
#
# 用法: python3 tests/batch_gfx1030_examples.py [--skip-build] [--only NAME]
# 产物: /tmp/batch_gfx1030/<example>/（device-only 变体 crate + .ll）
#       /tmp/batch_gfx1030/report.json（逐例机器可读结果）

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/zhuo/workspace/src/cuda-oxide")
EX = REPO / "crates/rustc-codegen-cuda/examples"
NIGHTLY = "nightly-2026-08-28"
LLC = Path.home() / (
    ".rustup/toolchains/" + NIGHTLY + "-x86_64-unknown-linux-gnu"
    "/lib/rustlib/x86_64-unknown-linux-gnu/bin/llc"
)
sys.path.insert(0, str(REPO / "tests"))
from translate_amdgcn import translate_nvvm_to_amdgcn  # noqa: E402

# v1 同一批 17 例（examples-report.md）
EXAMPLES = [
    "vecadd", "map_sum", "array_for_loop", "unroll_bounds_check",
    "checked_arith", "device_closures", "dotprod", "sharedmem", "barrier",
    "counted_barrier", "atomics", "redux_sum", "warp_reduce", "math_tan",
    # v1 手动排除的 3 例（抽取器局限，非移植边界证据）
    "array_index", "shared_memory_helper", "const_generic",
]

# v1 已知排除原因（照抄 examples-report.md，保持口径一致）
V1_EXCLUDED = {
    "array_index": "抽取器 brace 计数被字符串内 {} 干扰",
    "shared_memory_helper": "kernel 体依赖跨模块宏 shared_memory_helper_lib::shared_probe_body!",
    "const_generic": "#[kernel] 宏展开产生 cuda_host 引用，device-only 变体无法满足",
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
        if re.match(r"^(pub\s+)?(const|static|type)\b", line):
            depth = 0
            buf = []
            while i < len(lines):
                buf.append(lines[i])
                depth += sum(lines[i].count(c) for c in "{[(")
                depth -= sum(lines[i].count(c) for c in "}])")
                i += 1
                if lines[i - 1].rstrip().endswith(";") and depth <= 0:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    env = dict(os.environ, CUDA_TOOLKIT_PATH=os.path.expanduser("~/opt/cuda13"))
    results = {}

    for name in EXAMPLES:
        if args.only and name != args.only:
            continue
        d = Path("/tmp/batch_gfx1030") / name
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
            proc = run(["cargo", f"+{NIGHTLY}", "oxide", "build"], cwd=d, env=env)
            if proc.returncode != 0:
                rec.update(status="build_fail",
                           reason=(proc.stdout + proc.stderr)[-800:])
                results[name] = rec
                print(f"[build-fail] {name}")
                continue

        lls = sorted(d.glob("*.opt.ll")) or sorted(
            p for p in d.glob("*.ll") if not p.name.endswith(".opt.ll")
        )
        if not lls:
            rec.update(status="no_ll", reason="cargo oxide 成功但无 .ll 产物")
            results[name] = rec
            print(f"[no-ll] {name}")
            continue
        src_ll = lls[0]
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

    Path("/tmp/batch_gfx1030/report.json").write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results.values() if r.get("status") == "pass")
    ll_ct = sum(1 for r in results.values()
                if r.get("status") in ("pass", "llc_fail") or r.get("residual"))
    print(f"\nSUMMARY: llc-pass {ok} / 17; ll-produced {ll_ct} / 17")


if __name__ == "__main__":
    main()
