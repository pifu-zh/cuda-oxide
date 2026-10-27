#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible probe: does the llc amdgcn-amd-amdhsa / gfx1030 target emit
# DWARF when the IR carries !dbg metadata, and does our pipeline do so?
# Evidence recorded in docs/gfx1030_support_matrix.md (设备端 DWARF row).
set -u

LLVM_BIN="${LLVM_BIN:-$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin}"
DW_DUMP="${DW_DUMP:-/opt/rocm/core-10.1/lib/llvm/bin/llvm-dwarfdump}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

cat > "$OUT/dwarf_min.ll" <<'EOF'
define amdgpu_kernel void @k(ptr addrspace(1) %out) !dbg !8 {
entry:
  store i32 42, ptr addrspace(1) %out, !dbg !10
  ret void, !dbg !10
}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1}
!0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !2, producer: "probe", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !DIFile(filename: "k.rs", directory: "/tmp")
!8 = distinct !DISubprogram(name: "k", scope: !2, file: !2, line: 1, type: !9, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !{null})
!10 = !DILocation(line: 2, column: 1, scope: !8)
EOF

echo "== [1] llc --debugger-tune=gdb, gfx1030 object =="
"$LLVM_BIN/llc" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 \
  -filetype=obj --debugger-tune=gdb "$OUT/dwarf_min.ll" -o "$OUT/dwarf_min.o" || exit 1

echo "== [2] .debug_* sections in the object =="
"$LLVM_BIN/llvm-readobj" -S "$OUT/dwarf_min.o" | grep -o '\.debug_[a-z_]*' | sort -u

echo "== [3] file(1) verdict =="
file "$OUT/dwarf_min.o"

echo "== [4] DWARF CU/subprogram parse =="
"$DW_DUMP" --debug-info "$OUT/dwarf_min.o" 2>/dev/null | \
  grep -E 'DW_TAG_compile_unit|DW_TAG_subprogram|DW_AT_name' | head -4

echo "== [5] our pipeline: !dbg count in a built .amdgcn.ll and hsaco debug sections =="
EX="${1:-cvt_packed}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LL="$HERE/crates/rustc-codegen-cuda/examples/$EX/$EX.amdgcn.ll"
HSACO="$HERE/crates/rustc-codegen-cuda/examples/$EX/$EX.hsaco"
[ -f "$LL" ] && echo "$EX.amdgcn.ll !dbg count: $(grep -c '!dbg' "$LL")"
[ -f "$HSACO" ] && { echo "$EX.hsaco .debug_* sections:"; \
  "$LLVM_BIN/llvm-readobj" -S "$HSACO" | grep -c '\.debug_' || true; }
