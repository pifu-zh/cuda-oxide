#!/usr/bin/env python3
"""Inject mbarrier state dumps at bridge wait sites in the lowered .ll.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""
import re

lines = open("/tmp/cp_async_small.lowered.ll").read().splitlines()
out = []
in_bridges = False
wait_count = 0
# bridges kernel 的 out 指针：从第一个 out store 提取（store i32 %vN, ptr %vM）
out_ptr = None
injected = 0

# pass 1: find the bridges kernel's out generic ptr name
for i, l in enumerate(lines):
    if l.startswith("define "):
        in_bridges = "test_cp_async_mbarrier_bridges" in l
    if in_bridges:
        m = re.match(r"^\s*store i32 (%v\d+), ptr (%v\d+), align 4$", l)
        if m:
            out_ptr = m.group(2)
            break

in_bridges = False
for i, l in enumerate(lines):
    if l.startswith("define "):
        in_bridges = "test_cp_async_mbarrier_bridges" in l
    if in_bridges and "mbarrier.test_wait.shared.b64" in l:
        wait_count += 1
        mm = re.search(r"\(ptr addrspace\(3\) ([%\w.$]+), i64 ", l)
        if mm and wait_count >= 2 and out_ptr:
            bar = mm.group(1)
            tag = wait_count
            # 在 test_wait 前插：读状态 8B + 读 bar 地址 + 写到 out 尾部
            # out 区：块 stride 4B×4 = 16B 用于数据；dump 放 out+32 起（ memset 64B 内）
            out.append(f"  %dbg{tag}.st = load i64, ptr addrspace(3) {bar}, align 8")
            out.append(f"  %dbg{tag}.bi = ptrtoint ptr addrspace(3) {bar} to i64")
            out.append(f"  %dbg{tag}.o1 = getelementptr i8, ptr {out_ptr}, i64 {32 + (tag-2)*16}")
            out.append(f"  %dbg{tag}.o2 = getelementptr i8, ptr {out_ptr}, i64 {32 + (tag-2)*16 + 8}")
            out.append(f"  store i64 %dbg{tag}.st, ptr %dbg{tag}.o1, align 8")
            out.append(f"  store i64 %dbg{tag}.bi, ptr %dbg{tag}.o2, align 8")
            injected += 1
    out.append(l)

open("/tmp/cp_dbg8.ll", "w").write("\n".join(out) + "\n")
print(f"wait sites: {wait_count}, injected: {injected}, out_ptr: {out_ptr}")
