; SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
; SPDX-License-Identifier: Apache-2.0
; Panic-message transport probe: direct __ockl_printf_append_string_n with
; (a) a module-level constant pointer and (b) an addrspace(1) global cast to
; generic. SPDX-License-Identifier: Apache-2.0
target triple = "amdgcn-amd-amdhsa"

@__probe_const = unnamed_addr constant [15 x i8] c"const msg probe"
@__probe_global = addrspace(1) constant [16 x i8] c"global msg probe"
@__probe_nl = unnamed_addr constant [1 x i8] c"\0A"

declare i64 @__ockl_printf_begin(i64)
declare i64 @__ockl_printf_append_string_n(i64, ptr, i64, i32)

define amdgpu_kernel void @probe_const() {
entry:
  %h0 = call i64 @__ockl_printf_begin(i64 0)
  %h1 = call i64 @__ockl_printf_append_string_n(i64 %h0, ptr @__probe_const, i64 15, i32 0)
  %h2 = call i64 @__ockl_printf_append_string_n(i64 %h1, ptr @__probe_nl, i64 1, i32 1)
  ret void
}

define amdgpu_kernel void @probe_global() {
entry:
  %gen = addrspacecast ptr addrspace(1) @__probe_global to ptr
  %h0 = call i64 @__ockl_printf_begin(i64 0)
  %h1 = call i64 @__ockl_printf_append_string_n(i64 %h0, ptr %gen, i64 16, i32 0)
  %h2 = call i64 @__ockl_printf_append_string_n(i64 %h1, ptr @__probe_nl, i64 1, i32 1)
  ret void
}

!amdgpu_printf_kind = !{!1000}
!1000 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
