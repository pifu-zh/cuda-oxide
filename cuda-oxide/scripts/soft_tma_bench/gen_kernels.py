#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate gfx1030 hand-written ISA kernels for the soft-TMA microbenchmark.

All variants: 32 lanes/wave (wave32), one wave per block, each lane moves one
16B quad (float4) per instruction.

  plain1   gmem -> VGPR -> LDS -> VGPR -> gmem... fully serialized per quad
           (reference: no pipelining). Reduction consumption.
  df4_bN   gmem -> VGPR -> gmem full copy, deep pipeline (B quads/group,
           2 groups). Measures pure load/store pipelining at 2N traffic.
  lds_bN   gmem -> VGPR -> LDS (ds_write_b128) deep pipeline; consumption =
           ds_read + per-lane F32 reduction (TMA-like: data lands in LDS and
           is consumed, nothing written back). Measures gmem->LDS throughput
           at N traffic. One 16B reduction result per lane stored at the end.

Pipeline structure per body (deep variants), B = quads per group:
  issue next group's B loads; s_waitcnt vmcnt(B) (oldest = previous group's
  loads land, plus the previous body's stores for df4); consume the landed
  group. Max outstanding VM = 3B (<= 63 -> B <= 21).
"""
import sys

HDR = '''\t.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
\t.amdhsa_code_object_version 6
\t.text
\t.p2align\t8
'''

TAB = '\t'

def issue_group(group_base_vgpr, b):
    """B quad loads into VGPR group; address = s[12:13] + lane_off; advances s[12:13]."""
    out = []
    for k in range(b):
        out.append(f'{TAB}v_add_co_u32 v4, vcc_lo, s12, v1')
        out.append(f'{TAB}v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo')
        out.append(f'{TAB}global_load_dwordx4 v[{group_base_vgpr+4*k}:{group_base_vgpr+4*k+3}], v[4:5], off')
        out.append(f'{TAB}s_add_u32 s12, s12, 512')
        out.append(f'{TAB}s_addc_u32 s13, s13, 0')
    return out

def store_group(group_base_vgpr, b):
    """B quad stores from VGPR group to s[14:15]; advances it."""
    out = []
    for k in range(b):
        out.append(f'{TAB}v_add_co_u32 v4, vcc_lo, s14, v1')
        out.append(f'{TAB}v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo')
        out.append(f'{TAB}global_store_dwordx4 v[4:5], v[{group_base_vgpr+4*k}:{group_base_vgpr+4*k+3}], off')
        out.append(f'{TAB}s_add_u32 s14, s14, 512')
        out.append(f'{TAB}s_addc_u32 s15, s15, 0')
    return out

def ds_write_group(group_base_vgpr, b):
    """B quad writes into LDS slot s16; per-quad offset k*512."""
    out = []
    for k in range(b):
        out.append(f'{TAB}v_add_co_u32 v6, vcc_lo, s16, v1')
        if k:
            out.append(f'{TAB}v_add_nc_u32 v6, v6, {k*512}')
        out.append(f'{TAB}ds_write_b128 v6, v[{group_base_vgpr+4*k}:{group_base_vgpr+4*k+3}]')
    return out

def ds_read_group(group_base_vgpr, b):
    out = []
    for k in range(b):
        out.append(f'{TAB}v_add_co_u32 v6, vcc_lo, s16, v1')
        if k:
            out.append(f'{TAB}v_add_nc_u32 v6, v6, {k*512}')
        out.append(f'{TAB}ds_read_b128 v[{group_base_vgpr+4*k}:{group_base_vgpr+4*k+3}], v6')
    return out

ACC = 150  # accumulator VGPRs v150..v153, counter v154 (beyond group range)

def reduce_group(group_base_vgpr, b):
    """Accumulate the group's 4B lanes into v[150:153] (4 F32 accumulators)."""
    out = []
    for k in range(b):
        for j in range(4):
            out.append(f'{TAB}v_add_f32 v{ACC+j}, v{group_base_vgpr+4*k+j}, v{ACC+j}')
    return out

def prologue(name, b):
    """Common head: load kernargs, lane id, per-block base, pointers."""
    g0 = 16
    out = []
    out.append(f'{name}:')
    out.append(f'{TAB}s_load_dwordx2 s[4:5], s[0:1], 0x0')      # src
    out.append(f'{TAB}s_load_dwordx2 s[6:7], s[0:1], 0x8')      # dst
    out.append(f'{TAB}s_load_dwordx2 s[8:9], s[0:1], 0x10')     # quads_per_block
    out.append(f'{TAB}s_waitcnt lgkmcnt(0)')
    out.append(f'{TAB}v_mbcnt_lo_u32_b32 v0, -1, 0')            # lane id
    out.append(f'{TAB}v_mbcnt_hi_u32_b32 v0, -1, v0')
    out.append(f'{TAB}v_lshlrev_b32 v1, 4, v0')                 # lane byte off (16B/lane)
    out.append(f'{TAB}v_mov_b32 v3, 0')                         # high word of lane off
    # block byte base (u64) = (u64)wg_x * quads_per_block * 512
    # (a quad = 32 lanes x 16B = 512B; the old <<4 made 32 blocks overlap in a
    #  9.16MB prefix and left the rest of dst untouched)
    out.append(f'{TAB}s_mul_i32 s10, s2, s8')
    out.append(f'{TAB}s_lshl_b32 s11, s10, 9')                  # low  = base << 9
    out.append(f'{TAB}s_lshr_b32 s12, s10, 23')                 # high = base >> 23
    out.append(f'{TAB}s_mov_b32 s20, s6')                       # s[20:21] = raw dst (results base)
    out.append(f'{TAB}s_mov_b32 s21, s7')
    out.append(f'{TAB}s_add_u32 s4, s4, s11')
    out.append(f'{TAB}s_addc_u32 s5, s5, s12')                  # s[4:5] = src + base
    out.append(f'{TAB}s_add_u32 s6, s6, s11')
    out.append(f'{TAB}s_addc_u32 s7, s7, s12')                  # s[6:7] = dst + base
    out.append(f'{TAB}s_mov_b32 s12, s4')                       # s[12:13] = cur src quad base
    out.append(f'{TAB}s_mov_b32 s13, s5')
    out.append(f'{TAB}s_mov_b32 s14, s6')                       # s[14:15] = cur dst quad base
    out.append(f'{TAB}s_mov_b32 s15, s7')
    return out

def kernel_lds(name, b):
    """deep pipeline gmem->LDS, reduction consumption. Group0 base 16, group1 16+4B."""
    g0, g1 = 16, 16 + 4 * b
    lds_bytes = 2 * b * 512
    vgprs = 24 + 8 * b
    body = prologue(name, b)
    body.append(f'{TAB}s_movk_i32 s16, 0x0')                    # LDS slot base (slot A)
    body.append(f'{TAB}s_movk_i32 s18, {b*512}')                # slot stride (flip const)
    body.append(f'{TAB}s_mov_b32 s17, s8')                      # remaining quads to issue
    body.append(f'{TAB}v_mov_b32 v{ACC}, 0')                    # accumulators
    body.append(f'{TAB}v_mov_b32 v{ACC+1}, 0')
    body.append(f'{TAB}v_mov_b32 v{ACC+2}, 0')
    body.append(f'{TAB}v_mov_b32 v{ACC+3}, 0')
    body.append(f'{TAB}v_mov_b32 v{ACC+4}, 0')                  # debug: quads reduced
    # prologue: issue group0
    body += issue_group(g0, b)
    body.append(f'{TAB}s_sub_u32 s17, s17, {b}')
    # loop_a invariant: last issued = g0 (undrained); s17 = remaining to issue
    body.append(f'{name}_loop_a:')
    body.append(f'{TAB}s_cmp_eq_u32 s17, 0')
    body.append(f'{TAB}s_cbranch_scc1 {name}_tail_g0')
    body += issue_group(g1, b)
    body.append(f'{TAB}s_sub_u32 s17, s17, {b}')
    body.append(f'{TAB}s_waitcnt vmcnt({b})')                   # g0 lands
    body += ds_write_group(g0, b)
    body.append(f'{TAB}s_waitcnt lgkmcnt(0)')
    body += ds_read_group(g0, b)
    body.append(f'{TAB}s_waitcnt lgkmcnt(0)')
    body += reduce_group(g0, b)
    body.append(f'{TAB}v_add_nc_u32 v{ACC+4}, v{ACC+4}, {b}')
    body.append(f'{TAB}s_xor_b32 s16, s16, s18')                # flip LDS slot
    body.append(f'{TAB}s_cmp_eq_u32 s17, 0')
    body.append(f'{TAB}s_cbranch_scc1 {name}_tail_g1')
    body += issue_group(g0, b)
    body.append(f'{TAB}s_sub_u32 s17, s17, {b}')
    body.append(f'{TAB}s_waitcnt vmcnt({b})')                   # g1 lands
    body += ds_write_group(g1, b)
    body.append(f'{TAB}s_waitcnt lgkmcnt(0)')
    body += ds_read_group(g1, b)
    body.append(f'{TAB}s_waitcnt lgkmcnt(0)')
    body += reduce_group(g1, b)
    body.append(f'{TAB}v_add_nc_u32 v{ACC+4}, v{ACC+4}, {b}')
    body.append(f'{TAB}s_xor_b32 s16, s16, s18')
    body.append(f'{TAB}s_branch {name}_loop_a')
    # tails: reduce the last-landed group, store one 16B result per lane
    body.append(f'{name}_tail_g0:')
    body.append(f'{TAB}s_waitcnt vmcnt(0)')
    body += ds_write_group(g0, b)
    body.append(f'{TAB}s_waitcnt lgkmcnt(0)')
    body += ds_read_group(g0, b)
    body.append(f'{TAB}s_waitcnt lgkmcnt(0)')
    body += reduce_group(g0, b)
    body.append(f'{TAB}v_add_nc_u32 v{ACC+4}, v{ACC+4}, {b}')
    body += finish_reduce(name)
    body.append(f'{name}_tail_g1:')
    body.append(f'{TAB}s_waitcnt vmcnt(0)')
    body += ds_write_group(g1, b)
    body.append(f'{TAB}s_waitcnt lgkmcnt(0)')
    body += ds_read_group(g1, b)
    body.append(f'{TAB}s_waitcnt lgkmcnt(0)')
    body += reduce_group(g1, b)
    body.append(f'{TAB}v_add_nc_u32 v{ACC+4}, v{ACC+4}, {b}')
    body += finish_reduce(name)
    meta = dict(group_fixed=lds_bytes, vgpr=max(vgprs, ACC + 6), sgpr=22, reduce=True)
    return '\n'.join(body) + '\n', meta

def finish_reduce(name):
    """Per-block result block (640B): sums at lane*16, count at 512 + lane*4.
    Base = raw dst + wg_x*640."""
    out = []
    out.append(f'{TAB}s_mul_i32 s11, s2, 1024')                 # wg_x * 1024
    out.append(f'{TAB}s_add_u32 s14, s20, s11')
    out.append(f'{TAB}s_addc_u32 s15, s21, 0')
    out.append(f'{TAB}v_add_co_u32 v4, vcc_lo, s14, v1')
    out.append(f'{TAB}v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo')
    out.append(f'{TAB}global_store_dwordx4 v[4:5], v[{ACC}:{ACC+3}], off')
    out.append(f'{TAB}v_lshlrev_b32 v7, 2, v0')                 # lane*4
    out.append(f'{TAB}s_add_u32 s14, s14, 512')                 # count region
    out.append(f'{TAB}v_add_co_u32 v4, vcc_lo, s14, v7')
    out.append(f'{TAB}v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo')
    out.append(f'{TAB}global_store_dword v[4:5], v{ACC+4}, off offset:4')
    out.append(f'{TAB}v_mov_b32 v7, s17')
    out.append(f'{TAB}global_store_dword v[4:5], v7, off offset:8')
    out.append(f'{TAB}v_mov_b32 v7, s16')
    out.append(f'{TAB}global_store_dword v[4:5], v7, off offset:12')
    out.append(f'{TAB}s_endpgm')
    return out

def kernel_deep_f4(name, b):
    """gmem -> VGPR -> gmem full copy, no LDS. Group0 16, group1 16+4B."""
    g0, g1 = 16, 16 + 4 * b
    vgprs = 16 + 8 * b
    body = prologue(name, b)
    body.append(f'{TAB}s_mov_b32 s17, s8')
    body += issue_group(g0, b)
    body.append(f'{TAB}s_sub_u32 s17, s17, {b}')
    body.append(f'{name}_loop_a:')
    body.append(f'{TAB}s_cmp_eq_u32 s17, 0')
    body.append(f'{TAB}s_cbranch_scc1 {name}_tail_g0')
    body += issue_group(g1, b)
    body.append(f'{TAB}s_sub_u32 s17, s17, {b}')
    body.append(f'{TAB}s_waitcnt vmcnt({b})')                   # g0 lands
    body += store_group(g0, b)
    body.append(f'{TAB}s_cmp_eq_u32 s17, 0')
    body.append(f'{TAB}s_cbranch_scc1 {name}_tail_g1')
    body += issue_group(g0, b)
    body.append(f'{TAB}s_sub_u32 s17, s17, {b}')
    body.append(f'{TAB}s_waitcnt vmcnt({b})')                   # g1 lands
    body += store_group(g1, b)
    body.append(f'{TAB}s_branch {name}_loop_a')
    body.append(f'{name}_tail_g0:')
    body.append(f'{TAB}s_waitcnt vmcnt(0)')
    body += store_group(g0, b)
    body.append(f'{TAB}s_endpgm')
    body.append(f'{name}_tail_g1:')
    body.append(f'{TAB}s_waitcnt vmcnt(0)')
    body += store_group(g1, b)
    body.append(f'{TAB}s_endpgm')
    meta = dict(group_fixed=0, vgpr=max(vgprs, ACC + 6), sgpr=22, reduce=False)
    return '\n'.join(body) + '\n', meta

def kernel_plain(name):
    """Serialized reference: one quad at a time, full gmem->LDS->reduce path."""
    body = prologue(name, 1)
    body.append(f'{TAB}s_movk_i32 s16, 0x0')
    body.append(f'{TAB}s_mov_b32 s17, s8')
    body.append(f'{TAB}v_mov_b32 v{ACC}, 0')
    body.append(f'{TAB}v_mov_b32 v{ACC+1}, 0')
    body.append(f'{TAB}v_mov_b32 v{ACC+2}, 0')
    body.append(f'{TAB}v_mov_b32 v{ACC+3}, 0')
    body.append(f'{TAB}v_mov_b32 v{ACC+4}, 0')
    body.append(f'{name}_loop:')
    body.append(f'{TAB}s_cmp_eq_u32 s17, 0')
    body.append(f'{TAB}s_cbranch_scc1 {name}_done')
    body.append(f'{TAB}v_add_co_u32 v4, vcc_lo, s12, v1')
    body.append(f'{TAB}v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo')
    body.append(f'{TAB}global_load_dwordx4 v[16:19], v[4:5], off')
    body.append(f'{TAB}s_add_u32 s12, s12, 512')
    body.append(f'{TAB}s_addc_u32 s13, s13, 0')
    body.append(f'{TAB}s_waitcnt vmcnt(0)')
    body.append(f'{TAB}v_add_co_u32 v6, vcc_lo, s16, v1')
    body.append(f'{TAB}ds_write_b128 v6, v[16:19]')
    body.append(f'{TAB}s_waitcnt lgkmcnt(0)')
    body.append(f'{TAB}ds_read_b128 v[16:19], v6')
    body.append(f'{TAB}s_waitcnt lgkmcnt(0)')
    body.append(f'{TAB}v_add_f32 v{ACC}, v16, v{ACC}')
    body.append(f'{TAB}v_add_f32 v{ACC+1}, v17, v{ACC+1}')
    body.append(f'{TAB}v_add_f32 v{ACC+2}, v18, v{ACC+2}')
    body.append(f'{TAB}v_add_f32 v{ACC+3}, v19, v{ACC+3}')
    body.append(f'{TAB}v_add_nc_u32 v{ACC+4}, v{ACC+4}, 1')
    body.append(f'{TAB}s_sub_u32 s17, s17, 1')
    body.append(f'{TAB}s_branch {name}_loop')
    body.append(f'{name}_done:')
    body += finish_reduce(name)
    meta = dict(group_fixed=512, vgpr=max(26, ACC + 6), sgpr=22, reduce=True)
    return '\n'.join(body) + '\n', meta

def kernel_directive(name, meta):
    """.amdhsa_kernel directive block (COv6) -- field set mirrors the
    clang-emitted stub for this toolchain (7.2.4)."""
    return f'''\t.section\t.rodata,"a",@progbits
\t.p2align\t6, 0x0
\t.amdhsa_kernel {name}
\t\t.amdhsa_group_segment_fixed_size {meta['group_fixed']}
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_kernarg_size 24
\t\t.amdhsa_user_sgpr_count 2
\t\t.amdhsa_user_sgpr_private_segment_buffer 0
\t\t.amdhsa_user_sgpr_dispatch_ptr 0
\t\t.amdhsa_user_sgpr_queue_ptr 0
\t\t.amdhsa_user_sgpr_kernarg_segment_ptr 1
\t\t.amdhsa_user_sgpr_dispatch_id 0
\t\t.amdhsa_user_sgpr_flat_scratch_init 0
\t\t.amdhsa_user_sgpr_private_segment_size 0
\t\t.amdhsa_wavefront_size32 1
\t\t.amdhsa_uses_dynamic_stack 0
\t\t.amdhsa_system_sgpr_private_segment_wavefront_offset 0
\t\t.amdhsa_system_sgpr_workgroup_id_x 1
\t\t.amdhsa_system_sgpr_workgroup_id_y 0
\t\t.amdhsa_system_sgpr_workgroup_id_z 0
\t\t.amdhsa_system_sgpr_workgroup_info 0
\t\t.amdhsa_system_vgpr_workitem_id 0
\t\t.amdhsa_next_free_vgpr {meta['vgpr']}
\t\t.amdhsa_next_free_sgpr {meta['sgpr']}
\t\t.amdhsa_reserve_vcc 1
\t\t.amdhsa_reserve_flat_scratch 0
\t\t.amdhsa_float_round_mode_32 0
\t\t.amdhsa_float_round_mode_16_64 0
\t\t.amdhsa_float_denorm_mode_32 3
\t\t.amdhsa_float_denorm_mode_16_64 3
\t\t.amdhsa_dx10_clamp 1
\t\t.amdhsa_ieee_mode 1
\t\t.amdhsa_fp16_overflow 0
\t\t.amdhsa_workgroup_processor_mode 1
\t\t.amdhsa_memory_ordered 1
\t\t.amdhsa_forward_progress 1
\t\t.amdhsa_shared_vgpr_count 0
\t\t.amdhsa_exception_fp_ieee_invalid_op 0
\t\t.amdhsa_exception_fp_denorm_src 0
\t\t.amdhsa_exception_fp_ieee_div_zero 0
\t\t.amdhsa_exception_fp_ieee_overflow 0
\t\t.amdhsa_exception_fp_ieee_underflow 0
\t\t.amdhsa_exception_fp_ieee_inexact 0
\t\t.amdhsa_exception_int_div_zero 0
\t.end_amdhsa_kernel
'''

def metadata_yaml(metas):
    """COv6 NT_AMDGPU_METADATA note via .amdgpu_metadata YAML."""
    lines = ['\t.amdgpu_metadata', '---', 'amdhsa.kernels:']
    for name, meta in metas:
        lines.append('  - .args:')
        lines.append('      - .address_space:  global')
        lines.append('        .offset:         0')
        lines.append('        .size:           8')
        lines.append('        .value_kind:     global_buffer')
        lines.append('      - .address_space:  global')
        lines.append('        .offset:         8')
        lines.append('        .size:           8')
        lines.append('        .value_kind:     global_buffer')
        lines.append('      - .offset:         16')
        lines.append('        .size:           8')
        lines.append('        .value_kind:     by_value')
        lines.append(f"    .group_segment_fixed_size: {meta['group_fixed']}")
        lines.append('    .kernarg_segment_align: 8')
        lines.append('    .kernarg_segment_size: 24')
        lines.append('    .language:       OpenCL C')
        lines.append('    .language_version:')
        lines.append('      - 2')
        lines.append('      - 0')
        lines.append('    .max_flat_workgroup_size: 32')
        lines.append(f'    .name:           {name}')
        lines.append('    .private_segment_fixed_size: 0')
        lines.append(f"    .sgpr_count:     {meta['sgpr']}")
        lines.append('    .sgpr_spill_count: 0')
        lines.append(f'    .symbol:         {name}.kd')
        lines.append('    .uniform_work_group_size: 1')
        lines.append('    .uses_dynamic_stack: false')
        lines.append(f"    .vgpr_count:     {meta['vgpr']}")
        lines.append('    .vgpr_spill_count: 0')
        lines.append('    .wavefront_size: 32')
        lines.append('    .workgroup_processor_mode: 1')
    lines.append('amdhsa.target:   amdgcn-amd-amdhsa--gfx1030')
    lines.append('amdhsa.version:')
    lines.append('  - 1')
    lines.append('  - 2')
    lines.append('...')
    lines.append('\t.end_amdgpu_metadata')
    return '\n'.join(lines) + '\n'


def main():
    variants = [('plain1', kernel_plain('plain1'))]
    for b in (4, 8, 16):
        variants.append((f'df4_b{b}', kernel_deep_f4(f'df4_b{b}', b)))
    for b in (4, 8, 16):
        variants.append((f'lds_b{b}', kernel_lds(f'lds_b{b}', b)))
    out = [HDR]
    metas = []
    for name, (body, meta) in variants:
        metas.append((name, meta))
        out.append(f'\t.globl\t{name}')
        out.append(f'\t.type\t{name},@function')
        out.append(body)
        out.append(kernel_directive(name, meta))
        out.append('\t.text')
        out.append('\t.p2align\t8')
    out.append(metadata_yaml(metas))
    sys.stdout.write('\n'.join(out))

if __name__ == '__main__':
    main()
