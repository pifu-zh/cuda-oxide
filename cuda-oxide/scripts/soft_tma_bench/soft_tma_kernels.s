	.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
	.amdhsa_code_object_version 6
	.text
	.p2align	8

	.globl	plain1
	.type	plain1,@function
plain1:
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[6:7], s[0:1], 0x8
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_lshlrev_b32 v1, 4, v0
	v_mov_b32 v3, 0
	s_mul_i32 s10, s2, s8
	s_lshl_b32 s11, s10, 4
	s_lshr_b32 s12, s10, 28
	s_mov_b32 s20, s6
	s_mov_b32 s21, s7
	s_add_u32 s4, s4, s11
	s_addc_u32 s5, s5, s12
	s_add_u32 s6, s6, s11
	s_addc_u32 s7, s7, s12
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mov_b32 s14, s6
	s_mov_b32 s15, s7
	s_movk_i32 s16, 0x0
	s_mov_b32 s17, s8
	v_mov_b32 v150, 0
	v_mov_b32 v151, 0
	v_mov_b32 v152, 0
	v_mov_b32 v153, 0
	v_mov_b32 v154, 0
plain1_loop:
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 plain1_done
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_waitcnt vmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[16:19]
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[16:19], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v16, v150
	v_add_f32 v151, v17, v151
	v_add_f32 v152, v18, v152
	v_add_f32 v153, v19, v153
	v_add_nc_u32 v154, v154, 1
	s_sub_u32 s17, s17, 1
	s_branch plain1_loop
plain1_done:
	s_mul_i32 s11, s2, 1024
	s_add_u32 s14, s20, s11
	s_addc_u32 s15, s21, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[150:153], off
	v_lshlrev_b32 v7, 2, v0
	s_add_u32 s14, s14, 512
	v_add_co_u32 v4, vcc_lo, s14, v7
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dword v[4:5], v154, off offset:4
	v_mov_b32 v7, s17
	global_store_dword v[4:5], v7, off offset:8
	v_mov_b32 v7, s16
	global_store_dword v[4:5], v7, off offset:12
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel plain1
		.amdhsa_group_segment_fixed_size 512
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_private_segment_buffer 0
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 156
		.amdhsa_next_free_sgpr 22
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel

	.text
	.p2align	8
	.globl	df4_b4
	.type	df4_b4,@function
df4_b4:
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[6:7], s[0:1], 0x8
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_lshlrev_b32 v1, 4, v0
	v_mov_b32 v3, 0
	s_mul_i32 s10, s2, s8
	s_lshl_b32 s11, s10, 4
	s_lshr_b32 s12, s10, 28
	s_mov_b32 s20, s6
	s_mov_b32 s21, s7
	s_add_u32 s4, s4, s11
	s_addc_u32 s5, s5, s12
	s_add_u32 s6, s6, s11
	s_addc_u32 s7, s7, s12
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mov_b32 s14, s6
	s_mov_b32 s15, s7
	s_mov_b32 s17, s8
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[20:23], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[24:27], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[28:31], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 4
df4_b4_loop_a:
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 df4_b4_tail_g0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[32:35], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[36:39], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[40:43], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[44:47], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 4
	s_waitcnt vmcnt(4)
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[16:19], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[20:23], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[24:27], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[28:31], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 df4_b4_tail_g1
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[20:23], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[24:27], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[28:31], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 4
	s_waitcnt vmcnt(4)
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[32:35], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[36:39], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[40:43], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[44:47], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	s_branch df4_b4_loop_a
df4_b4_tail_g0:
	s_waitcnt vmcnt(0)
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[16:19], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[20:23], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[24:27], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[28:31], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	s_endpgm
df4_b4_tail_g1:
	s_waitcnt vmcnt(0)
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[32:35], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[36:39], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[40:43], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[44:47], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel df4_b4
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_private_segment_buffer 0
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 156
		.amdhsa_next_free_sgpr 22
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel

	.text
	.p2align	8
	.globl	df4_b8
	.type	df4_b8,@function
df4_b8:
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[6:7], s[0:1], 0x8
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_lshlrev_b32 v1, 4, v0
	v_mov_b32 v3, 0
	s_mul_i32 s10, s2, s8
	s_lshl_b32 s11, s10, 4
	s_lshr_b32 s12, s10, 28
	s_mov_b32 s20, s6
	s_mov_b32 s21, s7
	s_add_u32 s4, s4, s11
	s_addc_u32 s5, s5, s12
	s_add_u32 s6, s6, s11
	s_addc_u32 s7, s7, s12
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mov_b32 s14, s6
	s_mov_b32 s15, s7
	s_mov_b32 s17, s8
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[20:23], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[24:27], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[28:31], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[32:35], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[36:39], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[40:43], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[44:47], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 8
df4_b8_loop_a:
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 df4_b8_tail_g0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[48:51], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[52:55], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[56:59], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[60:63], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[64:67], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[68:71], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[72:75], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[76:79], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 8
	s_waitcnt vmcnt(8)
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[16:19], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[20:23], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[24:27], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[28:31], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[32:35], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[36:39], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[40:43], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[44:47], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 df4_b8_tail_g1
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[20:23], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[24:27], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[28:31], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[32:35], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[36:39], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[40:43], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[44:47], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 8
	s_waitcnt vmcnt(8)
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[48:51], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[52:55], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[56:59], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[60:63], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[64:67], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[68:71], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[72:75], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[76:79], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	s_branch df4_b8_loop_a
df4_b8_tail_g0:
	s_waitcnt vmcnt(0)
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[16:19], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[20:23], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[24:27], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[28:31], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[32:35], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[36:39], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[40:43], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[44:47], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	s_endpgm
df4_b8_tail_g1:
	s_waitcnt vmcnt(0)
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[48:51], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[52:55], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[56:59], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[60:63], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[64:67], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[68:71], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[72:75], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[76:79], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel df4_b8
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_private_segment_buffer 0
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 156
		.amdhsa_next_free_sgpr 22
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel

	.text
	.p2align	8
	.globl	df4_b16
	.type	df4_b16,@function
df4_b16:
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[6:7], s[0:1], 0x8
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_lshlrev_b32 v1, 4, v0
	v_mov_b32 v3, 0
	s_mul_i32 s10, s2, s8
	s_lshl_b32 s11, s10, 4
	s_lshr_b32 s12, s10, 28
	s_mov_b32 s20, s6
	s_mov_b32 s21, s7
	s_add_u32 s4, s4, s11
	s_addc_u32 s5, s5, s12
	s_add_u32 s6, s6, s11
	s_addc_u32 s7, s7, s12
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mov_b32 s14, s6
	s_mov_b32 s15, s7
	s_mov_b32 s17, s8
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[20:23], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[24:27], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[28:31], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[32:35], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[36:39], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[40:43], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[44:47], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[48:51], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[52:55], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[56:59], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[60:63], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[64:67], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[68:71], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[72:75], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[76:79], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 16
df4_b16_loop_a:
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 df4_b16_tail_g0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[80:83], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[84:87], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[88:91], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[92:95], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[96:99], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[100:103], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[104:107], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[108:111], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[112:115], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[116:119], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[120:123], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[124:127], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[128:131], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[132:135], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[136:139], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[140:143], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 16
	s_waitcnt vmcnt(16)
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[16:19], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[20:23], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[24:27], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[28:31], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[32:35], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[36:39], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[40:43], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[44:47], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[48:51], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[52:55], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[56:59], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[60:63], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[64:67], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[68:71], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[72:75], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[76:79], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 df4_b16_tail_g1
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[20:23], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[24:27], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[28:31], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[32:35], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[36:39], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[40:43], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[44:47], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[48:51], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[52:55], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[56:59], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[60:63], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[64:67], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[68:71], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[72:75], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[76:79], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 16
	s_waitcnt vmcnt(16)
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[80:83], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[84:87], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[88:91], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[92:95], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[96:99], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[100:103], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[104:107], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[108:111], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[112:115], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[116:119], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[120:123], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[124:127], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[128:131], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[132:135], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[136:139], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[140:143], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	s_branch df4_b16_loop_a
df4_b16_tail_g0:
	s_waitcnt vmcnt(0)
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[16:19], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[20:23], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[24:27], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[28:31], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[32:35], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[36:39], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[40:43], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[44:47], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[48:51], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[52:55], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[56:59], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[60:63], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[64:67], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[68:71], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[72:75], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[76:79], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	s_endpgm
df4_b16_tail_g1:
	s_waitcnt vmcnt(0)
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[80:83], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[84:87], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[88:91], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[92:95], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[96:99], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[100:103], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[104:107], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[108:111], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[112:115], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[116:119], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[120:123], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[124:127], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[128:131], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[132:135], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[136:139], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[140:143], off
	s_add_u32 s14, s14, 512
	s_addc_u32 s15, s15, 0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel df4_b16
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_private_segment_buffer 0
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 156
		.amdhsa_next_free_sgpr 22
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel

	.text
	.p2align	8
	.globl	lds_b4
	.type	lds_b4,@function
lds_b4:
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[6:7], s[0:1], 0x8
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_lshlrev_b32 v1, 4, v0
	v_mov_b32 v3, 0
	s_mul_i32 s10, s2, s8
	s_lshl_b32 s11, s10, 4
	s_lshr_b32 s12, s10, 28
	s_mov_b32 s20, s6
	s_mov_b32 s21, s7
	s_add_u32 s4, s4, s11
	s_addc_u32 s5, s5, s12
	s_add_u32 s6, s6, s11
	s_addc_u32 s7, s7, s12
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mov_b32 s14, s6
	s_mov_b32 s15, s7
	s_movk_i32 s16, 0x0
	s_movk_i32 s18, 2048
	s_mov_b32 s17, s8
	v_mov_b32 v150, 0
	v_mov_b32 v151, 0
	v_mov_b32 v152, 0
	v_mov_b32 v153, 0
	v_mov_b32 v154, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[20:23], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[24:27], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[28:31], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 4
lds_b4_loop_a:
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 lds_b4_tail_g0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[32:35], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[36:39], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[40:43], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[44:47], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 4
	s_waitcnt vmcnt(4)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[16:19]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_write_b128 v6, v[20:23]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_write_b128 v6, v[24:27]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_write_b128 v6, v[28:31]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_read_b128 v[16:19], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_read_b128 v[20:23], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_read_b128 v[24:27], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_read_b128 v[28:31], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v16, v150
	v_add_f32 v151, v17, v151
	v_add_f32 v152, v18, v152
	v_add_f32 v153, v19, v153
	v_add_f32 v150, v20, v150
	v_add_f32 v151, v21, v151
	v_add_f32 v152, v22, v152
	v_add_f32 v153, v23, v153
	v_add_f32 v150, v24, v150
	v_add_f32 v151, v25, v151
	v_add_f32 v152, v26, v152
	v_add_f32 v153, v27, v153
	v_add_f32 v150, v28, v150
	v_add_f32 v151, v29, v151
	v_add_f32 v152, v30, v152
	v_add_f32 v153, v31, v153
	v_add_nc_u32 v154, v154, 4
	s_xor_b32 s16, s16, s18
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 lds_b4_tail_g1
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[20:23], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[24:27], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[28:31], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 4
	s_waitcnt vmcnt(4)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[32:35]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_write_b128 v6, v[36:39]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_write_b128 v6, v[40:43]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_write_b128 v6, v[44:47]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_read_b128 v[32:35], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_read_b128 v[36:39], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_read_b128 v[40:43], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_read_b128 v[44:47], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v32, v150
	v_add_f32 v151, v33, v151
	v_add_f32 v152, v34, v152
	v_add_f32 v153, v35, v153
	v_add_f32 v150, v36, v150
	v_add_f32 v151, v37, v151
	v_add_f32 v152, v38, v152
	v_add_f32 v153, v39, v153
	v_add_f32 v150, v40, v150
	v_add_f32 v151, v41, v151
	v_add_f32 v152, v42, v152
	v_add_f32 v153, v43, v153
	v_add_f32 v150, v44, v150
	v_add_f32 v151, v45, v151
	v_add_f32 v152, v46, v152
	v_add_f32 v153, v47, v153
	s_xor_b32 s16, s16, s18
	s_branch lds_b4_loop_a
lds_b4_tail_g0:
	s_waitcnt vmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[16:19]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_write_b128 v6, v[20:23]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_write_b128 v6, v[24:27]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_write_b128 v6, v[28:31]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_read_b128 v[16:19], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_read_b128 v[20:23], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_read_b128 v[24:27], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_read_b128 v[28:31], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v16, v150
	v_add_f32 v151, v17, v151
	v_add_f32 v152, v18, v152
	v_add_f32 v153, v19, v153
	v_add_f32 v150, v20, v150
	v_add_f32 v151, v21, v151
	v_add_f32 v152, v22, v152
	v_add_f32 v153, v23, v153
	v_add_f32 v150, v24, v150
	v_add_f32 v151, v25, v151
	v_add_f32 v152, v26, v152
	v_add_f32 v153, v27, v153
	v_add_f32 v150, v28, v150
	v_add_f32 v151, v29, v151
	v_add_f32 v152, v30, v152
	v_add_f32 v153, v31, v153
	v_add_nc_u32 v154, v154, 4
	s_mul_i32 s11, s2, 1024
	s_add_u32 s14, s20, s11
	s_addc_u32 s15, s21, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[150:153], off
	v_lshlrev_b32 v7, 2, v0
	s_add_u32 s14, s14, 512
	v_add_co_u32 v4, vcc_lo, s14, v7
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dword v[4:5], v154, off offset:4
	v_mov_b32 v7, s17
	global_store_dword v[4:5], v7, off offset:8
	v_mov_b32 v7, s16
	global_store_dword v[4:5], v7, off offset:12
	s_endpgm
lds_b4_tail_g1:
	s_waitcnt vmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[32:35]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_write_b128 v6, v[36:39]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_write_b128 v6, v[40:43]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_write_b128 v6, v[44:47]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_read_b128 v[32:35], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_read_b128 v[36:39], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_read_b128 v[40:43], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_read_b128 v[44:47], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v32, v150
	v_add_f32 v151, v33, v151
	v_add_f32 v152, v34, v152
	v_add_f32 v153, v35, v153
	v_add_f32 v150, v36, v150
	v_add_f32 v151, v37, v151
	v_add_f32 v152, v38, v152
	v_add_f32 v153, v39, v153
	v_add_f32 v150, v40, v150
	v_add_f32 v151, v41, v151
	v_add_f32 v152, v42, v152
	v_add_f32 v153, v43, v153
	v_add_f32 v150, v44, v150
	v_add_f32 v151, v45, v151
	v_add_f32 v152, v46, v152
	v_add_f32 v153, v47, v153
	v_add_nc_u32 v154, v154, 4
	s_mul_i32 s11, s2, 1024
	s_add_u32 s14, s20, s11
	s_addc_u32 s15, s21, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[150:153], off
	v_lshlrev_b32 v7, 2, v0
	s_add_u32 s14, s14, 512
	v_add_co_u32 v4, vcc_lo, s14, v7
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dword v[4:5], v154, off offset:4
	v_mov_b32 v7, s17
	global_store_dword v[4:5], v7, off offset:8
	v_mov_b32 v7, s16
	global_store_dword v[4:5], v7, off offset:12
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel lds_b4
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_private_segment_buffer 0
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 156
		.amdhsa_next_free_sgpr 22
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel

	.text
	.p2align	8
	.globl	lds_b8
	.type	lds_b8,@function
lds_b8:
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[6:7], s[0:1], 0x8
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_lshlrev_b32 v1, 4, v0
	v_mov_b32 v3, 0
	s_mul_i32 s10, s2, s8
	s_lshl_b32 s11, s10, 4
	s_lshr_b32 s12, s10, 28
	s_mov_b32 s20, s6
	s_mov_b32 s21, s7
	s_add_u32 s4, s4, s11
	s_addc_u32 s5, s5, s12
	s_add_u32 s6, s6, s11
	s_addc_u32 s7, s7, s12
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mov_b32 s14, s6
	s_mov_b32 s15, s7
	s_movk_i32 s16, 0x0
	s_movk_i32 s18, 4096
	s_mov_b32 s17, s8
	v_mov_b32 v150, 0
	v_mov_b32 v151, 0
	v_mov_b32 v152, 0
	v_mov_b32 v153, 0
	v_mov_b32 v154, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[20:23], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[24:27], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[28:31], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[32:35], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[36:39], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[40:43], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[44:47], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 8
lds_b8_loop_a:
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 lds_b8_tail_g0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[48:51], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[52:55], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[56:59], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[60:63], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[64:67], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[68:71], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[72:75], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[76:79], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 8
	s_waitcnt vmcnt(8)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[16:19]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_write_b128 v6, v[20:23]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_write_b128 v6, v[24:27]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_write_b128 v6, v[28:31]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_write_b128 v6, v[32:35]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_write_b128 v6, v[36:39]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_write_b128 v6, v[40:43]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_write_b128 v6, v[44:47]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_read_b128 v[16:19], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_read_b128 v[20:23], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_read_b128 v[24:27], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_read_b128 v[28:31], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_read_b128 v[32:35], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_read_b128 v[36:39], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_read_b128 v[40:43], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_read_b128 v[44:47], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v16, v150
	v_add_f32 v151, v17, v151
	v_add_f32 v152, v18, v152
	v_add_f32 v153, v19, v153
	v_add_f32 v150, v20, v150
	v_add_f32 v151, v21, v151
	v_add_f32 v152, v22, v152
	v_add_f32 v153, v23, v153
	v_add_f32 v150, v24, v150
	v_add_f32 v151, v25, v151
	v_add_f32 v152, v26, v152
	v_add_f32 v153, v27, v153
	v_add_f32 v150, v28, v150
	v_add_f32 v151, v29, v151
	v_add_f32 v152, v30, v152
	v_add_f32 v153, v31, v153
	v_add_f32 v150, v32, v150
	v_add_f32 v151, v33, v151
	v_add_f32 v152, v34, v152
	v_add_f32 v153, v35, v153
	v_add_f32 v150, v36, v150
	v_add_f32 v151, v37, v151
	v_add_f32 v152, v38, v152
	v_add_f32 v153, v39, v153
	v_add_f32 v150, v40, v150
	v_add_f32 v151, v41, v151
	v_add_f32 v152, v42, v152
	v_add_f32 v153, v43, v153
	v_add_f32 v150, v44, v150
	v_add_f32 v151, v45, v151
	v_add_f32 v152, v46, v152
	v_add_f32 v153, v47, v153
	v_add_nc_u32 v154, v154, 8
	s_xor_b32 s16, s16, s18
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 lds_b8_tail_g1
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[20:23], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[24:27], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[28:31], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[32:35], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[36:39], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[40:43], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[44:47], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 8
	s_waitcnt vmcnt(8)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[48:51]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_write_b128 v6, v[52:55]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_write_b128 v6, v[56:59]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_write_b128 v6, v[60:63]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_write_b128 v6, v[64:67]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_write_b128 v6, v[68:71]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_write_b128 v6, v[72:75]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_write_b128 v6, v[76:79]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_read_b128 v[48:51], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_read_b128 v[52:55], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_read_b128 v[56:59], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_read_b128 v[60:63], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_read_b128 v[64:67], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_read_b128 v[68:71], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_read_b128 v[72:75], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_read_b128 v[76:79], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v48, v150
	v_add_f32 v151, v49, v151
	v_add_f32 v152, v50, v152
	v_add_f32 v153, v51, v153
	v_add_f32 v150, v52, v150
	v_add_f32 v151, v53, v151
	v_add_f32 v152, v54, v152
	v_add_f32 v153, v55, v153
	v_add_f32 v150, v56, v150
	v_add_f32 v151, v57, v151
	v_add_f32 v152, v58, v152
	v_add_f32 v153, v59, v153
	v_add_f32 v150, v60, v150
	v_add_f32 v151, v61, v151
	v_add_f32 v152, v62, v152
	v_add_f32 v153, v63, v153
	v_add_f32 v150, v64, v150
	v_add_f32 v151, v65, v151
	v_add_f32 v152, v66, v152
	v_add_f32 v153, v67, v153
	v_add_f32 v150, v68, v150
	v_add_f32 v151, v69, v151
	v_add_f32 v152, v70, v152
	v_add_f32 v153, v71, v153
	v_add_f32 v150, v72, v150
	v_add_f32 v151, v73, v151
	v_add_f32 v152, v74, v152
	v_add_f32 v153, v75, v153
	v_add_f32 v150, v76, v150
	v_add_f32 v151, v77, v151
	v_add_f32 v152, v78, v152
	v_add_f32 v153, v79, v153
	s_xor_b32 s16, s16, s18
	s_branch lds_b8_loop_a
lds_b8_tail_g0:
	s_waitcnt vmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[16:19]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_write_b128 v6, v[20:23]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_write_b128 v6, v[24:27]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_write_b128 v6, v[28:31]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_write_b128 v6, v[32:35]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_write_b128 v6, v[36:39]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_write_b128 v6, v[40:43]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_write_b128 v6, v[44:47]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_read_b128 v[16:19], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_read_b128 v[20:23], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_read_b128 v[24:27], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_read_b128 v[28:31], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_read_b128 v[32:35], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_read_b128 v[36:39], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_read_b128 v[40:43], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_read_b128 v[44:47], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v16, v150
	v_add_f32 v151, v17, v151
	v_add_f32 v152, v18, v152
	v_add_f32 v153, v19, v153
	v_add_f32 v150, v20, v150
	v_add_f32 v151, v21, v151
	v_add_f32 v152, v22, v152
	v_add_f32 v153, v23, v153
	v_add_f32 v150, v24, v150
	v_add_f32 v151, v25, v151
	v_add_f32 v152, v26, v152
	v_add_f32 v153, v27, v153
	v_add_f32 v150, v28, v150
	v_add_f32 v151, v29, v151
	v_add_f32 v152, v30, v152
	v_add_f32 v153, v31, v153
	v_add_f32 v150, v32, v150
	v_add_f32 v151, v33, v151
	v_add_f32 v152, v34, v152
	v_add_f32 v153, v35, v153
	v_add_f32 v150, v36, v150
	v_add_f32 v151, v37, v151
	v_add_f32 v152, v38, v152
	v_add_f32 v153, v39, v153
	v_add_f32 v150, v40, v150
	v_add_f32 v151, v41, v151
	v_add_f32 v152, v42, v152
	v_add_f32 v153, v43, v153
	v_add_f32 v150, v44, v150
	v_add_f32 v151, v45, v151
	v_add_f32 v152, v46, v152
	v_add_f32 v153, v47, v153
	v_add_nc_u32 v154, v154, 8
	s_mul_i32 s11, s2, 1024
	s_add_u32 s14, s20, s11
	s_addc_u32 s15, s21, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[150:153], off
	v_lshlrev_b32 v7, 2, v0
	s_add_u32 s14, s14, 512
	v_add_co_u32 v4, vcc_lo, s14, v7
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dword v[4:5], v154, off offset:4
	v_mov_b32 v7, s17
	global_store_dword v[4:5], v7, off offset:8
	v_mov_b32 v7, s16
	global_store_dword v[4:5], v7, off offset:12
	s_endpgm
lds_b8_tail_g1:
	s_waitcnt vmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[48:51]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_write_b128 v6, v[52:55]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_write_b128 v6, v[56:59]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_write_b128 v6, v[60:63]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_write_b128 v6, v[64:67]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_write_b128 v6, v[68:71]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_write_b128 v6, v[72:75]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_write_b128 v6, v[76:79]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_read_b128 v[48:51], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_read_b128 v[52:55], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_read_b128 v[56:59], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_read_b128 v[60:63], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_read_b128 v[64:67], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_read_b128 v[68:71], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_read_b128 v[72:75], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_read_b128 v[76:79], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v48, v150
	v_add_f32 v151, v49, v151
	v_add_f32 v152, v50, v152
	v_add_f32 v153, v51, v153
	v_add_f32 v150, v52, v150
	v_add_f32 v151, v53, v151
	v_add_f32 v152, v54, v152
	v_add_f32 v153, v55, v153
	v_add_f32 v150, v56, v150
	v_add_f32 v151, v57, v151
	v_add_f32 v152, v58, v152
	v_add_f32 v153, v59, v153
	v_add_f32 v150, v60, v150
	v_add_f32 v151, v61, v151
	v_add_f32 v152, v62, v152
	v_add_f32 v153, v63, v153
	v_add_f32 v150, v64, v150
	v_add_f32 v151, v65, v151
	v_add_f32 v152, v66, v152
	v_add_f32 v153, v67, v153
	v_add_f32 v150, v68, v150
	v_add_f32 v151, v69, v151
	v_add_f32 v152, v70, v152
	v_add_f32 v153, v71, v153
	v_add_f32 v150, v72, v150
	v_add_f32 v151, v73, v151
	v_add_f32 v152, v74, v152
	v_add_f32 v153, v75, v153
	v_add_f32 v150, v76, v150
	v_add_f32 v151, v77, v151
	v_add_f32 v152, v78, v152
	v_add_f32 v153, v79, v153
	v_add_nc_u32 v154, v154, 8
	s_mul_i32 s11, s2, 1024
	s_add_u32 s14, s20, s11
	s_addc_u32 s15, s21, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[150:153], off
	v_lshlrev_b32 v7, 2, v0
	s_add_u32 s14, s14, 512
	v_add_co_u32 v4, vcc_lo, s14, v7
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dword v[4:5], v154, off offset:4
	v_mov_b32 v7, s17
	global_store_dword v[4:5], v7, off offset:8
	v_mov_b32 v7, s16
	global_store_dword v[4:5], v7, off offset:12
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel lds_b8
		.amdhsa_group_segment_fixed_size 8192
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_private_segment_buffer 0
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 156
		.amdhsa_next_free_sgpr 22
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel

	.text
	.p2align	8
	.globl	lds_b16
	.type	lds_b16,@function
lds_b16:
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[6:7], s[0:1], 0x8
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mbcnt_hi_u32_b32 v0, -1, v0
	v_lshlrev_b32 v1, 4, v0
	v_mov_b32 v3, 0
	s_mul_i32 s10, s2, s8
	s_lshl_b32 s11, s10, 4
	s_lshr_b32 s12, s10, 28
	s_mov_b32 s20, s6
	s_mov_b32 s21, s7
	s_add_u32 s4, s4, s11
	s_addc_u32 s5, s5, s12
	s_add_u32 s6, s6, s11
	s_addc_u32 s7, s7, s12
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mov_b32 s14, s6
	s_mov_b32 s15, s7
	s_movk_i32 s16, 0x0
	s_movk_i32 s18, 8192
	s_mov_b32 s17, s8
	v_mov_b32 v150, 0
	v_mov_b32 v151, 0
	v_mov_b32 v152, 0
	v_mov_b32 v153, 0
	v_mov_b32 v154, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[20:23], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[24:27], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[28:31], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[32:35], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[36:39], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[40:43], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[44:47], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[48:51], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[52:55], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[56:59], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[60:63], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[64:67], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[68:71], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[72:75], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[76:79], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 16
lds_b16_loop_a:
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 lds_b16_tail_g0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[80:83], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[84:87], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[88:91], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[92:95], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[96:99], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[100:103], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[104:107], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[108:111], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[112:115], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[116:119], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[120:123], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[124:127], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[128:131], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[132:135], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[136:139], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[140:143], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 16
	s_waitcnt vmcnt(16)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[16:19]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_write_b128 v6, v[20:23]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_write_b128 v6, v[24:27]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_write_b128 v6, v[28:31]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_write_b128 v6, v[32:35]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_write_b128 v6, v[36:39]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_write_b128 v6, v[40:43]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_write_b128 v6, v[44:47]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4096
	ds_write_b128 v6, v[48:51]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4608
	ds_write_b128 v6, v[52:55]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5120
	ds_write_b128 v6, v[56:59]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5632
	ds_write_b128 v6, v[60:63]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6144
	ds_write_b128 v6, v[64:67]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6656
	ds_write_b128 v6, v[68:71]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7168
	ds_write_b128 v6, v[72:75]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7680
	ds_write_b128 v6, v[76:79]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_read_b128 v[16:19], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_read_b128 v[20:23], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_read_b128 v[24:27], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_read_b128 v[28:31], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_read_b128 v[32:35], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_read_b128 v[36:39], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_read_b128 v[40:43], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_read_b128 v[44:47], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4096
	ds_read_b128 v[48:51], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4608
	ds_read_b128 v[52:55], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5120
	ds_read_b128 v[56:59], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5632
	ds_read_b128 v[60:63], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6144
	ds_read_b128 v[64:67], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6656
	ds_read_b128 v[68:71], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7168
	ds_read_b128 v[72:75], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7680
	ds_read_b128 v[76:79], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v16, v150
	v_add_f32 v151, v17, v151
	v_add_f32 v152, v18, v152
	v_add_f32 v153, v19, v153
	v_add_f32 v150, v20, v150
	v_add_f32 v151, v21, v151
	v_add_f32 v152, v22, v152
	v_add_f32 v153, v23, v153
	v_add_f32 v150, v24, v150
	v_add_f32 v151, v25, v151
	v_add_f32 v152, v26, v152
	v_add_f32 v153, v27, v153
	v_add_f32 v150, v28, v150
	v_add_f32 v151, v29, v151
	v_add_f32 v152, v30, v152
	v_add_f32 v153, v31, v153
	v_add_f32 v150, v32, v150
	v_add_f32 v151, v33, v151
	v_add_f32 v152, v34, v152
	v_add_f32 v153, v35, v153
	v_add_f32 v150, v36, v150
	v_add_f32 v151, v37, v151
	v_add_f32 v152, v38, v152
	v_add_f32 v153, v39, v153
	v_add_f32 v150, v40, v150
	v_add_f32 v151, v41, v151
	v_add_f32 v152, v42, v152
	v_add_f32 v153, v43, v153
	v_add_f32 v150, v44, v150
	v_add_f32 v151, v45, v151
	v_add_f32 v152, v46, v152
	v_add_f32 v153, v47, v153
	v_add_f32 v150, v48, v150
	v_add_f32 v151, v49, v151
	v_add_f32 v152, v50, v152
	v_add_f32 v153, v51, v153
	v_add_f32 v150, v52, v150
	v_add_f32 v151, v53, v151
	v_add_f32 v152, v54, v152
	v_add_f32 v153, v55, v153
	v_add_f32 v150, v56, v150
	v_add_f32 v151, v57, v151
	v_add_f32 v152, v58, v152
	v_add_f32 v153, v59, v153
	v_add_f32 v150, v60, v150
	v_add_f32 v151, v61, v151
	v_add_f32 v152, v62, v152
	v_add_f32 v153, v63, v153
	v_add_f32 v150, v64, v150
	v_add_f32 v151, v65, v151
	v_add_f32 v152, v66, v152
	v_add_f32 v153, v67, v153
	v_add_f32 v150, v68, v150
	v_add_f32 v151, v69, v151
	v_add_f32 v152, v70, v152
	v_add_f32 v153, v71, v153
	v_add_f32 v150, v72, v150
	v_add_f32 v151, v73, v151
	v_add_f32 v152, v74, v152
	v_add_f32 v153, v75, v153
	v_add_f32 v150, v76, v150
	v_add_f32 v151, v77, v151
	v_add_f32 v152, v78, v152
	v_add_f32 v153, v79, v153
	v_add_nc_u32 v154, v154, 16
	s_xor_b32 s16, s16, s18
	s_cmp_eq_u32 s17, 0
	s_cbranch_scc1 lds_b16_tail_g1
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[16:19], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[20:23], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[24:27], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[28:31], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[32:35], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[36:39], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[40:43], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[44:47], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[48:51], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[52:55], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[56:59], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[60:63], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[64:67], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[68:71], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[72:75], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	v_add_co_u32 v4, vcc_lo, s12, v1
	v_add_co_ci_u32_e64 v5, null, s13, v3, vcc_lo
	global_load_dwordx4 v[76:79], v[4:5], off
	s_add_u32 s12, s12, 512
	s_addc_u32 s13, s13, 0
	s_sub_u32 s17, s17, 16
	s_waitcnt vmcnt(16)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[80:83]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_write_b128 v6, v[84:87]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_write_b128 v6, v[88:91]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_write_b128 v6, v[92:95]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_write_b128 v6, v[96:99]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_write_b128 v6, v[100:103]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_write_b128 v6, v[104:107]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_write_b128 v6, v[108:111]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4096
	ds_write_b128 v6, v[112:115]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4608
	ds_write_b128 v6, v[116:119]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5120
	ds_write_b128 v6, v[120:123]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5632
	ds_write_b128 v6, v[124:127]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6144
	ds_write_b128 v6, v[128:131]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6656
	ds_write_b128 v6, v[132:135]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7168
	ds_write_b128 v6, v[136:139]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7680
	ds_write_b128 v6, v[140:143]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_read_b128 v[80:83], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_read_b128 v[84:87], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_read_b128 v[88:91], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_read_b128 v[92:95], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_read_b128 v[96:99], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_read_b128 v[100:103], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_read_b128 v[104:107], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_read_b128 v[108:111], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4096
	ds_read_b128 v[112:115], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4608
	ds_read_b128 v[116:119], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5120
	ds_read_b128 v[120:123], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5632
	ds_read_b128 v[124:127], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6144
	ds_read_b128 v[128:131], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6656
	ds_read_b128 v[132:135], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7168
	ds_read_b128 v[136:139], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7680
	ds_read_b128 v[140:143], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v80, v150
	v_add_f32 v151, v81, v151
	v_add_f32 v152, v82, v152
	v_add_f32 v153, v83, v153
	v_add_f32 v150, v84, v150
	v_add_f32 v151, v85, v151
	v_add_f32 v152, v86, v152
	v_add_f32 v153, v87, v153
	v_add_f32 v150, v88, v150
	v_add_f32 v151, v89, v151
	v_add_f32 v152, v90, v152
	v_add_f32 v153, v91, v153
	v_add_f32 v150, v92, v150
	v_add_f32 v151, v93, v151
	v_add_f32 v152, v94, v152
	v_add_f32 v153, v95, v153
	v_add_f32 v150, v96, v150
	v_add_f32 v151, v97, v151
	v_add_f32 v152, v98, v152
	v_add_f32 v153, v99, v153
	v_add_f32 v150, v100, v150
	v_add_f32 v151, v101, v151
	v_add_f32 v152, v102, v152
	v_add_f32 v153, v103, v153
	v_add_f32 v150, v104, v150
	v_add_f32 v151, v105, v151
	v_add_f32 v152, v106, v152
	v_add_f32 v153, v107, v153
	v_add_f32 v150, v108, v150
	v_add_f32 v151, v109, v151
	v_add_f32 v152, v110, v152
	v_add_f32 v153, v111, v153
	v_add_f32 v150, v112, v150
	v_add_f32 v151, v113, v151
	v_add_f32 v152, v114, v152
	v_add_f32 v153, v115, v153
	v_add_f32 v150, v116, v150
	v_add_f32 v151, v117, v151
	v_add_f32 v152, v118, v152
	v_add_f32 v153, v119, v153
	v_add_f32 v150, v120, v150
	v_add_f32 v151, v121, v151
	v_add_f32 v152, v122, v152
	v_add_f32 v153, v123, v153
	v_add_f32 v150, v124, v150
	v_add_f32 v151, v125, v151
	v_add_f32 v152, v126, v152
	v_add_f32 v153, v127, v153
	v_add_f32 v150, v128, v150
	v_add_f32 v151, v129, v151
	v_add_f32 v152, v130, v152
	v_add_f32 v153, v131, v153
	v_add_f32 v150, v132, v150
	v_add_f32 v151, v133, v151
	v_add_f32 v152, v134, v152
	v_add_f32 v153, v135, v153
	v_add_f32 v150, v136, v150
	v_add_f32 v151, v137, v151
	v_add_f32 v152, v138, v152
	v_add_f32 v153, v139, v153
	v_add_f32 v150, v140, v150
	v_add_f32 v151, v141, v151
	v_add_f32 v152, v142, v152
	v_add_f32 v153, v143, v153
	s_xor_b32 s16, s16, s18
	s_branch lds_b16_loop_a
lds_b16_tail_g0:
	s_waitcnt vmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[16:19]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_write_b128 v6, v[20:23]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_write_b128 v6, v[24:27]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_write_b128 v6, v[28:31]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_write_b128 v6, v[32:35]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_write_b128 v6, v[36:39]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_write_b128 v6, v[40:43]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_write_b128 v6, v[44:47]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4096
	ds_write_b128 v6, v[48:51]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4608
	ds_write_b128 v6, v[52:55]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5120
	ds_write_b128 v6, v[56:59]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5632
	ds_write_b128 v6, v[60:63]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6144
	ds_write_b128 v6, v[64:67]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6656
	ds_write_b128 v6, v[68:71]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7168
	ds_write_b128 v6, v[72:75]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7680
	ds_write_b128 v6, v[76:79]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_read_b128 v[16:19], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_read_b128 v[20:23], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_read_b128 v[24:27], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_read_b128 v[28:31], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_read_b128 v[32:35], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_read_b128 v[36:39], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_read_b128 v[40:43], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_read_b128 v[44:47], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4096
	ds_read_b128 v[48:51], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4608
	ds_read_b128 v[52:55], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5120
	ds_read_b128 v[56:59], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5632
	ds_read_b128 v[60:63], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6144
	ds_read_b128 v[64:67], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6656
	ds_read_b128 v[68:71], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7168
	ds_read_b128 v[72:75], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7680
	ds_read_b128 v[76:79], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v16, v150
	v_add_f32 v151, v17, v151
	v_add_f32 v152, v18, v152
	v_add_f32 v153, v19, v153
	v_add_f32 v150, v20, v150
	v_add_f32 v151, v21, v151
	v_add_f32 v152, v22, v152
	v_add_f32 v153, v23, v153
	v_add_f32 v150, v24, v150
	v_add_f32 v151, v25, v151
	v_add_f32 v152, v26, v152
	v_add_f32 v153, v27, v153
	v_add_f32 v150, v28, v150
	v_add_f32 v151, v29, v151
	v_add_f32 v152, v30, v152
	v_add_f32 v153, v31, v153
	v_add_f32 v150, v32, v150
	v_add_f32 v151, v33, v151
	v_add_f32 v152, v34, v152
	v_add_f32 v153, v35, v153
	v_add_f32 v150, v36, v150
	v_add_f32 v151, v37, v151
	v_add_f32 v152, v38, v152
	v_add_f32 v153, v39, v153
	v_add_f32 v150, v40, v150
	v_add_f32 v151, v41, v151
	v_add_f32 v152, v42, v152
	v_add_f32 v153, v43, v153
	v_add_f32 v150, v44, v150
	v_add_f32 v151, v45, v151
	v_add_f32 v152, v46, v152
	v_add_f32 v153, v47, v153
	v_add_f32 v150, v48, v150
	v_add_f32 v151, v49, v151
	v_add_f32 v152, v50, v152
	v_add_f32 v153, v51, v153
	v_add_f32 v150, v52, v150
	v_add_f32 v151, v53, v151
	v_add_f32 v152, v54, v152
	v_add_f32 v153, v55, v153
	v_add_f32 v150, v56, v150
	v_add_f32 v151, v57, v151
	v_add_f32 v152, v58, v152
	v_add_f32 v153, v59, v153
	v_add_f32 v150, v60, v150
	v_add_f32 v151, v61, v151
	v_add_f32 v152, v62, v152
	v_add_f32 v153, v63, v153
	v_add_f32 v150, v64, v150
	v_add_f32 v151, v65, v151
	v_add_f32 v152, v66, v152
	v_add_f32 v153, v67, v153
	v_add_f32 v150, v68, v150
	v_add_f32 v151, v69, v151
	v_add_f32 v152, v70, v152
	v_add_f32 v153, v71, v153
	v_add_f32 v150, v72, v150
	v_add_f32 v151, v73, v151
	v_add_f32 v152, v74, v152
	v_add_f32 v153, v75, v153
	v_add_f32 v150, v76, v150
	v_add_f32 v151, v77, v151
	v_add_f32 v152, v78, v152
	v_add_f32 v153, v79, v153
	v_add_nc_u32 v154, v154, 16
	s_mul_i32 s11, s2, 1024
	s_add_u32 s14, s20, s11
	s_addc_u32 s15, s21, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[150:153], off
	v_lshlrev_b32 v7, 2, v0
	s_add_u32 s14, s14, 512
	v_add_co_u32 v4, vcc_lo, s14, v7
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dword v[4:5], v154, off offset:4
	v_mov_b32 v7, s17
	global_store_dword v[4:5], v7, off offset:8
	v_mov_b32 v7, s16
	global_store_dword v[4:5], v7, off offset:12
	s_endpgm
lds_b16_tail_g1:
	s_waitcnt vmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_write_b128 v6, v[80:83]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_write_b128 v6, v[84:87]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_write_b128 v6, v[88:91]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_write_b128 v6, v[92:95]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_write_b128 v6, v[96:99]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_write_b128 v6, v[100:103]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_write_b128 v6, v[104:107]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_write_b128 v6, v[108:111]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4096
	ds_write_b128 v6, v[112:115]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4608
	ds_write_b128 v6, v[116:119]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5120
	ds_write_b128 v6, v[120:123]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5632
	ds_write_b128 v6, v[124:127]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6144
	ds_write_b128 v6, v[128:131]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6656
	ds_write_b128 v6, v[132:135]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7168
	ds_write_b128 v6, v[136:139]
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7680
	ds_write_b128 v6, v[140:143]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v6, vcc_lo, s16, v1
	ds_read_b128 v[80:83], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 512
	ds_read_b128 v[84:87], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1024
	ds_read_b128 v[88:91], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 1536
	ds_read_b128 v[92:95], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2048
	ds_read_b128 v[96:99], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 2560
	ds_read_b128 v[100:103], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3072
	ds_read_b128 v[104:107], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 3584
	ds_read_b128 v[108:111], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4096
	ds_read_b128 v[112:115], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 4608
	ds_read_b128 v[116:119], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5120
	ds_read_b128 v[120:123], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 5632
	ds_read_b128 v[124:127], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6144
	ds_read_b128 v[128:131], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 6656
	ds_read_b128 v[132:135], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7168
	ds_read_b128 v[136:139], v6
	v_add_co_u32 v6, vcc_lo, s16, v1
	v_add_nc_u32 v6, v6, 7680
	ds_read_b128 v[140:143], v6
	s_waitcnt lgkmcnt(0)
	v_add_f32 v150, v80, v150
	v_add_f32 v151, v81, v151
	v_add_f32 v152, v82, v152
	v_add_f32 v153, v83, v153
	v_add_f32 v150, v84, v150
	v_add_f32 v151, v85, v151
	v_add_f32 v152, v86, v152
	v_add_f32 v153, v87, v153
	v_add_f32 v150, v88, v150
	v_add_f32 v151, v89, v151
	v_add_f32 v152, v90, v152
	v_add_f32 v153, v91, v153
	v_add_f32 v150, v92, v150
	v_add_f32 v151, v93, v151
	v_add_f32 v152, v94, v152
	v_add_f32 v153, v95, v153
	v_add_f32 v150, v96, v150
	v_add_f32 v151, v97, v151
	v_add_f32 v152, v98, v152
	v_add_f32 v153, v99, v153
	v_add_f32 v150, v100, v150
	v_add_f32 v151, v101, v151
	v_add_f32 v152, v102, v152
	v_add_f32 v153, v103, v153
	v_add_f32 v150, v104, v150
	v_add_f32 v151, v105, v151
	v_add_f32 v152, v106, v152
	v_add_f32 v153, v107, v153
	v_add_f32 v150, v108, v150
	v_add_f32 v151, v109, v151
	v_add_f32 v152, v110, v152
	v_add_f32 v153, v111, v153
	v_add_f32 v150, v112, v150
	v_add_f32 v151, v113, v151
	v_add_f32 v152, v114, v152
	v_add_f32 v153, v115, v153
	v_add_f32 v150, v116, v150
	v_add_f32 v151, v117, v151
	v_add_f32 v152, v118, v152
	v_add_f32 v153, v119, v153
	v_add_f32 v150, v120, v150
	v_add_f32 v151, v121, v151
	v_add_f32 v152, v122, v152
	v_add_f32 v153, v123, v153
	v_add_f32 v150, v124, v150
	v_add_f32 v151, v125, v151
	v_add_f32 v152, v126, v152
	v_add_f32 v153, v127, v153
	v_add_f32 v150, v128, v150
	v_add_f32 v151, v129, v151
	v_add_f32 v152, v130, v152
	v_add_f32 v153, v131, v153
	v_add_f32 v150, v132, v150
	v_add_f32 v151, v133, v151
	v_add_f32 v152, v134, v152
	v_add_f32 v153, v135, v153
	v_add_f32 v150, v136, v150
	v_add_f32 v151, v137, v151
	v_add_f32 v152, v138, v152
	v_add_f32 v153, v139, v153
	v_add_f32 v150, v140, v150
	v_add_f32 v151, v141, v151
	v_add_f32 v152, v142, v152
	v_add_f32 v153, v143, v153
	v_add_nc_u32 v154, v154, 16
	s_mul_i32 s11, s2, 1024
	s_add_u32 s14, s20, s11
	s_addc_u32 s15, s21, 0
	v_add_co_u32 v4, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dwordx4 v[4:5], v[150:153], off
	v_lshlrev_b32 v7, 2, v0
	s_add_u32 s14, s14, 512
	v_add_co_u32 v4, vcc_lo, s14, v7
	v_add_co_ci_u32_e64 v5, null, s15, v3, vcc_lo
	global_store_dword v[4:5], v154, off offset:4
	v_mov_b32 v7, s17
	global_store_dword v[4:5], v7, off offset:8
	v_mov_b32 v7, s16
	global_store_dword v[4:5], v7, off offset:12
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel lds_b16
		.amdhsa_group_segment_fixed_size 16384
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_private_segment_buffer 0
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 156
		.amdhsa_next_free_sgpr 22
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel

	.text
	.p2align	8
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 512
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           plain1
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         plain1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     156
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           df4_b4
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         df4_b4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     156
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           df4_b8
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         df4_b8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     156
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           df4_b16
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         df4_b16.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     156
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           lds_b4
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         lds_b4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     156
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           lds_b8
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         lds_b8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     156
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           lds_b16
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         lds_b16.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     156
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1030
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
