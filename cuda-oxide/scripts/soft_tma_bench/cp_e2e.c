/* E2E driver for the cp.async soft-emulation kernels.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do { hipError_t _e=(x); if (_e) { printf("ERR %s @%d\n", hipGetErrorString(_e), __LINE__); return 1; } } while(0)

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: cp_e2e <hsaco> <kernel>\n"); return 1; }
  CK(hipInit(0));
  hipModule_t mod;
  {
    FILE* f = fopen(argv[1], "rb");
    if (!f) { printf("no file\n"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* img = malloc(n);
    if (fread(img, 1, n, f) != (size_t)n) return 1;
    fclose(f);
    CK(hipModuleLoadData(&mod, img));
  }
  hipFunction_t fn;
  CK(hipModuleGetFunction(&fn, mod, argv[2]));

  unsigned h_in[32];
  for (int i = 0; i < 32; i++) h_in[i] = 100 + i;
  unsigned *d_in, *d_out;
  CK(hipMalloc((void**)&d_in, sizeof(h_in)));
  CK(hipMalloc((void**)&d_out, 64));
  CK(hipMemcpy(d_in, h_in, sizeof(h_in), hipMemcpyHostToDevice));
  CK(hipMemset(d_out, 0, 64));

  struct { void* in; unsigned long long in_len; void* out; unsigned long long out_len;
           unsigned nx, ny, nz; } ka = { d_in, sizeof(h_in), d_out, 64, 1, 1, 1 };
  void* args[] = { &ka.in, &ka.in_len, &ka.out, &ka.out_len, &ka.nx, &ka.ny, &ka.nz };
  CK(hipModuleLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, 0, args, NULL));
  hipError_t se = hipDeviceSynchronize();
  if (se) { printf("sync: %s\n", hipGetErrorString(se)); return 1; }

  unsigned h_out[16];
  CK(hipMemcpy(h_out, d_out, 64, hipMemcpyDeviceToHost));

  /* per-kernel expectations:
   * test_cp_async_4      writes out[0] only (one 4B cp.async)
   * compile_all          has NO out param (forms exercise; no crash = pass)
   * mbarrier_bridges     writes out[0..3] through three bridge sequences   */
  int ok; unsigned exp0;
  if (strcmp(argv[2], "test_cp_async_4") == 0) {
    ok = (h_out[0] == h_in[0]); exp0 = h_in[0];
  } else if (strcmp(argv[2], "compile_all_classic_cp_async") == 0) {
    ok = 1; exp0 = 0;
  } else {
    ok = memcmp(h_out, h_in, 16) == 0; exp0 = h_in[0];
  }
  printf("%s: out[0] = %u (expect %u) %s\n", argv[2], h_out[0], exp0, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
