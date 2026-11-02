/* Debug dump for soft-TMA microbenchmark kernels.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
  if (hipInit(0)) { printf("init fail\n"); return 1; }
  hipModule_t mod;
  {
    FILE* f = fopen("soft_v5.hsaco", "rb");
    if (!f) { printf("no hsaco\n"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* img = malloc(n);
    if (fread(img, 1, n, f) != (size_t)n) { printf("short read\n"); return 1; }
    fclose(f);
    if (hipModuleLoadData(&mod, img)) { printf("load fail\n"); return 1; }
  }
  hipFunction_t fn;
  if (hipModuleGetFunction(&fn, mod, argv[1])) { printf("fn fail: %s\n", argv[1]); return 1; }

  int blocks = 4;
  long long qpb = 512;
  float *src, *dst;
  if (hipMalloc((void**)&src, (size_t)(blocks * qpb * 16))) { printf("malloc src\n"); return 1; }
  if (hipMalloc((void**)&dst, (size_t)(blocks * 1024))) { printf("malloc dst\n"); return 1; }
  {
    float* h = (float*)malloc((size_t)(blocks * qpb * 16));
    for (long long i = 0; i < blocks * qpb * 4; i++) h[i] = (float)(i % 1000) * 0.5f;
    if (hipMemcpy(src, h, (size_t)(blocks * qpb * 16), hipMemcpyHostToDevice)) { printf("cpy src\n"); return 1; }
    free(h);
  }
  if (hipMemset(dst, 0, (size_t)(blocks * 1024))) { printf("memset\n"); return 1; }

  struct { void* s; void* d; long long q; } ka = { src, dst, qpb };
  void* args[] = { &ka.s, &ka.d, &ka.q };
  hipError_t le = hipModuleLaunchKernel(fn, blocks, 1, 1, 32, 1, 1, 0, 0, args, NULL);
  if (le) { printf("launch: %s\n", hipGetErrorString(le)); return 1; }
  hipError_t se = hipDeviceSynchronize();
  if (se) { printf("sync: %s\n", hipGetErrorString(se)); return 1; }

  float* r = (float*)malloc((size_t)(blocks * 1024));
  if (hipMemcpy(r, dst, (size_t)(blocks * 1024), hipMemcpyDeviceToHost)) { printf("cpy r\n"); return 1; }

  for (int b = 0; b < blocks; b++) {
    float* base = r + b * 256;              /* 1024B block region */
    unsigned cnt = *(unsigned*)(base + 129); /* count region byte 516: lane0 */
    unsigned s17v = *(unsigned*)(base + 130);
    unsigned s16v = *(unsigned*)(base + 131);
    printf("blk%d: lane0 sum=[%g %g %g %g] lane1 sum=[%g %g %g %g] cnt=%u s17=%u s16=%u\n",
           b, base[0], base[1], base[2], base[3],
           base[4], base[5], base[6], base[7], cnt, s17v, s16v);
  }
  /* expected per block/lane: sum over qpb quads of src_f[(b*qpb+k)*4 + lane*4 + j] */
  {
    float* h = (float*)malloc((size_t)(blocks * qpb * 16));
    if (hipMemcpy(h, src, (size_t)(blocks * qpb * 16), hipMemcpyDeviceToHost)) return 1;
    for (int b = 0; b < 2; b++) {
      printf("exp blk%d lane0=[", b);
      for (int j = 0; j < 4; j++) {
        double sum = 0;
        for (long long k = 0; k < qpb; k++) sum += h[((long long)b * qpb + k) * 4 + j];
        printf("%g ", sum);
      }
      printf("] lane1=[");
      for (int j = 0; j < 4; j++) {
        double sum = 0;
        for (long long k = 0; k < qpb; k++) sum += h[((long long)b * qpb + k) * 4 + 4 + j];
        printf("%g ", sum);
      }
      printf("]\n");
    }
    free(h);
  }
  return 0;
}
