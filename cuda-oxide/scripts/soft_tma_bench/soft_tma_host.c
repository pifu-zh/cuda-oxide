/* Host driver for the hand-written ISA soft-TMA microbenchmark.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Loads soft_tma_kernels.o (one code object, seven kernels), runs each
 * variant over the same buffers, verifies memcpy semantics, reports best-of
 * effective bandwidth (read N + write N = 2N DRAM bytes).
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(_e), __LINE__); return 1; } } while (0)

static const struct { const char* name; int batch; int reduce; } VARIANTS[] = {
    {"plain1", 1, 1}, {"df4_b4", 4, 0}, {"df4_b8", 8, 0}, {"df4_b16", 16, 0},
    {"lds_b4", 4, 1}, {"lds_b8", 8, 1}, {"lds_b16", 16, 1},
};
#define NV ((int)(sizeof(VARIANTS)/sizeof(VARIANTS[0])))

int main(int argc, char** argv) {
    CK(hipInit(0));
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    printf("device: %s SMs=%d sclk=%dkHz mclk=%dkHz\n", p.name,
           p.multiProcessorCount, p.clockRate, p.memoryClockRate);
    /* Empirical bandwidth ceiling of this card is ~576 GB/s: pure-read peaked at
     * 98.3% of 575.5 (2026-11 archive), triad at 90-95%, and the soft-TMA gmem->LDS
     * kernel below at 96.1% (553/575.5) over cold 255MiB regions. The spec-sheet
     * "6950 XT" name implies 512-bit@18G (1152), but device 0x73a5 with 16GB is a
     * 256-bit Navi21 memory config; the ROCm-reported memoryBusWidth=256 is correct. */
    double theory = 16.0 * p.memoryClockRate * 1000.0 * (p.memoryBusWidth / 8.0);
    printf("theory: %.1f GB/s\n", theory / 1e9);

    int blocks = p.multiProcessorCount * 8;   /* 8 waves/CU */
    if (argc > 2) blocks = atoi(argv[2]);
    const long long total_bytes = 256ll << 20;  /* per-rotation region: 2x L2 */
    const int ROTATIONS = 8;                    /* 8 regions x 256MiB = 2GiB: defeats L2 */
    /* kernel-side "quad" = one wave instruction = 32 lanes x 16B = 512B */
    long long n4 = total_bytes / 512;               /* wave-quads */
    long long qpb = n4 / blocks;                    /* wave-quads per block */
    qpb -= qpb % 32;                                /* divisible by max batch 16 */
    if (argc > 3) qpb = atoll(argv[3]);
    n4 = qpb * blocks;
    const long long bytes = n4 * 512;
    printf("grid=%d blocks, quads/block=%lld, moved=%lld MiB/block-set (r+w)\n",
           blocks, qpb, bytes >> 20);

    float *src, *dst;
    CK(hipMalloc((void**)&src, bytes * ROTATIONS));  /* 2 GiB */
    CK(hipMalloc((void**)&dst, bytes));
    {
        float* h = (float*)malloc(bytes * ROTATIONS);
        for (long long i = 0; i < bytes * ROTATIONS / 4; i++) h[i] = (float)(i % 1000) * 0.5f;
        CK(hipMemcpy(src, h, bytes * ROTATIONS, hipMemcpyHostToDevice));
        free(h);
    }

    void* image = NULL; long long image_len = 0;
    {
        FILE* f = fopen("/home/zhuo/tmp-prof/stma/soft_v5.hsaco", "rb");
        if (!f) { printf("cannot open code object\n"); return 1; }
        fseek(f, 0, SEEK_END); image_len = ftell(f); fseek(f, 0, SEEK_SET);
        image = malloc(image_len);
        if (fread(image, 1, image_len, f) != (size_t)image_len) { printf("short read\n"); return 1; }
        fclose(f);
    }

    hipModule_t mod; CK(hipModuleLoadData(&mod, image));

    struct { void* src; void* src0; void* dst; long long qpb; } kargs = { src, src, dst, qpb };

    /* warm the DVFS state: long loop using the first kernel */
    { const char* nw = getenv("STMA_NOWARM");
    if (!nw) {
        hipFunction_t fn; CK(hipModuleGetFunction(&fn, mod, "plain1"));
        void* args[] = { &kargs.src, &kargs.dst, &kargs.qpb };
        for (int w = 0; w < 1500; w++)
            CK(hipModuleLaunchKernel(fn, blocks, 1, 1, 32, 1, 1, 0, 0, args, NULL));
        CK(hipDeviceSynchronize());
    } }

    printf("\n%-10s %10s %12s %9s %9s\n", "variant", "ms", "GB/s(r+w)", "%theory", "verify");
    for (int v = 0; v < NV; v++) {
        if (argc > 1 && strcmp(VARIANTS[v].name, argv[1]) != 0) continue;
        hipFunction_t fn;
        hipError_t e = hipModuleGetFunction(&fn, mod, VARIANTS[v].name);
        if (e != hipSuccess) { printf("%-10s  (not found: %s)\n", VARIANTS[v].name, hipGetErrorString(e)); continue; }
        void* args[] = { &kargs.src, &kargs.dst, &kargs.qpb };
        float times[ROTATIONS];
        hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
        int timed_n = ROTATIONS; { const char* tn = getenv("STMA_TIMED_N"); if (tn) timed_n = atoi(tn); }
        int timed_rot = 1; { const char* tr = getenv("STMA_TIMED_ROT"); if (tr) timed_rot = atoi(tr); }
        for (int r = 0; r < timed_n; r++) {
            kargs.src = (void*)((float*)kargs.src0 + (long long)(timed_rot ? r : 0) * (bytes / 4));
            CK(hipEventRecord(ev0, 0));
            CK(hipModuleLaunchKernel(fn, blocks, 1, 1, 32, 1, 1, 0, 0, args, NULL));
            CK(hipEventRecord(ev1, 0)); CK(hipEventSynchronize(ev1));
            CK(hipDeviceSynchronize());  /* hard drain: overlapping passes corrupt dst and the clock */
            CK(hipEventElapsedTime(&times[r], ev0, ev1));
        }
        CK(hipEventDestroy(ev0)); CK(hipEventDestroy(ev1));
        /* median of one pass per region: no pass re-reads a warm region */
        for (int a = 0; a < ROTATIONS; a++) for (int b2 = a+1; b2 < ROTATIONS; b2++)
            if (times[b2] < times[a]) { float t = times[a]; times[a] = times[b2]; times[b2] = t; }
        float best = times[ROTATIONS/2];
        /* reduce variants move N (gmem->LDS); full-copy variants move 2N */
        double traffic = VARIANTS[v].reduce ? (double)bytes : (2.0 * bytes);
        double gbs = traffic / (best * 1e6);
        int bad = 0; long long firstbad = -1; float gv = 0, ev = 0;
        kargs.src = kargs.src0;
        if (VARIANTS[v].reduce) {
            /* per-lane 16B result at dst[block*512 + lane*16]:
             * sum over this block's quads of src_f[quad_base + lane*4 + j] */
            CK(hipMemset(dst, 0, blocks * 1024));
            CK(hipModuleLaunchKernel(fn, blocks, 1, 1, 32, 1, 1, 0, 0, args, NULL));
            CK(hipDeviceSynchronize());
            float* h = (float*)malloc(blocks * 1024);
            float* hsrc = (float*)malloc(bytes);
            CK(hipMemcpy(h, dst, blocks * 1024, hipMemcpyDeviceToHost));
            CK(hipMemcpy(hsrc, src, bytes, hipMemcpyDeviceToHost));
            for (int blk = 0; blk < blocks && !bad; blk++) {
                long long q0 = (long long)blk * qpb;
                for (int lane = 0; lane < 32 && !bad; lane++) {
                    for (int j = 0; j < 4; j++) {
                        double sum = 0;
                        for (long long k = 0; k < qpb; k++)
                            sum += hsrc[(q0 + k) * 128 + lane * 4 + j];
                        double expect = sum;
                        double got = h[blk * 256 + lane * 4 + j];
                        if (got - expect > expect * 0.05 || expect - got > expect * 0.05) {
                            bad = 1; firstbad = (long long)blk * 256 + lane * 4 + j;
                            gv = got; ev = expect;
                        }
                    }
                }
            }
            free(h); free(hsrc);
        } else {
            /* verify launch: region 0 (the timed passes rotated regions --
               their dst writes are per-region and would false-FAIL the
               memcmp against region 0's reference) */
            kargs.src = kargs.src0;
            CK(hipMemset(dst, 0, bytes));  /* sentinel: whole dst */
            CK(hipDeviceSynchronize());  /* barrier: the big fill races the kernel otherwise */
            CK(hipModuleLaunchKernel(fn, blocks, 1, 1, 32, 1, 1, 0, 0, args, NULL));
            CK(hipDeviceSynchronize());
            float* h = (float*)malloc(bytes);
            float* hsrc = (float*)malloc(bytes);
            CK(hipMemcpy(h, dst, bytes, hipMemcpyDeviceToHost));
            CK(hipMemcpy(hsrc, src, bytes, hipMemcpyDeviceToHost));
            bad = memcmp(h, hsrc, bytes) != 0;
            if (bad) {
                for (long long i = 0; i < bytes/4; i++) if (h[i] != hsrc[i]) { firstbad = i; gv = h[i]; ev = hsrc[i]; break; }
            }
            free(h); free(hsrc);
        }
        printf("%-10s %10.3f %12.1f %8.1f%% %9s", VARIANTS[v].name, best, gbs,
               100.0 * gbs / (theory / 1e9), bad ? "FAIL" : "ok");
        if (bad) printf(" (mismatch @ %lld: %g != %g)", firstbad, gv, ev);
        printf("  passes:");
        for (int a = 0; a < ROTATIONS; a++) printf(" %.3f", times[a]);
        printf("\n");
    }
    return 0;
}
