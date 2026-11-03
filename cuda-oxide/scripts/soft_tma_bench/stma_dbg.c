/* Per-wave completion debugger for the soft-TMA kernels.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Answers, per launch and per block: did the wave complete its loop?
 *   reduce (lds_bN): result region carries cnt (quads reduced), s17 (left to
 *     issue), s16 (LDS slot) per block at +512+4 (lane 0). cnt==qpb complete.
 *   copy (df4_bN): per-block memcmp against the launch's source region.
 * Modes:
 *   dump  memset/sync between launches: single-launch behavior, sleep-recheck
 *         on region 0 (readback, sleep, readback again: a difference proves
 *         the kernel was still running after sync returned).
 *   back  all launches back-to-back, no sync/memset between (the benchmark's
 *         timed-loop pattern minus events); dst after = last region exactly
 *         iff launches serialized on the device.
 *   timed the benchmark's exact event pattern per launch.
 * mclk DPM state is sampled around every launch (index 3 = 1124 MHz max here);
 * DVFS state is reported per launch so confounded passes are visible.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    printf("HIP err %s @%d\n", hipGetErrorString(_e), __LINE__); return 1; } } while (0)

typedef struct { void* src; void* src0; void* dst; long long qpb; } KA;

static int mclk_state(void) {
    FILE* f = fopen("/sys/class/drm/card1/device/pp_dpm_mclk", "r");
    if (!f) return -1;
    char buf[256];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    int cur = -1;
    for (char* p = strtok(buf, "\n"); p; p = strtok(NULL, "\n"))
        if (strchr(p, '*')) sscanf(p, "%d:", &cur);
    return cur;
}

static void* img_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("no hsaco\n"); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc(n);
    if (fread(p, 1, n, f) != (size_t)n) exit(1);
    fclose(f);
    return p;
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: stma_dbg <variant> [blocks] [qpb] [mode]\n"); return 1; }
    const char* variant = argv[1];
    const char* mode = argc > 4 ? argv[4] : "dump";
    int reduce = !strncmp(variant, "lds_", 4) || !strcmp(variant, "plain1");
    int pinr = argc > 5 ? atoi(argv[5]) : -1;   /* >=0: pin region for all launches */
    int R = argc > 7 ? atoi(argv[7]) : 8;
    int warm = argc > 8 ? atoi(argv[8]) : 1;
    CK(hipInit(0));
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    int blocks = argc > 2 ? atoi(argv[2]) : p.multiProcessorCount * 8;
    long long qpb = argc > 3 ? atoll(argv[3]) : 1632;
    long long bytes = qpb * 512 * (long long)blocks;  /* copy footprint per region: a quad = 32 lanes x 16B */
    long long resb = (long long)blocks * 1024;        /* result region (reduce) */
    long long dstb = reduce ? resb : bytes;
    printf("variant=%s mode=%s blocks=%d qpb=%lld dst=%.1fMiB\n",
           variant, mode, blocks, qpb, dstb / 1048576.0);

    float* src, * dst;
    CK(hipMalloc((void**)&src, bytes * 8));
    CK(hipMalloc((void**)&dst, dstb));
    printf("VA: src=%p (span %.1f MiB) dst=%p (span %.1f MiB) dst_end_gap_to_src=%lld\n",
           (void*)src, bytes * 8 / 1048576.0, (void*)dst, dstb / 1048576.0,
           (long long)((char*)src - ((char*)dst + dstb)));
    {
        float* h = (float*)malloc(bytes * 8);
        for (long long i = 0; i < bytes * 8 / 4; i++) h[i] = (float)(i % 1000) * 0.5f;
        CK(hipMemcpy(src, h, bytes * 8, hipMemcpyHostToDevice));
        free(h);
    }
    hipModule_t mod; CK(hipModuleLoadData(&mod, img_load("/home/zhuo/tmp-prof/stma/soft_v5.hsaco")));
    hipFunction_t fn; CK(hipModuleGetFunction(&fn, mod, variant));
    KA ka = { src, src, dst, qpb };
    void* args[] = { &ka.src, &ka.dst, &ka.qpb };
    static KA kas;                              /* static storage: detect deferred kernarg deref */
    kas = ka;
    void* args2[] = { &kas.src, &kas.dst, &kas.qpb };
    void** argp = argc > 6 ? args2 : args;      /* argv[6] present -> static struct */
    /* DVFS warm, mirroring the benchmark */
    if (warm) {
        hipFunction_t w; CK(hipModuleGetFunction(&w, mod, "plain1"));
        for (int i = 0; i < 1500; i++)
            CK(hipModuleLaunchKernel(w, blocks, 1, 1, 32, 1, 1, 0, 0, args, NULL));
        CK(hipDeviceSynchronize());
    }
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    if (!strcmp(mode, "bench")) {
        /* exact benchmark predecessor: 8 rotated timed passes, no memsets */
        int npass = 8; { const char* tn = getenv("DBG_TIMED_N"); if (tn) npass = atoi(tn); }
        for (int r = 0; r < npass; r++) {
            ka.src = (void*)((float*)ka.src0 + r * (bytes / 4));
            CK(hipEventRecord(e0, 0));
            CK(hipModuleLaunchKernel(fn, blocks, 1, 1, 32, 1, 1, 0, 0, args, NULL));
            CK(hipEventRecord(e1, 0)); CK(hipEventSynchronize(e1));
            CK(hipDeviceSynchronize());
        }
        /* verify, benchmark-style */
        { hipEvent_t d0, d1; CK(hipEventCreate(&d0)); CK(hipEventCreate(&d1)); CK(hipEventDestroy(d0)); CK(hipEventDestroy(d1)); }
        ka.src = ka.src0;
        CK(hipMemset(dst, 0, dstb));
        CK(hipDeviceSynchronize());
        CK(hipModuleLaunchKernel(fn, blocks, 1, 1, 32, 1, 1, 0, 0, args, NULL));
        CK(hipDeviceSynchronize());
        void* hh = (void*)malloc(dstb);
        CK(hipMemcpy(hh, dst, dstb, hipMemcpyDeviceToHost));
        int comp = 0, part = 0, shown2 = 0;
        printf("    BENCH hh[1000]=%g hh[20000000]=%g (pattern: 500 10000)\n",
               ((const float*)hh)[1000], ((const float*)hh)[20000000]);
        for (int b = 0; b < blocks; b++) {
            const unsigned char* hb = (const unsigned char*)hh + (long long)b * qpb * 512;
            const unsigned char* rref = (const unsigned char*)src + (long long)b * qpb * 512;
            if (memcmp(hb, rref, qpb * 16) == 0) comp++;
            else {
                part++;
                if (shown2++ < 3) {
                    long long bad = -1;
                    for (long long i = 0; i < qpb * 16; i += 4)
                        if (*(const unsigned*)(hb + i) != *(const unsigned*)(rref + i)) { bad = i; break; }
                    printf("    BENCH blk%d hole at byte %lld (quad %lld) dstval=%u refval=%u\n",
                           b, bad, bad / 512, *(const unsigned*)(hb + bad), *(const unsigned*)(rref + bad));
                }
            }
        }
        printf("  BENCH verify: complete=%d partial=%d\n", comp, part);
        free(hh);
        return 0;
    }
    for (int r = 0; r < R; r++) {
        long long idx = pinr >= 0 ? pinr : r;
        ka.src = (void*)((float*)ka.src0 + idx * (bytes / 4));
        kas.src = ka.src;
        if (strcmp(mode, "bench") && strcmp(mode, "back") != 0) CK(hipMemset(dst, 0, dstb));
        if (!strcmp(mode, "synced")) CK(hipDeviceSynchronize());   /* SDMA/compute ordering barrier */
        float ms = -1;
        int clk0 = mclk_state();
        if (!strcmp(mode, "timed")) {
            CK(hipEventRecord(e0, 0));
            CK(hipModuleLaunchKernel(fn, blocks, 1, 1, 32, 1, 1, 0, 0, argp, NULL));
            CK(hipEventRecord(e1, 0)); CK(hipEventSynchronize(e1));
            CK(hipEventElapsedTime(&ms, e0, e1));
        } else {
            CK(hipModuleLaunchKernel(fn, blocks, 1, 1, 32, 1, 1, 0, 0, argp, NULL));
            if (!strcmp(mode, "back") && r < R - 1) { printf("  r%d enqueued (no sync)\n", r); continue; }
            CK(hipDeviceSynchronize());
        }
        if (!strcmp(mode, "synced")) {
            /* full drain between EVERY op: memset -> sync, launch -> sync */
            CK(hipDeviceSynchronize());
        }
        int clk1 = mclk_state();
        void* h = (void*)malloc(dstb);
        CK(hipMemcpy(h, dst, dstb, hipMemcpyDeviceToHost));
        int changed = 0;
        void* h2 = NULL;
        if (!strcmp(mode, "dump") && r == 0) {
            usleep(200000);
            h2 = (void*)malloc(dstb);
            CK(hipMemcpy(h2, dst, dstb, hipMemcpyDeviceToHost));
            changed = memcmp(h, h2, dstb) != 0;
        }
        int complete = 0, partial = 0, untouched = 0, shown = 0;
        long long firstbad_global = -1;
        for (int b = 0; b < blocks; b++) {
            const unsigned char* hb = (const unsigned char*)h +
                (long long)b * (reduce ? 1024 : qpb * 512);
            const unsigned char* ref = (const unsigned char*)src +
                ((pinr >= 0 ? pinr : r) * bytes + (long long)b * qpb * 512);
            int is_complete = 0, is_untouched = 0;
            long long bad_off = -1;
            if (reduce) {
                unsigned cnt = *(const unsigned*)(hb + 516);
                is_complete = (cnt == (unsigned)qpb);
                is_untouched = (cnt == 0);
                if (!is_complete && !is_untouched) bad_off = cnt;
            } else {
                long long z = 0;
                for (long long i = 0; i < qpb * 16; i += 64)
                    z += *(const unsigned*)(hb + i) == 0;
                is_untouched = (z == qpb * 16 / 64);
                if (!is_untouched) {
                    if (memcmp(hb, ref, qpb * 16) != 0) {
                        for (long long i = 0; i < qpb * 16; i += 4)
                            if (*(const unsigned*)(hb + i) != *(const unsigned*)(ref + i)) { bad_off = i; break; }
                        /* identify which region (if any) the block actually holds */
                        for (int rr = 0; rr < 8 && b == shown; rr++) {
                            const unsigned char* rref = (const unsigned char*)src +
                                ((long long)rr * (bytes / 4) + (long long)b * qpb * 512);
                            if (memcmp(hb, rref, qpb * 16) == 0) { printf("    blk%d actually = region %d\n", b, rr); break; }
                        }
                    } else is_complete = 1;
                }
                if (firstbad_global < 0 && bad_off >= 0)
                    firstbad_global = (long long)b * qpb * 512 + bad_off;
            }
            if (is_complete) complete++;
            else if (is_untouched) untouched++;
            else {
                partial++;
                if (shown < 3) {
                    shown++;
                    if (reduce) {
                        unsigned cnt = *(const unsigned*)(hb + 516);
                        unsigned s17 = *(const unsigned*)(hb + 520);
                        unsigned s16 = *(const unsigned*)(hb + 524);
                        printf("    blk%d PARTIAL cnt=%u (qpb=%lld) s17=%u s16=%u\n", b, cnt, qpb, s17, s16);
                    } else {
                        printf("    blk%d PARTIAL firstbad=%lld (byte in block, quad %lld)\n",
                               b, bad_off, bad_off / 512);
                    }
                }
            }
        }
        printf("  r%d clk[%d->%d]", r, clk0, clk1);
        if (ms >= 0) printf(" ms=%.3f", ms);
        printf(" complete=%d partial=%d untouched=%d firstbad=%lld\n",
               complete, partial, untouched, firstbad_global);
        if (reduce) {   /* block 0 lane 0 sum fingerprint + counter region */
            const float* b0 = (const float*)h;
            const unsigned char* cb = (const unsigned char*)h + 512;
            printf("    blk0 lane0 sums=[%g %g %g %g] cnt=%u s17=%u s16=%u\n",
                   b0[0], b0[1], b0[2], b0[3],
                   *(const unsigned*)(cb + 4), *(const unsigned*)(cb + 8), *(const unsigned*)(cb + 12));
        }
        if (changed) printf("    !! dst CHANGED after 200ms sleep post-sync: kernel was still running\n");
        {   /* raw value observation: dst[0..3] vs every region's expected head */
            const float* d0 = (const float*)h;
            printf("    dst[0..3]=[%g %g %g %g] expect r%d=[", d0[0], d0[1], d0[2], d0[3],
                   pinr >= 0 ? pinr : r);
            for (int rr = 0; rr < 8; rr++) {
                long long off = (long long)rr * (bytes / 4);
                printf("%s%g", rr ? " " : "", ((const float*)src)[off] + 0.0f);
            }
            printf("] (region head values)\n");
        }
        free(h);
        free(h2);
    }
    return 0;
}
