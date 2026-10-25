/* [goal Phase D] gfx1030 performance baseline bench (ROCm 10.1).
 * Direct HIP link (headers from ~/opt/rocm-headers, lib from /opt/rocm/lib).
 * Measures: H2D/D2H/D2D bandwidth (64 MB, best-of-20), triad kernel
 * effective bandwidth (c = a + 2*b, f32, 64M elems), FFMA peak estimate.
 * Timing via hipEvent — DVFS-aware: report measured numbers only.
 */
#include <hip/hip_runtime.h>
/* hipcc 编译（gcc 无设备码前端）；CPATH 提供 HIP 头 */
#include <stdio.h>
#include <stdlib.h>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); return 1; } } while (0)

__global__ void triad(const float *a, const float *b, float *c, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + 2.0f * b[i];
}

/* FFMA: each thread chains K independent FMA streams on registers. */
#define K 64
__global__ void ffma_peak(float *out, float seed, int iters) {
    float v[K];
    for (int j = 0; j < K; j++) v[j] = seed + threadIdx.x * 1e-9f + j;
    for (int it = 0; it < iters; it++)
        for (int j = 0; j < K; j++) v[j] = v[j] * 1.0000001f + 1e-7f;
    float s = 0;
    for (int j = 0; j < K; j++) s += v[j];
    if (s == 12345.678f) out[0] = s; /* never true; defeats DCE */
}

static double best_ms, med_ms;
static void time_events(hipEvent_t s, hipEvent_t e, double *out) {
    float ms; hipEventElapsedTime(&ms, s, e); *out = ms;
}

int main(void) {
    CK(hipInit(0));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    printf("device: %s  SMs=%d  clock=%dkHz  memClock=%dkHz  busWidth=%dbit\n",
           prop.name, prop.multiProcessorCount, prop.clockRate,
           prop.memoryClockRate, prop.memoryBusWidth);
    /* GDDR6: 16 transfers per pin per DRAM clock (memClockRate is the DRAM
     * clock in kHz); 1124 MHz x 16 x 32B = 575.5 GB/s = the 576 GB/s spec. */
    double peak_bw = 16.0 * prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0);
    printf("theoretical mem BW: %.0f GB/s (GDDR6 16T/pin/clk)\n", peak_bw / 1e9);

    const size_t MB = 64ull << 20;
    const int REPS = 20;
    float *ha = (float*)malloc(MB), *hb = (float*)malloc(MB);
    for (size_t i = 0; i < MB / 4; i++) { ha[i] = 1.f; hb[i] = 2.f; }
    float *da, *db, *dc;
    CK(hipMalloc((void**)&da, MB)); CK(hipMalloc((void**)&db, MB)); CK(hipMalloc((void**)&dc, MB));

    hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
    double gb = (double)MB / 1e9;

    /* warmup + best-of-REPS, timed per rep */
    double best;
    for (int pass = 0; pass < 2; pass++) { /* pass0 warm */
        best = 1e9;
        for (int r = 0; r < REPS; r++) {
            CK(hipEventRecord(ev0, 0));
            CK(hipMemcpy(da, pass ? ha : ha, MB, hipMemcpyHostToDevice));
            CK(hipEventRecord(ev1, 0)); CK(hipEventSynchronize(ev1));
            if (pass) { float ms; CK(hipEventElapsedTime(&ms, ev0, ev1));
                if (ms < best) best = ms; }
        }
    }
    printf("H2D  best %.1f GB/s (%.3f ms / 64MB)\n", gb / (best / 1e3), best);

    best = 1e9;
    for (int r = 0; r < REPS; r++) {
        CK(hipEventRecord(ev0, 0));
        CK(hipMemcpy(ha, da, MB, hipMemcpyDeviceToHost));
        CK(hipEventRecord(ev1, 0)); CK(hipEventSynchronize(ev1));
        float ms; CK(hipEventElapsedTime(&ms, ev0, ev1));
        if (ms < best) best = ms;
    }
    printf("D2H  best %.1f GB/s (%.3f ms / 64MB)\n", gb / (best / 1e3), best);

    best = 1e9;
    for (int r = 0; r < REPS; r++) {
        CK(hipEventRecord(ev0, 0));
        CK(hipMemcpy(dc, da, MB, hipMemcpyDeviceToDevice));
        CK(hipEventRecord(ev1, 0)); CK(hipEventSynchronize(ev1));
        float ms; CK(hipEventElapsedTime(&ms, ev0, ev1));
        if (ms < best) best = ms;
    }
    printf("D2D  best %.1f GB/s copy (%.3f ms / 64MB)  [单向有效 = /2]\n",
           gb / (best / 1e3), best);

    /* triad: 64M f32 = 256MB buffers, 3 traffic per elem */
    const long long N = 64ll << 20;
    float *ta, *tb, *tc;
    CK(hipMalloc((void**)&ta, N * 4)); CK(hipMalloc((void**)&tb, N * 4)); CK(hipMalloc((void**)&tc, N * 4));
    CK(hipMemset(ta, 1, N * 4)); CK(hipMemset(tb, 2, N * 4));
    int blocks = (int)((N + 255) / 256);
    best = 1e9;
    for (int r = 0; r < 10; r++) {
        CK(hipEventRecord(ev0, 0));
        hipLaunchKernelGGL(triad, dim3(blocks), dim3(256), 0, 0, ta, tb, tc, N);
        CK(hipEventRecord(ev1, 0)); CK(hipEventSynchronize(ev1));
        float ms; CK(hipEventElapsedTime(&ms, ev0, ev1));
        if (ms < best) best = ms;
    }
    double triad_bw = 3.0 * (double)N * 4 / (best / 1e3) / 1e9;
    printf("triad 64M: %.3f ms -> effective %.1f GB/s (3N traffic) = %.0f%% of theoretical\n",
           best, triad_bw, 100.0 * triad_bw / (peak_bw / 1e9));

    /* FFMA peak: blocks*256 threads * K lanes * iters * 2 flops */
    int fblocks = prop.multiProcessorCount * 20;
    int iters = 20000;
    best = 1e9;
    for (int r = 0; r < 5; r++) {
        CK(hipEventRecord(ev0, 0));
        hipLaunchKernelGGL(ffma_peak, dim3(fblocks), dim3(256), 0, 0, dc, 1.f, iters);
        CK(hipEventRecord(ev1, 0)); CK(hipEventSynchronize(ev1));
        float ms; CK(hipEventElapsedTime(&ms, ev0, ev1));
        if (ms < best) best = ms;
    }
    double flops = (double)fblocks * 256 * K * iters * 2.0;
    printf("ffma: %.1f GFLOP in %.1f ms -> %.2f TFLOPS f32 (RDNA2 dual-issue 上界参考 ~2x)\n",
           flops / 1e9, best, flops / (best / 1e3) / 1e12);
    return 0;
}
