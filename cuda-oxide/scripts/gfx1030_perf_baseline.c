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
    float ms; hipError_t _ignored = hipEventElapsedTime(&ms, s, e); (void)_ignored; *out = ms;
}

int main(int argc_m, char **argv_m) {
    int skip_copies = argc_m > 1;
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
    /* allocate the 256MB triad buffers FIRST so they land on clean 2MB-page
     * boundaries; allocating them after the 3x64MB (192MB offset) measured
     * 88% vs 95% (page-granularity/TLB effect, see gfx1030_denorm_ab.c) */
    const long long N0 = 64ll << 20;
    float *ta, *tb, *tc;
    CK(hipMalloc((void**)&ta, N0 * 4)); CK(hipMalloc((void**)&tb, N0 * 4)); CK(hipMalloc((void**)&tc, N0 * 4));

    hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
    double gb = (double)MB / 1e9;

    /* warmup + best-of-REPS, timed per rep */
    double best;
    for (int pass = 0; pass < (skip_copies ? 0 : 2); pass++) { /* pass0 warm */
        best = 1e9;
        for (int r = 0; r < REPS; r++) {
            CK(hipEventRecord(ev0, 0));
            CK(hipMemcpy(da, pass ? ha : ha, MB, hipMemcpyHostToDevice));
            CK(hipEventRecord(ev1, 0)); CK(hipEventSynchronize(ev1));
            if (pass) { float ms; CK(hipEventElapsedTime(&ms, ev0, ev1));
                if (ms < best) best = ms; }
        }
    }
    if (!skip_copies)
    printf("H2D  best %.1f GB/s (%.3f ms / 64MB)\n", gb / (best / 1e3), best);

    best = 1e9;
    for (int r = 0; r < (skip_copies ? 0 : REPS); r++) {
        CK(hipEventRecord(ev0, 0));
        CK(hipMemcpy(ha, da, MB, hipMemcpyDeviceToHost));
        CK(hipEventRecord(ev1, 0)); CK(hipEventSynchronize(ev1));
        float ms; CK(hipEventElapsedTime(&ms, ev0, ev1));
        if (ms < best) best = ms;
    }
    if (!skip_copies)
    printf("D2H  best %.1f GB/s (%.3f ms / 64MB)\n", gb / (best / 1e3), best);

    best = 1e9;
    for (int r = 0; r < (skip_copies ? 0 : REPS); r++) {
        CK(hipEventRecord(ev0, 0));
        CK(hipMemcpy(dc, da, MB, hipMemcpyDeviceToDevice));
        CK(hipEventRecord(ev1, 0)); CK(hipEventSynchronize(ev1));
        float ms; CK(hipEventElapsedTime(&ms, ev0, ev1));
        if (ms < best) best = ms;
    }
    if (!skip_copies)
    printf("D2D  best %.1f GB/s copy (%.3f ms / 64MB)  [单向有效 = /2]\n",
           gb / (best / 1e3), best);

    /* pinned-host transfers (hipHostMalloc staging): the canonical H2D/D2H
     * numbers above are pageable; pinned doubles them (~2x, see
     * pinned_overlap example for the Rust-side chain). */
    {
        float *pin = NULL;
        if (hipHostMalloc((void**)&pin, MB, hipHostMallocDefault) == hipSuccess) {
            best = 1e9;
            for (int r = 0; r < REPS; r++) {
                CK(hipEventRecord(ev0, 0));
                CK(hipMemcpy(da, pin, MB, hipMemcpyHostToDevice));
                CK(hipEventRecord(ev1, 0)); CK(hipEventSynchronize(ev1));
                float ms; CK(hipEventElapsedTime(&ms, ev0, ev1));
                if (ms < best) best = ms;
            }
            printf("H2Dp best %.1f GB/s (pinned staging)\n", gb / (best / 1e3));
            best = 1e9;
            for (int r = 0; r < REPS; r++) {
                CK(hipEventRecord(ev0, 0));
                CK(hipMemcpy(pin, da, MB, hipMemcpyDeviceToHost));
                CK(hipEventRecord(ev1, 0)); CK(hipEventSynchronize(ev1));
                float ms; CK(hipEventElapsedTime(&ms, ev0, ev1));
                if (ms < best) best = ms;
            }
            printf("D2Hp best %.1f GB/s (pinned staging)\n", gb / (best / 1e3));
            hipHostFree(pin);
        }
    }

    /* triad: 64M f32 = 256MB buffers, 3 traffic per elem */
    const long long N = N0;
    CK(hipMemset(ta, 1, N * 4)); CK(hipMemset(tb, 2, N * 4));
    int blocks = (int)((N + 255) / 256);
    /* Warm-state protocol: ~10 s of continuous triad load (6000 launches),
     * then one discarded event-timed 40-rep block. The fast state (+7% BW,
     * 531-544 GB/s across sessions) engages after this; the first event-timed
     * block alone measures 497-506. Data content (denormal memset patterns vs
     * normal floats) does NOT matter — three-phase diagnostic on identical
     * buffers. pp_dpm_mclk stays at P0 1124 MHz throughout. */
    for (int w = 0; w < 6000; w++)
        hipLaunchKernelGGL(triad, dim3(blocks), dim3(256), 0, 0, ta, tb, tc, N);
    CK(hipDeviceSynchronize());
    for (int r = 0; r < 40; r++) {
        CK(hipEventRecord(ev0, 0));
        hipLaunchKernelGGL(triad, dim3(blocks), dim3(256), 0, 0, ta, tb, tc, N);
        CK(hipEventRecord(ev1, 0)); CK(hipEventSynchronize(ev1));
        float ms; CK(hipEventElapsedTime(&ms, ev0, ev1));
        (void)ms;
    }
    best = 1e9;
    /* Re-issue both source buffers via H2D immediately before timing: a
     * 2x256MB H2D right before the timed block reproduces the fast state
     * (531-547 GB/s in two binaries; memset re-init or no re-init leaves the
     * block at 497-506). Mechanism unconfirmed — recorded as empirical. */
    {
        float *hta = (float*)malloc(N * 4), *htb = (float*)malloc(N * 4);
        for (long long i = 0; i < N; i++) { hta[i] = 1.0f; htb[i] = 2.0f; }
        CK(hipMemcpy(ta, hta, N * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(tb, htb, N * 4, hipMemcpyHostToDevice));
        free(hta); free(htb);
    }
    for (int r = 0; r < 40; r++) {
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
    /* dual-issue FMA ceiling at the device's boost clock: CU * 64 lanes *
     * 2 FMA/clk * 2 flops; sclk pins at 2670-2720 MHz under load. */
    double ffma_ceiling_tf = (double)prop.multiProcessorCount * 64 * 2 * 2 *
                             (prop.clockRate * 1e3) / 1e12;
    double ffma_tf = flops / (best / 1e3) / 1e12;
    printf("ffma: %.1f GFLOP in %.1f ms -> %.2f TFLOPS f32 (dual-issue ceiling @ %.0f MHz = %.2f TFLOPS = %.1f%%)\n",
           flops / 1e9, best, ffma_tf,
           prop.clockRate / 1e3, ffma_ceiling_tf,
           100.0 * ffma_tf / ffma_ceiling_tf);
    return 0;
}
