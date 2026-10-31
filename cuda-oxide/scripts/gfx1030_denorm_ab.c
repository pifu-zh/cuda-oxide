/* Diagnostic: identical triad kernel measured in two back-to-back 40-rep
 * blocks (memset-pattern data first, float 1.0f/2.0f re-init second).
 * NOTE: the apparent "denormal data" effect was later REFUTED -- the
 * three-phase diagnostic in perf_baseline (normal->denorm->normal on the
 * same buffers: 497.5/496.9/531.0 GB/s) showed data content does NOT matter;
 * the block-to-block delta comes from the measurement-position effect and,
 * decisively, a 2x256MB H2D re-init immediately before timing (see matrix
 * section 8 fast-state protocol). Kept as the position-effect A/B record. */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(_e), __LINE__); return 1; } } while (0)

__global__ void triad(const float *a, const float *b, float *c, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + 2.0f * b[i];
}

static double measure(const float *a, const float *b, float *c, long long n, int reps) {
    int blocks = (int)((n + 255) / 256);
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        CK(hipEventRecord(e0, 0));
        hipLaunchKernelGGL(triad, dim3(blocks), dim3(256), 0, 0, a, b, c, n);
        CK(hipEventRecord(e1, 0)); CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        if (ms < best) best = ms;
    }
    hipEventDestroy(e0); hipEventDestroy(e1);
    return best;
}

int main(int argc, char **argv) {
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    double theory = 16.0 * p.memoryClockRate * 1000.0 * (p.memoryBusWidth / 8.0);
    printf("theory %.1f GB/s\n", theory / 1e9);
    const long long N = 64ll << 20;
    /* ARGV 1: pre-allocate a 192MB filler first (mimics perf_baseline layout) */
    float *filler = NULL;
    if (argc > 1) { CK(hipMalloc((void**)&filler, 192ull<<20)); }
    float *a, *b, *c;
    CK(hipMalloc((void**)&a, N*4)); CK(hipMalloc((void**)&b, N*4)); CK(hipMalloc((void**)&c, N*4));
    if (filler) printf("(with 192MB pre-alloc)\n");
    /* denormal data, exactly like perf_baseline's memset(1)/memset(2) */
    CK(hipMemset(a, 1, N*4)); CK(hipMemset(b, 2, N*4));
    for (int w = 0; w < 3000; w++) hipLaunchKernelGGL(triad, dim3((int)((N+255)/256)), dim3(256), 0, 0, a, b, c, N);
    CK(hipDeviceSynchronize());
    double ms_den = measure(a, b, c, N, 40);
    printf("denormal data : %.3f ms -> %.1f GB/s (%.1f%%)\n",
           ms_den, 3.0*N*4/(ms_den*1e6), 300.0*N*4/(ms_den*1e6)/(theory/1e9));
    /* normal data */
    {
        float *ha = (float*)malloc(N*4), *hb = (float*)malloc(N*4);
        for (long long i = 0; i < N; i++) { ha[i] = 1.0f; hb[i] = 2.0f; }
        CK(hipMemcpy(a, ha, N*4, hipMemcpyHostToDevice));
        CK(hipMemcpy(b, hb, N*4, hipMemcpyHostToDevice));
        free(ha); free(hb);
    }
    double ms_nor = measure(a, b, c, N, 40);
    printf("normal data   : %.3f ms -> %.1f GB/s (%.1f%%)\n",
           ms_nor, 3.0*N*4/(ms_nor*1e6), 300.0*N*4/(ms_nor*1e6)/(theory/1e9));
    float chk[2]; CK(hipMemcpy(chk, c, 8, hipMemcpyDeviceToHost));
    printf("verify c[0]=%g (expect 5)\n", chk[0]);
    return 0;
}
