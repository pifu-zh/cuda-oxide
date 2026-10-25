/* [goal §8] tuned bandwidth + latency bench (gfx1030, ROCm 10.1).
 * Working set 576 MB per triad pass (3 x 192 MB arrays) — far above the
 * 128 MB Infinity Cache so the numbers are DRAM-bound, not L2 artifacts
 * (the earlier 64 MB D2D "820 GB/s" was cache-resident).
 * Configs: block {128,256,512} x vector {1,2,4} x unroll {1,2}.
 * Also: read-only / write-only / copy direction probes and latency probes
 * (empty-kernel round trip, small D2D copy).
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); return 1; } } while (0)

__global__ void triad_v1(const float* a, const float* b, float* c, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + 2.0f * b[i];
}
__global__ void triad_v2(const float* a, const float* b, float* c, long long n2) { // float2
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n2) ((float2*)c)[i] = ((const float2*)a)[i] + 2.0f * ((const float2*)b)[i];
}
__global__ void triad_v4(const float* a, const float* b, float* c, long long n4) { // float4
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) ((float4*)c)[i] = ((const float4*)a)[i] + 2.0f * ((const float4*)b)[i];
}
template <int U> __global__ void triad_v4u(const float* a, const float* b, float* c, long long n4) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * U;
    #pragma unroll
    for (int u = 0; u < U; u++) {
        long long i = base + u;
        if (i < n4) ((float4*)c)[i] = ((const float4*)a)[i] + 2.0f * ((const float4*)b)[i];
    }
}
__global__ void readonly(const float4* a, float* sink, long long n4) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float4 v = (i < n4) ? a[i] : make_float4(0,0,0,0);
    if (v.x == 1e30f) sink[0] = v.x;
}
__global__ void writeonly(float4* a, long long n4) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) a[i] = make_float4(1.f, 2.f, 3.f, 4.f);
}
__global__ void copy_v4(const float4* a, float4* c, long long n4) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) c[i] = a[i];
}

template <int U> __global__ void readonly_u(const float4* a, float* sink, long long n4) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * U;
    #pragma unroll
    for (int u = 0; u < U; u++) {
        long long i = base + u;
        if (i < n4) { float4 v = a[i]; if (v.x == 1e30f) sink[0] = v.x; }
    }
}
__global__ void copy_v2(const float2* a, float2* c, long long n2) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n2) c[i] = a[i];
}
__global__ void empty() {}

typedef std::chrono::high_resolution_clock clk;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

int main(void) {
    CK(hipInit(0));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    const double SPEC = 16.0 * prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0);
    printf("device %s  theoretical %.0f GB/s\n", prop.name, SPEC / 1e9);

    const long long N = 48ll << 20;            // 48M floats = 192 MB/array
    float *a, *b, *c, *sink;
    CK(hipMalloc((void**)&a, N * 4));
    CK(hipMalloc((void**)&b, N * 4));
    CK(hipMalloc((void**)&c, N * 4));
    CK(hipMalloc((void**)&sink, 4));
    CK(hipMemset(a, 1, N * 4)); CK(hipMemset(b, 2, N * 4));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));

    double bytes_triad = 3.0 * N * 4, bytes_copy = 2.0 * N * 4, bytes_1w = 1.0 * N * 4;
    auto best_of = [&](auto launch, double bytes, int reps = 8) -> double {
        launch(); CK(hipDeviceSynchronize());           // warm
        double best = 1e9;
        for (int r = 0; r < reps; r++) {
            CK(hipEventRecord(e0, 0));
            launch();
            CK(hipEventRecord(e1, 0)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            if (ms < best) best = ms;
        }
        return bytes / (best / 1e3) / 1e9;              // GB/s
    };

    printf("\n-- triad (3N traffic), config scan --\n");
    double best_triad = 0; int best_cfg = 0;
    for (int blk : {128, 256, 512}) {
        long long g1 = (N + blk - 1) / blk, g2 = (N/2 + blk - 1) / blk, g4 = (N/4 + blk - 1) / blk;
        double r;
        r = best_of([&]{ hipLaunchKernelGGL(triad_v1, dim3((unsigned)g1), dim3(blk), 0, 0, a, b, c, N); }, bytes_triad);
        printf("v1 blk=%-3d %7.1f GB/s (%.1f%%)\n", blk, r, 100*r/(SPEC/1e9));
        if (r > best_triad) { best_triad = r; best_cfg = blk*10+1; }
        r = best_of([&]{ hipLaunchKernelGGL(triad_v2, dim3((unsigned)g2), dim3(blk), 0, 0, a, b, c, N/2); }, bytes_triad);
        printf("v2 blk=%-3d %7.1f GB/s (%.1f%%)\n", blk, r, 100*r/(SPEC/1e9));
        if (r > best_triad) { best_triad = r; best_cfg = blk*10+2; }
        r = best_of([&]{ hipLaunchKernelGGL(triad_v4, dim3((unsigned)g4), dim3(blk), 0, 0, a, b, c, N/4); }, bytes_triad);
        printf("v4 blk=%-3d %7.1f GB/s (%.1f%%)\n", blk, r, 100*r/(SPEC/1e9));
        if (r > best_triad) { best_triad = r; best_cfg = blk*10+4; }
    }
    for (int blk : {128, 256}) {
        long long g4u = (N/4 + 2*blk - 1) / (2*blk);
        double r = best_of([&]{ hipLaunchKernelGGL((triad_v4u<2>), dim3((unsigned)g4u), dim3(blk), 0, 0, a, b, c, N/4); }, bytes_triad);
        printf("v4u2 blk=%-3d %7.1f GB/s (%.1f%%)\n", blk, r, 100*r/(SPEC/1e9));
        if (r > best_triad) { best_triad = r; best_cfg = blk*10+6; }
    }
    printf("BEST triad %.1f GB/s = %.1f%% of spec (cfg %d)\n",
           best_triad, 100*best_triad/(SPEC/1e9), best_cfg);

    printf("\n-- direction probes (blk=256, v4) --\n");
    long long g4 = (N/4 + 255) / 256;
    double r;
    r = best_of([&]{ hipLaunchKernelGGL(readonly, dim3((unsigned)g4), dim3(256), 0, 0, (const float4*)a, sink, N/4); }, bytes_1w);
    printf("read-only  %7.1f GB/s (%.1f%%)\n", r, 100*r/(SPEC/1e9));
    r = best_of([&]{ hipLaunchKernelGGL(writeonly, dim3((unsigned)g4), dim3(256), 0, 0, (float4*)c, N/4); }, bytes_1w);
    printf("write-only %7.1f GB/s (%.1f%%)\n", r, 100*r/(SPEC/1e9));
    r = best_of([&]{ hipLaunchKernelGGL(copy_v4, dim3((unsigned)g4), dim3(256), 0, 0, (const float4*)a, (float4*)c, N/4); }, bytes_copy);
    printf("copy(2N)   %7.1f GB/s (%.1f%%)\n", r, 100*r/(SPEC/1e9));

    printf("\n-- tuned second pass --\n");
    for (int blk : {256, 512}) {
        long long g4u4 = (N/4 + 4*blk - 1) / (4*blk);
        double r = best_of([&]{ hipLaunchKernelGGL((readonly_u<4>), dim3((unsigned)g4u4), dim3(blk), 0, 0, (const float4*)a, sink, N/4); }, bytes_1w);
        printf("read-v4u4 blk=%-3d %7.1f GB/s (%.1f%%)\n", blk, r, 100*r/(SPEC/1e9));
    }
    {
        long long g2 = (N/2 + 511) / 512;
        double r = best_of([&]{ hipLaunchKernelGGL(copy_v2, dim3((unsigned)g2), dim3(512), 0, 0, (const float2*)a, (float2*)c, N/2); }, bytes_copy);
        printf("copy-v2 blk=512 %7.1f GB/s (%.1f%%)\n", r, 100*r/(SPEC/1e9));
    }
    printf("\n-- latency probes --\n");
    // empty-kernel round trip: wall clock min/median over 2000
    {
        double best = 1e9; double hist[2000];
        hipLaunchKernelGGL(empty, 1, 1, 0, 0); CK(hipDeviceSynchronize());
        for (int i = 0; i < 2000; i++) {
            auto t0 = clk::now();
            hipLaunchKernelGGL(empty, 1, 1, 0, 0);
            CK(hipDeviceSynchronize());
            double us = ms_since(t0) * 1e3;
            hist[i] = us;
            if (us < best) best = us;
        }
        for (int i = 1; i < 2000; i++) { double k = hist[i]; int j = i-1;
            while (j >= 0 && hist[j] > k) { hist[j+1] = hist[j]; j--; } hist[j+1] = k; }
        printf("empty-kernel round trip: min %.1f us  median %.1f us\n", best, hist[1000]);
    }
    // small D2D copy (4 KB) latency
    {
        void *s4k, *d4k;
        CK(hipMalloc(&s4k, 4096)); CK(hipMalloc(&d4k, 4096));
        CK(hipMemcpy(d4k, s4k, 4096, hipMemcpyDeviceToDevice)); CK(hipDeviceSynchronize());
        double best = 1e9;
        for (int i = 0; i < 2000; i++) {
            auto t0 = clk::now();
            CK(hipMemcpy(d4k, s4k, 4096, hipMemcpyDeviceToDevice));
            CK(hipDeviceSynchronize());
            double us = ms_since(t0) * 1e3;
            if (us < best) best = us;
        }
        printf("4KB D2D memcpy (sync): min %.1f us\n", best);
    }
    return 0;
}
