/* triad vector-width probe: scalar f32 vs float4 vs 2x-float4.
 * c = a + 2*b, 64M elems, hipEvent best-of-N. DVFS-aware: measured only.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { hipError_t _ck_e = (x); if (_ck_e != hipSuccess) { \
    printf("HIP error %s at %d\n", hipGetErrorString(_ck_e), __LINE__); return 1; } } while (0)

__global__ void triad_s(const float *a, const float *b, float *c, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + 2.0f * b[i];
}
__global__ void triad_f4(const float4 *a, const float4 *b, float4 *c, long long n4) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 x = a[i], y = b[i];
        c[i] = make_float4(x.x + 2.0f*y.x, x.y + 2.0f*y.y, x.z + 2.0f*y.z, x.w + 2.0f*y.w);
    }
}
__global__ void triad_f4x2(const float4 *a, const float4 *b, float4 *c, long long n4) {
    long long i0 = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i0 + 1 < n4) {
        float4 x0 = a[i0], y0 = b[i0], x1 = a[i0+1], y1 = b[i0+1];
        c[i0]   = make_float4(x0.x + 2.0f*y0.x, x0.y + 2.0f*y0.y, x0.z + 2.0f*y0.z, x0.w + 2.0f*y0.w);
        c[i0+1] = make_float4(x1.x + 2.0f*y1.x, x1.y + 2.0f*y1.y, x1.z + 2.0f*y1.z, x1.w + 2.0f*y1.w);
    }
}

static float best_of(int reps, void (*launch)(float*,float*,float*,long long,int,int),
                     float *a, float *b, float *c, long long n, int grid, int block) {
    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
        hipEvent_t ev0, ev1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
        CK(hipEventRecord(ev0));
        launch(a, b, c, n, grid, block);
        CK(hipEventRecord(ev1)); CK(hipEventSynchronize(ev1));
        float ms; CK(hipEventElapsedTime(&ms, ev0, ev1));
        if (ms < best) best = ms;
        hipEventDestroy(ev0); hipEventDestroy(ev1);
    }
    return best;
}
static void l_s(float*a,float*b,float*c,long long n,int g,int bl){ triad_s<<<g,bl>>>(a,b,c,n); }
static void l_f4(float*a,float*b,float*c,long long n,int g,int bl){ triad_f4<<<g,bl>>>((const float4*)a,(const float4*)b,(float4*)c,n/4); }
static void l_f4x2(float*a,float*b,float*c,long long n,int g,int bl){ triad_f4x2<<<g,bl>>>((const float4*)a,(const float4*)b,(float4*)c,n/4); }
/* ceiling decomposition: write-only fill, read-only reduce */
__global__ void fill_f4(float4 *c, float4 v, long long n4) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) c[i] = v;
}
__global__ void red_f4(const float4 *a, const float4 *b, float4 *out, long long n4) {
    /* true read-only: consume both arrays, store only under an unsatisfiable
       guard so the loads survive but no write traffic exists */
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float4 s = make_float4(0,0,0,0);
    if (i < n4) {
        float4 x = a[i], y = b[i];
        s = make_float4(x.x+y.x, x.y+y.y, x.z+y.z, x.w+y.w);
    }
    /* runtime-opaque guard: loads survive, no store fires */
    if (s.x == 12345.678f) out[0] = s;
}
static void l_fill(float*a,float*b,float*c,long long n,int g,int bl){ float4 v=make_float4(1.f,1.f,1.f,1.f); fill_f4<<<g,bl>>>((float4*)c,v,n/4); }
static void l_red(float*a,float*b,float*c,long long n,int g,int bl){ red_f4<<<g,bl>>>((const float4*)a,(const float4*)b,(float4*)c,n/4); }
/* grid-stride float4, ILP4: classic STREAM shape */
__global__ void triad_gs(const float4 *a, const float4 *b, float4 *c, long long n4) {
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    long long stride = (long long)gridDim.x * blockDim.x * 4;
    for (long long i = base; i + 3 < n4; i += stride) {
        float4 x0=a[i],   y0=b[i],   x1=a[i+1], y1=b[i+1];
        float4 x2=a[i+2], y2=b[i+2], x3=a[i+3], y3=b[i+3];
        c[i]   = make_float4(x0.x+2.0f*y0.x, x0.y+2.0f*y0.y, x0.z+2.0f*y0.z, x0.w+2.0f*y0.w);
        c[i+1] = make_float4(x1.x+2.0f*y1.x, x1.y+2.0f*y1.y, x1.z+2.0f*y1.z, x1.w+2.0f*y1.w);
        c[i+2] = make_float4(x2.x+2.0f*y2.x, x2.y+2.0f*y2.y, x2.z+2.0f*y2.z, x2.w+2.0f*y2.w);
        c[i+3] = make_float4(x3.x+2.0f*y3.x, x3.y+2.0f*y3.y, x3.z+2.0f*y3.z, x3.w+2.0f*y3.w);
    }
}
static void l_gs(float*a,float*b,float*c,long long n,int g,int bl){ triad_gs<<<g,bl>>>((const float4*)a,(const float4*)b,(float4*)c,n/4); }

int main(int argc, char **argv) {
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    printf("device: %s  SMs=%d clk=%dkHz memClock=%dkHz busWidth=%dbit\n",
           p.name, p.multiProcessorCount, p.clockRate, p.memoryClockRate, p.memoryBusWidth);
    double theory = (double)p.memoryClockRate * 16 * (p.memoryBusWidth/8);
    printf("theoretical mem BW: %.1f GB/s\n", theory / 1e6);
    const long long N = 64ll << 20;
    float *a, *b, *c;
    CK(hipMalloc((void**)&a, N*4)); CK(hipMalloc((void**)&b, N*4)); CK(hipMalloc((void**)&c, N*4));
    /* true float init: a=1.0f, b=2.0f so c must be 5.0f */
    {
        float *ha = (float*)malloc(N*4), *hb = (float*)malloc(N*4);
        for (long long i = 0; i < N; i++) { ha[i] = 1.0f; hb[i] = 2.0f; }
        CK(hipMemcpy(a, ha, N*4, hipMemcpyHostToDevice));
        CK(hipMemcpy(b, hb, N*4, hipMemcpyHostToDevice));
        free(ha); free(hb);
    }
    int blk = 256;
    struct { const char *name; void (*fn)(float*,float*,float*,long long,int,int); long long grid; } cases[] = {
        {"scalar f32     ", l_s,    N/blk},
        {"float4         ", l_f4,   N/(blk*4)},
        {"float4 x2/lane ", l_f4x2, N/(blk*8)},
        {"gs f4 ILP4 40CU ", l_gs,   40*8},  /* 256 threads x 8 CTUs per CU */
        {"write-only fill ", l_fill, N/(blk*4)},
        {"read-only true  ", l_red,  N/(blk*4)},
    };
    /* warmup 5x each to pin DVFS clocks */
    for (int w = 0; w < 5; w++) for (unsigned t = 0; t < 6; t++)
        cases[t].fn(a, b, c, N, (int)cases[t].grid, blk);
    for (unsigned t = 0; t < 6; t++) {
        float ms = best_of(40, cases[t].fn, a, b, c, N, (int)cases[t].grid, blk);
        double bytes = (t == 4) ? 1.0 * N * 4 : (t == 5 ? 2.0 * N * 4 : 3.0 * N * 4); /* t5: pure read 2N */
        double gbs = bytes / (ms * 1e6);
        printf("%s %.3f ms -> %.1f GB/s (%.1f%% of theoretical)\n",
               cases[t].name, ms, gbs, 100.0 * gbs / (theory / 1e6));
    }
    /* loop mode for clock sampling: 3000 f4 iterations */
    if (argc > 1 && (strcmp(argv[1], "loop") == 0 || strcmp(argv[1], "loops") == 0)) {
        int f4mode = strcmp(argv[1], "loop") == 0;
        int g4 = (int)(N/(blk*4)), gs = (int)((N + blk - 1)/blk);
        for (int it = 0; it < 6000; it++) {
            if (f4mode) triad_f4<<<g4, blk>>>((const float4*)a, (const float4*)b, (float4*)c, N/4);
            else        triad_s <<<gs, blk>>>(a, b, c, N);
        }
        CK(hipDeviceSynchronize());
        printf("loop(%s) done\n", f4mode ? "f4" : "scalar");
        return 0;
    }
    /* verify correctness once */
    l_f4(a, b, c, N, (int)(N/(blk*4)), blk);
    CK(hipDeviceSynchronize());
    float h[8]; CK(hipMemcpy(h, c, 32, hipMemcpyDeviceToHost));
    printf("verify f4 c[0..7] = %g %g %g %g %g %g %g %g (expect 5)\n",
           h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    l_gs(a, b, c, N, 40*8, blk);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h, c, 32, hipMemcpyDeviceToHost));
    printf("verify gs c[0..7] = %g %g %g %g %g %g %g %g (expect 5)\n",
           h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    l_f4x2(a, b, c, N, (int)(N/(blk*8)), blk);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h, c, 32, hipMemcpyDeviceToHost));
    printf("verify f4x2 c[0..7] = %g %g %g %g %g %g %g %g (expect 5)\n",
           h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    return 0;
}
