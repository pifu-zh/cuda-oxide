/* [goal §8 latency] minimal decomposition probe: empty-kernel launch.
 * Segments:
 *  (1) enqueue-only amortized: N launches back-to-back WITHOUT sync / N
 *  (2) kernel duration floor: hipEvent pair around ONE empty kernel (GPU
 *      timestamps - includes event overhead, upper bound of hw floor)
 *  (3) full round trip: launch + hipDeviceSynchronize, min & median
 *  (4) event-query spin: launch + spin hipEventQuery until done (no block)
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP err %s @%d\n", hipGetErrorString(e), __LINE__); return 1; } } while (0)

__global__ void empty() {}

typedef std::chrono::high_resolution_clock clk;
static double us_since(clk::time_point t) {
    return std::chrono::duration<double, std::micro>(clk::now() - t).count();
}

int main(void) {
    CK(hipInit(0));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    empty<<<1,1>>>(); CK(hipDeviceSynchronize());

    const int N = 2000;
    /* (1) enqueue amortized */
    {
        auto t0 = clk::now();
        for (int i = 0; i < N; i++) empty<<<1,1>>>();
        double enq = us_since(t0);
        CK(hipDeviceSynchronize());
        printf("(1) enqueue-only amortized: %.2f us/launch (%d launches, %0.f us total)\n", enq/N, N, enq);
    }
    /* (2) kernel duration via GPU events (single kernel) */
    {
        double best = 1e9; float ms;
        for (int r = 0; r < 200; r++) {
            CK(hipEventRecord(e0, 0));
            empty<<<1,1>>>();
            CK(hipEventRecord(e1, 0));
            CK(hipEventSynchronize(e1));
            CK(hipEventElapsedTime(&ms, e0, e1));
            if (ms * 1e3 < best) best = ms * 1e3;
        }
        printf("(2) empty-kernel duration (GPU events, min of 200): %.2f us\n", best);
    }
    /* (3) full round trip */
    {
        double v[N]; double best = 1e9;
        for (int i = 0; i < N; i++) {
            auto t0 = clk::now();
            empty<<<1,1>>>();
            CK(hipDeviceSynchronize());
            v[i] = us_since(t0);
            if (v[i] < best) best = v[i];
        }
        for (int i = 1; i < N; i++) { double k = v[i]; int j = i-1;
            while (j >= 0 && v[j] > k) { v[j+1] = v[j]; j--; } v[j+1] = k; }
        printf("(3) full round trip: min %.2f us  median %.2f us  p99 %.2f us\n",
               best, v[N/2], v[(int)(N*0.99)]);
    }
    /* (4) event-query spin round trip */
    {
        double best = 1e9;
        for (int i = 0; i < 500; i++) {
            auto t0 = clk::now();
            empty<<<1,1>>>();
            CK(hipEventRecord(e1, 0));
            while (hipEventQuery(e1) == hipErrorNotReady) { /* spin */ }
            double us = us_since(t0);
            if (us < best) best = us;
        }
        printf("(4) round trip via event-query spin: min %.2f us\n", best);
    }
    return 0;
}
