/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Event-based per-kernel attribution (rocprofiler-free).
//!
//! HIP event pairs around each benchmark section give device-side start/end
//! timestamps; the CSV lines feed the SS8 counter-adjacent attribution
//! baseline when rocprofiler capture is unavailable (non-root + no
//! amdgpu.perfmon).
//!
//! CSV columns: section,ms,theoretical_ref_units

#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(cmd)                                                             \
    do {                                                                       \
        hipError_t e_ = (cmd);                                                 \
        if (e_ != hipSuccess)                                                  \
        {                                                                      \
            printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__);   \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void triad_kernel(float* a, float* b, float* c, float s, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t k = 0; k < 16; ++k)
    {
        size_t idx = i + k * stride;
        if (idx < n) c[idx] = a[idx] + s * b[idx];
    }
}

__global__ void ffma_kernel(float* x, float a, size_t ops_per_thread)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float a0 = 0.5f, a1 = 0.25f, a2 = 0.75f, a3 = 1.5f;
    for (size_t k = 0; k < ops_per_thread; k += 4)
    {
        a0 = fmaf(a0, 1.000001f, 1.0f);
        a1 = fmaf(a1, 1.000002f, 2.0f);
        a2 = fmaf(a2, 0.999999f, 3.0f);
        a3 = fmaf(a3, 0.500001f, 4.0f);
    }
    float acc = a0 + a1 + a2 + a3;
    (void)acc;
    if (acc == 1e9f) x[i % 1024] = acc;
}

int main()
{
    CHECK(hipInit(0));
    const size_t N = 64u << 20; // 64M floats
    float *a, *b, *c;
    CHECK(hipMalloc(&a, N * 4));
    CHECK(hipMalloc(&b, N * 4));
    CHECK(hipMalloc(&c, N * 4));
    CHECK(hipMemset(a, 0, N * 4));
    CHECK(hipMemset(b, 0, N * 4));
    const unsigned blocks = N / 16 / 256;

    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));

    printf("section,ms,ref\n");

    // triad: 3N traffic, theoretical 575 GB/s on this card
    float triad_best = 1e9f;
    for (int it = 0; it < 5; ++it)
    {
        CHECK(hipEventRecord(t0, 0));
        triad_kernel<<<blocks, 256>>>(a, b, c, 2.0f, N);
        CHECK(hipEventRecord(t1, 0));
        CHECK(hipEventSynchronize(t1));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, t0, t1));
        if (ms < triad_best) triad_best = ms;
    }
    double gbs = (3.0 * N * 4) / (triad_best * 1e6);
    printf("triad_64M,%.3f,%.1f GB/s (theory 575)\n", triad_best, gbs);

    // ffma: 2 flops per fma
    float ffma_best = 1e9f;
    for (int it = 0; it < 3; ++it)
    {
        CHECK(hipEventRecord(t0, 0));
        // 32M threads x 512 fma each = 16G fma = 32G flops per launch
        ffma_kernel<<<N / 512, 256>>>(c, 2.0f, 512);
        CHECK(hipEventRecord(t1, 0));
        CHECK(hipEventSynchronize(t1));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, t0, t1));
        if (ms < ffma_best) ffma_best = ms;
    }
    double tflops = 2.0 * (double)N * 512 / (ffma_best * 1e9);
    printf("ffma_32M,%.3f,%.2f TFLOPS (theory 21.5 dual-issue)\n", ffma_best, tflops);

    CHECK(hipFree(a));
    CHECK(hipFree(b));
    CHECK(hipFree(c));
    CHECK(hipEventDestroy(t0));
    CHECK(hipEventDestroy(t1));
    return 0;
}
