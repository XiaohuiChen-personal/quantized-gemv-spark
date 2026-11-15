// Achievable DRAM bandwidth on this GPU: a copy kernel (read + write) and a
// read-only kernel (the access pattern of a weight-streaming GEMV).
// Each buffer is 1 GiB, far larger than L2, so every pass comes from DRAM.
//
// This is prelim/bandwidth.cu from the course repository (the proposal's 241 GB/s), with
// one change: the source buffer is filled with non-zero pseudo-random data instead of
// zeros, so the reference reads the same kind of data the GEMV kernels do.
#include <algorithm>
#include <vector>
#include "cuda_check.h"

__global__ void copyKernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

// Sums every element so the loads cannot be optimized away; the result is
// written only if it equals an impossible value, so the kernel does no stores.
__global__ void readKernel(const float4* __restrict__ src, size_t n, float* sink)
{
    float s = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 v = src[i];
        s += v.x + v.y + v.z + v.w;
    }
    if (s == 12345.678f) *sink = s;
}

__global__ void emptyKernel() {}

// Fills n float4 with a hash of the index: finite, non-zero, not compressible.
__global__ void fillKernel(float4* dst, size_t n)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ (uint32_t)(i >> 32);
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        const float f = 1.0f + (h & 0xFFFFFF) * (1.0f / 16777216.0f);  // [1, 2)
        dst[i] = make_float4(f, -f, 0.5f * f, 2.0f * f);
    }
}

int main()
{
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int memClockKHz = 0, busBits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&busBits, cudaDevAttrGlobalMemoryBusWidth, 0));
    printf("device: %s  cc %d.%d  SMs %d  L2 %d bytes\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.l2CacheSize);
    printf("memory bus %d bits, reported memory clock %d kHz -> %.1f GB/s theoretical\n",
           busBits, memClockKHz, (busBits / 8.0) * memClockKHz * 1e3 / 1e9);

    const size_t bytes = 1ull << 30;   // 1 GiB per buffer
    const size_t n = bytes / sizeof(float4);
    float4 *a, *b;
    float* sink;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
    fillKernel<<<p.multiProcessorCount * 16, 256>>>(a, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(b, 0, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Each kernel gets its own warm-up and timed loop (interleaving the copy and
    // the read lowered the read figure by a few GB/s in an earlier version).
    const int warmup = 3, timed = 21;   // odd count: median = middle element
    auto timeKernel = [&](auto launch) {
        std::vector<float> t;
        for (int r = 0; r < warmup + timed; r++) {
            float ms;
            CUDA_CHECK(cudaEventRecord(start));
            launch();
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (r >= warmup) t.push_back(ms);
        }
        std::sort(t.begin(), t.end());
        return t;   // t[timed / 2] is the median, t[0] the best
    };

    for (int blocksPerSM : {4, 8, 16, 32}) {
        int grid = p.multiProcessorCount * blocksPerSM;
        auto tCopy = timeKernel([&] { copyKernel<<<grid, 256>>>(a, b, n); });
        auto tRead = timeKernel([&] { readKernel<<<grid, 256>>>(a, n, sink); });
        // Copy moves 2x bytes (read + write); read-only moves 1x. GB = 1e9 bytes.
        printf("grid %5d x 256 | copy  median %6.1f GB/s  best %6.1f | "
               "read-only  median %6.1f GB/s  best %6.1f\n",
               grid, 2.0 * bytes / (tCopy[timed / 2] * 1e6), 2.0 * bytes / (tCopy[0] * 1e6),
               bytes / (tRead[timed / 2] * 1e6), bytes / (tRead[0] * 1e6));
    }

    // Launch overhead: an empty kernel, timed one launch at a time with events.
    auto tEmpty = timeKernel([&] { emptyKernel<<<p.multiProcessorCount, 256>>>(); });
    printf("empty kernel (event-timed, one launch): median %.1f us  best %.1f us\n",
           tEmpty[timed / 2] * 1e3, tEmpty[0] * 1e3);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(sink));
    return 0;
}
