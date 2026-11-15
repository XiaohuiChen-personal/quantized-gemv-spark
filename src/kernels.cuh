// GEMV kernels y = W x for one input vector (batch 1).
//
// W is M rows x K columns, row-major. x is BF16 (K values), y is BF16 (M values), and
// every kernel accumulates in FP32. The ladder:
//   v0  one thread per row                        (uncoalesced)
//   v1  one warp per row, shuffle reduction       (coalesced, 2-byte loads)
//   v2  BF16, 16-byte loads, x in shared memory, persistent grid, rows per warp, unroll
//   v3  v2's structure with INT8 weights, one FP16 scale per row
//   v4  v2's structure with INT4 weights, one FP16 scale per group of 128 along K
// v2-v4 are one template (gemv_vec) parameterised by the weight format, so they differ
// only in how a 16-byte chunk of weights is decoded.
//
// Requirements (checked by the host launchers): K % 128 == 0 for v2-v4 (every chunk
// lies inside one INT4 group, and rows stay 16-byte aligned); M is arbitrary.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

typedef __nv_bfloat16 bf16;

constexpr int kWarp = 32;
constexpr int kGroup = 128;  // INT4 group size along K

__device__ __forceinline__ float warp_sum(float v)
{
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// A 32-bit word holding two BF16 values (element 2i in the low half, 2i+1 in the high
// half, because memory is little-endian). BF16 -> FP32 is a 16-bit shift.
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

// ---------------------------------------------------------------- v0, v1

__global__ void gemv_v0(const bf16* __restrict__ W, const bf16* __restrict__ x,
                        bf16* __restrict__ y, int M, int K)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const bf16* w = W + (size_t)row * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc = fmaf(__bfloat162float(w[k]), __bfloat162float(x[k]), acc);
    y[row] = __float2bfloat16(acc);
}

// blockDim.x must be a multiple of 32 so a warp never straddles the `warp >= M` exit.
__global__ void gemv_v1(const bf16* __restrict__ W, const bf16* __restrict__ x,
                        bf16* __restrict__ y, int M, int K)
{
    const int warp = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) / kWarp);
    const int lane = threadIdx.x % kWarp;
    if (warp >= M) return;
    const bf16* w = W + (size_t)warp * K;
    float acc = 0.f;
    for (int k = lane; k < K; k += kWarp)
        acc = fmaf(__bfloat162float(w[k]), __bfloat162float(x[k]), acc);
    acc = warp_sum(acc);
    if (lane == 0) y[warp] = __float2bfloat16(acc);
}

// ---------------------------------------------------------------- formats for gemv_vec

// Read x either from shared memory (a plain load) or from global through the read-only
// path. XS is a compile-time flag, so the branch disappears.
template <bool XS, class T>
__device__ __forceinline__ T load_x(const T* p, int i)
{
    if constexpr (XS) return p[i];
    else return __ldg(p + i);
}

// BF16 weights. CHUNK is the load width: uint4 (16 B = 8 weights) for v2; uint2 and
// uint32_t exist only for the load-width ablation.
template <class CHUNK>
struct FmtBF16 {
    using Chunk = CHUNK;
    static constexpr int kWeightsPerChunk = sizeof(CHUNK) / 2;
    struct Params {
        const CHUNK* W;
    };
    __device__ static Chunk load(const Params& p, int row, int c, int chunks_per_row)
    {
        return __ldg(p.W + (size_t)row * chunks_per_row + c);
    }
    template <bool XS>
    __device__ static float dot(const Params&, const Chunk& w, const bf16* xs, int, int c,
                                float acc)
    {
        const Chunk xv = load_x<XS>(reinterpret_cast<const CHUNK*>(xs), c);
        const uint32_t* ww = reinterpret_cast<const uint32_t*>(&w);
        const uint32_t* xx = reinterpret_cast<const uint32_t*>(&xv);
#pragma unroll
        for (int i = 0; i < (int)(sizeof(CHUNK) / 4); ++i) {
            acc = fmaf(bf_lo(ww[i]), bf_lo(xx[i]), acc);
            acc = fmaf(bf_hi(ww[i]), bf_hi(xx[i]), acc);
        }
        return acc;
    }
    __device__ static float row_scale(const Params&, int) { return 1.f; }
};

// INT8 weights, one FP16 scale per row: y_r = s_r * sum_k q_rk x_k. A 16-byte chunk
// holds 16 weights; byte j of word i is weight 4i + j of the chunk.
struct FmtI8 {
    using Chunk = uint4;
    static constexpr int kWeightsPerChunk = 16;
    struct Params {
        const uint4* Q;
        const __half* scale;  // [M]
    };
    __device__ static Chunk load(const Params& p, int row, int c, int chunks_per_row)
    {
        return __ldg(p.Q + (size_t)row * chunks_per_row + c);
    }
    // Four signed bytes of q against four BF16 x values (two words).
    __device__ static float dot4(uint32_t q, uint32_t xa, uint32_t xb, float acc)
    {
        acc = fmaf((float)(static_cast<int>(q << 24) >> 24), bf_lo(xa), acc);
        acc = fmaf((float)(static_cast<int>(q << 16) >> 24), bf_hi(xa), acc);
        acc = fmaf((float)(static_cast<int>(q << 8) >> 24), bf_lo(xb), acc);
        acc = fmaf((float)(static_cast<int>(q) >> 24), bf_hi(xb), acc);
        return acc;
    }
    template <bool XS>
    __device__ static float dot(const Params&, const Chunk& w, const bf16* xs, int, int c,
                                float acc)
    {
        const uint4* xv = reinterpret_cast<const uint4*>(xs);
        const uint4 x0 = load_x<XS>(xv, 2 * c), x1 = load_x<XS>(xv, 2 * c + 1);
        acc = dot4(w.x, x0.x, x0.y, acc);
        acc = dot4(w.y, x0.z, x0.w, acc);
        acc = dot4(w.z, x1.x, x1.y, acc);
        acc = dot4(w.w, x1.z, x1.w, acc);
        return acc;
    }
    __device__ static float row_scale(const Params& p, int row)
    {
        return __half2float(__ldg(p.scale + row));
    }
};

// INT4 weights, two's-complement nibbles, low nibble = even column. A 16-byte chunk
// holds 32 weights; nibble n (from the least significant) of word i is weight 8i + n.
// Chunk c covers weights 32c .. 32c+31, all inside group c / 4, so one scale per chunk.
struct FmtI4 {
    using Chunk = uint4;
    static constexpr int kWeightsPerChunk = 32;
    struct Params {
        const uint4* Q;
        const __half* scale;  // [M][K / 128]
        int groups_per_row;
    };
    __device__ static Chunk load(const Params& p, int row, int c, int chunks_per_row)
    {
        return __ldg(p.Q + (size_t)row * chunks_per_row + c);
    }
    // Eight signed nibbles of q against eight BF16 x values (one uint4 of x).
    __device__ static float dot8(uint32_t q, const uint4& x, float acc)
    {
        acc = fmaf((float)(static_cast<int>(q << 28) >> 28), bf_lo(x.x), acc);
        acc = fmaf((float)(static_cast<int>(q << 24) >> 28), bf_hi(x.x), acc);
        acc = fmaf((float)(static_cast<int>(q << 20) >> 28), bf_lo(x.y), acc);
        acc = fmaf((float)(static_cast<int>(q << 16) >> 28), bf_hi(x.y), acc);
        acc = fmaf((float)(static_cast<int>(q << 12) >> 28), bf_lo(x.z), acc);
        acc = fmaf((float)(static_cast<int>(q << 8) >> 28), bf_hi(x.z), acc);
        acc = fmaf((float)(static_cast<int>(q << 4) >> 28), bf_lo(x.w), acc);
        acc = fmaf((float)(static_cast<int>(q) >> 28), bf_hi(x.w), acc);
        return acc;
    }
    template <bool XS>
    __device__ static float dot(const Params& p, const Chunk& w, const bf16* xs, int row, int c,
                                float acc)
    {
        const uint4* xv = reinterpret_cast<const uint4*>(xs);
        float part = 0.f;
        part = dot8(w.x, load_x<XS>(xv, 4 * c + 0), part);
        part = dot8(w.y, load_x<XS>(xv, 4 * c + 1), part);
        part = dot8(w.z, load_x<XS>(xv, 4 * c + 2), part);
        part = dot8(w.w, load_x<XS>(xv, 4 * c + 3), part);
        const float s = __half2float(__ldg(p.scale + (size_t)row * p.groups_per_row + (c >> 2)));
        return fmaf(s, part, acc);
    }
    __device__ static float row_scale(const Params&, int) { return 1.f; }
};

// Diagnostic, not a rung: v4's exact loads (weights, one scale per chunk, and the four
// 16-byte pieces of x the chunk needs) with the decode and multiply-adds replaced by
// integer XORs and one add, so no load can be removed. Its output is meaningless; its
// time is what v4 would take if decoding were free (a no-counter way to see whether v4
// is limited by decode or by memory).
struct FmtI4LoadOnly {
    using Chunk = uint4;
    static constexpr int kWeightsPerChunk = 32;
    using Params = FmtI4::Params;
    __device__ static Chunk load(const Params& p, int row, int c, int chunks_per_row)
    {
        return __ldg(p.Q + (size_t)row * chunks_per_row + c);
    }
    template <bool XS>
    __device__ static float dot(const Params& p, const Chunk& w, const bf16* xs, int row, int c, float acc)
    {
        const uint4* xv = reinterpret_cast<const uint4*>(xs);
        uint32_t h = w.x ^ w.y ^ w.z ^ w.w;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint4 xt = load_x<XS>(xv, 4 * c + t);
            h ^= xt.x ^ xt.y ^ xt.z ^ xt.w;
        }
        const float s = __half2float(__ldg(p.scale + (size_t)row * p.groups_per_row + (c >> 2)));
        return acc + s + (float)h;
    }
    __device__ static float row_scale(const Params&, int) { return 1.f; }
};

// ---------------------------------------------------------------- v2 / v3 / v4

// Each warp computes RPW rows at a time. Lane l handles chunks l, l + 32, l + 64, ... of
// those rows, issuing UNROLL chunk loads per row before using any of them, so each lane
// has RPW * UNROLL independent 16-byte loads in flight. The grid strides over row groups,
// so any grid size is correct; the launcher picks either a persistent grid (as many
// blocks as fit on the SMs at once) or one row group per warp. With XS, the block first
// copies x into shared memory (K * 2 bytes of dynamic shared memory).
template <class F, int THREADS, int RPW, int UNROLL, bool XS>
__global__ void __launch_bounds__(THREADS)
    gemv_vec(typename F::Params p, const bf16* __restrict__ x, bf16* __restrict__ y, int M, int K)
{
    static_assert(THREADS % kWarp == 0, "whole warps only");
    using Chunk = typename F::Chunk;
    constexpr int kWarpsPerBlock = THREADS / kWarp;
    extern __shared__ uint4 x_smem[];

    const bf16* xs = x;
    if constexpr (XS) {
        const uint4* xg = reinterpret_cast<const uint4*>(x);
        for (int i = threadIdx.x; i < K / 8; i += THREADS) x_smem[i] = __ldg(xg + i);
        __syncthreads();
        xs = reinterpret_cast<const bf16*>(x_smem);
    }

    const int lane = threadIdx.x % kWarp;
    const int warp = threadIdx.x / kWarp;
    const int chunks_per_row = K / F::kWeightsPerChunk;
    const int row_step = gridDim.x * kWarpsPerBlock * RPW;

    for (int row0 = (blockIdx.x * kWarpsPerBlock + warp) * RPW; row0 < M; row0 += row_step) {
        float acc[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) acc[r] = 0.f;

        for (int c0 = lane; c0 < chunks_per_row; c0 += kWarp * UNROLL) {
            Chunk w[UNROLL][RPW];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int c = c0 + u * kWarp;
#pragma unroll
                for (int r = 0; r < RPW; ++r)
                    w[u][r] = (c < chunks_per_row && row0 + r < M)
                                  ? F::load(p, row0 + r, c, chunks_per_row)
                                  : Chunk{};
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int c = c0 + u * kWarp;
                if (c < chunks_per_row) {
#pragma unroll
                    for (int r = 0; r < RPW; ++r)
                        if (row0 + r < M)  // warp-uniform: skips rows past the end
                            acc[r] = F::template dot<XS>(p, w[u][r], xs, row0 + r, c, acc[r]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const float v = warp_sum(acc[r]);
            if (lane == 0 && row0 + r < M) y[row0 + r] = __float2bfloat16(v * F::row_scale(p, row0 + r));
        }
    }
}

// ---------------------------------------------------------------- read-only reference

// The read-only bandwidth kernel of prelim/bandwidth.cu: streams n 16-byte words and
// sums them so the loads cannot be removed; it stores only if the sum is an impossible
// value, so it does no writes. Used with the same rotation as the GEMV kernels to give a
// size-matched bandwidth reference.
__global__ void read_kernel(const float4* __restrict__ src, size_t n, float* sink)
{
    float s = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const float4 v = src[i];
        s += v.x + v.y + v.z + v.w;
    }
    if (s == 12345.678f) *sink = s;
}

// Busy-waits for ns nanoseconds on the global timer, touching no memory. Queued just
// before a timed launch's start event, it keeps the GPU busy while the host enqueues the
// start event, the kernel and the stop event, so the events bracket only the kernel's
// execution and not the host's launch latency.
__global__ void spin_kernel(unsigned long long ns)
{
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

