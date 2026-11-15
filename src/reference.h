// CPU reference GEMV in double precision, and an independent check of the quantized
// data written by scripts/quantize.py.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "host_util.h"

// Row decoders: write row r of W, as the exact float values the kernels multiply by.
// INT8 and INT4 products q * scale are exact in float (at most 8 + 11 significant bits).
struct RowBF16 {
    const uint16_t* W;
    int K;
    void operator()(int r, float* out) const
    {
        const uint16_t* w = W + (size_t)r * K;
        for (int k = 0; k < K; ++k) out[k] = bf16_to_float(w[k]);
    }
};

struct RowI8 {
    const int8_t* Q;
    const uint16_t* S;  // FP16 [M]
    int K;
    void operator()(int r, float* out) const
    {
        const int8_t* q = Q + (size_t)r * K;
        const float s = half_to_float(S[r]);
        for (int k = 0; k < K; ++k) out[k] = (float)q[k] * s;
    }
};

inline int int4_at(const uint8_t* row, int k)
{
    const int nib = (k & 1) ? (row[k >> 1] >> 4) : (row[k >> 1] & 0xF);
    return nib >= 8 ? nib - 16 : nib;  // two's complement
}

struct RowI4 {
    const uint8_t* Q;   // [M][K/2], low nibble = even column
    const uint16_t* S;  // FP16 [M][K/128]
    int K;
    void operator()(int r, float* out) const
    {
        const uint8_t* q = Q + (size_t)r * (K / 2);
        const uint16_t* s = S + (size_t)r * (K / 128);
        for (int k = 0; k < K; ++k) out[k] = (float)int4_at(q, k) * half_to_float(s[k / 128]);
    }
};

struct RefResult {
    std::vector<double> y;       // sum_k w_k x_k
    std::vector<double> absdot;  // sum_k |w_k x_k|, for the accumulation tolerance
};

template <class RowFn>
RefResult reference_gemv(int M, int K, const std::vector<float>& x, const RowFn& row_of)
{
    RefResult res;
    res.y.assign(M, 0.0);
    res.absdot.assign(M, 0.0);
#pragma omp parallel
    {
        std::vector<float> w(K);
#pragma omp for schedule(static)
        for (int r = 0; r < M; ++r) {
            row_of(r, w.data());
            double s = 0, a = 0;
            for (int k = 0; k < K; ++k) {
                const double p = (double)w[k] * (double)x[k];
                s += p;
                a += std::fabs(p);
            }
            res.y[r] = s;
            res.absdot[r] = a;
        }
    }
    return res;
}

// Checks of the quantizer's output against the original BF16 weights, computed here in
// C++ independently of the Python that wrote it:
//   every |q| <= q_max (so -8 never appears in INT4, -128 never in INT8), and
//   every |w - q s| <= s/2 (1 + 2^-10) when s > 0, and q = 0 when s = 0,
// i.e. round-to-nearest with the stored FP16 scale. Also counts FP16-subnormal scales.
struct QuantCheck {
    long violations = 0;
    long subnormal_scales = 0, zero_scales = 0;
    double max_err_over_scale = 0;  // max |w - q s| / s, ideally <= 0.5
};

template <class QAt, class SAt>
QuantCheck check_quant(int M, int K, const uint16_t* W, int qmax, int group, QAt q_at, SAt s_bits_at)
{
    QuantCheck c;
    const int groups = K / group;
#pragma omp parallel
    {
        QuantCheck local;
#pragma omp for schedule(static)
        for (int r = 0; r < M; ++r) {
            for (int g = 0; g < groups; ++g) {
                const uint16_t sb = s_bits_at(r, g);
                const float s = half_to_float(sb);
                if ((sb & 0x7C00u) == 0 && (sb & 0x03FFu) != 0) ++local.subnormal_scales;
                if (s == 0.f) ++local.zero_scales;
                for (int k = g * group; k < (g + 1) * group; ++k) {
                    const int q = q_at(r, k);
                    const float w = bf16_to_float(W[(size_t)r * K + k]);
                    if (q > qmax || q < -qmax) ++local.violations;
                    if (s == 0.f) {
                        if (q != 0 || w != 0.f) ++local.violations;
                        continue;
                    }
                    const double e = std::fabs((double)w - (double)q * s) / s;
                    if (e > 0.5 * (1.0 + 1.0 / 1024)) ++local.violations;
                    if (e > local.max_err_over_scale) local.max_err_over_scale = e;
                }
            }
        }
#pragma omp critical
        {
            c.violations += local.violations;
            c.subnormal_scales += local.subnormal_scales;
            c.zero_scales += local.zero_scales;
            c.max_err_over_scale = std::max(c.max_err_over_scale, local.max_err_over_scale);
        }
    }
    return c;
}
