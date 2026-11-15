// The kernel sets: the tuning grid, the final ladder with its ablation and diagnostics,
// and the rotation sweep. Shared by bench and test, so the unit tests cover every
// configuration the tuning run can choose.
#pragma once
#include <vector>
#include "gemv.cuh"
#include "tuned.h"

// ---------------------------------------------------------------- tuning grid

template <class F, int T, int R, int U, bool XS>
void add_vec(std::vector<KernelDef>& out, const std::string& name, Fmt fmt)
{
    out.push_back(def_vec<F, T, R, U, XS>(name, fmt, true));
    out.push_back(def_vec<F, T, R, U, XS>(name, fmt, false));
}

template <class F, int T, int R, int U>
void add_vec_xs(std::vector<KernelDef>& out, const std::string& name, Fmt fmt)
{
    add_vec<F, T, R, U, true>(out, name, fmt);
    add_vec<F, T, R, U, false>(out, name, fmt);
}

template <class F, int T, int R>
void add_vec_u(std::vector<KernelDef>& out, const std::string& name, Fmt fmt)
{
    add_vec_xs<F, T, R, 1>(out, name, fmt);
    add_vec_xs<F, T, R, 2>(out, name, fmt);
    add_vec_xs<F, T, R, 4>(out, name, fmt);
}

template <class F, int T>
void add_vec_r(std::vector<KernelDef>& out, const std::string& name, Fmt fmt)
{
    add_vec_u<F, T, 1>(out, name, fmt);
    add_vec_u<F, T, 2>(out, name, fmt);
    add_vec_u<F, T, 4>(out, name, fmt);
}

// Full factorial: threads {128, 256} x rows per warp {1, 2, 4} x unroll {1, 2, 4} x
// x-in-shared {yes, no} x grid {persistent, full} = 72 configurations per kernel.
template <class F>
void add_grid(std::vector<KernelDef>& out, const std::string& name, Fmt fmt)
{
    add_vec_r<F, 128>(out, name, fmt);
    add_vec_r<F, 256>(out, name, fmt);
}

inline std::vector<KernelDef> tune_set()
{
    std::vector<KernelDef> k;
    add_grid<FmtBF16<uint4>>(k, "v2", Fmt::BF16);
    add_grid<FmtI8>(k, "v3", Fmt::INT8);
    add_grid<FmtI4>(k, "v4", Fmt::INT4);
    return k;
}

// ---------------------------------------------------------------- final set

template <class F, VecConfig C>
KernelDef def_tuned(const std::string& name, Fmt fmt, bool checked = true)
{
    return def_vec<F, C.threads, C.rows_per_warp, C.unroll, C.x_shared>(name, fmt, C.persistent, checked);
}

// The v2 ablation: the frozen v2 with one factor changed at a time (threads, rows per
// warp, unroll, x in shared memory, grid, load width). Rows equal to v2 itself are
// skipped, so every "v2abl" row differs from v2 in exactly one factor.
template <VecConfig C>
void add_v2_ablation(std::vector<KernelDef>& k)
{
    using B = FmtBF16<uint4>;
    constexpr int T2 = C.threads == 256 ? 128 : 256;
    k.push_back(def_vec<B, T2, C.rows_per_warp, C.unroll, C.x_shared>("v2abl", Fmt::BF16, C.persistent));
    if (C.rows_per_warp != 1) k.push_back(def_vec<B, C.threads, 1, C.unroll, C.x_shared>("v2abl", Fmt::BF16, C.persistent));
    if (C.rows_per_warp != 2) k.push_back(def_vec<B, C.threads, 2, C.unroll, C.x_shared>("v2abl", Fmt::BF16, C.persistent));
    if (C.rows_per_warp != 4) k.push_back(def_vec<B, C.threads, 4, C.unroll, C.x_shared>("v2abl", Fmt::BF16, C.persistent));
    if (C.unroll != 1) k.push_back(def_vec<B, C.threads, C.rows_per_warp, 1, C.x_shared>("v2abl", Fmt::BF16, C.persistent));
    if (C.unroll != 2) k.push_back(def_vec<B, C.threads, C.rows_per_warp, 2, C.x_shared>("v2abl", Fmt::BF16, C.persistent));
    if (C.unroll != 4) k.push_back(def_vec<B, C.threads, C.rows_per_warp, 4, C.x_shared>("v2abl", Fmt::BF16, C.persistent));
    k.push_back(def_vec<B, C.threads, C.rows_per_warp, C.unroll, !C.x_shared>("v2abl", Fmt::BF16, C.persistent));
    k.push_back(def_vec<B, C.threads, C.rows_per_warp, C.unroll, C.x_shared>("v2abl", Fmt::BF16, !C.persistent));
    k.push_back(def_vec<FmtBF16<uint2>, C.threads, C.rows_per_warp, C.unroll, C.x_shared>("v2abl", Fmt::BF16, C.persistent));
    k.push_back(def_vec<FmtBF16<uint32_t>, C.threads, C.rows_per_warp, C.unroll, C.x_shared>("v2abl", Fmt::BF16, C.persistent));
}

struct FinalContext {
    cublasHandle_t blas;
    cublasLtHandle_t lt;
    void* ws;
    size_t ws_bytes;
    float* sink;
};

// The ladder (v0, v1, v2, cuBLAS, v3, v4), cuBLASLt heuristics 0-7, the v2 ablation, the
// load-only v4 diagnostic, and the size-matched read references.
inline std::vector<KernelDef> final_set(const FinalContext& c, bool with_extras = true)
{
    std::vector<KernelDef> k;
    k.push_back(def_v0());
    k.push_back(def_v1());
    k.push_back(def_tuned<FmtBF16<uint4>, kV2>("v2", Fmt::BF16));
    k.push_back(def_cublas(c.blas));
    k.push_back(def_tuned<FmtI8, kV3>("v3", Fmt::INT8));
    k.push_back(def_tuned<FmtI4, kV4>("v4", Fmt::INT4));
    if (with_extras) {
        for (int i = 0; i < 8; ++i) k.push_back(def_cublaslt(c.lt, i, c.ws, c.ws_bytes));
        add_v2_ablation<kV2>(k);
        k.push_back(def_tuned<FmtI4LoadOnly, kV4>("v4load", Fmt::INT4, false));
        for (Fmt f : {Fmt::BF16, Fmt::INT8, Fmt::INT4}) k.push_back(def_read(f, c.sink));
    }
    return k;
}

// The rotation sweep: the read reference, v2 and v4 at an explicit copy count.
inline std::vector<KernelDef> rotation_set(float* sink)
{
    std::vector<KernelDef> k;
    k.push_back(def_read(Fmt::BF16, sink));
    k.push_back(def_read(Fmt::INT4, sink));
    k.push_back(def_tuned<FmtBF16<uint4>, kV2>("v2", Fmt::BF16));
    k.push_back(def_tuned<FmtI4, kV4>("v4", Fmt::INT4));
    return k;
}
