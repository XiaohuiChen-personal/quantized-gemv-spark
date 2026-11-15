// Host side of the GEMV kernels: launchers, rotating device copies, timing, checking.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <cublasLt.h>
#include <nvtx3/nvToolsExt.h>
#include "cuda_check.h"
#include "host_util.h"
#include "kernels.cuh"

enum class Fmt { BF16, INT8, INT4 };
inline const char* fmt_name(Fmt f)
{
    return f == Fmt::BF16 ? "bf16" : f == Fmt::INT8 ? "int8" : "int4";
}

// One rotating device copy of a matrix in one format: a single allocation holding the
// weights, then (for INT8/INT4) the scales at a 256-byte-aligned offset.
struct Operand {
    const void* w = nullptr;
    const void* s = nullptr;
    size_t bytes = 0;  // whole allocation, read by the size-matched read reference
};

using Launch = std::function<void(const Operand&, const bf16* x, bf16* y, cudaStream_t)>;

// A kernel in the benchmark: its ladder name, its configuration, its weight format,
// whether its output is checked (diagnostics are not), a factory that computes the
// launch geometry once per shape (an empty Launch means "not available for this shape"),
// and whether a failed check fails the run. My kernels and cublasGemmEx: yes. cuBLASLt's
// heuristic algorithms: no; a failure is recorded as a finding (some add a second BF16
// rounding) and the analysis counts only algorithms that pass.
struct KernelDef {
    std::string name, config;
    Fmt fmt;
    bool checked;
    std::function<Launch(int M, int K)> make;
    bool fatal = true;
};

inline int num_sms()
{
    int dev = 0, n = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    return n;
}

// ---------------------------------------------------------------- launchers

inline KernelDef def_v0()
{
    return {"v0", "t256", Fmt::BF16, true, [](int M, int K) -> Launch {
                const int grid = (M + 255) / 256;
                return [=](const Operand& o, const bf16* x, bf16* y, cudaStream_t s) {
                    gemv_v0<<<grid, 256, 0, s>>>(static_cast<const bf16*>(o.w), x, y, M, K);
                };
            }};
}

inline KernelDef def_v1()
{
    return {"v1", "t256", Fmt::BF16, true, [](int M, int K) -> Launch {
                const int grid = (M + 7) / 8;  // 8 warps = 8 rows per block
                return [=](const Operand& o, const bf16* x, bf16* y, cudaStream_t s) {
                    gemv_v1<<<grid, 256, 0, s>>>(static_cast<const bf16*>(o.w), x, y, M, K);
                };
            }};
}

template <class F>
typename F::Params make_params(const Operand& o, int K)
{
    typename F::Params p{};
    if constexpr (std::is_same_v<typename F::Params, FmtI8::Params>) {
        p.Q = static_cast<const uint4*>(o.w);
        p.scale = static_cast<const __half*>(o.s);
    } else if constexpr (std::is_same_v<typename F::Params, FmtI4::Params>) {
        p.Q = static_cast<const uint4*>(o.w);
        p.scale = static_cast<const __half*>(o.s);
        p.groups_per_row = K / kGroup;
    } else {
        p.W = static_cast<const typename F::Chunk*>(o.w);
    }
    return p;
}

// gemv_vec launcher. persistent = true caps the grid at the number of blocks that fit on
// the GPU at once (from the occupancy API); false launches one row group per warp.
template <class F, int T, int RPW, int U, bool XS>
KernelDef def_vec(const std::string& name, Fmt fmt, bool persistent, bool checked = true)
{
    std::string cfg = "t" + std::to_string(T) + "_r" + std::to_string(RPW) + "_u" +
                      std::to_string(U) + (XS ? "_xs" : "_xg") + (persistent ? "_pers" : "_full");
    if constexpr (std::is_same_v<F, FmtBF16<uint32_t>>) cfg += "_ld4";
    if constexpr (std::is_same_v<F, FmtBF16<uint2>>) cfg += "_ld8";
    return {name, cfg, fmt, checked, [persistent](int M, int K) -> Launch {
                if (K % kGroup != 0) {
                    fprintf(stderr, "gemv_vec needs K %% %d == 0 (K = %d)\n", kGroup, K);
                    exit(EXIT_FAILURE);
                }
                auto kern = gemv_vec<F, T, RPW, U, XS>;
                const size_t smem = XS ? (size_t)K * sizeof(bf16) : 0;
                if (smem > 48 * 1024)
                    CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    (int)smem));
                int per_sm = 0;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, T, smem));
                if (per_sm < 1) {
                    fprintf(stderr, "gemv_vec: no block fits on an SM (smem %zu)\n", smem);
                    exit(EXIT_FAILURE);
                }
                const long groups = (M + RPW - 1) / RPW;
                const long full = (groups + T / kWarp - 1) / (T / kWarp);
                const int grid = (int)(persistent ? std::min<long>(full, (long)per_sm * num_sms()) : full);
                return [=](const Operand& o, const bf16* x, bf16* y, cudaStream_t s) {
                    kern<<<grid, T, smem, s>>>(make_params<F>(o, K), x, y, M, K);
                };
            }};
}

// cuBLAS: row-major W (M x K) is column-major W^T (K x M) with lda = K, so
// y = W x is C (M x 1) = op(A) B with op = transpose, m = M, n = 1, k = K.
inline KernelDef def_cublas(cublasHandle_t h)
{
    return {"cublas", "GemmEx_OP_T_32F_DEFAULT", Fmt::BF16, true, [h](int M, int K) -> Launch {
                // The handle's stream is set once, before its workspace, by the caller:
                // cublasSetStream resets the workspace, so it must not be called here.
                return [=](const Operand& o, const bf16* x, bf16* y, cudaStream_t) {
                    const float alpha = 1.f, beta = 0.f;
                    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, K, &alpha, o.w,
                                              CUDA_R_16BF, K, x, CUDA_R_16BF, K, &beta, y,
                                              CUDA_R_16BF, M, CUBLAS_COMPUTE_32F,
                                              CUBLAS_GEMM_DEFAULT));
                };
            }};
}

// cuBLASLt with the i-th algorithm its heuristic returns for this shape (up to 8). Same
// operation and types as def_cublas. Not available (empty Launch) if the heuristic
// returns fewer than i + 1 algorithms.
struct LtPlan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    ~LtPlan()
    {
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (op) cublasLtMatmulDescDestroy(op);
    }
};

inline KernelDef def_cublaslt(cublasLtHandle_t lt, int i, void* ws, size_t ws_bytes)
{
    KernelDef d{"cublaslt", "heur" + std::to_string(i), Fmt::BF16, true,
            [=](int M, int K) -> Launch {
                auto plan = std::make_shared<LtPlan>();
                CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
                const cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
                CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA));
                CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB));
                CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_16BF, K, M, K));
                CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_16BF, K, 1, K));
                CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->c, CUDA_R_16BF, M, 1, M));
                cublasLtMatmulPreference_t pref;
                CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
                CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
                    pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof ws_bytes));
                cublasLtMatmulHeuristicResult_t res[8];
                int n = 0;
                CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, plan->op, plan->a, plan->b, plan->c,
                                                            plan->c, pref, 8, res, &n));
                CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
                if (i >= n) return Launch{};
                plan->algo = res[i].algo;
                return [=](const Operand& o, const bf16* x, bf16* y, cudaStream_t s) {
                    const float alpha = 1.f, beta = 0.f;
                    CUBLAS_CHECK(cublasLtMatmul(lt, plan->op, &alpha, o.w, plan->a, x, plan->b, &beta, y,
                                                plan->c, y, plan->c, &plan->algo, ws, ws_bytes, s));
                };
            }};
    d.fatal = false;
    return d;
}

// Size-matched read reference: the read-only kernel over the whole stored allocation.
// Grid 768 x 256, the middle grid of prelim/bandwidth.cu (all four gave the same rate).
inline KernelDef def_read(Fmt fmt, float* sink)
{
    return {"read", "t256_g768", fmt, false, [sink](int, int) -> Launch {
                return [=](const Operand& o, const bf16*, bf16*, cudaStream_t s) {
                    read_kernel<<<768, 256, 0, s>>>(static_cast<const float4*>(o.w), o.bytes / 16, sink);
                };
            }};
}

// ---------------------------------------------------------------- device copies

// Host image of one matrix in one format: weight bytes and (optional) scale bytes.
struct HostOperand {
    const uint8_t* w = nullptr;
    size_t w_bytes = 0;
    const uint8_t* s = nullptr;
    size_t s_bytes = 0;
    size_t stored_bytes() const { return w_bytes + s_bytes; }
};

// The rotation rule: R = max(2, ceil(32 x L2 / stored bytes)) copies, read in turn.
// The planned 4 x L2 was not enough on the GB10: a rotation sweep (results/rotation/)
// showed rates up to 14% above the DRAM plateau at 4 x L2 (e.g. a 21 MB BF16 matrix,
// 5 copies: 257 vs 226 GB/s), settling only from about 16 x L2 on; 32 x L2 (805 MB)
// leaves a margin. Two copies minimum, so consecutive launches never read the same one.
constexpr size_t kFootprintL2Multiple = 32;
inline size_t rotation_copies(size_t stored, size_t l2_bytes)
{
    return std::max<size_t>(2, (kFootprintL2Multiple * l2_bytes + stored - 1) / stored);
}

struct DeviceCopies {
    std::vector<Operand> ops;
    std::vector<void*> allocs;
    DeviceCopies(const HostOperand& h, size_t copies)
    {
        const size_t s_off = (h.w_bytes + 255) / 256 * 256;
        const size_t total = h.s_bytes ? s_off + h.s_bytes : h.w_bytes;
        for (size_t i = 0; i < copies; ++i) {
            void* p = nullptr;
            CUDA_CHECK(cudaMalloc(&p, total));
            CUDA_CHECK(cudaMemset(p, 0, total));
            CUDA_CHECK(cudaMemcpy(p, h.w, h.w_bytes, cudaMemcpyHostToDevice));
            Operand o;
            o.w = p;
            o.bytes = total / 16 * 16;
            if (h.s_bytes) {
                CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(p) + s_off, h.s, h.s_bytes,
                                      cudaMemcpyHostToDevice));
                o.s = static_cast<uint8_t*>(p) + s_off;
            }
            for (void* q : allocs)
                if (q == p) {
                    fprintf(stderr, "rotation copies are not distinct\n");
                    exit(EXIT_FAILURE);
                }
            allocs.push_back(p);
            ops.push_back(o);
        }
    }
    ~DeviceCopies()
    {
        for (void* p : allocs) CUDA_CHECK(cudaFree(p));
    }
    DeviceCopies(const DeviceCopies&) = delete;
    DeviceCopies& operator=(const DeviceCopies&) = delete;
};

// ---------------------------------------------------------------- timing

struct Profile {
    int warmup;          // at least this many warm-up launches ...
    double warmup_ms;    // ... and at least this long
    int iso, graph_n, graph_reps;
};
constexpr Profile kFinalProfile{20, 200.0, 200, 200, 5};
constexpr Profile kTuneProfile{10, 100.0, 100, 100, 3};
constexpr unsigned long long kSpinNs = 100000;  // 100 us: longer than any host enqueue seen

struct Timing {
    std::vector<double> iso_samples, spin_samples, str_samples;  // raw, in launch order
    double iso_med = 0, iso_p10 = 0, iso_p90 = 0, iso_min = 0;  // microseconds
    double spin_med = 0;                                        // microseconds
    double str_med = 0, str_min = 0, str_max = 0;               // microseconds per launch
    int graph_launches = 0;                                     // launches per captured graph
    double host_enqueue_max_us = 0;  // spin diagnostic: longest host time to enqueue
    bool spin_covered = true;        // ... and whether the spin kernel covered it
};

inline double median_of(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline double elapsed_us(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms * 1e3;
}

// Three measurements of one launch configuration over its rotating copies:
//  isolated  each launch between its own pair of events on an otherwise idle GPU,
//            synchronized after each launch (the proposal's method; the pre-registered
//            measure). On the GB10 this adds a fixed ~1.3-1.7 us of event overhead to
//            the kernel's own duration (results/timing_method/).
//  spin      the same with a spin kernel queued before the start event: a diagnostic.
//            Meant to hide host launch latency, it instead adds ~1.3 us more and lands
//            on a ~1 us lattice on this machine, so it is recorded but not used.
//  streamed  whole rotation cycles (at least graph_n launches, a multiple of R)
//            captured into one CUDA graph and replayed graph_reps times; per-launch
//            time = replay time / launches (how a decode engine runs its kernels).
// Consecutive launches always read different copies: the counter steps through the
// copies in order and R >= 2; each graph replay starts where the previous one ended.
// label names the NVTX ranges ("iso <label>", "spin <label>", "str <label>") that let an
// Nsight Systems trace match its kernels to this measurement; free without a profiler.
inline Timing time_launch(const Launch& L, const std::vector<Operand>& ops, const bf16* x,
                          bf16* y, cudaStream_t s, const Profile& pr, const std::string& label = "")
{
    using clk = std::chrono::steady_clock;
    const size_t R = ops.size();
    size_t next = 1;  // copy 0 was just read by the correctness run
    auto run = [&] { L(ops[next++ % R], x, y, s); };
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    const auto w0 = clk::now();
    for (int i = 0; i < pr.warmup || std::chrono::duration<double, std::milli>(clk::now() - w0).count() < pr.warmup_ms;
         ++i) {
        run();
        if (i % 16 == 15) CUDA_CHECK(cudaStreamSynchronize(s));  // keep the queue short
    }
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaStreamSynchronize(s));

    Timing out;
    std::vector<double> t, sp;
    nvtxRangePushA(("iso " + label).c_str());
    for (int i = 0; i < pr.iso; ++i) {
        CUDA_CHECK(cudaEventRecord(e0, s));
        run();
        CUDA_CHECK(cudaEventRecord(e1, s));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK_LAUNCH();
        t.push_back(elapsed_us(e0, e1));
    }
    nvtxRangePop();
    nvtxRangePushA(("spin " + label).c_str());
    for (int i = 0; i < pr.iso; ++i) {
        spin_kernel<<<1, 1, 0, s>>>(kSpinNs);
        const auto h0 = clk::now();
        CUDA_CHECK(cudaEventRecord(e0, s));
        run();
        CUDA_CHECK(cudaEventRecord(e1, s));
        const double host_us = std::chrono::duration<double, std::micro>(clk::now() - h0).count();
        out.host_enqueue_max_us = std::max(out.host_enqueue_max_us, host_us);
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK_LAUNCH();
        sp.push_back(elapsed_us(e0, e1));
    }
    nvtxRangePop();
    out.spin_covered = out.host_enqueue_max_us < kSpinNs / 1e3;
    std::vector<double> ts = t;
    std::sort(ts.begin(), ts.end());
    out.iso_med = median_of(t);
    out.iso_p10 = ts[ts.size() / 10];
    out.iso_p90 = ts[ts.size() * 9 / 10];
    out.iso_min = ts.front();
    out.spin_med = median_of(sp);
    out.iso_samples = t;
    out.spin_samples = sp;

    const size_t n = (pr.graph_n + R - 1) / R * R;  // whole rotation cycles
    out.graph_launches = (int)n;
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (size_t i = 0; i < n; ++i) run();
    CUDA_CHECK(cudaStreamEndCapture(s, &g));
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
    CUDA_CHECK(cudaGraphLaunch(ge, s));  // warm-up replay
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<double> per;
    nvtxRangePushA(("str " + label).c_str());
    for (int r = 0; r < pr.graph_reps; ++r) {
        CUDA_CHECK(cudaEventRecord(e0, s));
        CUDA_CHECK(cudaGraphLaunch(ge, s));
        CUDA_CHECK(cudaEventRecord(e1, s));
        CUDA_CHECK(cudaEventSynchronize(e1));
        per.push_back(elapsed_us(e0, e1) / n);
    }
    nvtxRangePop();
    out.str_samples = per;
    out.str_med = median_of(per);
    out.str_min = *std::min_element(per.begin(), per.end());
    out.str_max = *std::max_element(per.begin(), per.end());
    CUDA_CHECK(cudaGraphExecDestroy(ge));
    CUDA_CHECK(cudaGraphDestroy(g));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return out;
}

// ---------------------------------------------------------------- checking

// Pass criteria against a double-precision reference built from the same inputs:
// every element within 2^-8 |ref| + 2^-14 sum_k |w_k x_k| (BF16 output rounding is at
// most 2^-8 relative; the second term is FP32 accumulation slack), and the relative L2
// error at most 4e-3 (BF16 rounding alone stays below 2^-8 = 3.9e-3).
struct CheckResult {
    bool pass = false;
    double rel_l2 = 0;       // ||y - ref|| / ||ref||
    double worst_ratio = 0;  // max over elements of |y - ref| / tolerance
    long n_bad = 0;
};

inline CheckResult check_output(const std::vector<uint16_t>& y, const std::vector<double>& ref,
                                const std::vector<double>& absdot)
{
    CheckResult c;
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs((double)bf16_to_float(y[i]) - ref[i]);
        const double tol = std::ldexp(std::fabs(ref[i]), -8) + std::ldexp(absdot[i], -14);
        const double ratio = tol > 0 ? d / tol : (d > 0 ? INFINITY : 0);
        if (!(d <= tol)) ++c.n_bad;  // also catches NaN
        if (std::isnan(ratio) || (!std::isnan(c.worst_ratio) && ratio > c.worst_ratio))
            c.worst_ratio = ratio;  // a NaN, once seen, stays
        num += d * d;
        den += ref[i] * ref[i];
    }
    c.rel_l2 = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    c.pass = c.n_bad == 0 && c.rel_l2 <= 4e-3;
    return c;
}

// Relative L2 and largest absolute error of y against a reference (accuracy, not a
// pass/fail check).
inline void error_vs(const std::vector<uint16_t>& y, const std::vector<double>& ref, double& rel,
                     double& maxabs)
{
    double num = 0, den = 0;
    maxabs = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double)bf16_to_float(y[i]) - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
        maxabs = std::max(maxabs, std::fabs(d));
    }
    rel = std::sqrt(num / den);
}
