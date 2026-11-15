// Unit tests on synthetic data, for every kernel the benchmark can run (the final ladder,
// cuBLAS/cuBLASLt and all 216 tuning configurations):
//   random x and +-1 x on awkward shapes  -> the bench's pass criteria (check_output)
//   one-hot x = e_k at chunk/group edges  -> y must equal bf16(w[:, k]) exactly
//   INT4 matrices containing all 16 codes -> covered by both of the above
//   mutations of the data or the launch   -> the checker must FAIL
// Plus the host-side number conversions. Exit status 0 only if everything passed.
#include <map>
#include <random>
#include "gemv.cuh"
#include "kernel_sets.cuh"
#include "reference.h"

static int g_fail = 0, g_pass = 0;
static void expect(bool ok, const std::string& what)
{
    if (ok) {
        ++g_pass;
        return;
    }
    if (++g_fail <= 40) printf("FAIL: %s\n", what.c_str());
}

// ---------------------------------------------------------------- synthetic matrices

struct Synth {
    int M, K;
    std::vector<uint16_t> w;   // BF16
    std::vector<int8_t> q8;    // INT8 codes
    std::vector<uint16_t> s8;  // FP16 [M]
    std::vector<uint8_t> q4;   // INT4 packed, low nibble = even column
    std::vector<uint16_t> s4;  // FP16 [M][K/128]

    HostOperand op(Fmt f) const
    {
        HostOperand h;
        if (f == Fmt::BF16) {
            h.w = reinterpret_cast<const uint8_t*>(w.data());
            h.w_bytes = w.size() * 2;
        } else if (f == Fmt::INT8) {
            h.w = reinterpret_cast<const uint8_t*>(q8.data());
            h.w_bytes = q8.size();
            h.s = reinterpret_cast<const uint8_t*>(s8.data());
            h.s_bytes = s8.size() * 2;
        } else {
            h.w = q4.data();
            h.w_bytes = q4.size();
            h.s = reinterpret_cast<const uint8_t*>(s4.data());
            h.s_bytes = s4.size() * 2;
        }
        return h;
    }
    void set4(int r, int k, int code)
    {
        uint8_t& b = q4[(size_t)r * (K / 2) + k / 2];
        const uint8_t nib = (uint8_t)(code & 0xF);
        b = (k & 1) ? (uint8_t)((b & 0x0F) | (nib << 4)) : (uint8_t)((b & 0xF0) | nib);
    }
    // The exact float value each format's kernel multiplies by at (r, k).
    float value(Fmt f, int r, int k) const
    {
        if (f == Fmt::BF16) return bf16_to_float(w[(size_t)r * K + k]);
        if (f == Fmt::INT8) return (float)q8[(size_t)r * K + k] * half_to_float(s8[r]);
        return (float)int4_at(q4.data() + (size_t)r * (K / 2), k) * half_to_float(s4[(size_t)r * (K / 128) + k / 128]);
    }
};

// Random FP16 scale bits between 2^-8 and 2^-4 (normal numbers).
static uint16_t rand_scale(std::mt19937& g)
{
    return (uint16_t)(0x1C00 + g() % 0x1000);
}

static Synth make_synth(int M, int K, uint32_t seed)
{
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.f, 0.05f);
    Synth s{M, K, {}, {}, {}, {}, {}};
    s.w.resize((size_t)M * K);
    for (auto& v : s.w) v = float_to_bf16(nd(g));
    s.q8.resize((size_t)M * K);
    for (auto& v : s.q8) v = (int8_t)((int)(g() % 255) - 127);
    s.s8.resize(M);
    for (auto& v : s.s8) v = rand_scale(g);
    if (K % 128 == 0) {
        s.q4.assign((size_t)M * K / 2, 0);
        // Every code -8..7 appears in every row at every nibble position over the rows.
        for (int r = 0; r < M; ++r)
            for (int k = 0; k < K; ++k) s.set4(r, k, ((r * 7 + k) % 16) - 8);
        s.s4.resize((size_t)M * (K / 128));
        for (auto& v : s.s4) v = rand_scale(g);
    }
    return s;
}

static std::vector<uint16_t> make_x(int K, uint32_t seed, int kind)  // 0 Gaussian, 1 +-1
{
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> x(K);
    for (auto& v : x) v = float_to_bf16(kind == 0 ? nd(g) : ((g() & 1) ? 1.f : -1.f));
    return x;
}

static std::vector<float> to_float(const std::vector<uint16_t>& b)
{
    std::vector<float> f(b.size());
    for (size_t i = 0; i < b.size(); ++i) f[i] = bf16_to_float(b[i]);
    return f;
}

static RefResult ref_of(const Synth& s, Fmt f, const std::vector<float>& x)
{
    if (f == Fmt::BF16) return reference_gemv(s.M, s.K, x, RowBF16{s.w.data(), s.K});
    if (f == Fmt::INT8) return reference_gemv(s.M, s.K, x, RowI8{s.q8.data(), s.s8.data(), s.K});
    return reference_gemv(s.M, s.K, x, RowI4{s.q4.data(), s.s4.data(), s.K});
}

// ---------------------------------------------------------------- device helpers

struct DevVec {
    bf16* p = nullptr;
    explicit DevVec(const std::vector<uint16_t>& h)
    {
        CUDA_CHECK(cudaMalloc(&p, h.size() * 2));
        CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    }
    explicit DevVec(size_t n) { CUDA_CHECK(cudaMalloc(&p, n * 2)); }
    ~DevVec() { CUDA_CHECK(cudaFree(p)); }
};

static std::vector<uint16_t> run(const Launch& L, const Operand& o, const bf16* x, int M, cudaStream_t s)
{
    DevVec y((size_t)M);
    CUDA_CHECK(cudaMemsetAsync(y.p, 0xFF, (size_t)M * 2, s));  // NaN poison
    L(o, x, y.p, s);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<uint16_t> out(M);
    CUDA_CHECK(cudaMemcpy(out.data(), y.p, (size_t)M * 2, cudaMemcpyDeviceToHost));
    return out;
}

// ---------------------------------------------------------------- tests

static void test_conversions()
{
    expect(half_to_float(0x3C00) == 1.f, "half 1.0");
    expect(half_to_float(0xC000) == -2.f, "half -2.0");
    expect(half_to_float(0x7BFF) == 65504.f, "half max");
    expect(half_to_float(0x0001) == std::ldexp(1.f, -24), "half min subnormal");
    expect(half_to_float(0x0400) == std::ldexp(1.f, -14), "half min normal");
    expect(half_to_float(0x3555) == 0.333251953125f, "half 1/3");
    expect(bf16_to_float(0x3F80) == 1.f && float_to_bf16(1.f) == 0x3F80, "bf16 1.0");
    expect(float_to_bf16(1.00390625f) == 0x3F80, "bf16 tie rounds to even (down)");
    expect(float_to_bf16(1.01171875f) == 0x3F82, "bf16 tie rounds to even (up)");
    expect(float_to_bf16(-0.5f) == 0xBF00, "bf16 -0.5");
    // The device decode must agree with the host decode on the stored FP16 scale bits.
    for (uint16_t h : {0x0001, 0x03FF, 0x0400, 0x1C00, 0x2BFF, 0x3C00, 0x7BFF})
        expect(std::isfinite(half_to_float(h)), "half finite");
}

static const std::vector<std::pair<int, int>> kShapes = {{1, 128}, {3, 256}, {257, 384}, {2, 4096}, {64, 9216}, {1031, 2560}};

static void test_kernel(const KernelDef& kd, const std::map<std::pair<int, int>, Synth>& synth, cudaStream_t s)
{
    for (const auto& [M, K] : kShapes) {
        const bool vec = kd.name != "v0" && kd.name != "v1" && kd.name != "cublas" && kd.name != "cublaslt";
        if (vec && K % 128) continue;
        const Synth& sy = synth.at({M, K});
        const Launch L = kd.make(M, K);
        if (!L) continue;
        const std::string tag = kd.name + " " + fmt_name(kd.fmt) + " " + kd.config + " M=" +
                                std::to_string(M) + " K=" + std::to_string(K);
        DeviceCopies dc(sy.op(kd.fmt), 2);
        for (int kind : {0, 1}) {
            const std::vector<uint16_t> xb = make_x(K, 1000 + K, kind);
            const RefResult ref = ref_of(sy, kd.fmt, to_float(xb));
            DevVec dx(xb);
            for (int copy : {0, 1}) {
                const CheckResult c = check_output(run(L, dc.ops[copy], dx.p, M, s), ref.y, ref.absdot);
                expect(c.pass, tag + (kind ? " +-1 x" : " gaussian x") + " copy " + std::to_string(copy) +
                                   " rel " + std::to_string(c.rel_l2) + " worst " + std::to_string(c.worst_ratio));
            }
        }
        std::vector<int> ks = {0, 1, 7, 8, 15, 16, 31, 32, 127, 128, K - 128, K - 33, K - 32, K - 1};
        for (int k : ks) {
            if (k < 0 || k >= K) continue;
            std::vector<uint16_t> xb(K, 0);
            xb[k] = 0x3F80;  // 1.0
            DevVec dx(xb);
            const std::vector<uint16_t> y = run(L, dc.ops[0], dx.p, M, s);
            long bad = 0;
            for (int r = 0; r < M; ++r)
                if (bf16_to_float(y[r]) != bf16_to_float(float_to_bf16(sy.value(kd.fmt, r, k)))) ++bad;
            expect(bad == 0, tag + " one-hot k=" + std::to_string(k) + " wrong rows " + std::to_string(bad));
        }
    }
}

// Each mutation must make the checker fail (on the input named).
static void test_mutations(const KernelDef& kd, cudaStream_t s)
{
    for (const auto& [M, K] : std::vector<std::pair<int, int>>{{1031, 2560}, {64, 9216}}) {
        const Synth base = make_synth(M, K, 77);
        const std::vector<uint16_t> xg = make_x(K, 5, 0), xr = make_x(K, 6, 1);
        const RefResult rg = ref_of(base, kd.fmt, to_float(xg)), rr = ref_of(base, kd.fmt, to_float(xr));
        DevVec dxg(xg), dxr(xr);
        const std::string tag = kd.name + " " + fmt_name(kd.fmt) + " K=" + std::to_string(K);
        auto fails = [&](const Synth& mut, bool rad, const std::string& what, int launch_m = -1) {
            DeviceCopies dc(mut.op(kd.fmt), 2);
            const Launch L = kd.make(launch_m < 0 ? M : launch_m, K);
            const CheckResult c = check_output(run(L, dc.ops[0], rad ? dxr.p : dxg.p, M, s),
                                               rad ? rr.y : rg.y, rad ? rr.absdot : rg.absdot);
            expect(!c.pass, tag + " mutation not detected: " + what);
        };
        // The unmutated data must pass (otherwise "detected" means nothing).
        {
            DeviceCopies dc(base.op(kd.fmt), 2);
            const CheckResult c = check_output(run(kd.make(M, K), dc.ops[0], dxg.p, M, s), rg.y, rg.absdot);
            expect(c.pass, tag + " unmutated baseline");
        }
        // 1. One weight changed by a large amount at a k with |x_k| > 0.5.
        int k1 = 0;
        while (std::fabs(bf16_to_float(xg[k1])) <= 0.5f) ++k1;
        {
            Synth m = base;
            const int r = M / 2;
            double rms = 0;
            for (int k = 0; k < K; ++k) rms += (double)m.value(kd.fmt, r, k) * m.value(kd.fmt, r, k);
            rms = std::sqrt(rms / K);
            if (kd.fmt == Fmt::BF16) m.w[(size_t)r * K + k1] = float_to_bf16(bf16_to_float(m.w[(size_t)r * K + k1]) + 64 * (float)rms);
            else if (kd.fmt == Fmt::INT8) { int8_t& q = m.q8[(size_t)r * K + k1]; q = (int8_t)(q >= 0 ? q - 100 : q + 100); }
            else m.set4(r, k1, int4_at(m.q4.data() + (size_t)r * (K / 2), k1) >= 0 ? -8 : 7);
            fails(m, false, "one weight");
        }
        // 2. The last 32 weights of every row zeroed (as if the last chunk were skipped).
        {
            Synth m = base;
            for (int r = 0; r < M; ++r)
                for (int k = K - 32; k < K; ++k) {
                    if (kd.fmt == Fmt::BF16) m.w[(size_t)r * K + k] = 0;
                    else if (kd.fmt == Fmt::INT8) m.q8[(size_t)r * K + k] = 0;
                    else m.set4(r, k, 0);
                }
            fails(m, false, "last 32 columns zeroed");
        }
        // 3. One column zeroed: must be caught with the +-1 x.
        {
            Synth m = base;
            const int k0 = K / 3;
            for (int r = 0; r < M; ++r) {
                if (kd.fmt == Fmt::BF16) m.w[(size_t)r * K + k0] = 0;
                else if (kd.fmt == Fmt::INT8) m.q8[(size_t)r * K + k0] = 0;
                else m.set4(r, k0, 0);
            }
            fails(m, true, "one column zeroed (+-1 x)");
        }
        // 4./5. Format-specific: INT4 nibbles swapped, scales swapped.
        if (kd.fmt == Fmt::INT4) {
            Synth m = base;
            for (auto& b : m.q4) b = (uint8_t)((b >> 4) | (b << 4));
            fails(m, false, "INT4 nibbles swapped");
            Synth m2 = base;
            for (int r = 0; r < M; ++r) std::swap(m2.s4[(size_t)r * (K / 128)], m2.s4[(size_t)r * (K / 128) + 1]);
            fails(m2, false, "INT4 group scales 0 and 1 swapped");
        }
        if (kd.fmt == Fmt::INT8) {
            Synth m = base;
            for (int r = 0; r + 1 < M; r += 2) std::swap(m.s8[r], m.s8[r + 1]);
            fails(m, false, "INT8 row scales swapped in pairs");
        }
        // 6. The last row not computed (launch with M - 1 rows, check M).
        fails(base, false, "last row not computed", M - 1);
    }
}

int main()
{
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    FinalContext ctx{};
    CUBLAS_CHECK(cublasCreate(&ctx.blas));
    CUBLAS_CHECK(cublasLtCreate(&ctx.lt));
    ctx.ws_bytes = 32u << 20;
    CUDA_CHECK(cudaMalloc(&ctx.ws, ctx.ws_bytes));
    CUBLAS_CHECK(cublasSetWorkspace(ctx.blas, ctx.ws, ctx.ws_bytes));
    CUBLAS_CHECK(cublasSetStream(ctx.blas, s));
    CUDA_CHECK(cudaMalloc(&ctx.sink, sizeof(float)));

    test_conversions();
    printf("conversions: %d passed, %d failed\n", g_pass, g_fail);

    std::map<std::pair<int, int>, Synth> synth;
    for (const auto& [M, K] : kShapes) synth.emplace(std::make_pair(M, K), make_synth(M, K, 31 * M + K));

    std::vector<KernelDef> ks = final_set(ctx);
    for (KernelDef& kd : tune_set()) ks.push_back(kd);
    int n_kernels = 0;
    for (const KernelDef& kd : ks) {
        if (!kd.checked) continue;  // read references and the load-only diagnostic
        const int before = g_fail;
        test_kernel(kd, synth, s);
        ++n_kernels;
        if (g_fail > before) printf("  ^ in %s %s %s\n", kd.name.c_str(), fmt_name(kd.fmt), kd.config.c_str());
    }
    printf("kernels: %d tested, %d checks passed, %d failed so far\n", n_kernels, g_pass, g_fail);

    for (const KernelDef& kd : final_set(ctx, false))
        if (kd.name != "v0" && kd.name != "v1") test_mutations(kd, s);
    test_mutations(def_v1(), s);
    printf("total: %d checks passed, %d failed\n", g_pass, g_fail);

    CUDA_CHECK(cudaFree(ctx.sink));
    CUDA_CHECK(cudaFree(ctx.ws));
    CUBLAS_CHECK(cublasLtDestroy(ctx.lt));
    CUBLAS_CHECK(cublasDestroy(ctx.blas));
    CUDA_CHECK(cudaStreamDestroy(s));
    printf(g_fail ? "TESTS FAILED\n" : "ALL TESTS PASSED\n");
    return g_fail ? EXIT_FAILURE : EXIT_SUCCESS;
}
