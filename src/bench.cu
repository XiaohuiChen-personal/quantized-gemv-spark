// Benchmark driver: for every matrix in data/manifest.tsv and every kernel of the chosen
// mode, check the output against the CPU reference (Gaussian x and +-1 x), then time it
// (isolated, spin diagnostic, streamed) with rotating device copies. Writes one CSV row per
// (matrix, kernel) and, in final mode, an accuracy CSV of the CPU-only quantization error.
//
//   bench --mode final|tune|quick|rotation|ncu [--data data] [--out results/run.csv]
//         [--only id,id] [--seed N]
//
// final:    the ladder (v0, v1, v2, cuBLAS, v3, v4), cuBLASLt heuristics, the v2
//           ablation, the load-only v4 diagnostic and the read references; final profile.
// tune:     the configuration grid of v2, v3, v4 on the six unique hypothesis shapes.
// quick:    the ladder only, tuning profile (development).
// rotation: read, v2 and v4 with copy counts giving total footprints of 1, 2, 4, ...,
//           128 x L2 (finds the footprint at which no cache helps any more).
// ncu:      the final set without the v2 ablation, with a short profile (3 warm-up and 5
//           isolated launches) for Nsight Compute: run under
//           ncu --nvtx --nvtx-include "regex:^iso /", which profiles only the isolated
//           launches (scripts/run_stage.sh ncu). Its timings are not used.
// --seed shuffles the order of formats and of kernels within a format (final mode).
#include <chrono>
#include <ctime>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include <unistd.h>
#include "gemv.cuh"
#include "kernel_sets.cuh"
#include "reference.h"

#ifndef GIT_DESC
#define GIT_DESC "unknown"
#endif

// The six unique hypothesis shapes, represented by one matrix each (tuning set).
static const std::set<std::string> kTuneMatrices = {"gate", "down", "qkv", "z", "out", "lmhead"};
// Rotation sweep: a 21 MB BF16 matrix and the 5.2 MB k_proj (1.3 MB in INT4).
static const std::set<std::string> kRotationMatrices = {"gate", "out", "k"};
// Nsight Compute mode: a few launches per kernel, so the profiler replays only those.
static constexpr Profile kNcuProfile{3, 0.0, 5, 1, 1};

struct MatrixData {
    MatrixInfo info;
    std::vector<uint8_t> bf16, q8, s8, q4, s4;
    HostOperand op(Fmt f) const
    {
        HostOperand h;
        if (f == Fmt::BF16) {
            h.w = bf16.data();
            h.w_bytes = bf16.size();
        } else if (f == Fmt::INT8) {
            h.w = q8.data();
            h.w_bytes = q8.size();
            h.s = s8.data();
            h.s_bytes = s8.size();
        } else {
            h.w = q4.data();
            h.w_bytes = q4.size();
            h.s = s4.data();
            h.s_bytes = s4.size();
        }
        return h;
    }
};

MatrixData load_matrix(const std::string& dir, const MatrixInfo& m)
{
    MatrixData d;
    d.info = m;
    const size_t mk = (size_t)m.M * m.K;
    const std::string base = dir + "/" + m.id;
    d.bf16 = read_file(base + ".bf16.bin", mk * 2);
    d.q8 = read_file(base + ".int8.bin", mk);
    d.s8 = read_file(base + ".int8.scale.bin", (size_t)m.M * 2);
    d.q4 = read_file(base + ".int4.bin", mk / 2);
    d.s4 = read_file(base + ".int4.scale.bin", (size_t)m.M * (m.K / kGroup) * 2);
    return d;
}

std::vector<float> bf16_file_to_float(const std::vector<uint8_t>& b)
{
    std::vector<float> f(b.size() / 2);
    for (size_t i = 0; i < f.size(); ++i) f[i] = bf16_to_float(reinterpret_cast<const uint16_t*>(b.data())[i]);
    return f;
}

std::string utc_now()
{
    char buf[32];
    const std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

// Expected relative error for Gaussian x, from the weights alone:
// sqrt(sum delta^2 / sum w^2) over the whole matrix, delta = w - q s.
template <class RowQ>
double predicted_rel_error(int M, int K, const uint16_t* W, const RowQ& rowq)
{
    double num = 0, den = 0;
#pragma omp parallel reduction(+ : num, den)
    {
        std::vector<float> wq(K);
#pragma omp for schedule(static)
        for (int r = 0; r < M; ++r) {
            rowq(r, wq.data());
            for (int k = 0; k < K; ++k) {
                const double w = bf16_to_float(W[(size_t)r * K + k]);
                num += (w - wq[k]) * (w - wq[k]);
                den += w * w;
            }
        }
    }
    return std::sqrt(num / den);
}

std::vector<uint16_t> run_once(const Launch& L, const Operand& o, const bf16* dx, bf16* dy, int M,
                               cudaStream_t s)
{
    std::vector<uint16_t> y(M);
    CUDA_CHECK(cudaMemsetAsync(dy, 0xFF, (size_t)M * 2, s));  // NaN poison: unwritten rows fail
    L(o, dx, dy, s);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(y.data(), dy, (size_t)M * 2, cudaMemcpyDeviceToHost));
    return y;
}

int main(int argc, char** argv)
{
    std::string mode = "final", data = "data", out = "results/bench.csv";
    std::set<std::string> only;
    int seed = 0;
    std::string samples_path;  // optional: every raw timing sample, for distributions
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value after %s\n", a.c_str());
                exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if (a == "--mode") mode = next();
        else if (a == "--data") data = next();
        else if (a == "--out") out = next();
        else if (a == "--seed") seed = std::stoi(next());
        else if (a == "--samples") samples_path = next();
        else if (a == "--only") {
            std::stringstream ss(next());
            std::string id;
            while (std::getline(ss, id, ',')) only.insert(id);
        } else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            return EXIT_FAILURE;
        }
    }
    if (mode != "final" && mode != "tune" && mode != "quick" && mode != "rotation" && mode != "ncu") {
        fprintf(stderr, "--mode must be final, tune, quick, rotation or ncu\n");
        return EXIT_FAILURE;
    }
    const Profile prof = mode == "ncu" ? kNcuProfile
                         : (mode == "final" || mode == "rotation") ? kFinalProfile : kTuneProfile;

    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int drv = 0, rt = 0, cublas_ver = 0;
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    FinalContext ctx{};
    CUBLAS_CHECK(cublasCreate(&ctx.blas));
    CUBLAS_CHECK(cublasLtCreate(&ctx.lt));
    CUBLAS_CHECK(cublasGetVersion(ctx.blas, &cublas_ver));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    // Stream first, then a fixed workspace (cublasSetStream resets the workspace), so
    // cuBLAS never allocates during graph capture.
    CUBLAS_CHECK(cublasSetStream(ctx.blas, s));
    ctx.ws_bytes = 32u << 20;
    CUDA_CHECK(cudaMalloc(&ctx.ws, ctx.ws_bytes));
    CUBLAS_CHECK(cublasSetWorkspace(ctx.blas, ctx.ws, ctx.ws_bytes));
    CUDA_CHECK(cudaMalloc(&ctx.sink, sizeof(float)));

    char host[256] = "?";
    gethostname(host, sizeof host - 1);
    char prov[1024];
    snprintf(prov, sizeof prov,
             "# utc=%s host=%s device=\"%s\" cc=%d.%d sms=%d l2_bytes=%d driver=%d runtime=%d "
             "cublas=%d git=%s mode=%s seed=%d profile=warmup%d+%.0fms_iso%d_graph%dx%d spin_ns=%llu",
             utc_now().c_str(), host, p.name, p.major, p.minor, p.multiProcessorCount,
             p.l2CacheSize, drv, rt, cublas_ver, GIT_DESC, mode.c_str(), seed, prof.warmup,
             prof.warmup_ms, prof.iso, prof.graph_n, prof.graph_reps, kSpinNs);
    printf("%s\n", prov);

    std::vector<KernelDef> kernels;
    if (mode == "tune") kernels = tune_set();
    else if (mode == "rotation") kernels = rotation_set(ctx.sink);
    else kernels = final_set(ctx, mode == "final" || mode == "ncu");
    if (mode == "ncu")
        kernels.erase(std::remove_if(kernels.begin(), kernels.end(), [](const KernelDef& k) { return k.name == "v2abl"; }),
                      kernels.end());
    std::vector<Fmt> fmts = {Fmt::BF16, Fmt::INT8, Fmt::INT4};
    if (seed) {
        std::mt19937 rng(seed);
        std::shuffle(kernels.begin(), kernels.end(), rng);
        std::shuffle(fmts.begin(), fmts.end(), rng);
    }
    const std::vector<size_t> rotation_footprints = {1, 2, 4, 8, 16, 32, 64, 128};  // x L2

    FILE* csv = fopen(out.c_str(), "w");
    const std::string acc_path = out.substr(0, out.rfind('.')) + "_accuracy.csv";
    const bool want_acc = mode == "final";
    FILE* acsv = want_acc ? fopen(acc_path.c_str(), "w") : nullptr;
    if (!csv || (want_acc && !acsv)) {
        fprintf(stderr, "cannot write %s\n", out.c_str());
        return EXIT_FAILURE;
    }
    fprintf(csv, "%s\n", prov);
    fprintf(csv,
            "matrix,M,K,fmt,kernel,config,copies,bytes_counted,iso_med_us,iso_p10_us,iso_p90_us,"
            "iso_min_us,spin_med_us,str_med_us,str_min_us,str_max_us,graph_launches,gbps_iso,gbps_str,"
            "host_enqueue_max_us,spin_covered,check_pass,check_rel_l2,check_worst_ratio,"
            "check_rad_pass,check_rad_rel_l2,recheck_pass,bitwise_same,acc_rel_vs_bf16,"
            "acc_maxabs_vs_bf16\n");
    if (acsv) {
        fprintf(acsv, "%s\n", prov);
        fprintf(acsv,
                "matrix,M,K,fmt,rel_err_cpu,maxabs_err_cpu,predicted_rel_err,quant_violations,"
                "max_err_over_scale,subnormal_scales,zero_scales\n");
    }

    FILE* scsv = samples_path.empty() ? nullptr : fopen(samples_path.c_str(), "w");
    if (!samples_path.empty() && !scsv) {
        fprintf(stderr, "cannot write %s\n", samples_path.c_str());
        return EXIT_FAILURE;
    }
    if (scsv) fprintf(scsv, "%s\nmatrix,fmt,kernel,config,copies,mode,i,us\n", prov);
    bool all_pass = true;
    long n_rows = 0;
    const auto t_start = std::chrono::steady_clock::now();
    for (const MatrixInfo& mi : read_manifest(data + "/manifest.tsv")) {
        if (!only.empty() && !only.count(mi.id)) continue;
        if (mode == "tune" && only.empty() && !kTuneMatrices.count(mi.id)) continue;
        if (mode == "rotation" && only.empty() && !kRotationMatrices.count(mi.id)) continue;
        const int M = mi.M, K = mi.K;
        printf("\n== %s  %s  M=%d K=%d\n", mi.id.c_str(), mi.tensor.c_str(), M, K);
        fflush(stdout);
        const MatrixData d = load_matrix(data, mi);
        const std::vector<uint8_t> xb = read_file(data + "/x_k" + std::to_string(K) + ".bin", (size_t)K * 2);
        const std::vector<uint8_t> xr = read_file(data + "/xrad_k" + std::to_string(K) + ".bin", (size_t)K * 2);
        const std::vector<float> x = bf16_file_to_float(xb), xrad = bf16_file_to_float(xr);

        const uint16_t* W = reinterpret_cast<const uint16_t*>(d.bf16.data());
        const RowBF16 rb{W, K};
        const RowI8 r8{reinterpret_cast<const int8_t*>(d.q8.data()),
                       reinterpret_cast<const uint16_t*>(d.s8.data()), K};
        const RowI4 r4{d.q4.data(), reinterpret_cast<const uint16_t*>(d.s4.data()), K};
        std::map<Fmt, RefResult> ref, ref_rad;
        ref[Fmt::BF16] = reference_gemv(M, K, x, rb);
        ref[Fmt::INT8] = reference_gemv(M, K, x, r8);
        ref[Fmt::INT4] = reference_gemv(M, K, x, r4);
        ref_rad[Fmt::BF16] = reference_gemv(M, K, xrad, rb);
        ref_rad[Fmt::INT8] = reference_gemv(M, K, xrad, r8);
        ref_rad[Fmt::INT4] = reference_gemv(M, K, xrad, r4);

        if (acsv) {
            const QuantCheck c8 = check_quant(
                M, K, W, 127, K, [&](int r, int k) { return (int)r8.Q[(size_t)r * K + k]; },
                [&](int r, int) { return r8.S[r]; });
            const QuantCheck c4 = check_quant(
                M, K, W, 7, kGroup, [&](int r, int k) { return int4_at(r4.Q + (size_t)r * (K / 2), k); },
                [&](int r, int g) { return r4.S[(size_t)r * (K / kGroup) + g]; });
            const double p8 = predicted_rel_error(M, K, W, r8), p4 = predicted_rel_error(M, K, W, r4);
            for (auto [f, c, pr] : {std::tuple{Fmt::INT8, c8, p8}, std::tuple{Fmt::INT4, c4, p4}}) {
                double num = 0, den = 0, mx = 0;
                for (int r = 0; r < M; ++r) {
                    const double e = ref[f].y[r] - ref[Fmt::BF16].y[r];
                    num += e * e;
                    den += ref[Fmt::BF16].y[r] * ref[Fmt::BF16].y[r];
                    mx = std::max(mx, std::fabs(e));
                }
                fprintf(acsv, "%s,%d,%d,%s,%.6e,%.6e,%.6e,%ld,%.6f,%ld,%ld\n", mi.id.c_str(), M, K,
                        fmt_name(f), std::sqrt(num / den), mx, pr, c.violations, c.max_err_over_scale,
                        c.subnormal_scales, c.zero_scales);
                printf("  quant %s: cpu rel err %.4e (predicted %.4e), violations %ld, max|w-qs|/s %.4f, "
                       "subnormal scales %ld, zero scales %ld\n",
                       fmt_name(f), std::sqrt(num / den), pr, c.violations, c.max_err_over_scale,
                       c.subnormal_scales, c.zero_scales);
                if (c.violations) all_pass = false;
            }
            fflush(acsv);
        }

        bf16 *dx = nullptr, *dxr = nullptr, *dy = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, (size_t)K * 2));
        CUDA_CHECK(cudaMalloc(&dxr, (size_t)K * 2));
        CUDA_CHECK(cudaMalloc(&dy, (size_t)M * 2));
        CUDA_CHECK(cudaMemcpy(dx, xb.data(), (size_t)K * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dxr, xr.data(), (size_t)K * 2, cudaMemcpyHostToDevice));

        for (Fmt f : fmts) {
            bool any = false;
            for (const KernelDef& kd : kernels) any |= kd.fmt == f;
            if (!any) continue;
            const HostOperand ho = d.op(f);
            const size_t rule = rotation_copies(ho.stored_bytes(), (size_t)p.l2CacheSize);
            std::vector<size_t> counts{rule};
            if (mode == "rotation") {
                counts.clear();
                for (size_t m : rotation_footprints) {
                    const size_t r = std::max<size_t>(1, (m * (size_t)p.l2CacheSize + ho.stored_bytes() / 2) / ho.stored_bytes());
                    if (counts.empty() || r != counts.back()) counts.push_back(r);
                }
            }
            for (size_t ncopies : counts) {
                DeviceCopies dc(ho, ncopies);
                for (const KernelDef& kd : kernels) {
                    if (kd.fmt != f) continue;
                    const Launch L = kd.make(M, K);
                    if (!L) continue;  // e.g. a cuBLASLt heuristic index past the returned list
                    const bool is_read = kd.name == "read";
                    const size_t counted = is_read ? dc.ops[0].bytes : ho.stored_bytes() + (size_t)K * 2 + (size_t)M * 2;

                    CheckResult c0, cr, c1;
                    double acc_rel = 0, acc_max = 0;
                    std::vector<uint16_t> y0;
                    if (kd.checked) {
                        cr = check_output(run_once(L, dc.ops[0], dxr, dy, M, s), ref_rad[f].y, ref_rad[f].absdot);
                        y0 = run_once(L, dc.ops[0], dx, dy, M, s);
                        c0 = check_output(y0, ref[f].y, ref[f].absdot);
                        error_vs(y0, ref[Fmt::BF16].y, acc_rel, acc_max);
                    }
                    const Timing t = time_launch(L, dc.ops, dx, dy, s, prof,
                                                 mi.id + " " + fmt_name(f) + " " + kd.name + " " + kd.config + " x" +
                                                     std::to_string(dc.ops.size()));
                    bool same = true;
                    if (kd.checked) {  // the last timed launch wrote y; it must still be right
                        std::vector<uint16_t> y1(M);
                        CUDA_CHECK(cudaMemcpy(y1.data(), dy, (size_t)M * 2, cudaMemcpyDeviceToHost));
                        c1 = check_output(y1, ref[f].y, ref[f].absdot);
                        same = std::memcmp(y0.data(), y1.data(), (size_t)M * 2) == 0;
                        if (kd.fatal && (!c0.pass || !cr.pass || !c1.pass || !same)) all_pass = false;
                    }
                    const double gi = counted / (t.iso_med * 1e3), gs = counted / (t.str_med * 1e3);
                    // Check and accuracy columns: empty for unchecked diagnostics (read, v4load).
                    char chk[256] = ",,,,,,,,";
                    if (kd.checked)
                        snprintf(chk, sizeof chk, "%s,%.3e,%.3f,%s,%.3e,%s,%d,%.6e,%.6e", c0.pass ? "pass" : "FAIL",
                                 c0.rel_l2, c0.worst_ratio, cr.pass ? "pass" : "FAIL", cr.rel_l2,
                                 c1.pass ? "pass" : "FAIL", (int)same, acc_rel, acc_max);
                    fprintf(csv,
                            "%s,%d,%d,%s,%s,%s,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%.2f,%.2f,"
                            "%.1f,%d,%s\n",
                            mi.id.c_str(), M, K, fmt_name(f), kd.name.c_str(), kd.config.c_str(), dc.ops.size(),
                            counted, t.iso_med, t.iso_p10, t.iso_p90, t.iso_min, t.spin_med, t.str_med,
                            t.str_min, t.str_max, t.graph_launches, gi, gs, t.host_enqueue_max_us,
                            (int)t.spin_covered, chk);
                    fflush(csv);
                    if (scsv) {
                        for (auto [mname, v] : {std::pair{"iso", &t.iso_samples}, std::pair{"spin", &t.spin_samples},
                                                std::pair{"str", &t.str_samples}})
                            for (size_t i = 0; i < v->size(); ++i)
                                fprintf(scsv, "%s,%s,%s,%s,%zu,%s,%zu,%.3f\n", mi.id.c_str(), fmt_name(f),
                                        kd.name.c_str(), kd.config.c_str(), dc.ops.size(), mname, i, (*v)[i]);
                        fflush(scsv);
                    }
                    ++n_rows;
                    const bool ok = !kd.checked || (c0.pass && cr.pass && c1.pass && same);
                    printf("  %-8s %-4s %-28s x%-3zu iso %9.2f us %6.1f GB/s | spin %9.2f | str %9.2f us %6.1f GB/s | %s%s%s\n",
                           kd.name.c_str(), fmt_name(f), kd.config.c_str(), dc.ops.size(), t.iso_med, gi,
                           t.spin_med, t.str_med, gs, kd.checked ? (ok ? "pass" : (kd.fatal ? "FAIL" : "FAIL (recorded, not fatal)")) : "n/a",
                           same ? "" : " (not bitwise repeatable)", t.spin_covered ? "" : " (spin too short)");
                    fflush(stdout);
                }
            }
        }
        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dxr));
        CUDA_CHECK(cudaFree(dy));
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("\n%s: %ld rows in %.0f s; every check %s\n", out.c_str(), n_rows, secs,
           all_pass ? "passed" : "DID NOT PASS");
    fclose(csv);
    if (acsv) fclose(acsv);
    if (scsv) fclose(scsv);
    CUDA_CHECK(cudaFree(ctx.sink));
    CUDA_CHECK(cudaFree(ctx.ws));
    CUDA_CHECK(cudaStreamDestroy(s));
    CUBLAS_CHECK(cublasLtDestroy(ctx.lt));
    CUBLAS_CHECK(cublasDestroy(ctx.blas));
    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
