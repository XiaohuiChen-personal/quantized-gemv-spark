# Notes for the report

Hand-written interpretation of the kept results, for writing the report. Every number
comes from a file under `results/` (named in brackets); `ANALYSIS.md` is the generated,
complete version. Statements marked *Reading* are interpretations, not measurements.
This version incorporates adversarial review #3 (2026-11-15), which independently
recomputed all verdicts from the raw CSVs and agreed with every one.

## 1. Outcome in one paragraph

On an NVIDIA GB10 (DGX Spark), batch-1 GEMV on real Qwen3.5-4B weights is bandwidth-bound
once the loads are coalesced. The tuned BF16 kernel reaches 226–238 GB/s on the matrices
of 21 MB and up, 97.5–101% of a plain read of the same bytes [analysis/ladder.csv].
Storing the weights as INT8 or INT4 speeds it up almost in proportion to the bytes saved:
1.98–2.02× for INT8 (byte ratio 2.00) and 3.55–3.88× for INT4 (byte ratio 3.87) on the
47 MB MLP matrices and the 1.27 GB LM head, for about 1% (INT8) and 12–14% (INT4) relative
output error with a random input. All pre-registered verdicts (H1, H2-INT8, H2-INT4, H3)
are confirmed and each held in all three executions; the stretch H4 is confirmed too —
my INT4 kernel runs at 0.95–1.00× the time of vLLM's Marlin on the matrices of 21 MB and
up [ANALYSIS.md §4–7].

## 2. Findings, in the order the report should tell them

1. **Coalescing is almost everything on this GPU.** v0 (thread per row) reaches 19–113
   GB/s; v1 (warp per row, 2-byte loads) already reaches 96–98% of v2's rate on every
   matrix (e.g. gate 225.4 vs 232.2 GB/s) [ladder.csv]. The proposal expected v1 at 50–70%
   of 241 GB/s; it reached 90–97% on the matrices of 21 MB and up (83% on the 5 MB ones).
2. **The v2 design space is flat.** All 72 v2 configurations span only 228–232 GB/s
   (geometric mean) [tuning/selection.md]. The pre-registered rule picked one row per warp,
   no unroll, x read from global memory and a full grid, but the winner is only 0.24%
   above the runner-up (v3 0.03%, v4 0.34%), and the v2 and v3 picks changed between the
   two tuning runs (tuning_spin_measure/ vs tuning/) — they are near-ties, chosen with a
   lighter profile (warm-up 10 launches + 100 ms, 100 isolated launches) than the final
   runs. The ablation shows that the proposal's v2 elements — x in shared memory, a
   persistent grid, several rows per warp — change the time by −0.6% to +1.3% on the
   matrices of 21 MB and up and by up to +6.5% on the 5 MB `k_proj`; narrower loads cost
   up to +3.3% [ANALYSIS.md §10]. *Reading:* x (5–18 KB) stays in L1/L2 anyway, so staging
   it in shared memory buys nothing.
3. **cuBLAS picks a slow kernel for most of these shapes.** `cublasGemmEx` reaches
   133.5–173.0 GB/s on the ten matrices with K = 2560 or 4096 (gate/up/qkv/q ≈ 166–167,
   z/out/o ≈ 159, k/v ≈ 134, LM head 173) but 226.7 GB/s on `down_proj` (K = 9216)
   [ladder.csv]. PyTorch's BF16 `F.linear` behaves the same (283 µs on gate, like GemmEx's
   283 µs) [marlin/marlin_run*.csv]. cuBLASLt's heuristic list for the same problem puts
   GemmEx's choice first (heur0) and a fast algorithm (heur1) second, which runs at v2's
   speed on 9 of 11 matrices (not `down`, where heur1 is slower, nor the LM head, whose
   fast one is heur3) [final/run*.csv]. Nsight Systems shows every GemmEx call runs one
   `internal::gemvx::kernel` instance; the default and the fast alternative differ in
   template parameters and launch shape (16 × 4 vs 32 × 4 threads per block on gate), but
   another algorithm with the fast one's launch shape is slow too, so the cause is not
   resolved [nsys/cublas_kernels.md]. Consequence: H1's "within 1.15× of cuBLAS" is easy
   (0.72× on most matrices), and speedups *vs cuBLAS* (up to 5.3× for INT4) overstate what
   quantization buys; the report should lead with speedups over v2 and over cuBLASLt's best
   passing algorithm, which agree within 1% [ANALYSIS.md §5].
4. **Some cuBLASLt algorithms fail the accuracy contract.** On 10 (matrix, algorithm)
   pairs, an algorithm cuBLASLt's heuristic returns exceeds the per-element bound by
   1.80–3.32× (q/heur2 is the worst), and down/heur3 also exceeds the relative-L2 limit
   (4.71e-3 > 4e-3) [ANALYSIS.md §2, final/run*.csv]. *Reading:* an extra BF16 rounding
   (e.g. split-K partial sums stored in BF16). They are recorded and excluded from
   "cuBLASLt best".
5. **Bytes buy speed, almost one-for-one.** INT8 v3: 1.98–2.02× over v2. INT4 v4:
   3.55–3.88× over v2 on the H2 matrices, 92–100% of the byte ratio [ANALYSIS.md §5]. The
   load-only diagnostic (v4's loads of weights, scales and x, without the decode) is within
   0–1.7% of v4 on every matrix, so the INT4 decode is effectively free and v4 is
   memory-bound [§11]. Part of the gap to the byte ratio is that a smaller transfer reaches
   lower bandwidth (a plain read of the 12.2 MB INT4 `gate` reaches 226 GB/s vs 236 GB/s for
   its 47 MB BF16 form); the rest is v4 itself, which reaches 93–100% of its own
   size-matched read on the H2 matrices (93.3% on `down`, the largest gap) [ladder.csv].
6. **Smaller matrices gain less — mostly a measurement effect, partly real.** INT4's
   isolated speedup over v2 is 3.64× (47 MB), 3.41× (21 MB) and 2.58× (5 MB). The 21-vs-47
   MB gap is 6.3% isolated (above the predicted ~4%, below the 10% trigger), 3.7% streamed
   and 0.7% with Nsight kernel durations: the excess is the per-launch event overhead, not
   the kernels. On the 5 MB matrices Nsight still gives 3.02×, and a plain read of 1.35 MB
   reaches only ~155 GB/s, so small transfers really are less efficient [ANALYSIS.md §8].
7. **Accuracy matches a two-line prediction.** On the eight hypothesis matrices other than
   the LM head, the GPU kernels' relative errors are within −0.4% to +4.2% (relative) of
   √(Σδ²/Σw²) (δ = w − q·s): INT8 0.90–1.12%, INT4 11.90–12.92% [ANALYSIS.md §6]. The
   proposal's Gaussian-weight estimate (≈ 0.8–0.9% and 11–12%) is close; heavier-tailed
   real weights (excess kurtosis 0.09–5.75, median 0.92) raise both a little (Spearman 0.39
   between INT4 error and kurtosis over 201 matrices) [accuracy/all_layers.csv]. Across all
   32 layers, with the fixed x, the worst errors are 1.62% (INT8, layer-27 `k_proj`) and
   14.17% (INT4, layer-31 `o_proj`; predicted 14.31%) — under both thresholds, INT4
   narrowly. The LM head needs §4's explanation.
8. **Per token, the kernels deliver what the bytes promise.** Summing the measured
   (streamed) times over one token's 201 large GEMVs gives GEMV-only speeds of 21.3
   (cuBLAS BF16), 28.2 (v2 BF16), 56.2 (INT8) and 103 tokens/s (INT4): 98.2%, 98.2% and
   92.4% of the byte-count ceilings at 241 GB/s for v2, v3 and v4
   [analysis/per_token.csv]. Every weight is in the given format, including the tied LM
   head (13.6% error in INT4). These are upper bounds for a real engine (no attention,
   norms, state or KV-cache traffic).
9. **A hand-written INT4 kernel matches production code here.** vLLM 0.16's Marlin
   (`marlin_gemm`, uint4b8, group 128), three executions, same rotation and modes: streamed
   t(v4)/t(Marlin) = 0.972 (gate), 0.965 (up), 0.990 (down), 0.985 (LM head); 0.95–1.00 on
   all matrices of 21 MB and up; 0.84 on the 5 MB `k`/`v` (0.90 against Marlin's
   atomic-add variant) [ANALYSIS.md §7]. Plain-event timing is unfair to Marlin (its Python
   dispatch adds ~9 µs per call), so H4 is judged on streamed times; every mode passes.
   *Reading:* at batch 1 Marlin's tensor-core tiling buys nothing on a memory-bound GEMV;
   both kernels sit at the same bandwidth ceiling. Marlin's quantizer is 1–7% (relative)
   more accurate than mine on every matrix (gate 11.32% vs 11.90%) because it uses
   scale = max|w|/7.5 and the full code range −8…7, while mine uses q_max = 7 and leaves −8
   unused — a design choice worth one sentence in the report.
10. **Ground truth agrees.** Recomputing every verdict ratio from Nsight Systems kernel
    durations changes nothing (H1 0.72–1.00; H2 INT8 2.01–2.78×, INT4 3.35–5.34× vs
    cuBLAS; the 3.35 is a `down` v4 row that the profiled pass slowed by 5%) [§15].

## 3. Method findings worth a paragraph each (the "assumptions" part of the report)

- **The 4 × L2 rotation rule was not enough.** On the GB10, cycling copies whose total
  footprint is 4 × L2 still let up to 14% of the rate come from a cache (`out` BF16:
  13.8% with the development kernels, 13.5% with the frozen ones; 0% for some small INT4
  cases), and some reads exceeded the rated 273 GB/s [rotation/, figures/fig_rotation.png].
  From 16 × L2 on, rates are within a few percent of the plateau; the benchmark uses
  32 × L2 (805 MB). Even then, about one launch per rotation cycle is ≥ 15% faster than the
  median (158 of 11 000 isolated samples of the ladder kernels in execution 1, in 25 of 55
  rows; e.g. `down` v2: 11 of 200 with R = 18): one copy evidently still meets a warm cache
  or a favourable placement. Medians ignore it; streamed means include it (≲ 1%).
- **How to time a 10 µs kernel with events.** A spin kernel before the start event (a
  common way to hide launch latency) made the micro-benchmark worse on the GB10: +1.3 µs
  and a ~1 µs lattice; across the final set it sits a median +0.34 µs above plain events
  [timing_method/README.md, ANALYSIS.md §9]. Plain events exceed the Nsight kernel
  durations by a median 2.4 µs (IQR 1.7–2.9) over the final set [§9], which is why the
  smallest kernels look slower in isolated timing than they are. cuBLAS is the exception:
  its plain-event time includes part of its host-side call path (≤ 1% on the H1/H2
  matrices, 1.6–1.8% on the 21 MB ones); the Nsight ratios in §15 are the fair comparison.
- **The bandwidth reference.** The same-session 1 GiB read-only reference was 236.1, 235.4
  and 234.7 GB/s in the three executions (median of four grids each; overall 235.4), 2.3%
  below the proposal's 241 [final/bandwidth_run*.txt]. Differences: another day, and the
  reference now reads non-zero data (the proposal's read zeros). Verdicts use the
  pre-registered 193 GB/s (80% of 241 = 192.8); the share of the same-session reference is
  shown alongside. A GEMV can slightly exceed this simple read kernel (LM head v2: 101%).
- **Run-to-run spread is small but within one session.** The three executions ran back to
  back (about 20 minutes). Their per-kernel medians differ by a median of 0.24% (max 2.7%,
  v0 on `up`) on the matrices of 21 MB and up, up to 3.7% on the 5 MB ones [ladder.csv,
  run spread]; the day-to-day shift of the reference (2.3%) is the larger uncertainty.
- **Pre-registration.** The verdict rules were fixed on 2026-11-14 before the tuning and
  final runs; development runs (quick runs and the first rotation sweep) preceded them and
  motivated two method changes before any kept measurement (rotation footprint, event
  timing), both documented with their evidence.

## 4. Open points and caveats (threats to validity)

- **LM head accuracy above the prediction — explained.** Measured 13.64% (INT4) and 1.12%
  (INT8) vs predicted 12.58% and 1.02%. The embedding matrix is strongly anisotropic
  (‖mean row‖² / mean ‖row‖² = 0.142, vs ≤ 0.020 for every other matrix), so one x is a
  noisy sample. Over 64 fresh Gaussian x its INT4 error averages 12.55% (5th–95th
  percentile 10.06–13.78%) and INT8 1.018%, matching the predictions; the fixed x sits near
  the top [accuracy/x_samples.csv]. The largest 95th percentile over the 11 benchmarked
  matrices is 13.78% (INT4). Not sampled: the all-layer worst case (layer-31 `o_proj`,
  14.17% with the fixed x, predicted 14.31%), whose own spread over x could reach 15%.
- Random Gaussian x measures weight-rounding error, not model quality; real activations
  have outliers that can make INT4 round-to-nearest much worse (the reason GPTQ/AWQ exist).
- One GPU, one model, batch 1, one decode step; no end-to-end tokens/s measured.
- Marlin ran in another process and session, in a container with a CUDA 13.1
  forward-compatibility runtime, timed with PyTorch events, on weights quantized by vLLM's
  own quantizer (not mine). The Nsight pass is a single profiled execution.
- The final runs and Marlin record `git describe` honestly: the finals are at a clean
  `96a367a`; the Marlin CSVs say `e1371d0-dirty` only because the stage rewrote tracked
  result files (`results/marlin/env.txt`, `gpu_log.csv`) while running — no code change.
- The INT4 scale overhead (FP16 per 128 weights, 1.6%) is counted in every byte figure.

## 5. Deviations from the proposal (for an honesty paragraph)

- y_ref in double rather than FP32; tolerance-based correctness instead of rel-L2 < 1e-2.
- Rotation footprint 32 × L2 instead of "> 4 × L2" (evidence above).
- Added: `k_proj`/`v_proj` shapes (per-token model only), streamed timing, size-matched
  read references, cuBLASLt, the ablation, the load-only diagnostic, the all-layer sweep,
  the many-x check, the Nsight cross-check.
- The final v2 is the tuned member of the v2 family, which turned out not to use shared
  memory or a persistent grid.
- H4 is judged on streamed times (both kernels without host cost) on the H2 matrices; every
  timing mode and every matrix passes. The H1 threshold 193 GB/s is coded as 0.8 × 241
  (192.8); no value falls between.
- Hypotheses and thresholds unchanged; verdicts on the median of three executions.

## 6. Suggested figures and tables for the report

| Report item | Source |
| --- | --- |
| Bandwidth ladder (figure) | `figures/fig_ladder.png` |
| Timing table per matrix (table) | `analysis/ladder.csv` (pick v1, v2, cuBLAS, cuBLASLt best, v3, v4) |
| H1–H4 verdict tables | `ANALYSIS.md` §4–7 |
| Speedup vs byte ratio (figure) | `figures/fig_speedup_bytes.png` |
| Speed vs accuracy (figure) | `figures/fig_speed_accuracy.png` |
| Accuracy over all layers (figure) | `figures/fig_accuracy_layers.png` |
| v2 ablation (figure or table) | `figures/fig_ablation.png`, `analysis/ablation.csv` |
| Rotation sweeps (figure, method section) | `figures/fig_rotation.png` |
| Timing modes / small matrices (figure) | `figures/fig_timing_modes.png`, `ANALYSIS.md` §8–9 |
| Per-token model (figure) | `figures/fig_per_token.png`, `analysis/per_token.csv` |
| Correctness evidence | `tests/test_output.txt`, `ANALYSIS.md` §2 |
