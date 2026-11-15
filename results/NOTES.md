# Notes for the report

Hand-written interpretation of the kept results, for writing the report. Every number
here comes from a file under `results/` (named in brackets); `ANALYSIS.md` is the
generated, complete version. Where a number is only a reading of the data, it says so.

## 1. Outcome in one paragraph

On an NVIDIA GB10 (DGX Spark), batch-1 GEMV on real Qwen3.5-4B weights is purely
bandwidth-bound once the loads are coalesced. A BF16 kernel reaches 226–238 GB/s on the
matrices of 21 MB and up, 97.5–101% of a plain read of the same bytes. Storing the weights as INT8 or INT4 makes
the kernel faster almost exactly in proportion to the bytes saved: 1.98–2.02× for INT8
(byte ratio 2.00) and 3.55–3.88× for INT4 (byte ratio 3.87) on the 47 MB MLP matrices and
the 1.27 GB LM head. That costs about 1% (INT8) and 12–14% (INT4) relative output error
with a random input. All four pre-registered verdicts (H1, H2-INT8, H2-INT4, H3) are
confirmed, and each held in all three executions [ANALYSIS.md §4–6].

## 2. Findings, in the order the report should tell them

1. **Coalescing is almost everything on this GPU.** v0 (thread per row) reaches 19–113
   GB/s; v1 (warp per row, 2-byte loads) already reaches 96–98% of v2 on every matrix
   (e.g. gate 225.4 vs 232.2 GB/s) [ladder.csv]. The proposal expected v1 at
   50–70% of 241 GB/s; it reached 90–97% on the matrices of 21 MB and up (83% on 5 MB). The GB10's L1/L2 and the sheer number of
   warps in flight hide the small 2-byte loads.
2. **The proposal's v2 design elements do not help.** Tuning (72 configurations,
   pre-registered rule) picked one row per warp, no unroll, x read from global memory,
   and a full (non-persistent) grid: effectively "v1 with 16-byte loads". The whole v2
   configuration space spans only 228–232 GB/s (geometric mean) [tuning/selection.md].
   The ablation shows each removed element costs 0–1.3% on the large matrices and up to
   3–6.5% only on the 5 MB `k_proj` [ANALYSIS.md §10]. Staging x in shared memory is not
   needed: x (5–18 KB) stays in L1/L2 anyway.
3. **cuBLAS picks a slow kernel for most of these shapes.** `cublasGemmEx` (and PyTorch's
   `F.linear`, which takes the same 279 µs on gate) reaches only ~167 GB/s on every
   matrix with K = 2560 or 4096, but 227 GB/s on `down_proj` (K = 9216) [ladder.csv,
   marlin/marlin.csv]. cuBLASLt's heuristic list for the same problem ranks that slow
   algorithm first (heur0 ≈ GemmEx) and a fast one second (heur1 runs at v2's speed on
   10 of 11 matrices) — reading: a heuristic tuned for other GPUs. So H1's
   "within 1.15× of cuBLAS" is easy (0.72×), and speedups *vs cuBLAS* (up to 5.3× for
   INT4) overstate what quantization buys; the report should lead with speedups over
   v2 and over cuBLASLt's best passing algorithm, which agree within 1%.
4. **Some cuBLASLt algorithms fail the accuracy contract.** On 10 (matrix, algorithm)
   pairs, an algorithm returned by cuBLASLt's heuristic exceeds the per-element bound
   (worst ≈ 1.8× the bound) while staying under the relative-L2 limit — the signature of
   a second BF16 rounding (e.g. split-K partial sums stored in BF16). They are recorded
   and excluded from "cuBLASLt best" [ANALYSIS.md §2].
5. **Bytes buy speed, almost one-for-one.** INT8 v3: 1.98–2.02× over v2. INT4 v4:
   3.55–3.88× over v2 on the H2 matrices, i.e. 92–100% of the byte ratio. The load-only
   diagnostic (same loads, no decode) is within 0–1.7% of v4 on every matrix, so
   decoding INT4 is free; v4 is memory-bound [ANALYSIS.md §5, §11]. The remaining gap to
   the byte ratio is that smaller transfers reach lower bandwidth: a plain read of
   11.8 MB of INT4 weights reaches 226 GB/s vs 236 GB/s for the 47 MB BF16 matrix
   [ladder.csv, read ref column].
6. **Small matrices pay a fixed per-launch cost.** INT4's speedup over v2 drops from
   3.64× (47 MB) to 3.41× (21 MB) to 2.58× (5 MB) [ANALYSIS.md §8]. The 21-vs-47 MB gap
   (6.3% isolated, 3.7% streamed) is between the predicted 4% and the 10% trigger: no
   sign of a cause beyond fixed costs. In a back-to-back stream (CUDA graph), launches
   overlap their start-up and the median kernel saves 3.2 µs [§9].
7. **Accuracy matches a two-line prediction.** Measured relative errors track
   √(Σδ²/Σw²) (δ = w − q·s) within about 3% on every layer-0/3 matrix: INT8 0.88–1.12%,
   INT4 11.9–12.9% [ANALYSIS.md §6]. The proposal's Gaussian-weight estimate (≈ 0.8–0.9%
   and 11–12%) is close; heavier-tailed real weights (excess kurtosis 0.1–5.8, median
   0.9) raise both a little (Spearman 0.39 between INT4 error and kurtosis over 201
   matrices) [accuracy/all_layers.csv]. Across all 32 layers the worst errors are 1.62%
   (INT8, layer-27 `k_proj`) and 14.17% (INT4, layer-31 `o_proj`) — under both
   thresholds, INT4 narrowly. The LM head is the exception to the prediction (measured
   13.64% vs predicted 12.58%); see §4 below.
8. **Per token, the kernels deliver what the bytes promise.** Summing measured
   (streamed) times over the 201 large GEMVs of one token gives GEMV-only speeds of 21
   (cuBLAS BF16), 28 (v2 BF16), 56 (INT8) and 103 tokens/s (INT4): 98%, 98% and 92% of
   the byte-count ceilings at 241 GB/s for v2, v3 and v4 [analysis/per_token.csv]. These
   are upper bounds for a real engine (no attention, norms, state or KV-cache traffic).

## 3. Method findings worth a paragraph each (the "assumptions" part of the report)

- **The 4 × L2 rotation rule was not enough.** On the GB10, cycling copies whose total
  footprint is 4 × L2 still let 3–14% of the rate come from a cache (21 MB BF16 matrix:
  257 vs 226 GB/s; some reads exceeded the rated 273 GB/s). Rates settle from ~16 × L2;
  the benchmark uses 32 × L2 (805 MB) [rotation/README.md, figures/fig_rotation.png].
  The repeat with the frozen kernels is `rotation/sweep_final.csv`.
- **How to time a 10 µs kernel with events.** A spin kernel before the start event
  (a common trick to hide launch latency) made things worse on the GB10: +1.3 µs and a
  ~1 µs lattice. Plain events around one launch add a fixed ~1.3–1.7 µs over the true
  kernel time (Nsight Systems) [timing_method/README.md]. cuBLAS is the exception: its
  host-side call path makes its plain-event time ~2–3 µs pessimistic (≤ 1% on the
  hypothesis matrices) [§9: spin − isolated down to −3.5 µs for cuBLAS only].
- **The bandwidth reference.** The same-session 1 GiB read-only reference was 236.0–237.1
  GB/s in all three executions [final/bandwidth_run*.txt], below the proposal's 241
  (2%). Differences: a different day, and the reference now reads non-zero data (the
  proposal's read zeros). Verdicts use the pre-registered 193 GB/s (80% of 241); the %
  of the same-session reference is shown alongside. A GEMV can slightly exceed this
  simple read kernel (LM head v2: 101%).
- **Run-to-run spread is small.** Across the three executions, the per-execution
  medians of a kernel differ by a median of 0.24% (max 2.7%, v0 on `up`) on the matrices
  of 21 MB and up; up to 3.7% on the 5 MB matrices [ladder.csv, run spread].

## 4. Open points and caveats (threats to validity)

- **LM head accuracy above the prediction** (13.64% vs 12.58% INT4; 1.12% vs 1.02%
  INT8). The prediction is the expectation over random x; one x is a single sample. An
  embedding matrix is anisotropic (a large shared direction), so xᵀWᵀWx for one x can
  sit well below its expectation, inflating the ratio. To verify: average the measured
  ratio over many x (planned check; see the end of this file).
- Random Gaussian x measures weight-rounding error, not model quality; real activations
  have outliers that can make INT4 RTN much worse (the reason GPTQ/AWQ exist).
- One GPU, one model, batch 1, one decode step; no end-to-end tokens/s measured.
- Quantization is round-to-nearest with FP16 scales; the INT4 scale overhead (FP16 per
  128 weights, 1.6%) is counted in every byte figure.
- Isolated cuBLAS times include ~2–3 µs of its host path (above); Nsight Systems kernel
  durations give the fair comparison [nsys/crosscheck.md].

## 5. Deviations from the proposal (for an honesty paragraph)

- y_ref in double rather than FP32; tolerance-based correctness instead of rel-L2 < 1e-2.
- Rotation footprint 32 × L2 instead of "> 4 × L2" (evidence above).
- Added: `k_proj`/`v_proj` shapes (per-token model only), streamed timing, size-matched
  read references, cuBLASLt, the ablation, the load-only diagnostic, the all-layer sweep.
- The final v2 is the tuned member of the v2 family, which turned out not to use shared
  memory or a persistent grid.
- Hypotheses and thresholds unchanged; verdicts on the median of three executions.

## 6. Suggested figures and tables for the report

| Report item | Source |
| --- | --- |
| Bandwidth ladder (figure) | `figures/fig_ladder.png` |
| Timing table per matrix (table) | `analysis/ladder.csv` (pick v1, v2, cuBLAS, cuBLASLt best, v3, v4) |
| H1, H2, H3 verdict tables | `ANALYSIS.md` §4–6 |
| Speedup vs byte ratio (figure) | `figures/fig_speedup_bytes.png` |
| Speed vs accuracy (figure) | `figures/fig_speed_accuracy.png` |
| Accuracy over all layers (figure) | `figures/fig_accuracy_layers.png` |
| v2 ablation (figure or table) | `figures/fig_ablation.png`, `analysis/ablation.csv` |
| Rotation sweep (figure, method section) | `figures/fig_rotation.png` |
| Timing modes / small matrices (figure) | `figures/fig_timing_modes.png`, `ANALYSIS.md` §8–9 |
| Per-token model (figure) | `figures/fig_per_token.png`, `analysis/per_token.csv` |
| Correctness evidence | `tests/test_output.txt`, `ANALYSIS.md` §2 |
