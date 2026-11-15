# Rotation sweep: how many copies keep every launch reading from DRAM?

The benchmark cycles each timed launch through R device copies of the matrix so that no
launch finds its weights in a cache. The plan's rule was R = max(2, ⌈4 × L2 / stored
bytes⌉). These files test it.

`sweep_footprint_pre_rule.*` (2026-11-14, development build, before any tuning or final
run): the read-only reference, v2 (BF16) and v4 (INT4) on `gate` (47 MB BF16), `out`
(21 MB BF16) and `k` (5.2 MB BF16), with R chosen so the total footprint R × stored bytes
is 1, 2, 4, …, 128 × L2 (L2 = 25 165 824 B). Columns as in every bench CSV; `copies` is
R; `_samples.csv.gz` has every raw timing sample.

Finding: at the planned 4 × L2 footprint, rates were up to 14% above the DRAM plateau
(`out` BF16, v2: 257 GB/s at R = 5 vs 226 GB/s from R = 38 on; some reads even exceeded
the rated 273 GB/s), so something beyond the 24 MiB L2 (another cache level, or a
replacement policy that keeps part of older copies) still served data. At R = 5 the
isolated samples repeat with period 5 — each copy has its own speed — which is the
bimodal spread first seen in the development run. Every case reached its plateau
(within about 1%) by 16 × L2, so the rule was changed to **R = max(2, ⌈32 × L2 / stored
bytes⌉)** (805 MB footprint) before tuning. The final runs repeat the sweep with the
frozen kernels (`sweep_final*`).
