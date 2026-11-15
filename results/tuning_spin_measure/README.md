# Superseded tuning run (spin-kernel isolated measure)

The first tuning run (commit `d78d95e`), whose isolated times were taken with a spin
kernel before the start event. Code review #1 and `results/timing_method/` showed that on
the GB10 this adds ~1.3 µs and lands on a ~1 µs lattice, so the tuning was repeated
with plain events (`results/tuning/`). Kept for the record; not used by any verdict. Its
picks (v2 `t256_r1_u1_xg_full`, v3 `t256_r1_u2_xg_full`, v4 `t128_r1_u2_xg_full`) beat
their runners-up by 0.03–0.27%, below one lattice step: effectively ties.
