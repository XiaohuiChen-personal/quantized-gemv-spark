// Frozen kernel configurations used by the final sweep.
//
// Chosen by scripts/select_tuned.py from results/tuning/tune.csv with the pre-registered rule:
// best geometric-mean isolated bandwidth over the six unique hypothesis shapes,
// one configuration per kernel. See results/tuning/selection.md.
#pragma once

struct VecConfig {
    int threads, rows_per_warp, unroll;
    bool x_shared, persistent;
};

constexpr VecConfig kV2{256, 1, 1, false, false};  // t256_r1_u1_xg_full
constexpr VecConfig kV3{256, 1, 2, false, false};  // t256_r1_u2_xg_full
constexpr VecConfig kV4{128, 1, 2, false, false};  // t128_r1_u2_xg_full
