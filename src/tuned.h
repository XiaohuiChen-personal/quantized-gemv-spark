// Frozen kernel configurations used by the final sweep.
//
// PLACEHOLDER until the tuning run (results/tuning/) picks them by the pre-registered
// rule: best geometric-mean isolated bandwidth over the six unique hypothesis shapes,
// one configuration per kernel.
#pragma once

struct VecConfig {
    int threads, rows_per_warp, unroll;
    bool x_shared, persistent;
};

constexpr VecConfig kV2{256, 2, 2, true, true};
constexpr VecConfig kV3{256, 2, 2, true, true};
constexpr VecConfig kV4{256, 2, 2, true, true};
