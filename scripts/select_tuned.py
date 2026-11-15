#!/usr/bin/env python3
"""Pick the frozen v2/v3/v4 configurations from a tuning run (the pre-registered rule).

Rule (fixed before the tuning run): for each kernel, among the configurations whose
every correctness check passed on every tuning matrix (including the bitwise repeat), choose the one with the highest
geometric-mean isolated bandwidth (gbps_iso) over the six unique hypothesis shapes
(gate, down, qkv, z, out, lmhead). One configuration per kernel. Ties are impossible
in practice; the first in sorted order wins.

Writes src/tuned.h and a selection report (top configurations and the runners-up gap).

Usage: python3 scripts/select_tuned.py results/tuning/tune.csv [--header src/tuned.h]
                                       [--report results/tuning/selection.md]
"""
import argparse
import csv
import math
import re
from collections import defaultdict

SHAPES = ("gate", "down", "qkv", "z", "out", "lmhead")


def read_rows(path):
    with open(path) as f:
        prov = [l.rstrip("\n") for l in f if l.startswith("#")]
    with open(path) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    return prov, rows


def parse_cfg(cfg):
    m = re.fullmatch(r"t(\d+)_r(\d+)_u(\d+)_(xs|xg)_(pers|full)", cfg)
    if not m:
        raise ValueError(cfg)
    t, r, u, xs, grid = m.groups()
    return int(t), int(r), int(u), xs == "xs", grid == "pers"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--header", default="src/tuned.h")
    ap.add_argument("--report", default="results/tuning/selection.md")
    a = ap.parse_args()
    prov, rows = read_rows(a.csv)

    by = defaultdict(dict)  # (kernel, config) -> matrix -> row
    for r in rows:
        by[(r["kernel"], r["config"])][r["matrix"]] = r
    scored = defaultdict(list)  # kernel -> [(geomean, config, per-shape)]
    excluded = []
    for (kernel, cfg), per in by.items():
        missing = [s for s in SHAPES if s not in per]
        if missing:
            excluded.append((kernel, cfg, "missing " + ",".join(missing)))
            continue
        bad = [s for s in SHAPES if not (per[s]["check_pass"] == per[s]["check_rad_pass"] == per[s]["recheck_pass"] == "pass"
                                         and str(per[s]["bitwise_same"]) == "1")]
        if bad:
            excluded.append((kernel, cfg, "failed check on " + ",".join(bad)))
            continue
        g = math.exp(sum(math.log(float(per[s]["gbps_iso"])) for s in SHAPES) / len(SHAPES))
        scored[kernel].append((g, cfg, {s: float(per[s]["gbps_iso"]) for s in SHAPES}))

    chosen = {}
    lines = ["# Tuning selection", "", "Source: `" + a.csv + "`", ""] + [f"    {p}" for p in prov] + [""]
    lines += ["Rule: highest geometric-mean isolated GB/s over " + ", ".join(SHAPES) +
              " among configurations that passed every check (fixed before the run).", ""]
    for kernel in sorted(scored):
        ranked = sorted(scored[kernel], key=lambda t: (-t[0], t[1]))
        chosen[kernel] = ranked[0][1]
        lines += [f"## {kernel}: `{ranked[0][1]}`", "",
                  "| rank | config | geomean GB/s | " + " | ".join(SHAPES) + " |",
                  "| --- | --- | --- | " + " | ".join("---" for _ in SHAPES) + " |"]
        for i, (g, cfg, per) in enumerate(ranked[:10]):
            lines.append(f"| {i + 1} | `{cfg}` | {g:.1f} | " + " | ".join(f"{per[s]:.1f}" for s in SHAPES) + " |")
        worst = ranked[-1]
        lines += ["", f"{len(ranked)} configurations ranked; worst `{worst[1]}` at {worst[0]:.1f} GB/s; "
                  f"best is {100 * (ranked[0][0] / ranked[1][0] - 1):.2f}% above the runner-up.", ""]
    if excluded:
        lines += ["## Excluded", ""] + [f"- {k} `{c}`: {why}" for k, c, why in sorted(excluded)] + [""]

    hdr = ["// Frozen kernel configurations used by the final sweep.", "//",
           f"// Chosen by scripts/select_tuned.py from {a.csv} with the pre-registered rule:",
           "// best geometric-mean isolated bandwidth over the six unique hypothesis shapes,",
           "// one configuration per kernel. See results/tuning/selection.md.", "#pragma once", "",
           "struct VecConfig {", "    int threads, rows_per_warp, unroll;", "    bool x_shared, persistent;", "};", ""]
    for kernel, name in (("v2", "kV2"), ("v3", "kV3"), ("v4", "kV4")):
        t, r, u, xs, pers = parse_cfg(chosen[kernel])
        hdr.append(f"constexpr VecConfig {name}{{{t}, {r}, {u}, {str(xs).lower()}, {str(pers).lower()}}};"
                   f"  // {chosen[kernel]}")
    with open(a.header, "w") as f:
        f.write("\n".join(hdr) + "\n")
    with open(a.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    for k in sorted(chosen):
        print(f"{k}: {chosen[k]}")


if __name__ == "__main__":
    main()
