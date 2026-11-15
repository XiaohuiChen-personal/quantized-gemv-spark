#!/usr/bin/env python3
"""Is one Gaussian x representative? Relative quantization error over many random x.

The H3 errors use one fixed Gaussian x per K. The prediction sqrt(sum delta^2 / sum w^2)
is the ratio of the expected squared errors over random x. For a strongly anisotropic
matrix (one dominant direction, typical of embedding matrices) a single x can sit far from
that expectation. For each benchmarked matrix this script reports the error for the fixed
x, the mean and spread over N fresh Gaussian x, and an anisotropy measure:
||mean row||^2 / mean(||row||^2) (0 for isotropic rows, 1 if all rows are equal).

Usage: python3 scripts/x_sample_check.py [--data data] [--n 64] [--out results/accuracy/x_samples.csv]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_x  # noqa: E402
import quantize as qz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="results/accuracy/x_samples.csv")
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.data, "manifest.json")))["matrices"]
    rng = np.random.default_rng(7)
    rows = []
    for m in man:
        M, K = m["M"], m["K"]
        raw = np.fromfile(os.path.join(a.data, f"{m['id']}.bf16.bin"), dtype=np.uint16).reshape(M, K)
        X = rng.standard_normal((K, a.n))
        x0 = qz.bf16_to_f32(make_x.gaussian_x(K)).astype(np.float64)[:, None]
        Xall = np.hstack([x0, X])
        yy = np.zeros(a.n + 1)
        e8 = np.zeros(a.n + 1)
        e4 = np.zeros(a.n + 1)
        rowsum = np.zeros(K)
        rownorm2 = 0.0
        for r0 in range(0, M, 8192):
            w = qz.bf16_to_f32(raw[r0:r0 + 8192])
            w64 = w.astype(np.float64)
            y = w64 @ Xall
            q8, s8, _ = qz.quantize_int8(w)
            p4, s4, _ = qz.quantize_int4(w)
            d8 = qz.dequant_int8(q8, s8).astype(np.float64) @ Xall - y
            d4 = qz.dequant_int4(p4, s4).astype(np.float64) @ Xall - y
            yy += (y * y).sum(0)
            e8 += (d8 * d8).sum(0)
            e4 += (d4 * d4).sum(0)
            rowsum += w64.sum(0)
            rownorm2 += (w64 * w64).sum()
        r8, r4 = np.sqrt(e8 / yy), np.sqrt(e4 / yy)
        aniso = float((rowsum / M) @ (rowsum / M) / (rownorm2 / M))
        row = {"matrix": m["id"], "M": M, "K": K, "int8_fixed_x": r8[0], "int8_mean": r8[1:].mean(),
               "int8_p5": np.percentile(r8[1:], 5), "int8_p95": np.percentile(r8[1:], 95),
               "int4_fixed_x": r4[0], "int4_mean": r4[1:].mean(), "int4_p5": np.percentile(r4[1:], 5),
               "int4_p95": np.percentile(r4[1:], 95), "anisotropy": aniso, "n_x": a.n}
        rows.append(row)
        print(f"{m['id']:7s} int4 fixed x {r4[0]:.4f}  mean {r4[1:].mean():.4f} [{row['int4_p5']:.4f}, {row['int4_p95']:.4f}]  "
              f"int8 fixed {r8[0]:.5f} mean {r8[1:].mean():.5f}  anisotropy {aniso:.4f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# scripts/x_sample_check.py: {a.n} fresh Gaussian x per matrix (seed 7) plus the fixed x of make_x\n")
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")


if __name__ == "__main__":
    main()
