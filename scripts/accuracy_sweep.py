#!/usr/bin/env python3
"""Quantization error of every large language-model matrix of Qwen3.5-4B (CPU, NumPy).

For each 2-D weight with at least 1024 rows in all 32 layers, plus the LM head, with the
same quantizer (scripts/quantize.py) and the same Gaussian x per K (scripts/make_x.py)
as the GPU benchmark:

  rel_int8, rel_int4    ||(W_q - W) x|| / ||W x||, in double precision
  pred_int8, pred_int4  sqrt(sum delta^2 / sum w^2), the expected value for Gaussian x
  excess_kurtosis       of the weights (0 for a Gaussian; heavier tails -> larger)
  max_over_rms          max|w| / RMS(w) over the matrix

The rows for the eleven benchmarked matrices double as an independent NumPy cross-check
of the C++ CPU-only errors (bench *_accuracy.csv).

Usage: python3 scripts/accuracy_sweep.py [--out results/accuracy/all_layers.csv]
"""
import argparse
import csv
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_weights as ew  # noqa: E402
import make_x  # noqa: E402
import quantize as qz  # noqa: E402

CHUNK = 8192  # rows per block: bounds memory for the 248 320-row LM head


def header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot")
    ap.add_argument("--out", default="results/accuracy/all_layers.csv")
    a = ap.parse_args()
    snap = ew.find_snapshot(a.snapshot)
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    headers = {}
    names = [n for n in sorted(wmap) if n.startswith(ew.P) and n.endswith(".weight")]
    xs = {}
    out_rows = []
    t0 = time.time()
    for name in names:
        fname = wmap[name]
        if fname not in headers:
            headers[fname] = header(os.path.join(snap, fname))
        hdr, base = headers[fname]
        t = hdr[name]
        if len(t["shape"]) != 2 or t["dtype"] != "BF16" or t["shape"][0] < 1024:
            continue
        M, K = t["shape"]
        if K not in xs:
            xs[K] = qz.bf16_to_f32(make_x.gaussian_x(K)).astype(np.float64)
        x = xs[K]
        mm = np.memmap(os.path.join(snap, fname), dtype=np.uint16, mode="r",
                       offset=base + t["data_offsets"][0], shape=(M, K))
        acc = dict(yy=0.0, e8=0.0, e4=0.0, d8=0.0, d4=0.0, s1=0.0, s2=0.0, s4=0.0, amax=0.0)
        for r0 in range(0, M, CHUNK):
            w = qz.bf16_to_f32(np.asarray(mm[r0:r0 + CHUNK]))
            w64 = w.astype(np.float64)
            y = w64 @ x
            q8, s8, _ = qz.quantize_int8(w)
            d8 = qz.dequant_int8(q8, s8).astype(np.float64)
            p4, s4, _ = qz.quantize_int4(w)
            d4 = qz.dequant_int4(p4, s4).astype(np.float64)
            acc["yy"] += float(y @ y)
            e8 = d8 @ x - y
            e4 = d4 @ x - y
            acc["e8"] += float(e8 @ e8)
            acc["e4"] += float(e4 @ e4)
            acc["d8"] += float(((w64 - d8) ** 2).sum())
            acc["d4"] += float(((w64 - d4) ** 2).sum())
            w2 = w64 * w64
            acc["s1"] += float(w64.sum())
            acc["s2"] += float(w2.sum())
            acc["s4"] += float((w2 * w2).sum())
            acc["amax"] = max(acc["amax"], float(np.abs(w).max()))
        n = M * K
        mean = acc["s1"] / n
        var = acc["s2"] / n - mean * mean
        # Excess kurtosis about the mean; the weights' mean is ~0, so E[w^4] ~ the 4th
        # central moment (the correction terms are below 1e-6 here).
        kurt = (acc["s4"] / n) / (acc["s2"] / n) ** 2 - 3.0
        m = __import__("re").search(r"layers\.(\d+)\.", name)
        row = {
            "tensor": name, "layer": m.group(1) if m else "", "type": ew.layer_type(name),
            "short": name.replace(ew.P, "").replace(".weight", ""), "M": M, "K": K,
            "rel_int8": (acc["e8"] / acc["yy"]) ** 0.5, "pred_int8": (acc["d8"] / acc["s2"]) ** 0.5,
            "rel_int4": (acc["e4"] / acc["yy"]) ** 0.5, "pred_int4": (acc["d4"] / acc["s2"]) ** 0.5,
            "excess_kurtosis": kurt, "max_over_rms": acc["amax"] / (acc["s2"] / n) ** 0.5,
            "mean_over_std": mean / var ** 0.5,
        }
        out_rows.append(row)
        print(f"{row['short']:45s} {M:6d}x{K:5d} int8 {row['rel_int8']:.4%} (pred {row['pred_int8']:.4%})  "
              f"int4 {row['rel_int4']:.3%} (pred {row['pred_int4']:.3%})  kurt {kurt:6.2f}  "
              f"max/rms {row['max_over_rms']:6.1f}   [{time.time() - t0:.0f} s]", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        f.write(f"# model={ew.REPO} revision={ew.REVISION} x=make_x.gaussian_x(K) quantizer=scripts/quantize.py\n")
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items()})
    worst8 = max(out_rows, key=lambda r: r["rel_int8"])
    worst4 = max(out_rows, key=lambda r: r["rel_int4"])
    print(f"{len(out_rows)} matrices in {time.time() - t0:.0f} s; worst int8 {worst8['rel_int8']:.4%} "
          f"({worst8['short']}), worst int4 {worst4['rel_int4']:.3%} ({worst4['short']})")


if __name__ == "__main__":
    main()
