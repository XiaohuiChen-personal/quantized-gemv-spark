#!/usr/bin/env python3
"""Is one Gaussian x representative? Relative quantization error over many random x.

The H3 errors use one fixed Gaussian x per K. The prediction sqrt(sum delta^2 / sum w^2)
is the ratio of the expected squared errors over random x. For a strongly anisotropic
matrix (one dominant direction, typical of embedding matrices) a single x can sit far from
that expectation. For each matrix this script reports the error for the fixed x, the mean,
spread and maximum over N fresh Gaussian x, and an anisotropy measure:
||mean row||^2 / mean(||row||^2) (0 for isotropic rows, 1 if all rows are equal).

Two inputs:
  default        the eleven benchmarked matrices exported to data/ (64 x, seed 7)
  --all-layers   every matrix of scripts/accuracy_sweep.py (all 32 layers + LM head),
                 read straight from the safetensors files (256 x, seed 7); reports how many
                 x exceed H3's thresholds, so H3 across all layers can be judged for typical
                 inputs and not only for the fixed x.

Usage: python3 scripts/x_sample_check.py [--data data] [--n 64] [--out results/accuracy/x_samples.csv]
       python3 scripts/x_sample_check.py --all-layers [--n 256] [--out results/accuracy/x_samples_all_layers.csv]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_x  # noqa: E402
import quantize as qz  # noqa: E402

CHUNK = 8192  # rows per block: bounds memory for the 248 320-row LM head
THRESH = {"int8": 0.02, "int4": 0.15}  # H3 thresholds (CONTEXT.md §6.3)


def sample_errors(raw, M, K, X):
    """raw: uint16 BF16 [M, K] (array or memmap); X: float64 [K, n] (column 0 = fixed x).
    Returns per-x relative errors (int8, int4), the predictions sqrt(sum d^2 / sum w^2) and
    the anisotropy, all over the whole matrix."""
    n = X.shape[1]
    yy, e8, e4 = np.zeros(n), np.zeros(n), np.zeros(n)
    d8s = d4s = w2s = 0.0
    rowsum = np.zeros(K)
    for r0 in range(0, M, CHUNK):
        w = qz.bf16_to_f32(np.asarray(raw[r0:r0 + CHUNK]))
        w64 = w.astype(np.float64)
        y = w64 @ X
        q8, s8, _ = qz.quantize_int8(w)
        p4, s4, _ = qz.quantize_int4(w)
        dw8 = qz.dequant_int8(q8, s8).astype(np.float64) - w64
        dw4 = qz.dequant_int4(p4, s4).astype(np.float64) - w64
        d8, d4 = dw8 @ X, dw4 @ X
        yy += (y * y).sum(0)
        e8 += (d8 * d8).sum(0)
        e4 += (d4 * d4).sum(0)
        d8s += float((dw8 * dw8).sum())
        d4s += float((dw4 * dw4).sum())
        w2s += float((w64 * w64).sum())
        rowsum += w64.sum(0)
    aniso = float((rowsum / M) @ (rowsum / M) / (w2s / M))
    return np.sqrt(e8 / yy), np.sqrt(e4 / yy), (d8s / w2s) ** 0.5, (d4s / w2s) ** 0.5, aniso


def summarize(r8, r4, p8, p4, aniso, n):
    row = {"int8_fixed_x": r8[0], "int8_mean": r8[1:].mean(), "int8_p5": np.percentile(r8[1:], 5),
           "int8_p95": np.percentile(r8[1:], 95), "int4_fixed_x": r4[0], "int4_mean": r4[1:].mean(),
           "int4_p5": np.percentile(r4[1:], 5), "int4_p95": np.percentile(r4[1:], 95), "anisotropy": aniso, "n_x": n}
    return row, {"int8_max": r8[1:].max(), "int4_max": r4[1:].max(), "pred_int8": p8, "pred_int4": p4,
                 "n_int8_over_2pct": int((r8[1:] > THRESH["int8"]).sum()),
                 "n_int4_over_15pct": int((r4[1:] > THRESH["int4"]).sum())}


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(header + "\n")
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")


def benchmarked(a):
    """The eleven exported matrices (the original check, same x; errors now computed as
    (W_q - W) x instead of W_q x - W x, which agrees to the six digits the CSV keeps)."""
    man = json.load(open(os.path.join(a.data, "manifest.json")))["matrices"]
    rng = np.random.default_rng(7)
    rows = []
    for m in man:
        M, K = m["M"], m["K"]
        raw = np.fromfile(os.path.join(a.data, f"{m['id']}.bf16.bin"), dtype=np.uint16).reshape(M, K)
        X = rng.standard_normal((K, a.n))
        x0 = qz.bf16_to_f32(make_x.gaussian_x(K)).astype(np.float64)[:, None]
        r8, r4, p8, p4, aniso = sample_errors(raw, M, K, np.hstack([x0, X]))
        row, _ = summarize(r8, r4, p8, p4, aniso, a.n)
        row = {"matrix": m["id"], "M": M, "K": K, **row}
        rows.append(row)
        print(f"{m['id']:7s} int4 fixed x {r4[0]:.4f}  mean {row['int4_mean']:.4f} [{row['int4_p5']:.4f}, {row['int4_p95']:.4f}]  "
              f"int8 fixed {r8[0]:.5f} mean {row['int8_mean']:.5f}  anisotropy {aniso:.4f}", flush=True)
    write_csv(a.out, f"# scripts/x_sample_check.py: {a.n} fresh Gaussian x per matrix (seed 7) plus the fixed x of make_x", rows)


def all_layers(a):
    """Every large matrix of all 32 layers + the LM head, from the safetensors files."""
    import accuracy_sweep as sw
    import export_weights as ew
    snap = ew.find_snapshot(a.snapshot)
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    names = [n for n in sorted(wmap) if n.startswith(ew.P) and n.endswith(".weight")]
    headers, x0s = {}, {}
    rng = np.random.default_rng(7)
    rows, t0 = [], time.time()
    for name in names:
        fname = wmap[name]
        if fname not in headers:
            headers[fname] = sw.header(os.path.join(snap, fname))
        hdr, base = headers[fname]
        t = hdr[name]
        if len(t["shape"]) != 2 or t["dtype"] != "BF16" or t["shape"][0] < 1024:
            continue  # the same selection as accuracy_sweep.py
        M, K = t["shape"]
        if K not in x0s:
            x0s[K] = qz.bf16_to_f32(make_x.gaussian_x(K)).astype(np.float64)[:, None]
        mm = np.memmap(os.path.join(snap, fname), dtype=np.uint16, mode="r",
                       offset=base + t["data_offsets"][0], shape=(M, K))
        X = np.hstack([x0s[K], rng.standard_normal((K, a.n))])
        r8, r4, p8, p4, aniso = sample_errors(mm, M, K, X)
        row, extra = summarize(r8, r4, p8, p4, aniso, a.n)
        short = name.replace(ew.P, "").replace(".weight", "")
        lm = __import__("re").search(r"layers\.(\d+)\.", name)
        rows.append({"tensor": name, "short": short, "layer": lm.group(1) if lm else "", "type": ew.layer_type(name),
                     "M": M, "K": K, **row, **extra})
        print(f"{short:42s} int4 fixed {r4[0]:.4f} mean {row['int4_mean']:.4f} p95 {row['int4_p95']:.4f} "
              f"max {extra['int4_max']:.4f} (>15%: {extra['n_int4_over_15pct']})  int8 p95 {row['int8_p95']:.5f} "
              f"max {extra['int8_max']:.5f}  [{time.time() - t0:.0f} s]", flush=True)
    write_csv(a.out, f"# scripts/x_sample_check.py --all-layers: model={ew.REPO} revision={ew.REVISION} "
                     f"{a.n} fresh Gaussian x per matrix (seed 7) plus the fixed x of make_x; quantizer=scripts/quantize.py",
              rows)
    worst = max(rows, key=lambda r: r["int4_p95"])
    print(f"{len(rows)} matrices in {time.time() - t0:.0f} s; largest INT4 95th percentile {worst['int4_p95']:.4%} "
          f"({worst['short']}); x above 15%: {sum(r['n_int4_over_15pct'] for r in rows)} of {len(rows) * a.n}; "
          f"x above 2% (INT8): {sum(r['n_int8_over_2pct'] for r in rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--snapshot")
    ap.add_argument("--n", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.all_layers:
        a.n = a.n or 256
        a.out = a.out or "results/accuracy/x_samples_all_layers.csv"
        all_layers(a)
    else:
        a.n = a.n or 64
        a.out = a.out or "results/accuracy/x_samples.csv"
        benchmarked(a)


if __name__ == "__main__":
    main()
