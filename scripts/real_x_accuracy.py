#!/usr/bin/env python3
"""INT8/INT4 output error with real activations (CPU, NumPy; host Python).

Uses the input vectors recorded by scripts/capture_activations.py (data/real_x/) instead of
a random Gaussian x. For each matrix and each recorded token x_t, the relative error is
||(W_q - W) x_t|| / ||W x_t||, in double precision, with the project's quantizer
(scripts/quantize.py). Reported per matrix: the pooled error sqrt(sum_t ||dW x_t||^2 /
sum_t ||W x_t||^2), the median, 95th percentile and maximum over tokens, the count above
H3's thresholds, the same without the first token of each window (position 0 carries the
model's "attention sink" activations), and two predictions from the weights:

  pred    sqrt(sum delta^2 / sum w^2)                      (expected value for Gaussian x)
  wpred   sqrt(sum_k m_k sum_n delta_nk^2 / sum_k m_k sum_n w_nk^2),  m_k = mean_t x_tk^2
          (the same expectation for an x whose channels have the measured mean squares
           m_k and are uncorrelated: how far channel outliers alone move the error)

Activation statistics per site: peak |x| / RMS(x) and the share of ||x||^2 in the 8 largest
channels (medians over tokens).

  --stride S        tokens used for the all-layer sweep (every S-th, default 4)
  --gpu-dirs DIR    also write one bench data directory per (benchmarked matrix, chosen
                    token) -- the token with the median INT4 error (positions >= 1) and the
                    one with the largest -- so the GPU kernels can be run on real x
  --collect-gpu DIR summarize the bench CSVs written for those directories

Outputs (results/accuracy/real_x/): benchmarked.csv, all_layers.csv, per_token.csv.gz,
gpu_choice.json; with --collect-gpu: gpu_check.csv.

Usage: python3 scripts/real_x_accuracy.py [--gpu-dirs data/real_x_gpu]
       python3 scripts/real_x_accuracy.py --collect-gpu results/accuracy/real_x
"""
import argparse
import csv
import gzip
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accuracy_sweep as sw  # noqa: E402
import export_weights as ew  # noqa: E402
import quantize as qz  # noqa: E402

CHUNK = 8192
THRESH = {"int8": 0.02, "int4": 0.15}
SITE = {"gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down_in", "in_proj_qkv": "attn_in",
        "in_proj_z": "attn_in", "q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in",
        "out_proj": "attn_out_in", "o_proj": "attn_out_in"}


def site_of(tensor):
    if "embed_tokens" in tensor:
        return -1, "lm_in"
    m = re.search(r"layers\.(\d+)\.\w+\.(\w+)\.weight$", tensor)
    return int(m.group(1)), SITE[m.group(2)]


class RealX:
    def __init__(self, d):
        self.d = d
        self.man = json.load(open(os.path.join(d, "manifest.json")))
        self.sites = {(e["layer"], e["site"]): e for e in self.man["sites"]}
        self.L = self.man["window_len"]

    def load(self, layer, site):
        e = self.sites[(layer, site)]
        return np.fromfile(os.path.join(self.d, e["file"]), np.uint16).reshape(e["tokens"], e["K"])


def errors(raw, M, K, xb):
    """raw: BF16 [M, K]; xb: BF16 [T, K]. Per-token relative errors and the predictions."""
    X = qz.bf16_to_f32(xb).astype(np.float64).T  # [K, T]
    T = X.shape[1]
    yy, e8, e4 = np.zeros(T), np.zeros(T), np.zeros(T)
    cd8, cd4, cw = np.zeros(K), np.zeros(K), np.zeros(K)
    for r0 in range(0, M, CHUNK):
        w = qz.bf16_to_f32(np.asarray(raw[r0:r0 + CHUNK]))
        w64 = w.astype(np.float64)
        q8, s8, _ = qz.quantize_int8(w)
        p4, s4, _ = qz.quantize_int4(w)
        dw8 = qz.dequant_int8(q8, s8).astype(np.float64) - w64
        dw4 = qz.dequant_int4(p4, s4).astype(np.float64) - w64
        y, d8, d4 = w64 @ X, dw8 @ X, dw4 @ X
        yy += (y * y).sum(0)
        e8 += (d8 * d8).sum(0)
        e4 += (d4 * d4).sum(0)
        cd8 += (dw8 * dw8).sum(0)
        cd4 += (dw4 * dw4).sum(0)
        cw += (w64 * w64).sum(0)
    m = (X * X).mean(1)
    return {"yy": yy, "e8": e8, "e4": e4, "r8": np.sqrt(e8 / yy), "r4": np.sqrt(e4 / yy),
            "pred_int8": np.sqrt(cd8.sum() / cw.sum()), "pred_int4": np.sqrt(cd4.sum() / cw.sum()),
            "wpred_int8": np.sqrt((m * cd8).sum() / (m * cw).sum()), "wpred_int4": np.sqrt((m * cd4).sum() / (m * cw).sum())}


def xstats(xb):
    x = qz.bf16_to_f32(xb).astype(np.float64)
    e = x * x
    rms = np.sqrt(e.mean(1))
    top8 = np.sort(e, axis=1)[:, -8:].sum(1) / e.sum(1)
    return np.abs(x).max(1) / rms, top8


def summarize(r, pos):
    out = {}
    keep = pos > 0
    for f, rr, ee in (("int8", r["r8"], r["e8"]), ("int4", r["r4"], r["e4"])):
        out[f"{f}_pooled"] = float(np.sqrt(ee.sum() / r["yy"].sum()))
        out[f"{f}_median"] = float(np.median(rr))
        out[f"{f}_p95"] = float(np.percentile(rr, 95))
        out[f"{f}_max"] = float(rr.max())
        out[f"{f}_pooled_pos1"] = float(np.sqrt(ee[keep].sum() / r["yy"][keep].sum()))
        out[f"{f}_max_pos1"] = float(rr[keep].max())
        out[f"n_{f}_over"] = int((rr > THRESH[f]).sum())
    for k in ("pred_int8", "pred_int4", "wpred_int8", "wpred_int4"):
        out[k] = float(r[k])
    return out


def write_csv(path, rows, comment):
    with open(path, "w", newline="") as f:
        f.write(f"# {comment}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def gaussian_x_errors():
    rows = [r for r in csv.DictReader(l for l in open("results/accuracy/all_layers.csv") if not l.startswith("#"))]
    return {r["tensor"]: (float(r["rel_int8"]), float(r["rel_int4"])) for r in rows}


def run(a):
    rx = RealX(a.real)
    T = rx.man["windows"][-1][1] - rx.man["windows"][0][0]
    pos = np.arange(T) % rx.L
    gx = gaussian_x_errors()
    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    xs_cache = {}

    def site_x(layer, site):
        if (layer, site) not in xs_cache:
            xs_cache.clear()  # keep one site in memory
            xs_cache[(layer, site)] = rx.load(layer, site)
        return xs_cache[(layer, site)]

    # 1. The eleven benchmarked matrices, every recorded token.
    man = json.load(open(os.path.join(a.data, "manifest.json")))["matrices"]
    bench_rows, per_tok, choice = [], [], {}
    for m in man:
        layer, site = site_of(m["tensor"])
        xb = site_x(layer, site)
        raw = np.fromfile(os.path.join(a.data, f"{m['id']}.bf16.bin"), np.uint16).reshape(m["M"], m["K"])
        r = errors(raw, m["M"], m["K"], xb)
        peak, top8 = xstats(xb)
        g8, g4 = gx[m["tensor"]]
        row = {"matrix": m["id"], "tensor": m["tensor"], "layer": layer, "site": site, "M": m["M"], "K": m["K"],
               "tokens": T, **summarize(r, pos), "gauss_int8": g8, "gauss_int4": g4,
               "x_peak_over_rms_median": float(np.median(peak)), "x_peak_over_rms_max": float(peak.max()),
               "x_top8_energy_median": float(np.median(top8))}
        bench_rows.append(row)
        for t in range(T):
            per_tok.append((m["id"], t // rx.L, int(pos[t]), r["r8"][t], r["r4"][t], peak[t], top8[t]))
        cand = np.where(pos > 0)[0]
        med = cand[np.argsort(r["r4"][cand])[len(cand) // 2]]
        worst = int(np.argmax(r["r4"]))
        choice[m["id"]] = {"median": {"token": int(med), "int8": float(r["r8"][med]), "int4": float(r["r4"][med])},
                           "worst": {"token": worst, "int8": float(r["r8"][worst]), "int4": float(r["r4"][worst])}}
        print(f"{m['id']:7s} {site:11s} int4 pooled {row['int4_pooled']:.4f} median {row['int4_median']:.4f} p95 "
              f"{row['int4_p95']:.4f} max {row['int4_max']:.4f} (gauss {g4:.4f}, pred {row['pred_int4']:.4f}, wpred "
              f"{row['wpred_int4']:.4f}) | int8 pooled {row['int8_pooled']:.5f} max {row['int8_max']:.5f} | x peak/rms "
              f"{row['x_peak_over_rms_median']:.1f} (max {row['x_peak_over_rms_max']:.1f}) [{time.time() - t0:.0f} s]", flush=True)
    comment = (f"scripts/real_x_accuracy.py: real inputs from {rx.man['text']['repo']}@{rx.man['text']['revision']} "
               f"(test split), {len(rx.man['windows'])} windows x {rx.L} tokens, model {rx.man['model']}")
    write_csv(os.path.join(out_dir, "benchmarked.csv"), bench_rows, comment)
    with gzip.open(os.path.join(out_dir, "per_token.csv.gz"), "wt") as f:
        f.write(f"# {comment}\nmatrix,window,pos,rel_int8,rel_int4,x_peak_over_rms,x_top8_energy\n")
        for row in per_tok:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]:.6g},{row[4]:.6g},{row[5]:.5g},{row[6]:.5g}\n")
    json.dump(choice, open(os.path.join(out_dir, "gpu_choice.json"), "w"), indent=1)

    # 2. Every large matrix of all 32 layers, every stride-th token.
    snap = ew.find_snapshot(None)
    wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    headers, all_rows, targets = {}, [], []
    for name in sorted(wmap):
        if not (name.startswith(ew.P) and name.endswith(".weight")):
            continue
        fname = wmap[name]
        if fname not in headers:
            headers[fname] = sw.header(os.path.join(snap, fname))
        t = headers[fname][0][name]
        if len(t["shape"]) == 2 and t["dtype"] == "BF16" and t["shape"][0] >= 1024:  # accuracy_sweep's selection
            targets.append(name)
    sel = np.arange(0, T, a.stride)
    for name in sorted(targets, key=site_of):  # matrices sharing an input in a row
        hdr, base = headers[wmap[name]]
        fname = wmap[name]
        t = hdr[name]
        M, K = t["shape"]
        layer, site = site_of(name)
        xb = site_x(layer, site)[sel]
        mm = np.memmap(os.path.join(snap, fname), dtype=np.uint16, mode="r", offset=base + t["data_offsets"][0], shape=(M, K))
        r = errors(mm, M, K, xb)
        g8, g4 = gx[name]
        row = {"tensor": name, "short": name.replace(ew.P, "").replace(".weight", ""), "layer": layer, "site": site,
               "type": ew.layer_type(name), "M": M, "K": K, "tokens": len(sel), **summarize(r, pos[sel]),
               "gauss_int8": g8, "gauss_int4": g4}
        all_rows.append(row)
        print(f"{row['short']:42s} int4 pooled {row['int4_pooled']:.4f} p95 {row['int4_p95']:.4f} max {row['int4_max']:.4f} "
              f"(>15%: {row['n_int4_over']}) gauss {g4:.4f} | int8 pooled {row['int8_pooled']:.5f} max {row['int8_max']:.5f} "
              f"[{time.time() - t0:.0f} s]", flush=True)
    write_csv(os.path.join(out_dir, "all_layers.csv"), all_rows, comment + f"; every {a.stride}th token")
    w4 = max(all_rows, key=lambda r: r["int4_pooled"])
    print(f"{len(all_rows)} matrices; worst pooled INT4 {w4['int4_pooled']:.4%} ({w4['short']}); tokens above 15%: "
          f"{sum(r['n_int4_over'] for r in all_rows)} of {len(all_rows) * len(sel)}; above 2% (INT8): "
          f"{sum(r['n_int8_over'] for r in all_rows)} [{time.time() - t0:.0f} s]")

    # 3. Bench data directories for the GPU kernels on real x.
    if a.gpu_dirs:
        for m in man:
            layer, site = site_of(m["tensor"])
            xb = rx.load(layer, site)
            for which in ("median", "worst"):
                d = os.path.join(a.gpu_dirs, f"{m['id']}_{which}")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "manifest.tsv"), "w") as f:
                    f.write(f"{m['id']}\t{m['tensor']}\t{m['M']}\t{m['K']}\treal_x\n")
                for suffix in ("bf16", "int8", "int8.scale", "int4", "int4.scale"):
                    link = os.path.join(d, f"{m['id']}.{suffix}.bin")
                    if not os.path.lexists(link):
                        os.symlink(os.path.abspath(os.path.join(a.data, f"{m['id']}.{suffix}.bin")), link)
                rad = os.path.join(d, f"xrad_k{m['K']}.bin")
                if not os.path.lexists(rad):
                    os.symlink(os.path.abspath(os.path.join(a.data, f"xrad_k{m['K']}.bin")), rad)
                xb[choice[m["id"]][which]["token"]].tofile(os.path.join(d, f"x_k{m['K']}.bin"))


def collect_gpu(a):
    choice = json.load(open(os.path.join(a.collect_gpu, "gpu_choice.json")))
    rows = []
    for mid, ch in choice.items():
        for which in ("median", "worst"):
            path = os.path.join(a.collect_gpu, "gpu", f"{mid}_{which}.csv")
            for r in csv.DictReader(l for l in open(path) if not l.startswith("#")):
                fmt = r["fmt"]
                numpy_err = ch[which][fmt] if fmt in ("int8", "int4") else 0.0
                rows.append({"matrix": mid, "token_choice": which, "token": ch[which]["token"], "fmt": fmt,
                             "kernel": r["kernel"], "check_pass": r["check_pass"], "check_rad_pass": r["check_rad_pass"],
                             "recheck_pass": r["recheck_pass"], "bitwise_same": r["bitwise_same"],
                             "check_worst_ratio": float(r["check_worst_ratio"]), "gpu_rel_err": float(r["acc_rel_vs_bf16"]),
                             "numpy_rel_err": numpy_err})
    write_csv(os.path.join(a.collect_gpu, "gpu_check.csv"), rows,
              "bench --mode quick on real x (data dirs from real_x_accuracy.py --gpu-dirs); numpy_rel_err from the CPU")
    q = [r for r in rows if r["fmt"] != "bf16"]
    dev = max(abs(r["gpu_rel_err"] / r["numpy_rel_err"] - 1) for r in q)
    fails = [r for r in rows if "FAIL" in (r["check_pass"], r["check_rad_pass"], r["recheck_pass"]) or r["bitwise_same"] != "1"]
    print(f"{len(rows)} GPU rows; checks failed: {len(fails)}; GPU vs NumPy relative error of v3/v4: max deviation {dev:.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="data/real_x")
    ap.add_argument("--data", default="data")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", default="results/accuracy/real_x")
    ap.add_argument("--gpu-dirs")
    ap.add_argument("--collect-gpu")
    a = ap.parse_args()
    if a.collect_gpu:
        collect_gpu(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
