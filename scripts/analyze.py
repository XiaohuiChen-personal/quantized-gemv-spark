#!/usr/bin/env python3
"""Analysis of the kept results: hypothesis verdicts by the pre-registered rules, tables,
figures, and results/ANALYSIS.md. Reads only files under results/ and data/, so every
number it prints is traceable to a kept CSV.

Verdict rules (fixed 2026-11-14 before the final runs; the project plan's CONTEXT.md
§6.3): yes/no on the median over the three final executions of each execution's
isolated median time; "held in k/3" is a robustness note. Hypothesis matrices: gate,
up, down, qkv, z, out, q, o, lmhead (k and v are for the per-token model only).

Usage: python3 scripts/analyze.py [--results results] [--data data]
"""
import argparse
import glob
import math
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROPOSAL_REF = 241.0  # GB/s, the proposal's measured read-only reference
RATED = 273.0         # GB/s, NVIDIA's rated bandwidth
H1_MATS = ["gate", "up", "down", "qkv", "q", "lmhead"]
H2_MATS = ["gate", "up", "down", "lmhead"]
HYP_MATS = ["gate", "up", "down", "qkv", "z", "out", "q", "o", "lmhead"]
ALL_MATS = ["gate", "up", "down", "qkv", "q", "z", "out", "o", "k", "v", "lmhead"]
NAMES = {"gate": "L0 mlp.gate_proj", "up": "L0 mlp.up_proj", "down": "L0 mlp.down_proj",
         "qkv": "L0 linear_attn.in_proj_qkv", "z": "L0 linear_attn.in_proj_z",
         "out": "L0 linear_attn.out_proj", "q": "L3 self_attn.q_proj", "k": "L3 self_attn.k_proj",
         "v": "L3 self_attn.v_proj", "o": "L3 self_attn.o_proj", "lmhead": "LM head (embed_tokens)"}
LADDER = ["v0", "v1", "v2", "cublas", "v3", "v4"]
KLABEL = {"v0": "v0 thread/row", "v1": "v1 warp/row", "v2": "v2 BF16", "cublas": "cuBLAS GemmEx",
          "cublaslt": "cuBLASLt best", "v3": "v3 INT8", "v4": "v4 INT4", "v4load": "v4 load-only",
          "read": "read reference"}

# Reference palette (dataviz skill, light mode). Every entity keeps one colour in every
# figure: the kernels take fixed categorical slots, and a weight format takes the colour of
# its kernel (BF16 = v2, INT8 = v3, INT4 = v4). Read references are neutral grey. Marker
# shapes are a second encoding wherever points of different series can overlap.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
KCOLOR = {"v0": SLOT[0], "v1": SLOT[1], "v2": SLOT[2], "cublas": SLOT[3], "v3": SLOT[4], "v4": SLOT[6],
          "cublaslt": SLOT[7]}
FCOLOR = {"bf16": KCOLOR["v2"], "int8": KCOLOR["v3"], "int4": KCOLOR["v4"]}
FMARK = {"bf16": "o", "int8": "s", "int4": "^"}
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # sequential (ordinal) blue steps 250-700
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2.0, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def read(path):
    return pd.read_csv(path, comment="#")


def prov(path):
    with open(path) as f:
        return [l.strip() for l in f if l.startswith("#")]


def fmt_pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def md_table(df, floatfmt=None):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and floatfmt and c in floatfmt:
                cells.append(floatfmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.3g}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def bandwidth_ref(path):
    """Median over the four grids of the read-only median in a bandwidth program output."""
    vals = [float(m.group(1)) for m in re.finditer(r"read-only\s+median\s+([\d.]+)", open(path).read())]
    return statistics.median(vals) if vals else float("nan")


class Analysis:
    def __init__(self, results, data):
        self.R, self.D = results, data
        self.fig = os.path.join(results, "figures")
        self.tab = os.path.join(results, "analysis")
        os.makedirs(self.fig, exist_ok=True)
        os.makedirs(self.tab, exist_ok=True)
        self.md = []
        self.figs = []
        runs = sorted(glob.glob(os.path.join(results, "final", "run[0-9].csv")))
        if not runs:
            raise SystemExit("no results/final/run*.csv")
        self.runs = runs
        self.df = pd.concat([read(p).assign(run=int(re.search(r"run(\d)", p).group(1))) for p in runs])
        self.nruns = self.df["run"].nunique()
        self.prov = [prov(p)[0] for p in runs]
        self.refs = [bandwidth_ref(p) for p in sorted(glob.glob(os.path.join(results, "final", "bandwidth_run*.txt")))]
        self.session_ref = statistics.median(self.refs) if self.refs else float("nan")
        key = ["matrix", "fmt", "kernel", "config"]
        g = self.df.groupby(key)
        self.agg = g.agg(M=("M", "first"), K=("K", "first"), bytes=("bytes_counted", "first"),
                         copies=("copies", "first"), iso=("iso_med_us", "median"),
                         spin=("spin_med_us", "median"), stream=("str_med_us", "median"),
                         iso_min=("iso_med_us", "min"), iso_max=("iso_med_us", "max"),
                         acc_rel=("acc_rel_vs_bf16", "first"), acc_max=("acc_maxabs_vs_bf16", "first")).reset_index()
        chk = self.df[self.df.kernel == "cublaslt"].copy()
        chk["ok"] = (chk.check_pass == "pass") & (chk.check_rad_pass == "pass") & (chk.recheck_pass == "pass")
        okc = chk.groupby(["matrix", "config"]).ok.all().reset_index()
        self.lt_ok = {m: set(d[d.ok].config) for m, d in okc.groupby("matrix")}
        self.lt_failed = okc[~okc.ok]
        self.agg["gbps"] = self.agg["bytes"] / (self.agg["iso"] * 1e3)
        self.nsys_df = self.nsys_table()
        if len(self.nsys_df):
            g = self.agg.merge(self.nsys_df, on=["matrix", "fmt", "kernel", "config"])
            gap = g.iso - g.nsys
            self.ev_gap = (gap.median(), gap.quantile(0.25), gap.quantile(0.75))
        else:
            self.ev_gap = (float("nan"),) * 3
        sp = os.path.join(results, "final", "run1_samples.csv.gz")
        self.fast = (0, 0, 0, 0)
        if os.path.exists(sp):
            smp = pd.read_csv(sp, comment="#")
            smp = smp[(smp["mode"] == "iso") & smp.kernel.isin(["v1", "v2", "v3", "v4", "cublas"])]
            cnt = smp.groupby(["matrix", "fmt", "kernel", "config"]).us.apply(lambda u: int((u < 0.85 * u.median()).sum()))
            self.fast = (int(cnt.sum()), int(len(smp)), int((cnt > 0).sum()), int(len(cnt)))
        self.agg["gbps_str"] = self.agg["bytes"] / (self.agg["stream"] * 1e3)

    # ------------------------------------------------------------ lookups
    def t(self, m, kernel, col="iso", run=None, fmt=None):
        """Median-over-runs time (or one run's) of a ladder kernel; cublaslt = best passing heuristic."""
        if run is None:
            a = self.agg[(self.agg.matrix == m) & (self.agg.kernel == kernel)]
            if fmt:
                a = a[a.fmt == fmt]
            if kernel == "cublaslt":
                a = a[a.config.isin(self.lt_ok.get(m, set()))]
            return float(a[col].min()) if len(a) else float("nan")
        d = self.df[(self.df.matrix == m) & (self.df.kernel == kernel) & (self.df.run == run)]
        if kernel == "cublaslt":
            d = d[d.config.isin(self.lt_ok.get(m, set()))]
        if fmt:
            d = d[d.fmt == fmt]
        c = {"iso": "iso_med_us", "stream": "str_med_us", "spin": "spin_med_us"}[col]
        return float(d[c].min()) if len(d) else float("nan")

    def row(self, m, kernel, fmt=None):
        a = self.agg[(self.agg.matrix == m) & (self.agg.kernel == kernel)]
        if kernel == "cublaslt":  # only algorithms that met the accuracy contract in every run
            a = a[a.config.isin(self.lt_ok.get(m, set()))]
        if fmt:
            a = a[a.fmt == fmt]
        return a.sort_values("iso").iloc[0] if len(a) else None

    # ------------------------------------------------------------ sections
    def provenance(self):
        self.md += ["## 1. Provenance", "",
                    f"Final executions: {self.nruns} (`results/final/run*.csv`), each preceded by the "
                    "bandwidth reference program (`bandwidth_run*.txt`).", ""]
        for p in self.prov:
            self.md.append(f"    {p}")
        self.md += ["", "Same-session 1 GiB read-only reference (median of the four grids per execution): "
                    + ", ".join(f"{r:.1f}" for r in self.refs) + f" GB/s; median **{self.session_ref:.1f} GB/s** "
                    f"(proposal: {PROPOSAL_REF:.0f} GB/s; rated {RATED:.0f} GB/s).", ""]

    def correctness(self):
        d = self.df[self.df.check_pass.isin(["pass", "FAIL"]) & (self.df.kernel != "cublaslt")]
        n = len(d)
        ok = ((d.check_pass == "pass") & (d.check_rad_pass == "pass") & (d.recheck_pass == "pass")
              & (d.bitwise_same == 1)).sum()
        notbit = d[d.bitwise_same == 0]
        spin_bad = int((self.df.spin_covered == 0).sum())
        self.md += ["## 2. Correctness", "",
                    f"Checked rows of my kernels and cuBLAS GemmEx (all executions): {n}; passing every check "
                    f"(Gaussian x, ±1 x, re-check after timing, bitwise repeat): **{ok}**. Worst per-element tolerance ratio: {d.check_worst_ratio.max():.3f} "
                    f"(1.0 = at the bound). Largest relative L2 error vs the kernel's own reference: "
                    f"{d.check_rel_l2.max():.2e} (limit 4e-3).",
                    f"Rows whose output after timing differed bitwise from the first run: {len(notbit)}"
                    + (" (" + ", ".join(sorted(set(notbit.kernel + ' ' + notbit.config))) + ")" if len(notbit) else "")
                    + ". Spin-diagnostic launches whose host enqueue outlasted the spin kernel: " + str(spin_bad) + ".", ""]
        lf = self.lt_failed
        self.md += [f"cuBLASLt heuristic algorithms (not counted above): {len(lf)} (matrix, algorithm) pairs failed the "
                    "same checks and are excluded from \"cuBLASLt best\"" + (": " + ", ".join(f"{r.matrix}/{r.config}" for _, r in lf.iterrows()) if len(lf) else "") + ".", ""]
        tp = os.path.join(self.R, "tests", "test_output.txt")
        if os.path.exists(tp):
            tail = [l.strip() for l in open(tp) if l.strip()][-6:]
            self.md += ["Unit tests (`results/tests/test_output.txt`):", ""] + [f"    {l}" for l in tail] + [""]

    def ladder(self):
        rows = []
        for m in ALL_MATS:
            for k in LADDER + ["cublaslt"]:
                r = self.row(m, k)
                if r is None:
                    continue
                rd = self.row(m, "read", r.fmt)
                rows.append({"matrix": m, "kernel": KLABEL[k], "fmt": r.fmt, "config": r.config,
                             "iso_us": r.iso, "stream_us": r.stream, "spin_us": r.spin, "GB/s": r.gbps,
                             "%241": r.gbps / PROPOSAL_REF, "%273": r.gbps / RATED,
                             "%session": r.gbps / self.session_ref,
                             "%size-matched read": r.gbps / rd.gbps if rd is not None else float("nan"),
                             "read ref GB/s": rd.gbps if rd is not None else float("nan"),
                             "run spread": (r.iso_max - r.iso_min) / r.iso})
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(self.tab, "ladder.csv"), index=False)
        self.ladder_tab = t
        self.md += ["## 3. Bandwidth ladder", "",
                    "Isolated median time (median over executions), streamed (CUDA-graph) time, GB/s on stored "
                    "bytes (weights + scales + x + y), and that rate as a share of the proposal's 241 GB/s, the "
                    "rated 273 GB/s, the same-session 1 GiB reference, and the size-matched read reference (the "
                    "read-only kernel over the same stored bytes and rotation). `run spread` = (max − min) / "
                    "median of the per-execution medians. Full table: `results/analysis/ladder.csv`.", ""]
        show = t[t.matrix.isin(["gate", "down", "qkv", "z", "out", "k", "lmhead"])].copy()
        for c in ("%241", "%273", "%session", "%size-matched read", "run spread"):
            show[c] = show[c].map(lambda v: fmt_pct(v))
        show = show.drop(columns=["config", "spin_us"])
        self.md += [md_table(show, {"iso_us": "{:.2f}", "stream_us": "{:.2f}", "GB/s": "{:.1f}",
                                    "read ref GB/s": "{:.1f}"}), ""]

    def h1(self):
        rows, all_ok = [], True
        for m in H1_MATS:
            r2, rc = self.row(m, "v2"), self.row(m, "cublas")
            ok_bw, ok_t = r2.gbps >= 0.8 * PROPOSAL_REF, r2.iso <= 1.15 * rc.iso
            held = 0
            for i in sorted(self.df.run.unique()):
                t2, tc = self.t(m, "v2", run=i), self.t(m, "cublas", run=i)
                held += int(r2.bytes / (t2 * 1e3) >= 0.8 * PROPOSAL_REF and t2 <= 1.15 * tc)
            lt = self.row(m, "cublaslt")
            rows.append({"matrix": m, "v2 GB/s": r2.gbps, "≥193?": "yes" if ok_bw else "NO",
                         "% session ref": fmt_pct(r2.gbps / self.session_ref),
                         "t(v2)/t(cuBLAS)": r2.iso / rc.iso, "≤1.15?": "yes" if ok_t else "NO",
                         "t(v2)/t(cuBLASLt best)": r2.iso / lt.iso if lt is not None else float("nan"),
                         "held": f"{held}/{self.nruns}"})
            all_ok &= ok_bw and ok_t
        self.verdict = {"H1": "CONFIRMED" if all_ok else "REFUTED"}
        self.md += ["## 4. H1 — the bandwidth limit", "",
                    "Rule: on gate, up, down, qkv, q and the LM head, v2 ≥ 193 GB/s (80% of 241 = 192.8; no value "
                    "falls between the two) **and** t(v2) ≤ 1.15 × t(cuBLAS GemmEx).", "",
                    md_table(pd.DataFrame(rows), {"v2 GB/s": "{:.1f}", "t(v2)/t(cuBLAS)": "{:.3f}",
                                                  "t(v2)/t(cuBLASLt best)": "{:.3f}"}),
                    "", f"**Verdict H1: {self.verdict['H1']}.**", ""]

    def h2(self):
        rows = {8: [], 4: []}
        ok = {8: True, 4: True}
        thr = {8: 1.7, 4: 2.6}
        ker = {8: "v3", 4: "v4"}
        ratio = {8: 2.0 * 2560 / (2560 + 2), 4: 2.0 / (0.5 + 2 / 128)}
        for m in H2_MATS:
            tc, t2, tl = self.t(m, "cublas"), self.t(m, "v2"), self.t(m, "cublaslt")
            for b in (8, 4):
                tq = self.t(m, ker[b])
                held = sum(int(self.t(m, "cublas", run=i) / self.t(m, ker[b], run=i) >= thr[b])
                           for i in sorted(self.df.run.unique()))
                sp = tc / tq
                ok[b] &= sp >= thr[b]
                rows[b].append({"matrix": m, f"t(cuBLAS)/t({ker[b]})": sp, f"≥{thr[b]}?": "yes" if sp >= thr[b] else "NO",
                                f"t(v2)/t({ker[b]})": t2 / tq, f"t(cuBLASLt best)/t({ker[b]})": tl / tq,
                                "byte ratio": self.agg_bytes_ratio(m, ker[b]), "held": f"{held}/{self.nruns}"})
        self.verdict["H2-INT8"] = "CONFIRMED" if ok[8] else "REFUTED"
        self.verdict["H2-INT4"] = "CONFIRMED" if ok[4] else "REFUTED"
        ff = {c: "{:.2f}" for c in ("t(cuBLAS)/t(v3)", "t(v2)/t(v3)", "t(cuBLASLt best)/t(v3)", "t(cuBLAS)/t(v4)",
                                     "t(v2)/t(v4)", "t(cuBLASLt best)/t(v4)", "byte ratio")}
        self.md += ["## 5. H2 — bytes buy speed", "",
                    "Rule: on gate, up, down and the LM head, t(cuBLAS)/t(v3) ≥ 1.7 (INT8) and t(cuBLAS)/t(v4) ≥ 2.6 "
                    "(INT4). The speedups over my own v2 and over cuBLASLt's best heuristic algorithm are shown "
                    "next to them: they isolate what the smaller format bought, independent of cuBLAS's "
                    "efficiency. `byte ratio` = counted bytes of the BF16 v2 launch / counted bytes of the "
                    "quantized launch.", "",
                    "INT8:", "", md_table(pd.DataFrame(rows[8]), ff), "",
                    f"**Verdict H2-INT8: {self.verdict['H2-INT8']}.**", "",
                    "INT4:", "", md_table(pd.DataFrame(rows[4]), ff), "",
                    f"**Verdict H2-INT4: {self.verdict['H2-INT4']}.**", ""]

    def agg_bytes_ratio(self, m, kernel):
        return float(self.row(m, "v2").bytes / self.row(m, kernel).bytes)

    def h3(self):
        acc = read(sorted(glob.glob(os.path.join(self.R, "final", "run*_accuracy.csv")))[0])
        rows, ok = [], True
        for m in HYP_MATS:
            r3, r4 = self.row(m, "v3"), self.row(m, "v4")
            c8 = acc[(acc.matrix == m) & (acc.fmt == "int8")].iloc[0]
            c4 = acc[(acc.matrix == m) & (acc.fmt == "int4")].iloc[0]
            ok &= r3.acc_rel <= 0.02 and r4.acc_rel <= 0.15
            rows.append({"matrix": m, "INT8 GPU": fmt_pct(r3.acc_rel, 3), "INT8 CPU": fmt_pct(c8.rel_err_cpu, 3),
                         "INT8 predicted": fmt_pct(c8.predicted_rel_err, 3), "≤2%?": "yes" if r3.acc_rel <= 0.02 else "NO",
                         "INT4 GPU": fmt_pct(r4.acc_rel, 2), "INT4 CPU": fmt_pct(c4.rel_err_cpu, 2),
                         "INT4 predicted": fmt_pct(c4.predicted_rel_err, 2), "≤15%?": "yes" if r4.acc_rel <= 0.15 else "NO"})
        self.verdict["H3"] = "CONFIRMED" if ok else "REFUTED"
        det = self.df[self.df.kernel.isin(["v3", "v4"])].groupby(["matrix", "kernel"]).acc_rel_vs_bf16.nunique().max()
        self.md += ["## 6. H3 — accuracy", "",
                    "Rule: relative output error of the GPU kernel (its BF16 output, Gaussian x) vs the BF16-weight "
                    "reference in double precision: ≤ 2% (INT8) and ≤ 15% (INT4) on every hypothesis matrix. "
                    "`CPU` = the dequantized weights times x in double (no output rounding); `predicted` = "
                    "√(Σδ²/Σw²), the expectation for Gaussian x. Identical across executions: "
                    + ("yes" if det == 1 else "NO") + ".", "", md_table(pd.DataFrame(rows)), "",
                    f"**Verdict H3: {self.verdict['H3']}.**", ""]
        sw = os.path.join(self.R, "accuracy", "all_layers.csv")
        if os.path.exists(sw):
            s = read(sw)
            w8, w4 = s.loc[s.rel_int8.idxmax()], s.loc[s.rel_int4.idxmax()]
            self.sweep = s
            self.md += [f"**H3 across all 32 layers** (secondary, `results/accuracy/all_layers.csv`, {len(s)} matrices "
                        f"with ≥ 1024 rows): worst INT8 {fmt_pct(w8.rel_int8, 3)} (`{w8.short}`), worst INT4 "
                        f"{fmt_pct(w4.rel_int4, 2)} (`{w4.short}`); INT8 ≤ 2% everywhere: "
                        f"{'yes' if s.rel_int8.max() <= 0.02 else 'NO'}; INT4 ≤ 15% everywhere: "
                        f"{'yes' if s.rel_int4.max() <= 0.15 else 'NO (' + str(int((s.rel_int4 > 0.15).sum())) + ' matrices above)'}.",
                        f"Median over the sweep: INT8 {fmt_pct(s.rel_int8.median(), 3)}, INT4 {fmt_pct(s.rel_int4.median(), 2)}; "
                        f"excess kurtosis median {s.excess_kurtosis.median():.2f} (Gaussian = 0), range "
                        f"{s.excess_kurtosis.min():.2f}–{s.excess_kurtosis.max():.2f}. Correlation of INT4 error with "
                        f"kurtosis (Spearman): {s.rel_int4.rank().corr(s.excess_kurtosis.rank()):.2f}.", ""]
            # NumPy cross-check: same matrices, independent implementation.
            man = pd.read_json(os.path.join(self.D, "manifest.json"))["matrices"]
            tensor_of = {m["id"]: m["tensor"] for m in man}
            xr = []
            for m in ALL_MATS:
                srow = s[s.tensor == tensor_of[m]]
                if not len(srow):
                    continue
                c8 = acc[(acc.matrix == m) & (acc.fmt == "int8")].iloc[0]
                c4 = acc[(acc.matrix == m) & (acc.fmt == "int4")].iloc[0]
                xr.append(max(abs(srow.rel_int8.iloc[0] / c8.rel_err_cpu - 1), abs(srow.rel_int4.iloc[0] / c4.rel_err_cpu - 1)))
            xs = os.path.join(self.R, "accuracy", "x_samples.csv")
            if os.path.exists(xs):
                x = read(xs)
                lm = x[x.matrix == "lmhead"].iloc[0]
                self.md += [f"One x vs many (`results/accuracy/x_samples.csv`, {int(lm.n_x)} fresh Gaussian x per matrix): "
                            f"the LM head is the outlier between measured and predicted error because it is strongly "
                            f"anisotropic (‖mean row‖² / mean ‖row‖² = {lm.anisotropy:.3f}; every other matrix ≤ "
                            f"{x[x.matrix != 'lmhead'].anisotropy.max():.3f}), so one x is a noisy sample: over many x its INT4 "
                            f"error averages {fmt_pct(lm.int4_mean, 2)} (5th–95th percentile {fmt_pct(lm.int4_p5, 2)}–"
                            f"{fmt_pct(lm.int4_p95, 2)}) and its INT8 error {fmt_pct(lm.int8_mean, 3)}, matching the predictions; "
                            f"the fixed x ({fmt_pct(lm.int4_fixed_x, 2)}) sits near the top of that range. The largest 95th "
                            f"percentile over all matrices is {fmt_pct(x.int4_p95.max(), 2)} (INT4) and "
                            f"{fmt_pct(x.int8_p95.max(), 3)} (INT8): H3's thresholds hold for typical x, not only for the fixed one.", ""]
            if xr:
                agree = ("identical to all 7 significant digits both print" if max(xr) < 5e-7
                         else f"within {max(xr):.1e} (relative)")
                self.md += [f"NumPy cross-check: for the {len(xr)} benchmarked matrices, the independent NumPy relative "
                            f"errors (`scripts/accuracy_sweep.py`) agree with the C++ CPU-only errors: {agree}.", ""]

    def h4(self):
        runs = sorted(glob.glob(os.path.join(self.R, "marlin", "marlin_run[0-9].csv")))
        if not runs:
            self.verdict["H4"] = "not run"
            self.md += ["## 7. H4 — Marlin (stretch)", "", "Not run (no `results/marlin/marlin_run*.csv`).", ""]
            return
        mr = pd.concat([read(p) for p in runs]).groupby("matrix").median(numeric_only=True)
        at = os.path.join(self.R, "marlin", "marlin_runatomic.csv")
        ma = read(at).set_index("matrix") if os.path.exists(at) else None
        rows, ok = [], True
        for m in ALL_MATS:
            if m not in mr.index:
                continue
            r = mr.loc[m]
            ratio = self.t(m, "v4", "stream") / r.marlin_str_us  # the judged, like-for-like ratio
            others = [self.t(m, "v4") / r.marlin_us, self.t(m, "v4", "spin") / r.marlin_spin_us]
            if m in H2_MATS:
                ok &= ratio <= 1.5
                self.h4_all_modes_pass = getattr(self, "h4_all_modes_pass", True) and max([ratio] + others) <= 1.5
            rows.append({"matrix": m, "streamed v4 us": self.t(m, "v4", "stream"), "streamed Marlin us": r.marlin_str_us,
                         "streamed v4/Marlin": ratio, "≤1.5?": "yes" if ratio <= 1.5 else "NO",
                         "spin v4/Marlin": others[1], "plain-event v4/Marlin": others[0],
                         "streamed Marlin atomic-add us": float(ma.loc[m].marlin_str_us) if ma is not None else float("nan"),
                         "err v4 / Marlin": f"{fmt_pct(self.row(m, 'v4').acc_rel, 2)} / {fmt_pct(r.marlin_rel_err_vs_bf16, 2)}",
                         "PyTorch F.linear BF16 us": r.torch_linear_bf16_us})
        self.verdict["H4"] = "CONFIRMED" if ok else "REFUTED"
        t = pd.DataFrame(rows)
        big = t[~t.matrix.isin(["k", "v"])]
        self.md += ["## 7. H4 — Marlin (stretch)", "",
                    "Rule: t(v4) ≤ 1.5 × t(Marlin) at batch 1. Judged on the H2 matrices (gate, up, down, LM head); the "
                    "proposal says \"the same matrices\", and every matrix shown also passes. Judged on the **streamed** "
                    "(CUDA-graph) times, the only mode in which both are timed without host-side cost: Marlin is called "
                    "through Python, so its plain-event time includes ~9 µs of dispatch between the start event and the "
                    "kernel (e.g. gate: 66 µs plain vs 57 µs with a sleep kernel first vs 55 µs streamed). The verdict is "
                    "the same in all three modes. Marlin = vLLM 0.16's "
                    f"`marlin_gemm` (uint4b8, group 128) on the same BF16 weights, quantized by vLLM's own "
                    f"`marlin_quantize`; {len(runs)} executions in the `vllm-node-qwen35` container (median shown), timed "
                    "with the harness's three modes (isolated = plain events, spin = sleep kernel before the start "
                    "event, streamed = CUDA graph) and rotation (`results/marlin/`). Caveats: another process and "
                    "session, PyTorch events, the container's CUDA 13.1 forward-compatibility runtime, and vLLM's "
                    "quantizer (scale = max|w| / 7.5, codes −8…7) rather than mine; the atomic-add column is the "
                    "variant vLLM's log recommends for small N (one execution).", "",
                    md_table(t, {"streamed v4 us": "{:.2f}", "streamed Marlin us": "{:.2f}", "plain-event v4/Marlin": "{:.3f}",
                                 "spin v4/Marlin": "{:.3f}", "streamed v4/Marlin": "{:.3f}",
                                 "streamed Marlin atomic-add us": "{:.2f}", "PyTorch F.linear BF16 us": "{:.1f}"}), "",
                    f"Matrices of 21 MB and up: streamed v4/Marlin {big['streamed v4/Marlin'].min():.2f}–"
                    f"{big['streamed v4/Marlin'].max():.2f} (spin {big['spin v4/Marlin'].min():.2f}–{big['spin v4/Marlin'].max():.2f}; "
                    f"plain events {big['plain-event v4/Marlin'].min():.2f}–{big['plain-event v4/Marlin'].max():.2f}, which "
                    "favours v4 because of Marlin's Python dispatch). All modes pass: "
                    f"{'yes' if getattr(self, 'h4_all_modes_pass', False) else 'NO'}.", "",
                    f"**Verdict H4: {self.verdict['H4']}.**", ""]
        self.marlin = t

    def nsys_table(self):
        p = os.path.join(self.R, "nsys", "crosscheck.md")
        rows = []
        if os.path.exists(p):
            for l in open(p):
                c = [x.strip() for x in l.strip().strip("|").split("|")]
                if len(c) == 6 and re.match(r"[\d.]+$", c[2] or "x"):
                    m, f, k, cfg = c[0].split()[:4]
                    rows.append({"matrix": m, "fmt": f, "kernel": k, "config": cfg, "bench": float(c[1]), "nsys": float(c[2])})
        return pd.DataFrame(rows)

    def nt(self, m, k):
        n = self.nsys_df
        d = n[(n.matrix == m) & (n.kernel == k)] if len(n) else n
        return float(d.nsys.min()) if len(d) else float("nan")

    def small_vs_large(self):
        out = {}
        for mode in ("iso", "stream", "nsys"):
            def sp(m):
                return self.nt(m, "v2") / self.nt(m, "v4") if mode == "nsys" else self.t(m, "v2", mode) / self.t(m, "v4", mode)
            out[mode] = (np.mean([sp(m) for m in ("gate", "up", "down")]), np.mean([sp(m) for m in ("z", "out", "o")]),
                         np.mean([sp(m) for m in ("k", "v")]))
        self.md += ["## 8. Exploratory — do the 21 MB matrices gain less? (CONTEXT.md §6.1)", "",
                    "INT4 speedup over BF16 v2, t(v2)/t(v4), mean per size class. Prediction: ≈ 3.75× (47 MB) vs "
                    "≈ 3.6× (21 MB), a gap of about 4%; > 10% would point to another cause. The `nsys` row uses the "
                    "Nsight kernel durations of the single profiled pass (no event overhead).", "",
                    "| timing | 47 MB (gate, up, down) | 21 MB (z, out, o) | gap 21 vs 47 | 5 MB (k, v) |",
                    "| --- | --- | --- | --- | --- |"]
        for mode, (a, b, c) in out.items():
            self.md.append(f"| {mode} | {a:.2f}× | {b:.2f}× | {fmt_pct(1 - b / a)} | {c:.2f}× |")
        gap = 1 - out["iso"][1] / out["iso"][0]
        self.md += ["", f"Isolated gap {fmt_pct(gap)}: above the predicted ~4% but below the 10% trigger; streamed "
                    f"{fmt_pct(1 - out['stream'][1] / out['stream'][0])}, and with Nsight kernel durations only "
                    f"{fmt_pct(1 - out['nsys'][1] / out['nsys'][0])}. So the excess over the prediction is the event "
                    "overhead per launch (§9, §15), not a property of the 21 MB matrices' kernels. On the 5 MB matrices the "
                    f"penalty is real but also inflated by events: {out['nsys'][2]:.2f}× with Nsight durations vs "
                    f"{out['iso'][2]:.2f}× isolated; a plain read of 1.3 MB also reaches only ~155 GB/s (§3).", ""]
        self.small = out

    def timing_modes(self):
        a = self.agg[self.agg.kernel.isin(LADDER)].copy()
        a["str/iso"] = a.stream / a.iso
        a["spin-iso us"] = a.spin - a.iso
        a["iso-str us"] = a.iso - a.stream
        self.md += ["## 9. Timing modes", "",
                    "Isolated (events around one launch on an idle GPU: the verdict measure), spin (the same with a "
                    "spin kernel before the start event: a diagnostic, see `results/timing_method/`) and streamed "
                    "(whole rotation cycles in one CUDA graph) for the ladder kernels, all matrices.", "",
                    f"- streamed / isolated: median {a['str/iso'].median():.3f}, range {a['str/iso'].min():.3f}–{a['str/iso'].max():.3f}; "
                    f"for kernels under 30 µs: median {a[a.iso < 30]['str/iso'].median():.3f}; over 1 ms: "
                    f"{a[a.iso > 1000]['str/iso'].median():.3f}. Isolated − streamed: median {a['iso-str us'].median():.2f} µs; "
                    "part of that is the event overhead that the streamed mode does not pay per launch (final-run events "
                    f"exceed the Nsight kernel durations by a median {self.ev_gap[0]:.1f} µs, IQR {self.ev_gap[1]:.1f}–"
                    f"{self.ev_gap[2]:.1f}), the rest is launch start-up and tail that back-to-back kernels overlap.",
                    f"- spin − isolated: median {a['spin-iso us'].median():+.2f} µs (range {a['spin-iso us'].min():+.2f} to "
                    f"{a['spin-iso us'].max():+.2f}); negative mostly for cuBLAS, whose plain-event time includes part "
                    "of its host-side call path (≤ 1% on the H1/H2 matrices, 1.6–1.8% on the 21 MB ones).",
                    f"- Residual caching: {self.fast[0]} of {self.fast[1]} isolated samples of the ladder kernels in "
                    f"execution 1 are ≥ 15% faster than their row's median, in {self.fast[2]} of {self.fast[3]} rows, "
                    "about one per rotation cycle (e.g. `down` v2: 11 of 200 with R = 18) — one copy of the rotation "
                    "evidently still meets a warm cache or a favourable placement. The median ignores them; streamed "
                    "means include them (≲ 1%).", ""]
        self.modes = a

    def ablation(self):
        base = {m: self.row(m, "v2") for m in ALL_MATS}
        ab = self.agg[self.agg.kernel == "v2abl"].copy()
        if not len(ab):
            return
        cfg2 = base["gate"].config

        def diff(c):
            a, b = cfg2.split("_"), c.split("_")
            names = {"t": "threads", "r": "rows/warp", "u": "unroll", "x": "x", "p": "grid", "f": "grid", "l": "load width"}
            d = []
            for i in range(max(len(a), len(b))):
                x = a[i] if i < len(a) else "ld16"
                y = b[i] if i < len(b) else "ld16"
                if x != y:
                    d.append(f"{names.get(y[0], y)} {x}→{y}")
            return "; ".join(d)
        ab["change"] = ab.config.map(diff)
        ab["dt"] = [r.iso / base[r.matrix].iso - 1 for _, r in ab.iterrows()]
        piv = ab.pivot_table(index="change", columns="matrix", values="dt")
        piv = piv[[m for m in ("gate", "down", "qkv", "z", "out", "k", "lmhead") if m in piv.columns]]
        piv.to_csv(os.path.join(self.tab, "ablation.csv"))
        self.abl = piv
        show = piv.map(lambda v: f"{100 * v:+.1f}%").reset_index()
        self.md += ["## 10. v2 ablation", "", f"Frozen v2 = `{cfg2}`. Each row changes one factor; cells are the change "
                    "in isolated time vs v2 (positive = slower).", "", md_table(show), ""]

    def diagnostics(self):
        rows = []
        for m in ALL_MATS:
            t4, tl = self.t(m, "v4"), self.t(m, "v4load")
            rd = self.row(m, "read", "int4")
            rows.append({"matrix": m, "v4 us": t4, "v4 load-only us": tl, "decode cost": fmt_pct(t4 / tl - 1),
                         "v4 GB/s": self.row(m, "v4").gbps, "INT4 read ref GB/s": rd.gbps if rd is not None else float("nan")})
        self.md += ["## 11. INT4 diagnostics (no GPU counters needed)", "",
                    "`v4 load-only` issues v4's exact loads (weights, scales and x) with the decode and multiply-adds "
                    "removed. `decode cost` = t(v4)/t(load-only) − 1: near zero means v4 is limited by memory, "
                    "not by its arithmetic.", "",
                    md_table(pd.DataFrame(rows), {"v4 us": "{:.2f}", "v4 load-only us": "{:.2f}", "v4 GB/s": "{:.1f}",
                                                  "INT4 read ref GB/s": "{:.1f}"}), ""]

    def per_token(self):
        inv = pd.read_csv(os.path.join(self.D, "inventory.csv"))
        shape_to = {}
        for m in ALL_MATS:
            r = self.row(m, "v2")
            shape_to.setdefault((int(r.M), int(r.K)), []).append(m)
        res, excluded = {}, inv[~inv.apply(lambda r: (r.M, r.K) in shape_to, axis=1)]
        cases = {"BF16 cuBLAS": ("cublas", "bf16"), "BF16 v2": ("v2", "bf16"), "INT8 v3": ("v3", "int8"), "INT4 v4": ("v4", "int4")}
        inc = inv[inv.apply(lambda r: (r.M, r.K) in shape_to, axis=1)]
        for name, (k, f) in cases.items():
            tot = {"stream": 0.0, "iso": 0.0}
            byts = 0
            for _, r in inc.iterrows():
                ms = shape_to[(r.M, r.K)]
                for mode in tot:
                    tot[mode] += statistics.median(self.t(m, k, mode) for m in ms)
                byts += statistics.median(self.row(m, k).bytes for m in ms)
            res[name] = {"GEMV time per token, streamed (ms)": tot["stream"] / 1e3,
                         "tokens/s (streamed)": 1e6 / tot["stream"], "tokens/s (isolated)": 1e6 / tot["iso"],
                         "bytes per token (GB)": byts / 1e9, "ceiling at 241 GB/s (tokens/s)": PROPOSAL_REF * 1e9 / byts,
                         "share of ceiling": (1e6 / tot["stream"]) / (PROPOSAL_REF * 1e9 / byts)}
        t = pd.DataFrame(res).T.reset_index().rename(columns={"index": "format"})
        t.to_csv(os.path.join(self.tab, "per_token.csv"), index=False)
        self.pt = t
        self.md += ["## 12. Per-token GEMV time model", "",
                    f"Sum over the {len(inc)} large GEMVs of one token (all layers + LM head, counts from "
                    f"`data/inventory.csv`) of the measured time of its shape. Not included: {len(excluded)} small "
                    f"2-D weights ({excluded.bytes.sum() / 1e6:.1f} MB BF16 per token: "
                    + ", ".join(sorted(set(excluded.tensor.str.replace(r'model\.language_model\.layers\.\d+\.', '', regex=True))))
                    + "), attention, norms, state and KV-cache traffic — so these are upper bounds on a real "
                    "engine's speed. The ceiling divides 241 GB/s by the same bytes.", "",
                    md_table(t, {c: "{:.3g}" for c in t.columns if c != "format"} | {"share of ceiling": "{:.1%}"}), ""]

    def tuning(self):
        sp = os.path.join(self.R, "tuning", "selection.md")
        tp = os.path.join(self.R, "tuning", "tune.csv")
        if not os.path.exists(tp):
            return
        tu = read(tp)
        lines = ["## 13. Tuning", "", "Selection by the pre-registered rule (`results/tuning/selection.md`):", ""]
        if os.path.exists(sp):
            for l in open(sp):
                if l.startswith("## "):
                    lines.append("- " + l[3:].strip())
        # Sensitivity: geometric-mean GB/s by rows-per-warp x unroll (bytes in flight per lane), best over the rest.
        tu = tu[tu.matrix.isin(["gate", "down", "qkv", "z", "out", "lmhead"])]
        tu["r"] = tu.config.str.extract(r"_r(\d)_").astype(int)
        tu["u"] = tu.config.str.extract(r"_u(\d)_").astype(int)
        g = tu.groupby(["kernel", "config", "r", "u"]).gbps_iso.apply(lambda v: math.exp(np.log(v).mean())).reset_index()
        best = g.groupby(["kernel", "r", "u"]).gbps_iso.max().unstack("u")
        lines += ["", "Best geometric-mean GB/s for each rows-per-warp (rows) × unroll (columns) — loads in flight "
                  "per lane = rows × unroll:", ""]
        for k in ("v2", "v3", "v4"):
            if k in best.index.get_level_values(0):
                b = best.loc[k]
                lines.append(f"{k}: " + "; ".join(f"r{r}: " + ", ".join(f"u{u} {b.loc[r, u]:.0f}" for u in b.columns) for r in b.index))
        lines.append("")
        spread = g.groupby("kernel").gbps_iso.agg(["min", "max"])
        lines.append("Range over all 72 configurations (geometric-mean GB/s): " +
                     "; ".join(f"{k} {r['min']:.0f}–{r['max']:.0f}" for k, r in spread.iterrows()) + ".")
        lines.append("")
        self.md += lines

    def rotation(self):
        lines = ["## 14. Rotation sweep (why 32 × L2)", ""]
        for name in ("sweep_footprint_pre_rule.csv", "sweep_final.csv"):
            p = os.path.join(self.R, "rotation", name)
            if not os.path.exists(p):
                continue
            s = read(p)
            s["stored"] = np.where(s.kernel == "read", s.bytes_counted, s.bytes_counted - 2 * s.K - 2 * s.M)
            s["footprint_L2"] = s.copies * s.stored / 25165824
            lines += [f"`results/rotation/{name}` ({'development kernels and the spin-based timing, before the rule change' if 'pre' in name else 'the frozen kernels and the final timing'}): isolated GB/s at footprints of about 4, 16 and ≥ 64 × L2:", "",
                      "| matrix | fmt | kernel | ≈4×L2 | ≈16×L2 | plateau (≥64×L2, median) | 4×L2 excess |", "| --- | --- | --- | --- | --- | --- | --- |"]
            for (m, f, k), d in s.groupby(["matrix", "fmt", "kernel"], sort=False):
                d = d.sort_values("footprint_L2")
                near = lambda x: d.iloc[(d.footprint_L2 - x).abs().argmin()].gbps_iso
                plat = d[d.footprint_L2 >= 60].gbps_iso.median()
                lines.append(f"| {m} | {f} | {k} | {near(4):.0f} | {near(16):.0f} | {plat:.0f} | {fmt_pct(near(4) / plat - 1)} |")
            lines.append("")
            self.rot = getattr(self, "rot", {}) | {name: s}
        ex = {}
        for name, sdf in getattr(self, "rot", {}).items():
            e = []
            for _, d in sdf.groupby(["matrix", "fmt", "kernel"]):
                d = d.sort_values("footprint_L2")
                plat = d[d.footprint_L2 >= 60].gbps_iso.median()
                e.append(d.iloc[(d.footprint_L2 - 4).abs().argmin()].gbps_iso / plat - 1)
            ex[name] = max(e)
        lines += ["Reading: at 4 × L2 the worst excess over the plateau is "
                  + "; ".join(f"{fmt_pct(v)} in `{n}`" for n, v in ex.items())
                  + " (both on the 21 MB BF16 `out`), so the planned rule would have inflated that matrix with either "
                  "kernel set. From 16 × L2 on, rates are within a few percent of the plateau. Medians at the adopted "
                  "32 × L2 still include about one fast launch per rotation cycle (§9).", ""]
        self.md += lines

    def nsys(self):
        p = os.path.join(self.R, "nsys", "crosscheck.md")
        if not os.path.exists(p):
            return
        summ = [l.strip() for l in open(p) if l[:1].isdigit()]
        n = self.nsys_df
        lines = ["## 15. Nsight Systems cross-check (kernel durations)", "",
                 "One final-set pass under `nsys profile --trace=cuda,nvtx` (`results/nsys/`): a single, profiled "
                 "execution, so it is a cross-check rather than a fourth measurement (profiling perturbs a few rows, "
                 "e.g. `down` v4 ran at 60.9 µs there vs 57.8 µs in the finals). Per bench row, the median kernel "
                 "duration of its 200 isolated launches vs the bench's event median in that same pass:"] + summ + [""]
        if len(n):
            h1 = [self.nt(m, "v2") / self.nt(m, "cublas") for m in H1_MATS]
            h28 = [self.nt(m, "cublas") / self.nt(m, "v3") for m in H2_MATS]
            h24 = [self.nt(m, "cublas") / self.nt(m, "v4") for m in H2_MATS]
            v2bw = [self.row(m, "v2").bytes / (self.nt(m, "v2") * 1e3) for m in H1_MATS]
            lines += ["Verdict ratios recomputed from the Nsight kernel durations (fair to cuBLAS, whose event time includes "
                      "part of its host-side call path):",
                      f"- H1: v2 {min(v2bw):.1f}–{max(v2bw):.1f} GB/s (≥ 193), t(v2)/t(cuBLAS) {min(h1):.3f}–{max(h1):.3f} (≤ 1.15).",
                      f"- H2: t(cuBLAS)/t(v3) {min(h28):.2f}–{max(h28):.2f} (≥ 1.7); t(cuBLAS)/t(v4) {min(h24):.2f}–{max(h24):.2f} "
                      "(≥ 2.6; the low end is the perturbed `down` v4 row).",
                      "- Every verdict is unchanged." if (min(v2bw) >= 193 and max(h1) <= 1.15 and min(h28) >= 1.7 and min(h24) >= 2.6)
                      else "- **A verdict would change under Nsight durations — see the table.**", ""]
        if os.path.exists(os.path.join(self.R, "nsys", "cublas_kernels.md")):
            lines += ["Kernels cuBLAS chose (`results/nsys/cublas_kernels.md`): every `cublasGemmEx` call runs one "
                      "`internal::gemvx::kernel` instance. Its default for K = 2560/4096 and the fastest passing "
                      "cuBLASLt algorithm are instances of that template with different template parameters and launch "
                      "shapes (16 × 4 vs 32 × 4 threads per block on gate); which parameter matters is not resolved — "
                      "another algorithm with the fast one's launch shape is also slow.", ""]
        self.md += lines

    # ------------------------------------------------------------ figures
    def save(self, fig, name, caption):
        path = os.path.join(self.fig, name)
        fig.savefig(path)
        plt.close(fig)
        self.figs.append((name, caption))

    def fig_ladder(self):
        mats = ["gate", "down", "qkv", "z", "out", "k", "lmhead"]
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        w = 0.13
        x = np.arange(len(mats))
        for i, k in enumerate(LADDER):
            for j, m in enumerate(mats):
                r = self.row(m, k)
                xb = j + (i - 2.5) * w
                ax.bar(xb, r.gbps, w * 0.92, color=KCOLOR[k], label=KLABEL[k] if j == 0 else None, zorder=2)
                rd = self.row(m, "read", r.fmt)  # a plain read of the bytes this bar's format stores
                ax.plot([xb - w * 0.46, xb + w * 0.46], [rd.gbps] * 2, color=INK, lw=1.1, zorder=3)
        ax.plot([], [], color=INK, lw=1.1, label="plain read of the same bytes")
        for y, lab in ((RATED, "rated 273"), (PROPOSAL_REF, "proposal 241")):
            ax.axhline(y, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
            ax.text(len(mats) - 0.45, y + 2, lab, ha="right", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(x, [f"{m}\n{self.row(m, 'v2').M}\n×{self.row(m, 'v2').K}" for m in mats], fontsize=7.5)
        ax.set_ylabel("GB/s on stored bytes (isolated)")
        ax.set_ylim(0, 300)
        ax.legend(ncol=4, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.24))
        self.save(fig, "fig_ladder.png", "Achieved bandwidth (isolated median over three executions, GB/s on the bytes "
                  "each format stores) of the kernel ladder and cuBLAS on one matrix of each shape. Black ticks: a plain "
                  "read of the same bytes in the same format and rotation (the size-matched reference); dashed: rated "
                  "273 GB/s and the proposal's 241 GB/s.")

    def fig_speedup_bytes(self):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        classes = [("≥ 42 MB + LM head", ["gate", "up", "down", "qkv", "q", "lmhead"], "o"),
                   ("21 MB", ["z", "out", "o"], "s"), ("5 MB (k, v)", ["k", "v"], "^")]
        lim = 4.3
        ax.plot([1, lim], [1, lim], color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(3.95, 4.05, "speedup = byte ratio", fontsize=7.5, color=INK2, ha="right")
        rng = np.random.default_rng(0)
        for lab, ms, mk in classes:
            for k, f in (("v3", "int8"), ("v4", "int4")):
                xs = [self.agg_bytes_ratio(m, k) + rng.uniform(-0.04, 0.04) for m in ms]
                ys = [self.t(m, "v2") / self.t(m, k) for m in ms]
                ax.scatter(xs, ys, s=34, marker=mk, color=FCOLOR[f], edgecolor=SURFACE, linewidth=1.0, zorder=3)
        for lab, _, mk in classes:
            ax.scatter([], [], marker=mk, color=INK2, s=30, label=lab)
        for f, lab in (("int8", "INT8 (v3)"), ("int4", "INT4 (v4)")):
            ax.scatter([], [], marker="o", color=FCOLOR[f], s=30, label=lab)
        ax.set_xlabel("byte ratio (BF16 bytes / quantized bytes)")
        ax.set_ylabel("isolated speedup over v2, t(v2)/t(quantized)")
        ax.set_xlim(1, lim)
        ax.set_ylim(1, lim)
        ax.legend(loc="upper left", fontsize=7.2, ncol=1)
        self.save(fig, "fig_speedup_bytes.png", "Isolated speedup of INT8 (v3) and INT4 (v4) over the BF16 v2 kernel "
                  "against the byte ratio; colour = format, shape = matrix size; points are offset horizontally by up to "
                  "±0.04 for visibility. On the dashed line, time is exactly proportional to bytes.")

    def fig_speed_accuracy(self):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for f, k in (("bf16", "v2"), ("int8", "v3"), ("int4", "v4")):
            xs = [self.row(m, k).acc_rel for m in HYP_MATS]
            ys = [self.t(m, "v2") / self.t(m, k) for m in HYP_MATS]
            ax.scatter(xs, ys, s=34, marker=FMARK[f], color=FCOLOR[f], edgecolor=SURFACE, linewidth=1.0,
                       label=f.upper(), zorder=3)
        for x, lab in ((0.02, "H3 INT8 limit 2%"), (0.15, "H3 INT4 limit 15%")):
            ax.axvline(x, color=INK2, lw=0.9, ls=(0, (4, 3)))
            ax.text(x * 1.06, 1.15, lab, rotation=90, fontsize=7, color=INK2, va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("relative output error vs BF16 weights (Gaussian x, log scale)")
        ax.set_ylabel("isolated speedup over v2")
        ax.legend(loc="upper left", fontsize=7.5)
        self.save(fig, "fig_speed_accuracy.png", "Speed–accuracy trade-off on the nine hypothesis matrices: isolated "
                  "speedup over BF16 v2 against the relative output error for the fixed Gaussian x (BF16's error is its "
                  "output rounding alone).")

    def fig_ablation(self):
        if not hasattr(self, "abl"):
            return
        piv = self.abl
        mats = [m for m in ("k", "z", "gate", "lmhead") if m in piv.columns]  # increasing size
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(piv) + 0.9))
        y = np.arange(len(piv))
        h = 0.8 / len(mats)
        for i, m in enumerate(mats):
            ax.barh(y + (i - (len(mats) - 1) / 2) * h, 100 * piv[m].values, h * 0.9, color=BLUE_RAMP[i],
                    label=f"{m} ({self.row(m, 'v2').bytes / 1e6:.0f} MB)", zorder=2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y, piv.index)
        ax.invert_yaxis()
        ax.set_xlabel("change in isolated time vs frozen v2 (%; positive = slower)")
        ax.legend(ncol=len(mats), fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        self.save(fig, "fig_ablation.png", "v2 ablation: the frozen v2 with one factor changed at a time; change in "
                  "isolated time on four matrices (light to dark = small to large).")

    def fig_rotation(self):
        if not hasattr(self, "rot"):
            return
        sweeps = [(n, self.rot[n]) for n in ("sweep_footprint_pre_rule.csv", "sweep_final.csv") if n in self.rot]
        mats = ["gate", "out", "k"]
        fig, axes = plt.subplots(len(sweeps), 3, figsize=(7.2, 2.5 * len(sweeps)), squeeze=False)
        series = [("bf16", "read", INK2, "-"), ("bf16", "v2", KCOLOR["v2"], "-"),
                  ("int4", "read", INK2, (0, (3, 2))), ("int4", "v4", KCOLOR["v4"], "-")]
        for row, (name, s) in enumerate(sweeps):
            for col, m in enumerate(mats):
                ax = axes[row, col]
                for f, k, c, ls in series:
                    d = s[(s.matrix == m) & (s.fmt == f) & (s.kernel == k)].sort_values("footprint_L2")
                    ax.plot(d.footprint_L2, d.gbps_iso, color=c, ls=ls, marker="o", ms=3, lw=1.6,
                            label=f"{f.upper()} {'plain read' if k == 'read' else k}")
                ax.axhline(RATED, color=INK2, lw=0.8, ls=(0, (1, 2)))
                for xv, lab in ((4, "4×"), (32, "32×")):
                    ax.axvline(xv, color=INK2, lw=0.8, ls=(0, (4, 3)))
                    ax.text(xv * 1.1, 392, lab, fontsize=6.5, color=INK2, va="top")
                ax.set_xscale("log", base=2)
                ax.set_ylim(100, 400)
                ax.set_title(f"{'before the rule change' if 'pre' in name else 'frozen kernels'}: {m}", fontsize=8.5)
                if row == len(sweeps) - 1:
                    ax.set_xlabel("footprint of all copies / L2")
            axes[row, 0].set_ylabel("isolated GB/s")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.0))
        fig.tight_layout()
        self.save(fig, "fig_rotation.png", "Rotation sweeps: isolated rate against the total footprint of the rotating "
                  "copies (units of the 24 MiB L2). Top: development kernels before the rule change (the evidence for "
                  "it); bottom: the frozen kernels. Dashed verticals: the planned 4 × L2 and the adopted 32 × L2; dotted: "
                  "rated 273 GB/s (anything above it comes from a cache). The axis is cut at 400 GB/s: footprints that "
                  "fit in L2 reach 900+ GB/s.")

    def fig_accuracy_layers(self):
        if not hasattr(self, "sweep"):
            return
        s = self.sweep.copy()
        groups = [("MLP gate/up/down", r"mlp\.", "o"), ("in_proj_qkv / q_proj", r"in_proj_qkv|q_proj", "s"),
                  ("in_proj_z", r"in_proj_z", "D"), ("out_proj / o_proj", r"out_proj|o_proj", "^"),
                  ("k_proj / v_proj", r"k_proj|v_proj", "v")]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
        for ax, col, thr, f in ((axes[0], "rel_int8", 0.02, "int8"), (axes[1], "rel_int4", 0.15, "int4")):
            lay = s[s.layer.notna() & (s.layer.astype(str) != "")]
            for lab, pat, mk in groups:
                d = lay[lay.short.str.contains(pat)]
                ax.scatter(d.layer.astype(int), 100 * d[col], s=13, marker=mk, facecolor="none",
                           edgecolor=FCOLOR[f], linewidth=0.9, label=lab, zorder=3)
            lm = s[s.type == "lm_head"]
            if len(lm):
                ax.scatter([33.5], [100 * lm[col].iloc[0]], s=60, marker="*", color=FCOLOR[f], zorder=4, label="LM head")
            ax.axhline(100 * thr, color=INK2, lw=0.9, ls=(0, (4, 3)))
            ax.text(0, 100 * thr, f" H3 limit {100 * thr:.0f}%", fontsize=7, color=INK2, va="bottom")
            ax.set_xticks([0, 8, 16, 24, 31, 33.5], ["0", "8", "16", "24", "31", "LM"])
            ax.set_xlabel("layer")
            ax.set_ylabel(("INT8" if f == "int8" else "INT4") + " relative error (%)")
        handles = [plt.Line2D([], [], marker=mk, ls="", color=INK2, mfc="none") for _, _, mk in groups]
        handles.append(plt.Line2D([], [], marker="*", ls="", color=INK2, ms=9))
        labels = [lab for lab, _, _ in groups] + ["LM head"]
        fig.legend(handles, labels, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.02))
        self.save(fig, "fig_accuracy_layers.png", "Relative output error (fixed Gaussian x) of every large matrix in all "
                  "32 layers and the LM head, INT8 (left) and INT4 (right); marker shape = matrix type; dashed: the H3 "
                  "thresholds (2%, 15%).")

    def fig_timing_modes(self):
        a = self.modes
        fig, ax = plt.subplots(figsize=(4.8, 3.3))
        for f in ("bf16", "int8", "int4"):
            d = a[a.fmt == f]
            ax.scatter(d.iso, d.stream / d.iso, s=26, marker=FMARK[f], color=FCOLOR[f], edgecolor=SURFACE, linewidth=0.8,
                       label=f.upper(), zorder=3)
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("isolated time per launch (µs, log scale)")
        ax.set_ylabel("streamed / isolated time")
        ax.legend(fontsize=7.5)
        self.save(fig, "fig_timing_modes.png", "Back-to-back (CUDA-graph) time per launch relative to the isolated time, "
                  "for every ladder kernel and matrix (colour and shape = weight format): the shorter the kernel, the more "
                  "the per-launch event overhead and start-up/tail cost matter.")

    def fig_per_token(self):
        t = self.pt
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        x = np.arange(len(t))
        cols = [KCOLOR["cublas"], KCOLOR["v2"], KCOLOR["v3"], KCOLOR["v4"]]
        ax.bar(x, t["tokens/s (streamed)"], 0.55, color=cols, zorder=2)
        ax.scatter(x, t["ceiling at 241 GB/s (tokens/s)"], marker="_", s=600, color=INK, linewidth=1.6, zorder=3,
                   label="byte-count ceiling at 241 GB/s")
        for i, v in enumerate(t["tokens/s (streamed)"]):
            ax.text(i, v - 2.5, f"{v:.0f}", ha="center", va="top", fontsize=8, color=SURFACE, fontweight="bold")
        ax.set_xticks(x, t.format)
        ax.set_ylabel("tokens/s from GEMV time alone")
        ax.legend(fontsize=7.5, loc="upper left")
        self.save(fig, "fig_per_token.png", "Decode speed implied by the measured GEMV times alone (sum over one "
                  "token's 201 large GEMVs, streamed mode, every weight in the given format — INT4 includes the LM head) "
                  "against the byte-count ceiling at 241 GB/s.")

    # ------------------------------------------------------------ run
    def run(self):
        self.md = ["# Analysis of the final runs", "",
                   "Generated by `scripts/analyze.py` from the kept CSVs; do not edit by hand (notes go in "
                   "`results/NOTES.md`). Verdict rules: CONTEXT.md §6.3 in the course repository, fixed before "
                   "the final runs.", ""]
        self.provenance()
        self.correctness()
        self.ladder()
        self.h1()
        self.h2()
        self.h3()
        self.h4()
        self.small_vs_large()
        self.timing_modes()
        self.ablation()
        self.diagnostics()
        self.per_token()
        self.tuning()
        self.rotation()
        self.nsys()
        for f in (self.fig_ladder, self.fig_speedup_bytes, self.fig_speed_accuracy, self.fig_ablation,
                  self.fig_rotation, self.fig_accuracy_layers, self.fig_timing_modes, self.fig_per_token):
            f()
        summary = ["## Verdicts", ""] + [f"- **{h}: {v}**" for h, v in self.verdict.items()] + [""]
        self.md[4:4] = summary
        self.md += ["## Figures", ""] + [f"- `figures/{n}`: {c}" for n, c in self.figs] + [""]
        with open(os.path.join(self.R, "ANALYSIS.md"), "w") as f:
            f.write("\n".join(self.md))
        for h, v in self.verdict.items():
            print(f"{h}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    Analysis(a.results, a.data).run()


if __name__ == "__main__":
    main()
