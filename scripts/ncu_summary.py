#!/usr/bin/env python3
"""Summarize an Nsight Compute run of `bench --mode ncu` (scripts/run_stage.sh ncu).

Input: the raw-page CSV export of the report
(`ncu --import r.ncu-rep --csv --page raw --print-units base`)
and the bench CSV written under the profiler (for the bytes each launch must read). Every
profiled kernel is matched to its (matrix, format, kernel, configuration) through the NVTX
range bench puts around the isolated launches ("iso <matrix> <fmt> <kernel> <config> x<R>").

DRAM traffic on the GB10: its memory is system memory (LPDDR5x), and Nsight Compute has no
dram__* counters on this chip. Every DRAM read the GPU makes fills an L2 sector from system
memory, so DRAM read bytes = lts__d_sectors_fill_sysmem.sum x 32 B (checked against the
sysmem-aperture L2 misses, which agree, and against the stored bytes of each matrix).

Per (matrix, fmt, kernel, config), the median over the profiled launches of each CUDA
kernel the call ran (cuBLAS/cuBLASLt may run a split-K reduction as a second kernel; bytes
and durations are summed over the kernels of one call):

  dram_read_bytes     DRAM -> L2 bytes          dram_over_stored  / bytes the launch must read
  l2_read_bytes       L1 -> L2 read requests     duration_us       gpu__time_duration
  dram_gbps           DRAM read bytes / duration (clocks not locked: --clock-control none)
  occupancy_pct       achieved occupancy (sm__warps_active, % of peak)
  sm_pct, mem_pct     SM and memory-pipeline throughput, % of peak (Speed Of Light)
  inst_per_kb         warp instructions executed per KB the launch must read
  long_scoreboard     share of warp stall reasons that are long scoreboard (waiting on
                      global/L2/DRAM loads); top_stall names the largest reason

Usage: python3 scripts/ncu_summary.py raw.csv bench_under_ncu.csv [--csv summary.csv] > summary.md
"""
import argparse
import csv
import re
import statistics as st
from collections import defaultdict

RANGE_COL = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
RANGE_RE = re.compile(r":iso (\S+) (\S+) (\S+) (\S+) x(\d+):")
STALL_RE = re.compile(r"^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$")
SECTOR = 32
ORDER = ["v0", "v1", "v2", "cublas", "cublaslt", "v3", "v4", "v4load", "read"]


def num(s):
    s = s.replace(",", "").strip()
    return float(s) if s not in ("", "n/a") else float("nan")


def short_kernel(name):
    name = re.sub(r"^void |^std::enable_if<!T7, void>::type ", "", name)
    return name.split("(")[0][:60]


TIME_TO_US = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3, "s": 1e6, "second": 1e6}


def load_raw(path):
    """The raw page, exported with --print-units base (one unit per column)."""
    rows = list(csv.reader(open(path)))
    h, units = rows[0], rows[1]
    for need in ("lts__d_sectors_fill_sysmem.sum", "lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum",
                 "lts__t_sectors_op_read.sum"):
        if units[h.index(need)] != "sector":
            raise SystemExit(f"{need}: expected unit 'sector', got {units[h.index(need)]!r}")
    tu = units[h.index("gpu__time_duration.sum")]
    if tu not in TIME_TO_US:
        raise SystemExit(f"gpu__time_duration.sum: unknown unit {tu!r}")
    to_us = TIME_TO_US[tu]
    stall_cols = [(c, STALL_RE.match(c).group(1)) for c in h if STALL_RE.match(c)]
    out = []
    for r in rows[2:]:
        d = dict(zip(h, r))
        m = RANGE_RE.search(d[RANGE_COL])
        if not m:
            continue
        stalls = {name: num(d[c]) for c, name in stall_cols}
        tot = sum(v for v in stalls.values() if v == v)
        out.append({
            "key": m.groups()[:4], "copies": int(m.group(5)), "cuda_kernel": short_kernel(d["Kernel Name"]),
            "fill": num(d["lts__d_sectors_fill_sysmem.sum"]) * SECTOR,
            "miss": num(d["lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum"]) * SECTOR,
            "l2_read": num(d["lts__t_sectors_op_read.sum"]) * SECTOR,
            "dur": num(d["gpu__time_duration.sum"]) * to_us,
            "occ": num(d["sm__warps_active.avg.pct_of_peak_sustained_active"]),
            "sm": num(d["sm__throughput.avg.pct_of_peak_sustained_elapsed"]),
            "mem": num(d["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]),
            "inst": num(d["smsp__inst_executed.sum"]),
            "regs": num(d["launch__registers_per_thread"]), "grid": d["launch__grid_size"], "block": d["launch__block_size"],
            "lsb": stalls.get("long_scoreboard", float("nan")) / tot if tot else float("nan"),
            "top": max(stalls, key=lambda k: stalls[k] if stalls[k] == stalls[k] else -1) if stalls else "",
        })
    return out


def load_bench(path):
    rows = [r for r in csv.DictReader(l for l in open(path) if not l.startswith("#"))]
    return {(r["matrix"], r["fmt"], r["kernel"], r["config"]): r for r in rows}


def summarize(raw, bench):
    groups = defaultdict(lambda: defaultdict(list))
    for r in raw:
        groups[r["key"]][r["cuda_kernel"]].append(r)
    out = []
    for key, kernels in groups.items():
        b = bench.get(key)
        if b is None:
            raise SystemExit(f"no bench row for {key}")
        med = {k: {f: st.median(x[f] for x in v) for f in ("fill", "miss", "l2_read", "dur", "occ", "sm", "mem", "inst", "lsb", "regs")}
               for k, v in kernels.items()}
        main = max(med, key=lambda k: med[k]["dur"])  # the kernel that does the work
        stored = float(b["bytes_counted"])
        fill = sum(m["fill"] for m in med.values())
        dur = sum(m["dur"] for m in med.values())
        first = kernels[main][0]
        out.append({
            "matrix": key[0], "M": int(b["M"]), "K": int(b["K"]), "fmt": key[1], "kernel": key[2], "config": key[3],
            "cuda_kernels": " + ".join(sorted(med, key=lambda k: -med[k]["dur"])),
            "launches": min(len(v) for v in kernels.values()),
            "stored_bytes": stored, "dram_read_bytes": fill, "dram_over_stored": fill / stored,
            "miss_over_fill": sum(m["miss"] for m in med.values()) / fill,
            "l2_read_bytes": sum(m["l2_read"] for m in med.values()), "duration_us": dur,
            "dram_gbps": fill / (dur * 1e3), "stored_gbps": stored / (dur * 1e3),
            "event_iso_us": float(b["iso_med_us"]),
            "occupancy_pct": med[main]["occ"], "sm_pct": med[main]["sm"], "mem_pct": med[main]["mem"],
            "registers": med[main]["regs"], "grid": first["grid"], "block": first["block"],
            "inst_per_kb": sum(m["inst"] for m in med.values()) / (stored / 1e3),
            "long_scoreboard": med[main]["lsb"], "top_stall": first["top"],
        })
    out.sort(key=lambda r: (r["matrix"], ["bf16", "int8", "int4"].index(r["fmt"]),
                            ORDER.index(r["kernel"]) if r["kernel"] in ORDER else 99, r["config"]))
    return out


def md(rows):
    cols = [("matrix", "{}"), ("fmt", "{}"), ("kernel", "{}"), ("config", "{}"), ("stored_bytes", "{:.2e}"),
            ("dram_over_stored", "{:.3f}"), ("l2_read_bytes", "{:.2e}"), ("duration_us", "{:.1f}"), ("dram_gbps", "{:.1f}"),
            ("occupancy_pct", "{:.0f}"), ("sm_pct", "{:.0f}"), ("mem_pct", "{:.0f}"), ("registers", "{:.0f}"),
            ("inst_per_kb", "{:.1f}"), ("long_scoreboard", "{:.2f}"), ("top_stall", "{}")]
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + " --- |" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f.format(r[c]) for c, f in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("bench")
    ap.add_argument("--csv")
    a = ap.parse_args()
    raw = load_raw(a.raw)
    rows = summarize(raw, load_bench(a.bench))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    checked = [r for r in rows if r["kernel"] != "read"]
    print(f"# Nsight Compute summary\n\n{len(raw)} profiled kernel launches, {len(rows)} (matrix, kernel) rows. "
          f"DRAM read bytes = lts__d_sectors_fill_sysmem x 32 B; sysmem-aperture L2 misses / fills: "
          f"{min(r['miss_over_fill'] for r in rows):.4f}–{max(r['miss_over_fill'] for r in rows):.4f}. "
          f"DRAM read bytes / bytes the launch must read: {min(r['dram_over_stored'] for r in checked):.4f}–"
          f"{max(r['dram_over_stored'] for r in checked):.4f} over the GEMV kernels.\n")
    print(md(rows))


if __name__ == "__main__":
    main()
