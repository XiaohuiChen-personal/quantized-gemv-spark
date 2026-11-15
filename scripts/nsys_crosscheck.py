#!/usr/bin/env python3
"""Cross-check the benchmark's event timings against Nsight Systems kernel durations.

Input: the SQLite export of an `nsys profile --trace=cuda,nvtx` run of `bench`, and the
CSV that same run wrote. For every "iso <label>" NVTX range (one per bench row), the
kernels that ran inside it are split into the range's ISO launches (200 in final mode;
cuBLAS may run more than one kernel per call, so the kernels are grouped in equal
consecutive chunks), and each launch's GPU time is the sum of its kernels' durations. The median per-launch time is compared with the bench's isolated
median for the same row, and the kernel names are listed (this is how the report names
the kernels cuBLAS chose).

Usage: python3 scripts/nsys_crosscheck.py trace.sqlite bench_under_nsys.csv > crosscheck.md
"""
import csv
import sqlite3
import statistics
import sys
from collections import Counter


ISO = 200  # isolated launches per bench row in final mode (gemv.cuh kFinalProfile.iso)


def main():
    db, bench_csv = sys.argv[1], sys.argv[2]
    con = sqlite3.connect(db)
    strings = dict(con.execute("SELECT id, value FROM StringIds"))
    kern = [(s, e, strings.get(n, str(n))) for s, e, n in
            con.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    cols = {r[1] for r in con.execute("PRAGMA table_info(NVTX_EVENTS)")}
    text = "COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))" if "textId" in cols else "text"
    ranges = con.execute(f"SELECT start, end, {text} FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchall()
    with open(bench_csv) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    bench = {f"{r['matrix']} {r['fmt']} {r['kernel']} {r['config']} x{r['copies']}": r for r in rows}

    out = ["# Nsight Systems cross-check of the isolated event timings", "",
           "Per bench row: the bench's isolated median (events, under nsys) and the median",
           "per-launch GPU kernel time from the trace (sum of the kernels between spin kernels).",
           "", "| row | bench iso median (us) | nsys kernel median (us) | bench / nsys | kernels per launch | kernel names |",
           "| --- | --- | --- | --- | --- | --- |"]
    ratios = []
    i0 = 0
    for start, end, label in sorted(r for r in ranges if r[2] and r[2].startswith("iso ")):
        key = label[4:]
        while i0 < len(kern) and kern[i0][0] < start:
            i0 += 1
        durs, names = [], Counter()
        i = i0
        while i < len(kern) and kern[i][0] <= end:
            s, e, n = kern[i]
            if "spin_kernel" not in n:
                durs.append(e - s)
                names[n] += 1
            i += 1
        if not durs or key not in bench:
            continue
        if len(durs) % ISO:
            out.append(f"| {key} | | | | {len(durs)} kernels, not a multiple of {ISO} | |")
            continue
        per_launch = len(durs) // ISO
        launches = [durs[j:j + per_launch] for j in range(0, len(durs), per_launch)]
        med_ns = statistics.median(sum(l) for l in launches)
        per = statistics.median(len(l) for l in launches)
        b = float(bench[key]["iso_med_us"])
        ratios.append((b / (med_ns / 1e3), key))
        out.append(f"| {key} | {b:.2f} | {med_ns / 1e3:.2f} | {b / (med_ns / 1e3):.3f} | {per:g} | "
                   + ", ".join(f"`{n}`" for n, _ in names.most_common(3)) + " |")
    if ratios:
        rs = sorted(r for r, _ in ratios)
        out += ["", f"{len(rs)} rows; bench/nsys ratio: min {rs[0]:.3f}, median {statistics.median(rs):.3f}, "
                    f"max {rs[-1]:.3f} (events add the event-record overhead around the kernel)."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
