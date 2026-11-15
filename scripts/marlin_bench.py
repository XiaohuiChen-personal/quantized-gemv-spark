#!/usr/bin/env python3
"""H4 (stretch): time vLLM's Marlin INT4 kernel at batch 1 on the benchmarked matrices.

Runs inside the `vllm-node-qwen35:latest` container (vLLM 0.16, PyTorch 2.10) with the
repository mounted at /work. Same rules as the C++ harness:
  * the same BF16 weights (data/<id>.bf16.bin) and the same Gaussian x (data/x_k<K>.bin);
  * INT4, symmetric (vLLM's uint4b8), group 128 along K, quantized by vLLM's own
    marlin_quantize (its round-to-nearest, scale = max|w| / 7.5 per group, codes -8..7);
  * rotating copies with a footprint >= 32 x L2;
  * isolated: a GPU sleep kernel before the start event, events around one call, median
    of 200; streamed: 200 calls captured in one CUDA graph, replayed 5 times.
The output is checked against Marlin's own dequantized weights (w_ref) times x in FP64.
PyTorch's BF16 F.linear (cuBLASLt via PyTorch) is timed the same way as a reference.

Usage (from the repository root on the Spark):
  docker run --rm --gpus all --ipc=host -v $PWD:/work -w /work vllm-node-qwen35:latest \
      python3 scripts/marlin_bench.py --out results/marlin/marlin.csv
"""
import argparse
import datetime
import json
import os
import statistics
import subprocess

import numpy as np
import torch

L2 = 25165824
FOOTPRINT = 32 * L2
MATS = ["gate", "up", "down", "qkv", "z", "out", "q", "k", "v", "o", "lmhead"]


def load_bf16(path, shape):
    u = np.fromfile(path, dtype=np.uint16).reshape(shape)
    return torch.from_numpy(u.view(np.int16).copy()).view(torch.bfloat16)


def time_op(fn_for_copy, ncopies, iso=200, graph_n=200, reps=5, warm_ms=200):
    """fn_for_copy(i) runs one call on copy i. Returns (isolated median us, streamed median us)."""
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    nxt = [1]

    def run():
        fn_for_copy(nxt[0] % ncopies)
        nxt[0] += 1
    t0 = datetime.datetime.now()
    i = 0
    while i < 20 or (datetime.datetime.now() - t0).total_seconds() * 1e3 < warm_ms:
        run()
        i += 1
        if i % 16 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iso):
        torch.cuda._sleep(400_000)  # ~130-200 us of spinning: covers the host enqueue
        s.record()
        run()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()  # warm the side stream before capture
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(g):
        for _ in range(graph_n):
            run()
    g.replay()
    torch.cuda.synchronize()
    per = []
    for _ in range(reps):
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        per.append(s.elapsed_time(e) * 1e3 / graph_n)
    return statistics.median(ts), statistics.median(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/marlin/marlin.csv")
    ap.add_argument("--only", default=",".join(MATS))
    a = ap.parse_args()
    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types

    dev = torch.device("cuda")
    man = {m["id"]: m for m in json.load(open(os.path.join(a.data, "manifest.json")))["matrices"]}
    try:  # the container may have no git; the host's commit is then recorded by run_stage.sh
        git = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
    except OSError:
        git = os.environ.get("GIT_DESC", "unknown")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    for mid in a.only.split(","):
        M, K = man[mid]["M"], man[mid]["K"]
        W = load_bf16(os.path.join(a.data, f"{mid}.bf16.bin"), (M, K)).to(dev)
        x = load_bf16(os.path.join(a.data, f"x_k{K}.bin"), (K,)).to(dev).reshape(1, K)
        # Marlin's layout is (size_k, size_n): the transpose of our row-major W.
        w_ref, q_w, s_w, g_idx, sort_idx, _ = marlin_quantize(W.t().contiguous(), scalar_types.uint4b8, 128, False)
        zp = torch.empty(0, dtype=torch.int, device=dev)
        ws = mu.marlin_make_workspace_new(dev)
        stored = q_w.numel() * q_w.element_size() + s_w.numel() * s_w.element_size()
        nc = max(2, -(-FOOTPRINT // stored))
        qs = [q_w] + [q_w.clone() for _ in range(nc - 1)]
        ss = [s_w] + [s_w.clone() for _ in range(nc - 1)]

        def marlin(i):
            return mu.apply_gptq_marlin_linear(x, qs[i], ss[i], zp, g_idx, sort_idx, ws, scalar_types.uint4b8,
                                               output_size_per_partition=M, input_size_per_partition=K, is_k_full=True)
        y = marlin(0).double().flatten()
        ref = (w_ref.double().t() @ x.double().flatten())
        y_bf16 = (W.double() @ x.double().flatten())
        rel_ref = float(torch.linalg.norm(y - ref) / torch.linalg.norm(ref))
        rel_bf16 = float(torch.linalg.norm(y - y_bf16) / torch.linalg.norm(y_bf16))
        t_iso, t_str = time_op(marlin, nc)
        del qs, ss

        nb = max(2, -(-FOOTPRINT // (M * K * 2)))
        Ws = [W] + [W.clone() for _ in range(nb - 1)]
        yb = torch.nn.functional.linear(x, W).double().flatten()
        rel_lin = float(torch.linalg.norm(yb - y_bf16) / torch.linalg.norm(y_bf16))
        tl_iso, tl_str = time_op(lambda i: torch.nn.functional.linear(x, Ws[i]), nb)
        del Ws
        row = {"matrix": mid, "M": M, "K": K, "marlin_stored_bytes": stored, "marlin_copies": nc,
               "marlin_us": t_iso, "marlin_str_us": t_str,
               "marlin_gbps": (stored + 2 * K + 2 * M) / (t_iso * 1e3),
               "marlin_rel_err_vs_own_ref": rel_ref, "marlin_rel_err_vs_bf16": rel_bf16,
               "torch_linear_bf16_us": tl_iso, "torch_linear_bf16_str_us": tl_str,
               "torch_linear_rel_err_vs_bf16": rel_lin}
        rows.append(row)
        print(f"{mid:7s} {M:6d}x{K:5d}  marlin iso {t_iso:9.2f} us str {t_str:9.2f} us  "
              f"{row['marlin_gbps']:6.1f} GB/s  err vs own ref {rel_ref:.2e}  vs bf16 {rel_bf16:.4f} | "
              f"torch linear bf16 iso {tl_iso:9.2f} us str {tl_str:9.2f} us", flush=True)
        del W, q_w, s_w, w_ref
        torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        f.write(f"# utc={datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ} device=\"{torch.cuda.get_device_name(0)}\" "
                f"vllm={vllm.__version__} torch={torch.__version__} cuda={torch.version.cuda} git={git} "
                f"image=vllm-node-qwen35:latest footprint_bytes={FOOTPRINT}\n")
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")


if __name__ == "__main__":
    main()
