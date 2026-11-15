#!/usr/bin/env python3
"""Export the benchmarked Qwen3.5-4B weight matrices as raw row-major BF16 files.

safetensors stores each tensor as contiguous little-endian bytes in row-major order, so
a BF16 matrix is exported by copying its byte range; no framework is needed. Writes:

  data/<id>.bf16.bin     M x K BF16, row-major
  data/manifest.tsv      id, tensor, M, K, role (read by the C++ benchmark)
  data/manifest.json     the same plus bytes, SHA-256, source file, model revision
  data/inventory.csv     every 2-D language-model weight (name, shape, bytes, layer type),
                         used by the per-token GEMV time model

Usage: python3 scripts/export_weights.py [--snapshot DIR] [--out data]
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys

REPO = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
P = "model.language_model."

# id, tensor, expected (M, K), role. "hyp" = a hypothesis matrix (CONTEXT.md §6.3);
# "model" = benchmarked only for the per-token time model.
EXPORTS = [
    ("gate", P + "layers.0.mlp.gate_proj.weight", (9216, 2560), "hyp"),
    ("up", P + "layers.0.mlp.up_proj.weight", (9216, 2560), "hyp"),
    ("down", P + "layers.0.mlp.down_proj.weight", (2560, 9216), "hyp"),
    ("qkv", P + "layers.0.linear_attn.in_proj_qkv.weight", (8192, 2560), "hyp"),
    ("z", P + "layers.0.linear_attn.in_proj_z.weight", (4096, 2560), "hyp"),
    ("out", P + "layers.0.linear_attn.out_proj.weight", (2560, 4096), "hyp"),
    ("q", P + "layers.3.self_attn.q_proj.weight", (8192, 2560), "hyp"),
    ("k", P + "layers.3.self_attn.k_proj.weight", (1024, 2560), "model"),
    ("v", P + "layers.3.self_attn.v_proj.weight", (1024, 2560), "model"),
    ("o", P + "layers.3.self_attn.o_proj.weight", (2560, 4096), "hyp"),
    ("lmhead", P + "embed_tokens.weight", (248320, 2560), "hyp"),
]


def find_snapshot(arg):
    if arg:
        return arg
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(REPO, revision=REVISION, local_files_only=True)
    except Exception as e:  # noqa: BLE001 - report and stop
        sys.exit(f"model not in the local cache ({e}); download it or pass --snapshot")


def read_header(path):
    """Return (header dict, byte offset where tensor data starts)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def copy_tensor(src, start, end, dst):
    """Copy bytes [start, end) of src into dst; return the SHA-256 of what was written."""
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(dst + ".tmp", "wb") as fo:
        fi.seek(start)
        left = end - start
        while left:
            chunk = fi.read(min(left, 64 << 20))
            if not chunk:
                sys.exit(f"{src}: file ended early")
            fo.write(chunk)
            h.update(chunk)
            left -= len(chunk)
    os.replace(dst + ".tmp", dst)  # never leave a truncated .bin under the final name
    return h.hexdigest()


def layer_type(name):
    if "linear_attn" in name:
        return "deltanet"
    if "self_attn" in name:
        return "full_attention"
    if "mlp" in name:
        return "mlp"
    if "embed_tokens" in name:
        return "lm_head"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    snap = find_snapshot(a.snapshot)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    headers = {}

    def tensor_info(name):
        fname = weight_map[name]
        if fname not in headers:
            headers[fname] = read_header(os.path.join(snap, fname))
        hdr, base = headers[fname]
        t = hdr[name]
        return fname, t["dtype"], t["shape"], base + t["data_offsets"][0], base + t["data_offsets"][1]

    manifest = []
    for mid, name, (M, K), role in EXPORTS:
        fname, dtype, shape, start, end = tensor_info(name)
        if dtype != "BF16" or shape != [M, K] or end - start != M * K * 2:
            sys.exit(f"{name}: {dtype} {shape} {end - start} B, expected BF16 [{M}, {K}]")
        dst = os.path.join(a.out, f"{mid}.bf16.bin")
        sha = copy_tensor(os.path.join(snap, fname), start, end, dst)
        if os.path.getsize(dst) != M * K * 2:
            sys.exit(f"{dst}: wrong size")
        manifest.append({"id": mid, "tensor": name, "M": M, "K": K, "role": role, "bytes": M * K * 2,
                         "sha256": sha, "source_file": fname})
        print(f"{mid:7s} {name:58s} {M:7d} x {K:5d}  {M * K * 2 / 1e6:9.2f} MB  {sha[:12]}")

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump({"model": REPO, "revision": REVISION, "matrices": manifest}, f, indent=1)
    with open(os.path.join(a.out, "manifest.tsv"), "w") as f:
        f.write("# id tensor M K role (written by scripts/export_weights.py)\n")
        for m in manifest:
            f.write(f"{m['id']}\t{m['tensor']}\t{m['M']}\t{m['K']}\t{m['role']}\n")

    # Inventory of every 2-D language-model weight, for the per-token model.
    rows = []
    for name in sorted(weight_map):
        if not name.startswith(P) or not name.endswith(".weight"):
            continue
        _, dtype, shape, start, end = tensor_info(name)
        if len(shape) != 2:
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        rows.append((name, m.group(1) if m else "", layer_type(name), shape[0], shape[1], dtype, end - start))
    with open(os.path.join(a.out, "inventory.csv"), "w") as f:
        f.write("tensor,layer,type,M,K,dtype,bytes\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print(f"manifest: {len(manifest)} matrices; inventory: {len(rows)} 2-D weights, "
          f"{sum(r[6] for r in rows) / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
