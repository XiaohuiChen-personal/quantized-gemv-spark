#!/usr/bin/env python3
"""Write a weight-only INT4 or INT8 checkpoint of Qwen3.5-4B that vLLM can serve.

The large language-model matrices -- every 2-D weight with at least 1024 rows in all 32
layers (MLP gate/up/down, DeltaNet in_proj_qkv/in_proj_z/out_proj, attention q/k/v/o),
the same set as scripts/accuracy_sweep.py minus the token embedding -- are quantized with
this project's quantizer (scripts/quantize.py: symmetric round-to-nearest, FP16 scales)
and stored in GPTQ format, which vLLM runs with its Marlin kernels (gptq_marlin):

  INT4  group 128 along K. GPTQ's symmetric 4-bit type is uint4b8 (w = (u - 8) s): u = q + 8
        with q in [-7, 7] from quantize_int4, so u in [1, 15]. qweight int32 [K/8, M]
        packs u[n, 8i + j] into bits 4j..4j+3 of qweight[i, n]; scales FP16 [K/128, M].
  INT8  one scale per output row (GPTQ group_size -1). uint8b128: u = q + 128, q in
        [-127, 127]; qweight int32 [K/4, M] packs u[n, 4i + j] into bits 8j..8j+7; scales
        FP16 [1, M].

g_idx (int32 [K], group of each column) and qzeros (int32, GPTQ v1 zero point minus one,
unused by vLLM for symmetric types) complete each layer. Everything else -- the token
embedding (tied to the LM head, so the LM head stays BF16), norms, the DeltaNet
conv1d/A_log/dt_bias and its small in_proj_a/in_proj_b, the vision encoder and the MTP
head -- is copied byte for byte. vLLM finds the quantized layers from the integer dtypes.

With --verify-data, the codes and scales of every matrix that bench exports (data/) are
unpacked again and compared with data/<id>.int4.bin / .int8.bin and their scales, bit for
bit, so the served model uses exactly the operands the kernels were timed and checked on.

Usage: python3 scripts/make_quant_checkpoint.py --bits 4 --out <dir> [--snapshot S] [--verify-data data]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_weights as ew  # noqa: E402
import quantize as qz  # noqa: E402

CHUNK = 8192
DT = {"BF16": (np.uint16, 2), "F16": (np.float16, 2), "F32": (np.float32, 4), "I32": (np.int32, 4),
      "I64": (np.int64, 8), "U8": (np.uint8, 1), "I8": (np.int8, 1), "BOOL": (np.uint8, 1)}


def is_target(name, info):
    return (name.startswith(ew.P) and name.endswith(".weight") and "embed_tokens" not in name
            and len(info["shape"]) == 2 and info["dtype"] == "BF16" and info["shape"][0] >= 1024)


def quantize_gptq(raw, M, K, bits):
    """raw: uint16 BF16 [M, K] -> dict of GPTQ tensors (numpy) for one linear layer."""
    per = 32 // bits
    groups = K // qz.GROUP if bits == 4 else 1
    qweight = np.empty((K // per, M), np.int32)
    scales = np.empty((groups, M), np.float16)
    for r0 in range(0, M, CHUNK):
        w = qz.bf16_to_f32(np.asarray(raw[r0:r0 + CHUNK]))
        n = len(w)
        if bits == 4:
            p, s, _ = qz.quantize_int4(w)
            u = (qz.unpack_int4(p).astype(np.int32) + 8).astype(np.uint32)  # [n, K] in 1..15
            scales[:, r0:r0 + n] = s.T
        else:
            q, s, _ = qz.quantize_int8(w)
            u = (q.astype(np.int32) + 128).astype(np.uint32)  # 1..255
            scales[0, r0:r0 + n] = s
        u = u.reshape(n, K // per, per)
        packed = np.zeros((n, K // per), np.uint32)
        for j in range(per):
            packed |= u[:, :, j] << np.uint32(bits * j)
        qweight[:, r0:r0 + n] = packed.T.view(np.int32)
    zero_word = 0x77777777 if bits == 4 else 0x7F7F7F7F  # zero point 8 (128) minus one, per GPTQ v1
    qzeros = np.full((groups, M // per), zero_word, np.uint32).view(np.int32)
    g_idx = (np.arange(K, dtype=np.int32) // qz.GROUP) if bits == 4 else np.zeros(K, np.int32)
    return {"qweight": qweight, "scales": scales, "qzeros": qzeros, "g_idx": g_idx}


def unpack_gptq(t, M, K, bits):
    """Inverse of quantize_gptq for checking: -> (codes int8 [M, K], scales float16 [M, groups])."""
    per = 32 // bits
    mask = (1 << bits) - 1
    qw = t["qweight"].view(np.uint32).T  # [M, K/per]
    u = np.stack([(qw >> np.uint32(bits * j)) & np.uint32(mask) for j in range(per)], axis=-1).reshape(M, K)
    q = u.astype(np.int32) - (8 if bits == 4 else 128)
    return q.astype(np.int8), t["scales"].T


class ShardWriter:
    """Minimal safetensors writer: tensors are appended to a data file, the header is
    written in front at the end (header length, JSON padded to 8 bytes, data)."""

    def __init__(self, path):
        self.path, self.tmp = path, path + ".data.tmp"
        self.f = open(self.tmp, "wb")
        self.hdr, self.off = {}, 0

    def add(self, name, dtype, shape, data):
        b = data if isinstance(data, (bytes, memoryview)) else np.ascontiguousarray(data).tobytes()
        self.f.write(b)
        self.hdr[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [self.off, self.off + len(b)]}
        self.off += len(b)

    def close(self):
        self.f.close()
        h = json.dumps({"__metadata__": {"format": "pt"}, **self.hdr}, separators=(",", ":")).encode()
        h += b" " * (-len(h) % 8)
        with open(self.path + ".tmp", "wb") as out, open(self.tmp, "rb") as src:
            out.write(struct.pack("<Q", len(h)))
            out.write(h)
            shutil.copyfileobj(src, out, 64 << 20)
        os.remove(self.tmp)
        os.replace(self.path + ".tmp", self.path)
        return self.off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, choices=(4, 8), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot")
    ap.add_argument("--verify-data")
    a = ap.parse_args()
    snap = ew.find_snapshot(a.snapshot)
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    os.makedirs(a.out, exist_ok=True)
    exported = {}
    if a.verify_data:
        for m in json.load(open(os.path.join(a.verify_data, "manifest.json")))["matrices"]:
            exported[m["tensor"]] = m
    t0, n_q, n_copy, new_map, total = time.time(), 0, 0, {}, 0
    verified = []
    for fname in sorted(set(wmap.values())):
        hdr, base = ew.read_header(os.path.join(snap, fname))
        w = ShardWriter(os.path.join(a.out, fname))
        mm = np.memmap(os.path.join(snap, fname), dtype=np.uint8, mode="r")
        for name, info in sorted((k, v) for k, v in hdr.items() if k != "__metadata__"):
            lo, hi = info["data_offsets"]
            if not is_target(name, info):
                w.add(name, info["dtype"], info["shape"], memoryview(mm[base + lo:base + hi]))
                new_map[name] = fname
                n_copy += 1
                continue
            M, K = info["shape"]
            raw = np.frombuffer(mm[base + lo:base + hi], dtype=np.uint16).reshape(M, K)
            t = quantize_gptq(raw, M, K, a.bits)
            stem = name[: -len(".weight")]
            for k, dtype in (("qweight", "I32"), ("qzeros", "I32"), ("scales", "F16"), ("g_idx", "I32")):
                w.add(f"{stem}.{k}", dtype, t[k].shape, t[k])
                new_map[f"{stem}.{k}"] = fname
            n_q += 1
            if name in exported:  # bit-exact against the benchmark's operands
                m = exported[name]
                q, s = unpack_gptq(t, M, K, a.bits)
                if a.bits == 4:
                    ref_q = qz.unpack_int4(np.fromfile(os.path.join(a.verify_data, f"{m['id']}.int4.bin"), np.uint8).reshape(M, K // 2))
                    ref_s = np.fromfile(os.path.join(a.verify_data, f"{m['id']}.int4.scale.bin"), np.float16).reshape(M, K // qz.GROUP)
                else:
                    ref_q = np.fromfile(os.path.join(a.verify_data, f"{m['id']}.int8.bin"), np.int8).reshape(M, K)
                    ref_s = np.fromfile(os.path.join(a.verify_data, f"{m['id']}.int8.scale.bin"), np.float16).reshape(M, 1)
                ok = np.array_equal(q, ref_q) and np.array_equal(s.view(np.uint16), ref_s.view(np.uint16))
                verified.append((m["id"], ok))
                if not ok:
                    sys.exit(f"{name}: GPTQ codes/scales differ from data/{m['id']}")
            print(f"{name.replace(ew.P, ''):45s} {M:6d}x{K:5d} -> int{a.bits} GPTQ  [{time.time() - t0:.0f} s]", flush=True)
        total += w.close()
    cfg = json.load(open(os.path.join(snap, "config.json")))
    cfg["quantization_config"] = {
        "quant_method": "gptq", "bits": a.bits, "group_size": qz.GROUP if a.bits == 4 else -1,
        "desc_act": False, "sym": True, "lm_head": False, "checkpoint_format": "gptq",
        "true_sequential": True, "damp_percent": 0.0,
        "quantizer": "round-to-nearest, scripts/quantize.py of quantized-gemv-spark (no calibration)",
    }
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    for f in os.listdir(snap):  # tokenizer, chat template, processor configs, licence
        if not f.endswith(".safetensors") and f not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(os.path.join(snap, f), os.path.join(a.out, f))
    info = {"source": f"{ew.REPO}@{ew.REVISION}", "bits": a.bits, "quantized_matrices": n_q, "copied_tensors": n_copy,
            "total_bytes": total, "verified_against_data": verified,
            "files": {f: hashlib.sha256(open(os.path.join(a.out, f), "rb").read()).hexdigest()
                      for f in sorted(os.listdir(a.out)) if f.endswith(".json")}}
    json.dump(info, open(os.path.join(a.out, "quantization_info.json"), "w"), indent=1)
    print(f"{n_q} matrices quantized to int{a.bits}, {n_copy} tensors copied, {total / 1e9:.2f} GB in "
          f"{time.time() - t0:.0f} s; verified bit-exact against data/: {sum(ok for _, ok in verified)}/{len(verified)}")


if __name__ == "__main__":
    main()
