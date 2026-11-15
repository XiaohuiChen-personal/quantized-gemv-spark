#!/usr/bin/env python3
"""Weight-only symmetric quantization of the exported BF16 matrices (round to nearest).

INT8: one FP16 scale per row,             s = max|w| / 127 over the row.
INT4: one FP16 scale per 128 weights of K, s = max|w| / 7 over the group.
The scale is rounded to FP16 first and q = rint(w / s) uses that FP16 value, so q * s is
exactly what the kernels compute. rint rounds half to even. A row or group whose
max|w| / q_max underflows FP16 to zero gets the smallest FP16 subnormal (2^-24) unless it
is all zeros, which gets s = 0 and q = 0. INT4 codes are two's complement, two per byte,
low nibble = even column k, high nibble = odd column k + 1; -8 is never produced.

  data/<id>.int8.bin  M x K int8         data/<id>.int8.scale.bin  M FP16
  data/<id>.int4.bin  M x K/2 uint8      data/<id>.int4.scale.bin  M x K/128 FP16

Usage: python3 scripts/quantize.py [--data data]
"""
import argparse
import hashlib
import json
import os

import numpy as np

GROUP = 128
TINY = np.float16(2.0 ** -24)  # smallest positive FP16 subnormal


def bf16_to_f32(u16):
    """BF16 bit patterns (uint16 array) -> float32: the BF16 bits are the top half."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def scales_fp16(amax, qmax):
    """FP16 scales for the given max|w| (float32 array); see the module docstring."""
    s = (amax / np.float32(qmax)).astype(np.float16)
    s = np.where((s == 0) & (amax > 0), TINY, s)
    return s


def quantize_blocks(w, qmax):
    """w: float32 [..., n]; quantize each last-axis block with one scale.
    Returns (q int8 same shape, s float16 [...], n_clipped)."""
    amax = np.max(np.abs(w), axis=-1)
    s16 = scales_fp16(amax, qmax)
    s = s16.astype(np.float32)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(s > 0, np.rint(w / np.where(s > 0, s, 1)), 0)
    clipped = int(np.count_nonzero(np.abs(r) > qmax))
    q = np.clip(r, -qmax, qmax).astype(np.int8)
    return q, s16, clipped


def quantize_int8(w):
    """w: float32 [M, K] -> (q int8 [M, K], s float16 [M], clipped)."""
    return quantize_blocks(w, 127)


def quantize_int4(w):
    """w: float32 [M, K], K % 128 == 0 -> (packed uint8 [M, K/2], s float16 [M, K/128], clipped)."""
    M, K = w.shape
    assert K % GROUP == 0
    q, s16, clipped = quantize_blocks(w.reshape(M, K // GROUP, GROUP), 7)
    return pack_int4(q.reshape(M, K)), s16, clipped


def pack_int4(q):
    """int8 codes in [-8, 7], [M, K] -> uint8 [M, K/2]; low nibble = even column."""
    u = q.astype(np.uint8) & 0x0F
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)


def unpack_int4(p):
    """uint8 [M, K/2] -> int8 codes [M, K] (inverse of pack_int4)."""
    lo = (p & 0x0F).astype(np.int8)
    hi = (p >> 4).astype(np.int8)
    q = np.empty((p.shape[0], p.shape[1] * 2), dtype=np.int8)
    q[:, 0::2] = np.where(lo >= 8, lo - 16, lo)
    q[:, 1::2] = np.where(hi >= 8, hi - 16, hi)
    return q


def dequant_int8(q, s):
    return q.astype(np.float32) * s.astype(np.float32)[:, None]


def dequant_int4(p, s):
    q = unpack_int4(p).astype(np.float32)
    M, K = q.shape
    return (q.reshape(M, K // GROUP, GROUP) * s.astype(np.float32)[:, :, None]).reshape(M, K)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, arr):
    arr.tofile(path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    mpath = os.path.join(a.data, "manifest.json")
    with open(mpath) as f:
        man = json.load(f)
    rows_per_chunk = 16384
    for m in man["matrices"]:
        M, K, mid = m["M"], m["K"], m["id"]
        raw = np.fromfile(os.path.join(a.data, f"{mid}.bf16.bin"), dtype=np.uint16).reshape(M, K)
        q8 = np.empty((M, K), np.int8)
        s8 = np.empty(M, np.float16)
        q4 = np.empty((M, K // 2), np.uint8)
        s4 = np.empty((M, K // GROUP), np.float16)
        clip8 = clip4 = 0
        for r0 in range(0, M, rows_per_chunk):
            w = bf16_to_f32(raw[r0:r0 + rows_per_chunk])
            q8[r0:r0 + len(w)], s8[r0:r0 + len(w)], c = quantize_int8(w)
            clip8 += c
            q4[r0:r0 + len(w)], s4[r0:r0 + len(w)], c = quantize_int4(w)
            clip4 += c
        stats = {}
        for name, arr in (("int8", q8), ("int8.scale", s8), ("int4", q4), ("int4.scale", s4)):
            path = os.path.join(a.data, f"{mid}.{name}.bin")
            write(path, arr)
            stats[name] = {"bytes": int(arr.nbytes), "sha256": sha256(path)}
        for name, s in (("int8.scale", s8), ("int4.scale", s4)):
            bits = s.view(np.uint16)
            stats[name]["subnormal"] = int(np.count_nonzero(((bits & 0x7C00) == 0) & ((bits & 0x03FF) != 0)))
            stats[name]["zero"] = int(np.count_nonzero(s == 0))
        stats["int8"]["clipped"], stats["int4"]["clipped"] = clip8, clip4
        m["quant"] = stats
        print(f"{mid:7s} int8 scales: {stats['int8.scale']['subnormal']} subnormal, {stats['int8.scale']['zero']} zero, "
              f"{clip8} clipped | int4 scales: {stats['int4.scale']['subnormal']} subnormal, "
              f"{stats['int4.scale']['zero']} zero, {clip4} clipped")
    with open(mpath, "w") as f:
        json.dump(man, f, indent=1)


if __name__ == "__main__":
    main()
