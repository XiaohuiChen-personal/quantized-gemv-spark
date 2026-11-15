#!/usr/bin/env python3
"""Write the input vectors x (BF16), one per K used by the benchmarked matrices.

  data/x_k<K>.bin     Gaussian N(0, 1), seed 20261114 + K   (timing and accuracy)
  data/xrad_k<K>.bin  +-1 with equal probability, seed 1 + K (correctness only: with
                      |x_k| = 1 a dropped column cannot hide under a small x_k)

Usage: python3 scripts/make_x.py [--data data]
"""
import argparse
import os

import numpy as np

SEED = 20261114
KS = (2560, 4096, 9216)


def f32_to_bf16(f):
    """float32 -> BF16 bit patterns with round-to-nearest-even (finite inputs)."""
    u = np.asarray(f, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def gaussian_x(K):
    return f32_to_bf16(np.random.default_rng(SEED + K).standard_normal(K, dtype=np.float32))


def rademacher_x(K):
    signs = np.random.default_rng(1 + K).integers(0, 2, K) * 2 - 1
    return f32_to_bf16(signs.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    for K in KS:
        gaussian_x(K).tofile(os.path.join(a.data, f"x_k{K}.bin"))
        rademacher_x(K).tofile(os.path.join(a.data, f"xrad_k{K}.bin"))
        print(f"x_k{K}.bin, xrad_k{K}.bin")


if __name__ == "__main__":
    main()
