#!/usr/bin/env python3
"""Unit tests of scripts/quantize.py and scripts/make_x.py (run by `make test`).

Also checks the proposal's error estimate on synthetic Gaussian weights: for Gaussian x
the relative output error should be close to sqrt(sum delta^2 / sum w^2), and for
Gaussian weights that is about 0.8-0.9% (INT8, K = 2560) and 11-12% (INT4, group 128).
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_x  # noqa: E402
import quantize as qz  # noqa: E402


class TestQuantize(unittest.TestCase):
    def test_bf16_roundtrip(self):
        f = np.array([1.0, -2.5, 2.0 ** -7, -(2.0 ** -15)], np.float32)  # exactly representable
        self.assertTrue(np.array_equal(qz.bf16_to_f32(make_x.f32_to_bf16(f)), f))
        # Ties round to even: 1 + 2^-8 -> 1.0, 1 + 3*2^-8 -> 1 + 2^-6.
        t = make_x.f32_to_bf16(np.array([1.00390625, 1.01171875], np.float32))
        self.assertEqual(list(t), [0x3F80, 0x3F82])

    def test_int8_known_row(self):
        w = np.array([[0.5, -1.0, 0.25, 0.0]], np.float32)
        q, s, clipped = qz.quantize_int8(w)
        s32 = s.astype(np.float32)
        self.assertEqual(s[0], np.float16(1.0 / 127))
        self.assertEqual(q[0, 1], -127)
        self.assertEqual(clipped, 0)
        self.assertTrue(np.all(np.abs(w - q * s32[:, None]) <= s32[:, None] / 2))

    def test_pack_order(self):
        q = np.array([[1, -2, 3, -8, 7, 0, -1, 4]], np.int8)
        p = qz.pack_int4(q)
        # low nibble = even column, high nibble = odd column, two's complement
        self.assertEqual(list(p[0]), [0xE1, 0x83, 0x07, 0x4F])
        self.assertTrue(np.array_equal(qz.unpack_int4(p), q))

    def test_all_codes_roundtrip(self):
        q = np.tile(np.arange(-8, 8, dtype=np.int8), (3, 8))
        self.assertTrue(np.array_equal(qz.unpack_int4(qz.pack_int4(q)), q))

    def test_zero_and_tiny_groups(self):
        w = np.zeros((2, 256), np.float32)
        w[1, 128] = 1e-9  # max|w| / 7 underflows FP16 to zero
        p, s, _ = qz.quantize_int4(w)
        self.assertEqual(s[0, 0], 0)
        self.assertEqual(s[0, 1], 0)
        self.assertEqual(s[1, 1], qz.TINY)
        self.assertTrue(np.all(np.isfinite(qz.dequant_int4(p, s))))
        self.assertTrue(np.all(qz.unpack_int4(p)[0] == 0))

    def test_random_bounds(self):
        rng = np.random.default_rng(0)
        w = (rng.standard_t(3, size=(64, 1024)) * 0.02).astype(np.float32)  # heavy tails
        for quant, deq, qmax, group in ((qz.quantize_int8, qz.dequant_int8, 127, 1024),
                                        (qz.quantize_int4, qz.dequant_int4, 7, 128)):
            q, s, clipped = quant(w)
            self.assertEqual(clipped, 0)
            d = deq(q, s)
            s_full = np.repeat(s.astype(np.float32).reshape(64, -1), group, axis=1)
            self.assertTrue(np.all(np.abs(w - d) <= s_full / 2 * (1 + 2.0 ** -10)))
            codes = q if qmax == 127 else qz.unpack_int4(q)
            self.assertLessEqual(int(np.abs(codes).max()), qmax)
            self.assertGreaterEqual(int(codes.min()), -qmax)  # -8 / -128 never produced

    def test_error_estimate_gaussian(self):
        rng = np.random.default_rng(1)
        M, K = 2048, 2560
        w = rng.standard_normal((M, K), dtype=np.float32) * 0.02
        x = rng.standard_normal(K).astype(np.float64)
        y = w.astype(np.float64) @ x
        for quant, deq, lo, hi in ((qz.quantize_int8, qz.dequant_int8, 0.006, 0.011),
                                   (qz.quantize_int4, qz.dequant_int4, 0.09, 0.14)):
            q, s, _ = quant(w)
            d = deq(q, s).astype(np.float64)
            rel = np.linalg.norm(d @ x - y) / np.linalg.norm(y)
            pred = np.sqrt(((w - d) ** 2).sum() / (w.astype(np.float64) ** 2).sum())
            self.assertTrue(lo < rel < hi, f"rel {rel}")
            self.assertAlmostEqual(rel / pred, 1.0, delta=0.05)

    def test_x_vectors(self):
        x = qz.bf16_to_f32(make_x.gaussian_x(2560))
        self.assertAlmostEqual(float(x.std()), 1.0, delta=0.05)
        r = qz.bf16_to_f32(make_x.rademacher_x(2560))
        self.assertTrue(set(np.unique(r)) == {-1.0, 1.0})
        self.assertTrue(np.array_equal(make_x.gaussian_x(2560), make_x.gaussian_x(2560)))  # fixed seed


if __name__ == "__main__":
    unittest.main(verbosity=1)
