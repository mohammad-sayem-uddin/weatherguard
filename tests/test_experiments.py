"""Tests for src/experiments — corruption functions and config validation."""
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.experiments import (
    CorruptionConfig,
    corrupt_image,
    default_corruption_grid,
    encode_pil,
)


class TestCorruptions(unittest.TestCase):
    def setUp(self):
        # Build a 64x64 RGB image with a known pattern.
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[16:48, 16:48] = (200, 100, 50)
        self.image = Image.fromarray(arr, mode="RGB")

    def test_identity_preserves_image(self):
        out = corrupt_image(self.image, CorruptionConfig("identity", "", {}))
        out_arr = np.asarray(out)
        self.assertTrue(np.array_equal(out_arr, np.asarray(self.image)))

    def test_brightness_reduce_darkens(self):
        out = corrupt_image(self.image, CorruptionConfig("brightness_reduce", "", {"factor": 0.3}))
        out_arr = np.asarray(out).astype(np.float32)
        orig_arr = np.asarray(self.image).astype(np.float32)
        self.assertLess(out_arr.mean(), orig_arr.mean() * 0.5)

    def test_contrast_reduce_decreases_std(self):
        out = corrupt_image(self.image, CorruptionConfig("contrast_reduce", "", {"factor": 0.3}))
        out_arr = np.asarray(out).astype(np.float32)
        orig_arr = np.asarray(self.image).astype(np.float32)
        self.assertLess(out_arr.std(), orig_arr.std() * 0.5)

    def test_blur_smooths(self):
        out = corrupt_image(self.image, CorruptionConfig("blur", "", {"radius": 4.0}))
        out_arr = np.asarray(out).astype(np.float32)
        # The central region should be smoother (lower local variance)
        orig_arr = np.asarray(self.image).astype(np.float32)
        orig_var = orig_arr[16:48, 16:48].var()
        out_var = out_arr[16:48, 16:48].var()
        self.assertLess(out_var, orig_var)

    def test_gaussian_noise_deterministic(self):
        cfg = CorruptionConfig("gaussian_noise", "", {"sigma": 10.0})
        a = np.asarray(corrupt_image(self.image, cfg, seed=0)).astype(np.int16)
        b = np.asarray(corrupt_image(self.image, cfg, seed=0)).astype(np.int16)
        # Same seed should produce identical noise.
        self.assertTrue(np.array_equal(a, b))
        # Different seed → different noise.
        c = np.asarray(corrupt_image(self.image, cfg, seed=1)).astype(np.int16)
        self.assertFalse(np.array_equal(a, c))

    def test_unknown_corruption_raises(self):
        with self.assertRaises(ValueError):
            corrupt_image(self.image, CorruptionConfig("mystery", "", {}))

    def test_default_grid_has_baseline(self):
        grid = default_corruption_grid()
        names = [c.name for c in grid]
        self.assertIn("identity", names)
        # And at least three controlled corruptions
        self.assertGreaterEqual(len(grid), 4)

    def test_encode_pil_returns_png(self):
        out = encode_pil(self.image)
        self.assertGreater(len(out), 0)
        self.assertEqual(out[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
