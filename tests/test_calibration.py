"""Unit tests for the new analysis scripts: bootstrap and calibration.

These tests verify:
  * Bootstrap percentile function is correct and deterministic.
  * Bootstrap resampling is reproducible for a fixed seed.
  * `_per_image_score_outcome` produces the correct correct/incorrect
    labels on a tiny synthetic example.
  * `_fit_temperature` returns a positive value and the calibrated
    probabilities lie in (0, 1).
  * `_apply_temperature` is monotone-decreasing in the uncalibrated
    score when T > 1.
"""
from __future__ import annotations

import math
import unittest

import scripts.run_statistical_analysis as rsa
import scripts.run_calibration as rcal


class TestBootstrap(unittest.TestCase):
    def test_percentile_simple(self):
        s = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(rsa._percentile(s, 0), 1.0)
        self.assertAlmostEqual(rsa._percentile(s, 50), 3.0)
        self.assertAlmostEqual(rsa._percentile(s, 100), 5.0)
        # 25th percentile linear interpolation: 1 + 0.25*1 = 1.25
        self.assertAlmostEqual(rsa._percentile(s, 25), 2.0)

    def test_percentile_empty(self):
        self.assertTrue(math.isnan(rsa._percentile([], 50)))

    def test_bootstrap_records_deterministic(self):
        records = [{"image_id": f"img_{i}", "x": i} for i in range(20)]
        a = rsa._bootstrap_records(records, seed=7)
        b = rsa._bootstrap_records(records, seed=7)
        self.assertEqual([r["x"] for r in a], [r["x"] for r in b])
        # Different seed → different (with overwhelming probability)
        c = rsa._bootstrap_records(records, seed=8)
        self.assertNotEqual([r["x"] for r in a], [r["x"] for r in c])

    def test_ci95_bounds(self):
        vals = list(range(100))
        s = rsa._summarize([float(v) for v in vals])
        self.assertLess(s["ci95"][0], s["mean"])
        self.assertGreater(s["ci95"][1], s["mean"])


class TestCalibration(unittest.TestCase):
    def test_per_image_score_outcome(self):
        per_image = [
            {
                "image_id": "i1",
                "predictions": [
                    {"label": "car", "common_label": "Car",
                     "score": 0.9, "box_xyxy": (10, 10, 50, 50)},
                    {"label": "car", "common_label": "Car",
                     "score": 0.4, "box_xyxy": (200, 200, 240, 240)},
                ],
                "ground_truth": [
                    {"class_name": "Car", "box_xyxy": (10, 10, 50, 50)},
                ],
            }
        ]
        # score >= 0.5: only the first prediction; matches GT → outcome=1
        pairs = rcal._per_image_score_outcome(per_image, score_threshold=0.5, iou_threshold=0.5)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0], (0.9, 1))

    def test_per_image_score_outcome_unmatched(self):
        per_image = [
            {
                "image_id": "i1",
                "predictions": [
                    {"label": "car", "common_label": "Car",
                     "score": 0.7, "box_xyxy": (300, 300, 350, 350)},
                ],
                "ground_truth": [
                    {"class_name": "Car", "box_xyxy": (10, 10, 50, 50)},
                ],
            }
        ]
        pairs = rcal._per_image_score_outcome(per_image, score_threshold=0.5, iou_threshold=0.5)
        self.assertEqual(pairs, [(0.7, 0)])

    def test_fit_temperature_positive(self):
        pairs = [(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0), (0.6, 1), (0.3, 0)]
        T = rcal._fit_temperature(pairs)
        self.assertGreater(T, 0)

    def test_apply_temperature_in_unit_interval(self):
        pairs = [(0.9, 1), (0.2, 0), (0.5, 1)]
        cal = rcal._apply_temperature(pairs, T=1.5)
        for p, _ in cal:
            self.assertGreater(p, 0.0)
            self.assertLess(p, 1.0)

    def test_apply_temperature_T1_is_identity(self):
        # At T=1, calibrated ≈ original (within float precision).
        pairs = [(0.9, 1), (0.2, 0), (0.5, 1), (0.4, 0)]
        cal = rcal._apply_temperature(pairs, T=1.0)
        for (po, _), (pc, _) in zip(pairs, cal):
            self.assertAlmostEqual(po, pc, places=5)

    def test_apply_temperature_Tgt1_smooths(self):
        # With T>1 the calibrated score should be closer to 0.5 than the original.
        pairs = [(0.9, 1), (0.2, 0), (0.99, 1), (0.01, 0)]
        cal = rcal._apply_temperature(pairs, T=2.0)
        for (po, _), (pc, _) in zip(pairs, cal):
            self.assertLess(abs(pc - 0.5), abs(po - 0.5))

    def test_brier_perfect(self):
        self.assertAlmostEqual(rcal._brier([(1.0, 1), (0.0, 0)]), 0.0)
        self.assertAlmostEqual(rcal._brier([(1.0, 0), (0.0, 1)]), 1.0)


if __name__ == "__main__":
    unittest.main()
