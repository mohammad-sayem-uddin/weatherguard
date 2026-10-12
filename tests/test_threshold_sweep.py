"""Unit tests for the threshold-sweep helper functions.

The script is exercised end-to-end elsewhere; here we just check the
helper selection logic in isolation.
"""
from __future__ import annotations

import unittest

import scripts.run_threshold_sweep as rts


class TestThresholdSweep(unittest.TestCase):
    def test_select_best_returns_highest(self):
        rows = [
            {"score_threshold": 0.3, "micro_f1": 0.5, "mAP_50": 0.4},
            {"score_threshold": 0.5, "micro_f1": 0.7, "mAP_50": 0.6},
            {"score_threshold": 0.7, "micro_f1": 0.6, "mAP_50": 0.5},
        ]
        self.assertEqual(rts._select_best(rows, "micro_f1")["score_threshold"], 0.5)
        self.assertEqual(rts._select_best(rows, "mAP_50")["score_threshold"], 0.5)

    def test_select_best_empty(self):
        self.assertEqual(rts._select_best([], "micro_f1"), {})

    def test_select_best_filters_none(self):
        rows = [
            {"score_threshold": 0.3, "micro_f1": None, "mAP_50": 0.4},
            {"score_threshold": 0.5, "micro_f1": 0.7, "mAP_50": 0.6},
        ]
        self.assertEqual(rts._select_best(rows, "micro_f1")["score_threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
