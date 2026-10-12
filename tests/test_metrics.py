"""Unit tests for src/metrics using small synthetic examples.

Run with:  .venv/bin/python -m unittest tests.test_metrics
"""
from __future__ import annotations

import math
import unittest

from src.metrics import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    ClassPR,
    EvalResult,
    _average_precision_101,
    evaluate_detections,
    greedy_match,
    iou_xyxy,
)


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint_boxes(self):
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_half_overlap(self):
        # Two 10x10 boxes overlapping in 5x10 area → inter=50, union=100+100-50=150 → IoU 1/3
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3, places=4)

    def test_contained_box(self):
        # 2x2 inside 10x10 → inter=4, union=100, IoU=0.04
        self.assertAlmostEqual(iou_xyxy((0, 0, 10, 10), (4, 4, 6, 6)), 4 / 100)

    def test_zero_area_box_returns_zero(self):
        self.assertEqual(iou_xyxy((0, 0, 0, 0), (0, 0, 10, 10)), 0.0)


class TestGreedyMatch(unittest.TestCase):
    def test_no_predictions_no_gt(self):
        self.assertEqual(greedy_match([], [], []), [])

    def test_perfect_match(self):
        m = greedy_match([(0, 0, 10, 10)], [0.9], [(0, 0, 10, 10)], iou_threshold=0.5)
        self.assertEqual(m, [(0, 0, 1.0)])

    def test_low_iou_rejected(self):
        m = greedy_match([(0, 0, 10, 10)], [0.9], [(20, 20, 30, 30)], iou_threshold=0.5)
        self.assertEqual(m, [])

    def test_score_order_picks_higher(self):
        # Two GT, two predictions; better-confidence pred should match the closer GT
        m = greedy_match(
            [(0, 0, 10, 10), (20, 20, 30, 30)],
            [0.5, 0.9],
            [(0, 0, 10, 10), (20, 20, 30, 30)],
            iou_threshold=0.5,
        )
        self.assertEqual(len(m), 2)
        # The first match in score order is the 0.9 box
        self.assertEqual(m[0][0], 1)
        self.assertEqual(m[0][1], 1)
        self.assertEqual(m[1][0], 0)
        self.assertEqual(m[1][1], 0)

    def test_greedy_uses_each_gt_once(self):
        # Two predictions and one GT → only one match
        m = greedy_match(
            [(0, 0, 10, 10), (1, 1, 11, 11)],
            [0.7, 0.9],
            [(0, 0, 10, 10)],
            iou_threshold=0.3,
        )
        self.assertEqual(len(m), 1)


class TestAveragePrecision(unittest.TestCase):
    def test_perfect_ap(self):
        scores = [0.9, 0.8, 0.7]
        # 3 GT, all detected correctly
        ap = _average_precision_101(scores, [True, True, True], n_gt=3)
        self.assertAlmostEqual(ap, 1.0, places=4)

    def test_zero_ap_when_no_recall(self):
        ap = _average_precision_101([0.9, 0.8], [False, False], n_gt=2)
        self.assertAlmostEqual(ap, 0.0, places=4)

    def test_no_gt_returns_nan(self):
        self.assertTrue(math.isnan(_average_precision_101([0.9], [True], n_gt=0)))


class TestEvaluateDetections(unittest.TestCase):
    def test_perfect_detector(self):
        per_image = [
            {
                "image_id": "img1",
                "predictions": [
                    {
                        "label": "car",
                        "common_label": "Car",
                        "score": 0.99,
                        "box_xyxy": (10, 20, 110, 120),
                    }
                ],
                "ground_truth": [{"class_name": "Car", "box_xyxy": (10, 20, 110, 120)}],
            }
        ]
        res = evaluate_detections(per_image, iou_threshold=0.5, score_threshold=0.5)
        self.assertEqual(res.n_images, 1)
        car = next(c for c in res.per_class if c.class_name == "Car")
        self.assertEqual(car.tp, 1)
        self.assertEqual(car.fp, 0)
        self.assertEqual(car.fn, 0)
        self.assertAlmostEqual(car.precision, 1.0)
        self.assertAlmostEqual(car.recall, 1.0)
        self.assertAlmostEqual(car.ap_50, 1.0)

    def test_fully_missed_class(self):
        per_image = [
            {
                "image_id": "img1",
                "predictions": [],
                "ground_truth": [{"class_name": "Truck", "box_xyxy": (0, 0, 50, 50)}],
            }
        ]
        res = evaluate_detections(per_image, iou_threshold=0.5, score_threshold=0.5)
        truck = next(c for c in res.per_class if c.class_name == "Truck")
        self.assertEqual(truck.n_gt, 1)
        self.assertEqual(truck.tp, 0)
        self.assertEqual(truck.fn, 1)
        self.assertAlmostEqual(truck.ap_50, 0.0)

    def test_false_positive_counted(self):
        per_image = [
            {
                "image_id": "img1",
                "predictions": [
                    {
                        "label": "car",
                        "common_label": "Car",
                        "score": 0.9,
                        "box_xyxy": (0, 0, 50, 50),
                    },
                    {
                        "label": "car",
                        "common_label": "Car",
                        "score": 0.9,
                        "box_xyxy": (200, 200, 250, 250),
                    },
                ],
                "ground_truth": [{"class_name": "Car", "box_xyxy": (0, 0, 50, 50)}],
            }
        ]
        res = evaluate_detections(per_image, iou_threshold=0.5, score_threshold=0.5)
        car = next(c for c in res.per_class if c.class_name == "Car")
        self.assertEqual(car.tp, 1)
        self.assertEqual(car.fp, 1)
        self.assertEqual(car.fn, 0)
        self.assertAlmostEqual(car.precision, 0.5)
        self.assertAlmostEqual(car.recall, 1.0)

    def test_score_threshold_filters_predictions(self):
        per_image = [
            {
                "image_id": "img1",
                "predictions": [
                    {
                        "label": "car",
                        "common_label": "Car",
                        "score": 0.2,
                        "box_xyxy": (0, 0, 50, 50),
                    }
                ],
                "ground_truth": [{"class_name": "Car", "box_xyxy": (0, 0, 50, 50)}],
            }
        ]
        # With score threshold 0.5 the prediction is filtered → FN
        res = evaluate_detections(per_image, iou_threshold=0.5, score_threshold=0.5)
        car = next(c for c in res.per_class if c.class_name == "Car")
        self.assertEqual(car.tp, 0)
        self.assertEqual(car.fp, 0)
        self.assertEqual(car.fn, 1)

    def test_oov_predictions_ignored(self):
        # 'potted plant' is not in DAWN label space → must be ignored
        per_image = [
            {
                "image_id": "img1",
                "predictions": [
                    {
                        "label": "potted plant",
                        "common_label": None,
                        "score": 0.99,
                        "box_xyxy": (0, 0, 50, 50),
                    }
                ],
                "ground_truth": [],
            }
        ]
        res = evaluate_detections(per_image, iou_threshold=0.5, score_threshold=0.5)
        self.assertEqual(res.micro["tp"], 0)
        self.assertEqual(res.micro["fp"], 0)
        self.assertEqual(res.micro["fn"], 0)

    def test_empty_input_returns_empty_eval(self):
        res = evaluate_detections([], iou_threshold=0.5, score_threshold=0.5)
        self.assertEqual(res.n_images, 0)
        self.assertTrue(math.isnan(res.macro["mAP_50"]))

    def test_invalid_entry_raises(self):
        with self.assertRaises(ValueError):
            evaluate_detections([{"image_id": "x"}], iou_threshold=0.5, score_threshold=0.5)

    def test_thresholds_constants_sensible(self):
        self.assertEqual(DEFAULT_IOU_THRESHOLD, 0.5)
        self.assertEqual(DEFAULT_SCORE_THRESHOLD, 0.5)


if __name__ == "__main__":
    unittest.main()
