"""Tests for the new analysis script: error taxonomy classifier.

Verifies the per-image categorisation function from
`scripts/run_error_taxonomy.py` produces the documented categories
on tiny synthetic examples.
"""
from __future__ import annotations

import unittest

import scripts.run_error_taxonomy as ret


def _rec(preds, gts, image_id="img1", condition="foggy"):
    return {
        "image_id": image_id,
        "condition": condition,
        "predictions": preds,
        "ground_truth": gts,
    }


def _p(label, common, score, box, raw="car"):
    return {"label": raw, "common_label": common, "score": score, "box_xyxy": box}


def _g(cls, box):
    return {"class_name": cls, "box_xyxy": box}


class TestErrorTaxonomy(unittest.TestCase):
    def test_perfect_match_yields_TP_and_Detected(self):
        rec = _rec(
            [_p("car", "Car", 0.9, (10, 10, 50, 50))],
            [_g("Car", (10, 10, 50, 50))],
        )
        cat = ret._categorise_image(rec)
        self.assertEqual(cat["pred_cats"], ["TP"])
        self.assertEqual(cat["gt_cats"], ["Detected"])

    def test_class_confusion_with_good_iou(self):
        rec = _rec(
            [_p("truck", "Truck", 0.9, (10, 10, 50, 50), raw="truck")],
            [_g("Car", (10, 10, 50, 50))],
        )
        cat = ret._categorise_image(rec)
        self.assertEqual(cat["pred_cats"], ["Class_confusion"])
        # The GT was matched but with the wrong class → Missed_class
        self.assertEqual(cat["gt_cats"], ["Missed_class"])

    def test_duplicate_detection_marked(self):
        rec = _rec(
            [
                _p("car", "Car", 0.9, (10, 10, 50, 50)),
                _p("car", "Car", 0.7, (10, 10, 50, 50)),  # same box
            ],
            [_g("Car", (10, 10, 50, 50))],
        )
        cat = ret._categorise_image(rec)
        self.assertEqual(cat["pred_cats"], ["TP", "Duplicate"])

    def test_localization_when_iou_between_01_and_05(self):
        # 10×10 vs 10×4 = IoU 4/100+100-4 = 0.04... no, let me build it.
        # a=(0,0,10,10) b=(0,0,10,4): inter=4*10=40, union=100+40-40=100 → IoU 0.4
        rec = _rec(
            [_p("car", "Car", 0.6, (0, 0, 10, 4))],
            [_g("Car", (0, 0, 10, 10))],
        )
        cat = ret._categorise_image(rec)
        self.assertEqual(cat["pred_cats"], ["Localization"])
        # Accounting rule: a Localization prediction does NOT consume the GT.
        # The GT therefore appears as Missed_class (its Detection column in
        # the per-class ground-truth table counts only TPs, which is the
        # standard mAP accounting).
        self.assertEqual(cat["gt_cats"], ["Missed_class"])

    def test_fp_other_when_no_overlap(self):
        rec = _rec(
            [_p("car", "Car", 0.7, (200, 200, 250, 250))],
            [_g("Car", (10, 10, 50, 50))],
        )
        cat = ret._categorise_image(rec)
        self.assertEqual(cat["pred_cats"], ["FP_other"])
        self.assertEqual(cat["gt_cats"], ["Missed_class"])

    def test_missed_gt_with_no_predictions(self):
        rec = _rec(
            [],
            [_g("Car", (10, 10, 50, 50)), _g("Person", (0, 0, 5, 5))],
        )
        cat = ret._categorise_image(rec)
        self.assertEqual(cat["pred_cats"], [])
        self.assertEqual(cat["gt_cats"], ["Missed_class", "Missed_class"])

    def test_out_of_vocabulary_predictions_ignored(self):
        rec = _rec(
            [_p("potted plant", None, 0.99, (10, 10, 50, 50))],
            [_g("Car", (10, 10, 50, 50))],
        )
        cat = ret._categorise_image(rec)
        self.assertEqual(cat["pred_cats"], [])  # OOV filtered before categorisation
        self.assertEqual(cat["gt_cats"], ["Missed_class"])

    def test_low_confidence_predictions_filtered(self):
        rec = _rec(
            [_p("car", "Car", 0.4, (10, 10, 50, 50))],  # below 0.5
            [_g("Car", (10, 10, 50, 50))],
        )
        cat = ret._categorise_image(rec)
        self.assertEqual(cat["pred_cats"], [])


class TestSizeBucket(unittest.TestCase):
    def test_size_buckets(self):
        self.assertEqual(ret._size_bucket(31 * 31), "small")  # 961 < 1024
        self.assertEqual(ret._size_bucket(32 * 32), "medium")  # 1024 >= 1024 and < 9216
        self.assertEqual(ret._size_bucket(96 * 96), "large")  # 9216 >= 9216
        self.assertEqual(ret._size_bucket(200 * 200), "large")


if __name__ == "__main__":
    unittest.main()
