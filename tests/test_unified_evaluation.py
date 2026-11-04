"""Unit tests for WeatherGuard unified evaluation framework.

Module: tests/test_unified_evaluation.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""
from pathlib import Path
import pytest

from scripts.evaluate_weatherguard import (
    DetectionBox,
    GroundTruthBox,
    WeatherGuardEvaluator,
    evaluate_predictions,
    get_object_scale,
    merge_two_tier_nms,
)


def test_object_scale_boundaries():
    assert get_object_scale(10, 15) == "ultra_small"
    assert get_object_scale(15.9, 8) == "ultra_small"
    assert get_object_scale(16.0, 10) == "small"
    assert get_object_scale(31.9, 20) == "small"
    assert get_object_scale(32.0, 50) == "medium"
    assert get_object_scale(95.9, 10) == "medium"
    assert get_object_scale(96.0, 50) == "large"
    assert get_object_scale(200, 300) == "large"


def test_evaluate_predictions_exact_math():
    # Image 1:
    # GT: car (medium), truck (small)
    # Pred: car (medium, matched, IoU=0.8), bicycle (fp, small)
    # Result: TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5, F1=0.5
    gt1 = [
        GroundTruthBox(class_name="car", x1=100, y1=100, x2=150, y2=150, scale="medium", max_dim=50),
        GroundTruthBox(class_name="truck", x1=200, y1=200, x2=220, y2=220, scale="small", max_dim=20),
    ]
    pred1 = [
        DetectionBox(class_name="car", confidence=0.9, x1=102, y1=102, x2=148, y2=148),
        DetectionBox(class_name="bicycle", confidence=0.8, x1=300, y1=300, x2=325, y2=325),
    ]

    # Image 2:
    # GT: person (ultra_small)
    # Pred: person (ultra_small, matched)
    gt2 = [
        GroundTruthBox(class_name="person", x1=50, y1=50, x2=60, y2=60, scale="ultra_small", max_dim=10),
    ]
    pred2 = [
        DetectionBox(class_name="person", confidence=0.75, x1=50, y1=50, x2=60, y2=60),
    ]

    all_preds = [pred1, pred2]
    all_gts = [gt1, gt2]

    res = evaluate_predictions(all_preds, all_gts, iou_thresh=0.50)

    # Total: TP=2, FP=1, FN=1
    # Prec = 2/3 = 0.6667
    # Rec = 2/3 = 0.6667
    # F1 = 0.6667
    assert res["overall"]["tp"] == 2
    assert res["overall"]["fp"] == 1
    assert res["overall"]["fn"] == 1
    assert res["overall"]["precision"] == 0.6667
    assert res["overall"]["recall"] == 0.6667
    assert res["overall"]["f1"] == 0.6667

    # Scale checks
    assert res["by_scale"]["ultra_small"]["tp"] == 1
    assert res["by_scale"]["ultra_small"]["fp"] == 0
    assert res["by_scale"]["ultra_small"]["f1"] == 1.0

    assert res["by_scale"]["small"]["fn"] == 1  # truck missed
    assert res["by_scale"]["small"]["fp"] == 1  # bicycle fp
    assert res["by_scale"]["small"]["f1"] == 0.0

    assert res["by_scale"]["medium"]["tp"] == 1
    assert res["by_scale"]["medium"]["f1"] == 1.0


def test_merge_two_tier_nms_deduplication():
    base = [
        DetectionBox(class_name="car", confidence=0.60, x1=100, y1=100, x2=200, y2=200),
    ]
    crop = [
        # Higher confidence, highly overlapping -> should replace or suppress base
        DetectionBox(class_name="car", confidence=0.85, x1=102, y1=102, x2=198, y2=198),
        # Distinct distant car -> should be kept
        DetectionBox(class_name="car", confidence=0.70, x1=400, y1=400, x2=450, y2=450),
    ]

    merged = merge_two_tier_nms(base, crop, iou_thresh=0.50)
    assert len(merged) == 2
    confidences = [d.confidence for d in merged]
    assert 0.85 in confidences
    assert 0.70 in confidences
    assert 0.60 not in confidences  # pruned by higher-confidence crop


def test_evaluator_uninitialized_weights():
    evaluator = WeatherGuardEvaluator(weights_path=Path("non_existent_weights.pt"))
    res = evaluator.evaluate_image_stream([{"file_path": "dummy.jpg", "gt_boxes": []}])
    assert res["status"] == "MODEL_WEIGHTS_NOT_FOUND"
