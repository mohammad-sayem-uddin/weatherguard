"""Mission 18 — Independent Evaluation Pipeline Integration and Metric Validation Tests.

Validates the correctness of WeatherGuard metric calculations (Precision, Recall, F1,
101-point interpolated mAP@0.50, mAP@0.50:0.95, and scale-stratified breakdowns)
against independently constructed synthetic ground-truth fixtures.

Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""

import math
from typing import List
import pytest

from scripts.evaluate_weatherguard import (
    CANONICAL_CLASSES,
    DetectionBox,
    GroundTruthBox,
    compute_ap_101,
    compute_iou,
    compute_map,
    evaluate_predictions,
    get_object_scale,
)


# ==============================================================================
# A. Perfect Detection
# ==============================================================================

def test_independent_perfect_detection():
    """Every ground truth object has exactly one correct prediction."""
    # 3 distinct objects: 1 car (medium), 1 person (small), 1 truck (large)
    gts = [[
        GroundTruthBox(x1=50, y1=50, x2=100, y2=100, class_name="car", scale="medium", max_dim=50),
        GroundTruthBox(x1=200, y1=200, x2=225, y2=225, class_name="person", scale="small", max_dim=25),
        GroundTruthBox(x1=300, y1=300, x2=450, y2=450, class_name="truck", scale="large", max_dim=150),
    ]]
    preds = [[
        DetectionBox(x1=50, y1=50, x2=100, y2=100, confidence=0.98, class_name="car"),
        DetectionBox(x1=200, y1=200, x2=225, y2=225, confidence=0.95, class_name="person"),
        DetectionBox(x1=300, y1=300, x2=450, y2=450, confidence=0.90, class_name="truck"),
    ]]

    metrics = evaluate_predictions(preds, gts, iou_thresh=0.50)

    # Overall metrics
    assert metrics["overall"]["precision"] == 1.0
    assert metrics["overall"]["recall"] == 1.0
    assert metrics["overall"]["f1"] == 1.0
    assert metrics["overall"]["tp"] == 3
    assert metrics["overall"]["fp"] == 0
    assert metrics["overall"]["fn"] == 0

    # mAP checks
    assert metrics["map"]["map_50"] == 1.0
    assert metrics["map"]["map_50_95"] == 1.0
    assert metrics["map"]["per_class_ap_50"]["car"] == 1.0
    assert metrics["map"]["per_class_ap_50"]["person"] == 1.0
    assert metrics["map"]["per_class_ap_50"]["truck"] == 1.0

    # Scale checks
    assert metrics["by_scale"]["medium"]["f1"] == 1.0
    assert metrics["by_scale"]["small"]["f1"] == 1.0
    assert metrics["by_scale"]["large"]["f1"] == 1.0


# ==============================================================================
# B. Complete Miss
# ==============================================================================

def test_independent_complete_miss():
    """Ground truth exists, but zero predictions are returned."""
    gts = [[
        GroundTruthBox(x1=50, y1=50, x2=100, y2=100, class_name="car", scale="medium", max_dim=50),
        GroundTruthBox(x1=200, y1=200, x2=220, y2=220, class_name="bicycle", scale="small", max_dim=20),
    ]]
    preds = [[]]

    metrics = evaluate_predictions(preds, gts, iou_thresh=0.50)

    assert metrics["overall"]["precision"] == 0.0
    assert metrics["overall"]["recall"] == 0.0
    assert metrics["overall"]["f1"] == 0.0
    assert metrics["overall"]["tp"] == 0
    assert metrics["overall"]["fp"] == 0
    assert metrics["overall"]["fn"] == 2

    assert metrics["map"]["map_50"] == 0.0
    assert metrics["map"]["map_50_95"] == 0.0
    assert metrics["map"]["per_class_ap_50"]["car"] == 0.0
    assert metrics["map"]["per_class_ap_50"]["bicycle"] == 0.0


# ==============================================================================
# C. False Positives & One-to-One Matching
# ==============================================================================

def test_independent_false_positives_and_duplicates():
    """Tests background FP, duplicate predictions on single GT, class confusion, and localization errors."""
    # Single GT car
    gts = [[
        GroundTruthBox(x1=100, y1=100, x2=160, y2=160, class_name="car", scale="medium", max_dim=60)
    ]]

    preds = [[
        # Pred 1: Correct high-conf match -> TP
        DetectionBox(x1=100, y1=100, x2=160, y2=160, confidence=0.95, class_name="car"),
        # Pred 2: Duplicate on same car -> Must be FP (GT already matched)
        DetectionBox(x1=102, y1=102, x2=158, y2=158, confidence=0.85, class_name="car"),
        # Pred 3: Background clutter -> Pure FP
        DetectionBox(x1=400, y1=400, x2=450, y2=450, confidence=0.80, class_name="car"),
        # Pred 4: Class confusion (overlaps GT car, but predicted as person) -> FP
        DetectionBox(x1=100, y1=100, x2=160, y2=160, confidence=0.75, class_name="person"),
        # Pred 5: Localization failure (correct class, but IoU < 0.50) -> FP
        # GT area: 60x60 = 3600. Pred: (140, 140, 200, 200) area 3600. Inter: (140..160) = 20x20 = 400.
        # IoU = 400 / (3600 + 3600 - 400) = 400 / 6800 = 0.0588 (< 0.50) -> FP
        DetectionBox(x1=140, y1=140, x2=200, y2=200, confidence=0.70, class_name="car"),
    ]]

    metrics = evaluate_predictions(preds, gts, iou_thresh=0.50)

    # 1 TP, 4 FP, 0 FN
    assert metrics["overall"]["tp"] == 1
    assert metrics["overall"]["fp"] == 4
    assert metrics["overall"]["fn"] == 0
    assert metrics["overall"]["precision"] == round(1 / 5, 4)
    assert metrics["overall"]["recall"] == 1.0

    # Verify per-class counts
    assert metrics["by_class"]["car"]["tp"] == 1
    assert metrics["by_class"]["car"]["fp"] == 3  # duplicate, background, bad localization
    assert metrics["by_class"]["person"]["tp"] == 0
    assert metrics["by_class"]["person"]["fp"] == 1  # class confusion


# ==============================================================================
# D. Confidence Ranking Impact on Average Precision
# ==============================================================================

def test_independent_confidence_ranking_ap_impact():
    """Verifies that confidence ranking globally orders candidates and impacts 101-point AP."""
    # Ground truth: 1 car
    gts = [[
        GroundTruthBox(x1=10, y1=10, x2=50, y2=50, class_name="car", scale="medium", max_dim=40)
    ]]

    # Case 1: High-confidence TP (0.90) and low-confidence FP (0.20)
    preds_case1 = [[
        DetectionBox(x1=10, y1=10, x2=50, y2=50, confidence=0.90, class_name="car"),
        DetectionBox(x1=200, y1=200, x2=250, y2=250, confidence=0.20, class_name="car"),
    ]]
    res1 = compute_map(preds_case1, gts, target_classes=["car"])
    ap1 = res1["per_class_ap_50"]["car"]

    # Case 2: Inverted confidence: High-confidence FP (0.90) and low-confidence TP (0.20)
    preds_case2 = [[
        DetectionBox(x1=200, y1=200, x2=250, y2=250, confidence=0.90, class_name="car"),
        DetectionBox(x1=10, y1=10, x2=50, y2=50, confidence=0.20, class_name="car"),
    ]]
    res2 = compute_map(preds_case2, gts, target_classes=["car"])
    ap2 = res2["per_class_ap_50"]["car"]

    # Case 1 should have perfect AP (1.0), whereas Case 2 must be penalised (~0.50)
    assert ap1 == 1.0, f"Expected AP=1.0 when TP precedes FP, got {ap1}"
    assert ap2 <= 0.55, f"Expected AP<=0.55 when FP precedes TP, got {ap2}"
    assert ap1 > ap2, f"Confidence ranking did not correctly penalize early false positive: {ap1} vs {ap2}"


# ==============================================================================
# E. IoU Boundary Conditions
# ==============================================================================

def test_independent_iou_exact_boundaries():
    """Verifies strict boundary adherence at IoU = 0.500."""
    # GT box: (0, 0, 100, 100), Area = 10,000
    gt = GroundTruthBox(x1=0, y1=0, x2=100, y2=100, class_name="car", scale="large", max_dim=100)

    # Box with exact 0.500 IoU:
    # Let pred be (0, 0, x, 100). Intersection = 100 * x. Union = 10,000 + 100*x - 100*x = 10,000.
    # IoU = 100*x / 10,000 = x / 100. For IoU = 0.50, x = 50.
    # Wait, area of pred is 100 * x.
    # Area GT = 10,000. Area Pred = 100 * x. Inter = 100 * x.
    # Union = 10,000 + 100*x - 100*x = 10,000. IoU = 100*x / 10,000 = x / 100.
    # When x = 50: IoU is exactly 50 / 100 = 0.5000.
    # When x = 49.9: IoU is 0.4990 (< 0.50).
    # When x = 50.1: IoU is 0.5010 (> 0.50).

    iou_exact = compute_iou((0, 0, 50, 100), (0, 0, 100, 100))
    iou_below = compute_iou((0, 0, 49.9, 100), (0, 0, 100, 100))
    iou_above = compute_iou((0, 0, 50.1, 100), (0, 0, 100, 100))

    assert math.isclose(iou_exact, 0.50, rel_tol=1e-5)
    assert iou_below < 0.50
    assert iou_above > 0.50

    # Test that exactly 0.50 is matched as TP (best_iou >= iou_thresh)
    res_exact = evaluate_predictions([[DetectionBox(x1=0, y1=0, x2=50, y2=100, confidence=0.9, class_name="car")]], [[gt]], iou_thresh=0.50)
    assert res_exact["overall"]["tp"] == 1
    assert res_exact["overall"]["fp"] == 0

    # Test that 0.499 is counted as FP
    res_below = evaluate_predictions([[DetectionBox(x1=0, y1=0, x2=49.9, y2=100, confidence=0.9, class_name="car")]], [[gt]], iou_thresh=0.50)
    assert res_below["overall"]["tp"] == 0
    assert res_below["overall"]["fp"] == 1

    # Test that 0.501 is counted as TP
    res_above = evaluate_predictions([[DetectionBox(x1=0, y1=0, x2=50.1, y2=100, confidence=0.9, class_name="car")]], [[gt]], iou_thresh=0.50)
    assert res_above["overall"]["tp"] == 1
    assert res_above["overall"]["fp"] == 0


# ==============================================================================
# F. Empty and Invalid Cases
# ==============================================================================

def test_independent_empty_and_invalid_cases():
    """Tests zero ground truth, zero predictions, invalid coordinates, and unsupported class IDs."""
    # 1. Both empty
    res_empty = evaluate_predictions([[]], [[]])
    assert res_empty["overall"]["f1"] == 0.0
    assert res_empty["overall"]["precision"] == 0.0
    assert res_empty["overall"]["recall"] == 0.0
    assert res_empty["map"]["map_50"] == 0.0
    assert res_empty["map"]["map_50_95"] == 0.0

    # 2. Zero-area box (width 0 or height 0)
    gt = GroundTruthBox(x1=10, y1=10, x2=50, y2=50, class_name="car", scale="medium", max_dim=40)
    zero_box = DetectionBox(x1=20, y1=20, x2=20, y2=40, confidence=0.9, class_name="car")  # width = 0
    res_zero = evaluate_predictions([[zero_box]], [[gt]])
    assert res_zero["overall"]["tp"] == 0
    assert res_zero["overall"]["fp"] == 1
    assert res_zero["overall"]["fn"] == 1

    # 3. Inverted bounding box (x2 < x1 or y2 < y1)
    inv_box = DetectionBox(x1=50, y1=50, x2=10, y2=10, confidence=0.9, class_name="car")
    res_inv = evaluate_predictions([[inv_box]], [[gt]])
    assert res_inv["overall"]["tp"] == 0
    assert res_inv["overall"]["fp"] == 1

    # 4. Out-of-vocabulary class ("airplane", "zebra")
    # Must NOT be silently mapped to "car" or canonical classes; must be counted as FP in overall
    oov_box = DetectionBox(x1=10, y1=10, x2=50, y2=50, confidence=0.9, class_name="airplane")
    res_oov = evaluate_predictions([[oov_box]], [[gt]])
    assert res_oov["overall"]["tp"] == 0
    assert res_oov["overall"]["fp"] == 1
    assert res_oov["overall"]["fn"] == 1
    # Check that canonical classes were not corrupted
    for cls in CANONICAL_CLASSES:
        assert res_oov["by_class"][cls]["tp"] == 0


# ==============================================================================
# G. Multi-Threshold Monotonicity of mAP@[0.50:0.95]
# ==============================================================================

def test_independent_map_multithreshold_monotonicity():
    """Verifies that mAP monotonically decreases or stays equal as IoU threshold increases."""
    # Synthetic multi-image predictions with varying IoU alignment
    gts = [
        [GroundTruthBox(x1=10, y1=10, x2=60, y2=60, class_name="car", scale="medium", max_dim=50)],
        [GroundTruthBox(x1=100, y1=100, x2=140, y2=140, class_name="person", scale="medium", max_dim=40)],
    ]
    # Car has IoU ~ 0.70, person has IoU ~ 0.55
    preds = [
        [DetectionBox(x1=15, y1=15, x2=65, y2=65, confidence=0.90, class_name="car")],
        [DetectionBox(x1=110, y1=110, x2=150, y2=150, confidence=0.85, class_name="person")],
    ]

    map_res = compute_map(preds, gts, target_classes=["car", "person"])
    assert "map_50" in map_res
    assert "map_50_95" in map_res
    # mAP@0.50 must be >= mAP@0.50:0.95 because tighter IoU thresholds reject more predictions
    assert map_res["map_50"] >= map_res["map_50_95"]
    assert map_res["map_50"] > 0.0
