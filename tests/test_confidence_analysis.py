"""Unit tests for src/confidence_analysis.py."""

import numpy as np
import pytest

from src.confidence_analysis import (
    compute_ece_and_brier,
    sweep_confidence_thresholds,
)


def test_ece_perfect_calibration():
    # Construct calibrated data: 100 samples with confidence 0.8, 80 are positive
    conf = np.array([0.8] * 100)
    labels = np.array([1.0] * 80 + [0.0] * 20)
    res = compute_ece_and_brier(conf, labels, n_bins=10)
    assert abs(res["ece"]) < 1e-4
    assert res["n_samples"] == 100
    assert "brier_score" in res


def test_ece_extreme_miscalibration():
    # Predict 1.0 confidence for completely false labels
    conf = np.array([1.0] * 50)
    labels = np.array([0.0] * 50)
    res = compute_ece_and_brier(conf, labels, n_bins=10)
    assert abs(res["ece"] - 1.0) < 1e-4
    assert abs(res["brier_score"] - 1.0) < 1e-4


def test_sweep_confidence_thresholds():
    eval_records = [
        {
            "image_id": "img_01",
            "ground_truth": [{"class_name": "Car", "box_xyxy": [10, 10, 50, 50]}],
            "predictions": [
                {"common_label": "Car", "score": 0.85, "box_xyxy": [10, 10, 50, 50]},
                {"common_label": "Car", "score": 0.25, "box_xyxy": [100, 100, 150, 150]},
            ],
        }
    ]
    res = sweep_confidence_thresholds(eval_records, thresholds=[0.20, 0.50, 0.90])
    assert len(res["threshold_curve"]) == 3
    # At th=0.50, 1 TP and 0 FP -> F1 = 1.0
    curve_by_th = {pt["threshold"]: pt for pt in res["threshold_curve"]}
    assert curve_by_th[0.50]["tp"] == 1
    assert curve_by_th[0.50]["fp"] == 0
    assert curve_by_th[0.50]["f1"] == 1.0
    assert res["best_threshold"] == 0.50
