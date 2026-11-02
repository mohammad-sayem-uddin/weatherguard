"""Unit tests for src/model_fusion.py."""

import numpy as np
import pytest

from src.model_fusion import (
    nms_boxes,
    fuse_predictions_simple_ensemble,
    fuse_predictions_wbf,
    fuse_predictions_selective_fallback,
)


def test_nms_boxes():
    boxes = [
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 11.0, 49.0, 48.0],  # Overlaps box 0 with high IoU
        [100.0, 100.0, 150.0, 150.0],  # Disjoint
    ]
    scores = [0.85, 0.70, 0.90]
    kept = nms_boxes(boxes, scores, iou_thresh=0.50)
    assert 2 in kept  # Box 2 kept (score 0.90)
    assert 0 in kept  # Box 0 kept (score 0.85)
    assert 1 not in kept  # Box 1 suppressed by Box 0


def test_fuse_predictions_wbf_agreement():
    preds_pre = [
        {"common_label": "Car", "score": 0.80, "box_xyxy": [10.0, 10.0, 50.0, 50.0]},
    ]
    preds_ft = [
        {"common_label": "Car", "score": 0.60, "box_xyxy": [12.0, 12.0, 52.0, 52.0]},
    ]
    fused, stats = fuse_predictions_wbf(preds_pre, preds_ft, threshold=0.20, iou_thresh=0.50)
    assert len(fused) == 1
    assert stats["agreed"] == 1
    assert fused[0]["models_agreed"] is True
    # Fused score is average: (0.8 + 0.6) / 2 = 0.70
    assert abs(fused[0]["score"] - 0.70) < 1e-4
    # Weighted box coordinates lie strictly between the two
    assert 10.0 < fused[0]["box_xyxy"][0] < 12.0


def test_fuse_predictions_wbf_unconfirmed_penalty():
    preds_pre = []
    # Adapted hallucinated an FP alone
    preds_ft = [
        {"common_label": "Bus", "score": 0.25, "box_xyxy": [20.0, 20.0, 80.0, 80.0]},
    ]
    # unconfirmed penalty 0.70 -> score becomes 0.25 * 0.70 = 0.175 (< threshold 0.20)
    fused, stats = fuse_predictions_wbf(
        preds_pre, preds_ft, threshold=0.20, unconfirmed_penalty=0.70
    )
    assert len(fused) == 0  # Successfully suppressed below operating threshold!
    assert stats["ft_only"] == 1


def test_fuse_predictions_selective_fallback():
    preds_pre = [
        # Small object (area = 20 * 20 = 400 < 1024)
        {"common_label": "Person", "score": 0.40, "box_xyxy": [5.0, 5.0, 25.0, 25.0]},
        # Large object (area = 100 * 100 = 10000 >= 1024)
        {"common_label": "Car", "score": 0.50, "box_xyxy": [100.0, 100.0, 200.0, 200.0]},
    ]
    preds_ft = [
        # High confidence salient weather target (score 0.75 >= 0.40)
        {"common_label": "Car", "score": 0.75, "box_xyxy": [102.0, 101.0, 198.0, 202.0]},
    ]
    fused, stats = fuse_predictions_selective_fallback(preds_pre, preds_ft, threshold=0.20)
    assert stats["pre_small"] == 1
    assert stats["ft_salient"] == 1
    labels = [p["common_label"] for p in fused]
    assert "Person" in labels
    assert "Car" in labels
