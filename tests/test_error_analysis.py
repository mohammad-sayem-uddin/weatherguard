"""Tests for WeatherGuard Error Taxonomy and Per-Condition Analysis Framework."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.error_analysis import (
    CategorizedGroundTruth,
    CategorizedPrediction,
    ErrorTaxonomyAnalyzer,
    categorize_image_errors,
    compute_box_iou,
    extract_box_contrast,
    get_scale_bucket,
)


def test_compute_box_iou():
    # Disjoint boxes
    assert compute_box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # Identical boxes
    assert compute_box_iou((10, 10, 50, 50), (10, 10, 50, 50)) == 1.0
    # 50% intersection
    # Box 1: (0, 0, 10, 10) area 100
    # Box 2: (0, 0, 10, 5) area 50, inter 50, union 100 -> 0.5
    assert compute_box_iou((0, 0, 10, 10), (0, 0, 10, 5)) == 0.5


def test_get_scale_bucket():
    assert get_scale_bucket(10, 15) == "ultra_small"
    assert get_scale_bucket(16, 25) == "small"
    assert get_scale_bucket(32, 80) == "medium"
    assert get_scale_bucket(100, 200) == "large"


def test_extract_box_contrast():
    # Uniform image -> std is 0.0
    img_flat = np.ones((100, 100, 3), dtype=np.uint8) * 128
    assert extract_box_contrast(img_flat, (10, 10, 50, 50)) == 0.0

    # High contrast image (checkerboard / gradient)
    img_grad = np.zeros((100, 100, 3), dtype=np.uint8)
    img_grad[:, 50:] = 255
    contrast = extract_box_contrast(img_grad, (10, 10, 90, 90))
    assert contrast is not None
    assert contrast > 50.0

    # None image handling
    assert extract_box_contrast(None, (0, 0, 10, 10)) is None


def test_categorize_image_errors_all_categories():
    gts = [
        {"class_name": "car", "box_xyxy": (10, 10, 50, 50)},       # matched by TP and Duplicate
        {"class_name": "person", "box_xyxy": (100, 100, 120, 120)}, # matched with wrong class -> Class_confusion
        {"class_name": "bicycle", "box_xyxy": (200, 200, 250, 250)}, # matched with low IoU -> Localization
        {"class_name": "truck", "box_xyxy": (300, 300, 380, 380)},   # never proposed -> Missed_unproposed
    ]

    preds = [
        {"class_name": "car", "score": 0.95, "box_xyxy": (10, 10, 50, 50)},         # TP
        {"class_name": "car", "score": 0.85, "box_xyxy": (12, 12, 48, 48)},         # Duplicate
        {"class_name": "truck", "score": 0.80, "box_xyxy": (101, 101, 119, 119)},   # Class_confusion
        {"class_name": "bicycle", "score": 0.70, "box_xyxy": (220, 220, 260, 260)}, # Localization
        {"class_name": "car", "score": 0.65, "box_xyxy": (500, 500, 550, 550)},     # FP_background
    ]

    cat_preds, cat_gts = categorize_image_errors(preds, gts, iou_thresh=0.50, loc_thresh=0.10)

    pred_map = {p.category: p for p in cat_preds}
    assert "TP" in pred_map
    assert "Duplicate" in pred_map
    assert "Class_confusion" in pred_map
    assert "Localization" in pred_map
    assert "FP_background" in pred_map

    gt_map = {g.class_name: g.category for g in cat_gts}
    assert gt_map["car"] == "Detected"
    assert gt_map["person"] == "Missed_class"
    assert gt_map["bicycle"] == "Missed_localization"
    assert gt_map["truck"] == "Missed_unproposed"


def test_error_taxonomy_analyzer_aggregation(tmp_path: Path):
    analyzer = ErrorTaxonomyAnalyzer(iou_thresh=0.50, loc_thresh=0.10)

    # Image 1: Foggy, 1 ultra-small unproposed GT
    analyzer.add_image_result(
        image_id="img_fog_1",
        predictions=[],
        ground_truths=[{"class_name": "car", "box_xyxy": (10, 10, 22, 22)}],  # max_dim = 12 (<16 -> ultra_small)
        condition="fog",
    )

    # Image 2: Day, 1 small detected TP
    analyzer.add_image_result(
        image_id="img_day_1",
        predictions=[{"class_name": "person", "score": 0.9, "box_xyxy": (50, 50, 75, 75)}], # 25 (<32 -> small)
        ground_truths=[{"class_name": "person", "box_xyxy": (50, 50, 75, 75)}],
        condition="day",
    )

    summary = analyzer.compute_summary()
    assert summary["total_images"] == 2
    assert summary["total_predictions"] == 1
    assert summary["total_ground_truths"] == 2

    # Scale check
    assert summary["scale_analysis"]["ultra_small"]["gt_count"] == 1
    assert summary["scale_analysis"]["ultra_small"]["recall"] == 0.0
    assert summary["scale_analysis"]["ultra_small"]["unproposed_fraction"] == 1.0

    assert summary["scale_analysis"]["small"]["gt_count"] == 1
    assert summary["scale_analysis"]["small"]["recall"] == 1.0

    # Condition check
    assert "fog" in summary["condition_analysis"]
    assert "day" in summary["condition_analysis"]
    assert summary["condition_analysis"]["fog"]["images"] == 1
    assert summary["condition_analysis"]["day"]["images"] == 1

    # Markdown export
    md = analyzer.to_markdown()
    assert "# WeatherGuard Error Taxonomy Analysis" in md
    assert "Fog" in md
    assert "Day" in md
    assert "ultra_small" in md

    # JSON export
    json_path = tmp_path / "taxonomy_test.json"
    analyzer.save_json(json_path)
    assert json_path.exists()
    with open(json_path, "r") as f:
        loaded = json.load(f)
    assert loaded["total_images"] == 2
