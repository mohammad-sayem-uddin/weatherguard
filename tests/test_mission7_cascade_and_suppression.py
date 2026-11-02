"""Unit tests for Cascade merging and False Alarm Suppression (src/conditional_cascade.py, src/false_alarm_suppression.py)."""

import pytest
from src.conditional_cascade import (
    CropRegion,
    construct_candidate_crops,
    map_crop_to_image_coords,
    merge_cascade_predictions,
)
from src.false_alarm_suppression import (
    filter_aspect_ratio_anomalies,
    filter_road_surface_reflections,
    filter_cross_resolution_unconfirmed,
)


def test_merge_cascade_predictions_priority():
    base_preds = [
        {"common_label": "Car", "score": 0.40, "box_xyxy": [10.0, 10.0, 50.0, 50.0]},
        {"common_label": "Person", "score": 0.50, "box_xyxy": [100.0, 100.0, 130.0, 180.0]},
    ]
    # Refined detection overlapping the Car box with higher resolution
    escalated_preds = [
        {"common_label": "Car", "score": 0.65, "box_xyxy": [12.0, 11.0, 49.0, 51.0]},
    ]
    merged = merge_cascade_predictions(base_preds, escalated_preds, nms_iou_threshold=0.50)
    # The overlapping Car from base_preds should be suppressed in favor of escalated_preds
    assert len(merged) == 2
    car_det = [p for p in merged if p["common_label"] == "Car"][0]
    assert car_det["score"] == 0.65


def test_map_crop_to_image_coords():
    region = CropRegion(x1=100.0, y1=100.0, x2=300.0, y2=300.0, orig_pred_score=0.3, orig_label="Car")
    # A box located in the center of a 640x640 crop
    crop_box = [160.0, 160.0, 480.0, 480.0]
    img_box = map_crop_to_image_coords(crop_box, region, crop_img_size=640)
    # Width of crop region is 200. scale is 200/640 = 0.3125
    # 100 + 160 * 0.3125 = 150
    assert abs(img_box[0] - 150.0) < 1e-4
    assert abs(img_box[1] - 150.0) < 1e-4
    assert abs(img_box[2] - 250.0) < 1e-4
    assert abs(img_box[3] - 250.0) < 1e-4


def test_filter_aspect_ratio_anomalies():
    preds = [
        # Normal vehicle
        {"common_label": "Car", "score": 0.30, "box_xyxy": [10, 10, 50, 40]},  # w=40, h=30, aspect=1.33
        # Extreme vertical sliver (noise/pole)
        {"common_label": "Car", "score": 0.30, "box_xyxy": [10, 10, 12, 100]},  # w=2, h=90, aspect = 2/90 < 0.25
        # Extreme horizontal sliver (noise/cable)
        {"common_label": "Car", "score": 0.30, "box_xyxy": [10, 10, 200, 12]},  # w=190, h=2, aspect = 190/2 > 4.5
        # High confidence override (should be preserved despite aspect ratio)
        {"common_label": "Car", "score": 0.85, "box_xyxy": [10, 10, 200, 12]},
    ]
    filtered = filter_aspect_ratio_anomalies(preds, min_aspect_ratio=0.25, max_aspect_ratio=4.5, high_conf_override=0.50)
    assert len(filtered) == 2
    scores = [p["score"] for p in filtered]
    assert 0.30 in scores
    assert 0.85 in scores


def test_filter_road_surface_reflections():
    preds = [
        # Normal vehicle in mid-scene
        {"common_label": "Car", "score": 0.30, "box_xyxy": [100, 200, 200, 300]},
        # Low-confidence reflection at bottom 20% of frame (height 720, zone_top = 576, y1 = 600 > 576, score < 0.28)
        {"common_label": "Car", "score": 0.22, "box_xyxy": [100, 600, 180, 680]},
        # High-confidence true vehicle in bottom zone (should be preserved)
        {"common_label": "Car", "score": 0.65, "box_xyxy": [250, 600, 350, 710]},
    ]
    filtered = filter_road_surface_reflections(preds, img_height=720, bottom_zone_frac=0.20, min_score_in_zone=0.28)
    assert len(filtered) == 2
    scores = [p["score"] for p in filtered]
    assert 0.30 in scores
    assert 0.65 in scores
    assert 0.22 not in scores
