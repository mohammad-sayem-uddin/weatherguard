"""Unit tests for WeatherGuard Mission 4 (Adaptive Inference & Error Recovery)."""

import pytest
import numpy as np

from src.adaptive_routing import (
    ImageRoutingFeatures,
    extract_routing_features,
    policy_fixed_640,
    policy_fixed_832,
    policy_fixed_1280,
    policy_small_box_trigger_k1,
    policy_small_box_trigger_k2,
    policy_low_conf_trigger_k2,
    policy_density_trigger_k8,
    policy_two_stage_hierarchical,
    policy_hierarchical_fast_832,
)
from src.crop_refinement import (
    CropRegion,
    generate_candidate_crops,
    transform_crop_to_original,
    merge_and_nms,
    iou_xyxy,
)


def test_extract_routing_features_empty():
    feats = extract_routing_features([], image_id="img_001")
    assert feats.image_id == "img_001"
    assert feats.n_small_boxes == 0
    assert feats.n_medium_boxes == 0
    assert feats.n_large_boxes == 0
    assert feats.n_low_conf == 0
    assert feats.has_small_box is False


def test_extract_routing_features_scales():
    preds = [
        {"label": "car", "common_label": "Car", "score": 0.80, "box_xyxy": [0, 0, 20, 20]},      # area 400 (<1024, small)
        {"label": "car", "common_label": "Car", "score": 0.50, "box_xyxy": [0, 0, 50, 50]},      # area 2500 (med)
        {"label": "truck", "common_label": "Truck", "score": 0.90, "box_xyxy": [0, 0, 150, 150]}, # area 22500 (large)
        {"label": "person", "common_label": "Person", "score": 0.10, "box_xyxy": [0, 0, 10, 10]}, # score 0.10 in [0.05, 0.25) low conf
    ]
    feats = extract_routing_features(preds, min_score=0.15)
    assert feats.n_small_boxes == 1
    assert feats.n_medium_boxes == 1
    assert feats.n_large_boxes == 1
    assert feats.n_low_conf == 1
    assert feats.has_small_box is True


def test_routing_policies_logic():
    # Feats with 0 small box
    f_empty = ImageRoutingFeatures(
        image_id="1", n_small_boxes=0, n_medium_boxes=1, n_large_boxes=2,
        n_low_conf=0, n_total_candidates=3, min_box_area=2000, mean_box_area=5000,
        has_small_box=False, uncertainty_score=0.0
    )
    assert policy_fixed_640(f_empty) == 640
    assert policy_fixed_832(f_empty) == 832
    assert policy_fixed_1280(f_empty) == 1280
    assert policy_small_box_trigger_k1(f_empty) == 640
    assert policy_two_stage_hierarchical(f_empty) == 640

    # Feats with 1 small box
    f_one_small = ImageRoutingFeatures(
        image_id="2", n_small_boxes=1, n_medium_boxes=1, n_large_boxes=1,
        n_low_conf=1, n_total_candidates=3, min_box_area=400, mean_box_area=3000,
        has_small_box=True, uncertainty_score=0.33
    )
    assert policy_small_box_trigger_k1(f_one_small) == 1280
    assert policy_small_box_trigger_k2(f_one_small) == 640
    assert policy_two_stage_hierarchical(f_one_small) == 832  # routes to tier 2
    assert policy_hierarchical_fast_832(f_one_small) == 832

    # Feats with 2 small boxes
    f_two_small = ImageRoutingFeatures(
        image_id="3", n_small_boxes=2, n_medium_boxes=0, n_large_boxes=1,
        n_low_conf=0, n_total_candidates=3, min_box_area=300, mean_box_area=1500,
        has_small_box=True, uncertainty_score=0.0
    )
    assert policy_small_box_trigger_k2(f_two_small) == 1280
    assert policy_two_stage_hierarchical(f_two_small) == 1280  # routes to tier 3


def test_crop_refinement_coordinate_projection():
    crop = CropRegion(x1=100.0, y1=100.0, x2=500.0, y2=500.0, candidate_indices=[0])
    # Crop size = 400x400. Model input = 640x640.
    box_in_crop = [0.0, 0.0, 640.0, 640.0]
    orig_box = transform_crop_to_original(box_in_crop, crop, crop_input_size=(640, 640))
    assert orig_box == [100.0, 100.0, 500.0, 500.0]


def test_crop_refinement_boundary_clamp():
    crops = generate_candidate_crops(
        [{"label": "car", "common_label": "Car", "score": 0.5, "box_xyxy": [1, 1, 10, 10]}],
        image_shape=(1280, 720),
        max_crops=1,
        context_margin=0.20,
    )
    assert len(crops) == 1
    assert crops[0].x1 >= 0.0
    assert crops[0].y1 >= 0.0
    assert crops[0].x2 <= 1280.0
    assert crops[0].y2 <= 720.0
