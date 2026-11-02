"""Unit tests for Dynamic Router (src/dynamic_router.py)."""

import pytest
from src.dynamic_router import (
    RoutingFeatures,
    extract_routing_features,
    PolicyFixedBaseline,
    PolicyLowConfidence,
    PolicySmallObject,
    PolicyDetectionDensity,
    PolicyCombinedUncertainty,
)


def test_extract_routing_features_empty():
    feats = extract_routing_features([], score_threshold=0.20)
    assert feats.num_detections == 0
    assert feats.mean_conf == 0.0
    assert feats.num_small == 0
    assert feats.small_ratio == 0.0
    assert feats.density == 0.0


def test_extract_routing_features_with_detections():
    preds = [
        # Small box with borderline confidence
        {"common_label": "Car", "score": 0.25, "box_xyxy": [10.0, 10.0, 30.0, 30.0]},  # area 400 < 1024
        # Medium box with high confidence
        {"common_label": "Person", "score": 0.85, "box_xyxy": [100.0, 100.0, 150.0, 160.0]},  # area 3000
    ]
    feats = extract_routing_features(preds, score_threshold=0.20)
    assert feats.num_detections == 2
    assert abs(feats.mean_conf - 0.55) < 1e-4
    assert feats.num_small == 1
    assert abs(feats.small_ratio - 0.50) < 1e-4
    assert feats.num_candidates_low == 1  # 0.25 is in [0.10, 0.30]


def test_policy_small_object():
    policy_1 = PolicySmallObject(k_small=1)
    policy_2 = PolicySmallObject(k_small=2)

    feats_empty = RoutingFeatures(num_small=0)
    assert policy_1.route(feats_empty) is False
    assert policy_2.route(feats_empty) is False

    feats_1_small = RoutingFeatures(num_small=1)
    assert policy_1.route(feats_1_small) is True
    assert policy_2.route(feats_1_small) is False

    feats_2_small = RoutingFeatures(num_small=2)
    assert policy_1.route(feats_2_small) is True
    assert policy_2.route(feats_2_small) is True


def test_policy_fixed_and_density():
    policy_fixed = PolicyFixedBaseline()
    feats = RoutingFeatures(num_small=5, num_candidates_low=10, density=3.0)
    assert policy_fixed.route(feats) is False

    policy_density = PolicyDetectionDensity(min_candidates=15)
    feats_sparse = RoutingFeatures(density=1.0)  # 10 candidates
    assert policy_density.route(feats_sparse) is False
    feats_dense = RoutingFeatures(density=1.6)   # 16 candidates
    assert policy_density.route(feats_dense) is True


def test_policy_combined_uncertainty():
    policy_comb = PolicyCombinedUncertainty(
        w_small=1.5,
        w_low=1.0,
        w_density=0.5,
        w_max_conf=1.0,
        threshold=2.0,
    )
    # High max_conf dampens escalation
    feats_confident = RoutingFeatures(
        num_small=1,            # 1.5
        num_candidates_low=0,   # 0.0
        density=1.0,            # 0.5
        max_conf=0.95,          # -0.95 -> total = 1.05 < 2.0
    )
    assert policy_comb.route(feats_confident) is False

    # Low max_conf allows escalation
    feats_ambiguous = RoutingFeatures(
        num_small=1,            # 1.5
        num_candidates_low=1,   # 1.0
        density=1.0,            # 0.5
        max_conf=0.35,          # -0.35 -> total = 2.65 >= 2.0
    )
    assert policy_comb.route(feats_ambiguous) is True
