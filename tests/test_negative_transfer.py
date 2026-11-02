"""Unit tests for src/negative_transfer.py."""

import pytest
from src.negative_transfer import (
    get_scale_category,
    match_image_predictions,
    analyze_image_transitions,
    summarize_transitions,
    InstanceTransition,
)


def test_scale_category():
    # area < 1024
    assert get_scale_category([0, 0, 30, 30]) == "small"  # 900
    # 1024 <= area < 9216
    assert get_scale_category([0, 0, 50, 50]) == "medium"  # 2500
    # area >= 9216
    assert get_scale_category([0, 0, 100, 100]) == "large"  # 10000


def test_transition_states():
    # Ground truth: 2 cars
    gt_objects = [
        {"class_name": "Car", "box_xyxy": [10, 10, 50, 50]},  # gt 0
        {"class_name": "Car", "box_xyxy": [100, 100, 150, 150]},  # gt 1
        {"class_name": "Person", "box_xyxy": [200, 200, 230, 250]},  # gt 2
    ]

    # Pretrained predictions:
    # - matches gt 0 (Car)
    # - matches gt 1 (Car)
    # - FP background (Car at [300, 300, 350, 350])
    pre_preds = [
        {"common_label": "Car", "score": 0.85, "box_xyxy": [11, 11, 49, 49]},
        {"common_label": "Car", "score": 0.70, "box_xyxy": [102, 101, 148, 149]},
        {"common_label": "Car", "score": 0.35, "box_xyxy": [300, 300, 350, 350]},
    ]

    # Adapted predictions:
    # - matches gt 0 (Car) -> State 1 (Correct to Correct)
    # - misses gt 1 -> State 2 (Correct to Missed)
    # - matches gt 2 (Person) -> State 3 (Missed to Recovered)
    # - new FP at [400, 400, 450, 450] -> State 5 (Additional FP)
    # Note: Pre FP [300, 300, 350, 350] is absent -> State 6 (FP to Corrected)
    adapt_preds = [
        {"common_label": "Car", "score": 0.90, "box_xyxy": [10, 10, 50, 50]},
        {"common_label": "Person", "score": 0.65, "box_xyxy": [200, 200, 230, 250]},
        {"common_label": "Bus", "score": 0.40, "box_xyxy": [400, 400, 450, 450]},
    ]

    transitions = analyze_image_transitions(
        image_id="img_test_001",
        weather="rainy",
        gt_objects=gt_objects,
        pre_preds=pre_preds,
        adapt_preds=adapt_preds,
        threshold=0.20,
        iou_thresh=0.50,
    )

    t_types = {t.transition_type for t in transitions}
    assert "1_correct_to_correct" in t_types
    assert "2_correct_to_missed" in t_types
    assert "3_missed_to_recovered" in t_types
    assert "5_additional_false_positive" in t_types
    assert "6_fp_to_corrected" in t_types

    summary = summarize_transitions(transitions)
    assert summary["total_transitions"] == len(transitions)
    assert summary["state_counts"]["1_correct_to_correct"] == 1
    assert summary["state_counts"]["2_correct_to_missed"] == 1
    assert summary["state_counts"]["3_missed_to_recovered"] == 1
    assert summary["state_counts"]["5_additional_false_positive"] == 1
    assert summary["state_counts"]["6_fp_to_corrected"] == 1


def test_retained_fp_and_wrong_class():
    gt_objects = [
        {"class_name": "Truck", "box_xyxy": [50, 50, 120, 120]},
    ]
    # Pretrained detects Truck correctly
    pre_preds = [
        {"common_label": "Truck", "score": 0.75, "box_xyxy": [51, 51, 119, 119]},
        {"common_label": "Car", "score": 0.30, "box_xyxy": [300, 300, 350, 350]},
    ]
    # Adapted predicts Bus for the Truck (wrong class) and also predicts the exact same Car FP
    adapt_preds = [
        {"common_label": "Bus", "score": 0.80, "box_xyxy": [50, 50, 120, 120]},
        {"common_label": "Car", "score": 0.45, "box_xyxy": [301, 301, 349, 349]},
    ]

    transitions = analyze_image_transitions(
        image_id="img_test_002",
        weather="snowy",
        gt_objects=gt_objects,
        pre_preds=pre_preds,
        adapt_preds=adapt_preds,
        threshold=0.20,
        iou_thresh=0.50,
    )

    t_types = {t.transition_type for t in transitions}
    assert "4_correct_to_wrong_class" in t_types
    assert "7_retained_false_positive" in t_types
