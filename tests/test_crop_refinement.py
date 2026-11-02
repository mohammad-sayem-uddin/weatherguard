"""Unit tests for Targeted Region Refinement (WeatherGuard Mission 4)."""

import pytest
from src.crop_refinement import (
    CropRegion,
    generate_candidate_crops,
    transform_crop_to_original,
    merge_and_nms,
    iou_xyxy,
)


def test_crop_generation_empty():
    """No candidate detections should produce 0 crops."""
    crops = generate_candidate_crops([], image_shape=(1280, 720))
    assert len(crops) == 0


def test_crop_generation_boundary_clamping():
    """Crops touching or exceeding boundaries must clamp within [0, W] and [0, H]."""
    preds = [
        {
            "label": "car",
            "common_label": "Car",
            "score": 0.50,
            "box_xyxy": [10.0, 10.0, 30.0, 30.0],  # near top-left
        },
        {
            "label": "person",
            "common_label": "Person",
            "score": 0.40,
            "box_xyxy": [1260.0, 700.0, 1275.0, 715.0],  # near bottom-right
        },
    ]
    crops = generate_candidate_crops(preds, image_shape=(1280, 720), max_crops=2)
    assert len(crops) <= 2
    for c in crops:
        assert c.x1 >= 0.0
        assert c.y1 >= 0.0
        assert c.x2 <= 1280.0
        assert c.y2 <= 720.0
        assert c.width > 0.0
        assert c.height > 0.0


def test_transform_crop_to_original_math():
    """Verify exact coordinate scaling and translation."""
    crop_region = CropRegion(x1=200.0, y1=100.0, x2=600.0, y2=500.0, candidate_indices=[0])
    # Crop size is 400x400. Model input is 640x640.
    # Scale factor = 400 / 640 = 0.625
    # Box inside crop: [160, 160, 320, 320]
    # Expected in native: x1 = 200 + 160*0.625 = 300, y1 = 100 + 160*0.625 = 200
    # Expected x2 = 200 + 320*0.625 = 400, y2 = 100 + 320*0.625 = 300
    crop_box = [160.0, 160.0, 320.0, 320.0]
    orig_box = transform_crop_to_original(crop_box, crop_region, crop_input_size=(640, 640))

    assert pytest.approx(orig_box[0], 1e-4) == 300.0
    assert pytest.approx(orig_box[1], 1e-4) == 200.0
    assert pytest.approx(orig_box[2], 1e-4) == 400.0
    assert pytest.approx(orig_box[3], 1e-4) == 300.0


def test_transform_crop_to_original_identity():
    """When crop region matches full image and input resolution, transformation is identity."""
    crop_region = CropRegion(x1=0.0, y1=0.0, x2=640.0, y2=640.0, candidate_indices=[])
    crop_box = [50.0, 60.0, 120.0, 180.0]
    orig_box = transform_crop_to_original(crop_box, crop_region, crop_input_size=(640, 640))
    assert orig_box == crop_box


def test_iou_xyxy():
    """Verify IoU calculation on known boxes."""
    b1 = [0.0, 0.0, 10.0, 10.0]  # area 100
    b2 = [5.0, 0.0, 15.0, 10.0]  # area 100, intersection 5x10 = 50, union = 150 -> IoU 1/3
    assert pytest.approx(iou_xyxy(b1, b2), 1e-4) == (50.0 / 150.0)

    # Disjoint
    b3 = [20.0, 20.0, 30.0, 30.0]
    assert iou_xyxy(b1, b3) == 0.0


def test_merge_and_nms_suppression():
    """Verify higher confidence duplicate is retained and lower is suppressed."""
    base_preds = [
        {
            "label": "car",
            "common_label": "Car",
            "score": 0.30,
            "box_xyxy": [100.0, 100.0, 200.0, 200.0],
        }
    ]
    crop_preds = [
        {
            "label": "car",
            "common_label": "Car",
            "score": 0.85,  # refined prediction with higher confidence
            "box_xyxy": [102.0, 101.0, 198.0, 199.0],  # near identical box
        },
        {
            "label": "car",
            "common_label": "Car",
            "score": 0.60,
            "box_xyxy": [500.0, 500.0, 600.0, 600.0],  # non-overlapping new box
        },
    ]

    merged = merge_and_nms(base_preds, crop_preds, nms_iou_threshold=0.50)
    assert len(merged) == 2
    # The overlapping car should have the refined higher score of 0.85
    overlapping_car = [p for p in merged if p["box_xyxy"][0] < 300.0][0]
    assert overlapping_car["score"] == 0.85
