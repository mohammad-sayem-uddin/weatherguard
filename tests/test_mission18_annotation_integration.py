"""Mission 18 — Independent Annotation Quality and Coordinate Integration Tests.

Validates Gemini Flash coordinate conversions, multi-schema adapters, canonical class
mappings, geometric non-inversion, bounding box normalization reversibility, and
recalculated object scale taxonomy for WeatherGuard-D1.

Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""

import math
from pathlib import Path
import pytest

from scripts.validate_d1_annotations import (
    CLASS_NAME_TO_ID,
    VALID_CLASSES,
    WeatherGuardD1AnnotationValidator,
    classify_object_size,
    compute_iou_xyxy,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "weatherguard_d1.yaml"


@pytest.fixture
def validator():
    return WeatherGuardD1AnnotationValidator(config_path=CONFIG_PATH)


# ==============================================================================
# 1. Multi-Schema Coordinate Format Parsing & Non-Inversion
# ==============================================================================

def test_canonical_dict_schema(validator):
    """Verifies standard dictionary with [x_min, y_min, x_max, y_max] in [0..1000]."""
    raw = {
        "image_id": "test_dict_01",
        "annotations": [
            {
                "class_id": 3,
                "class_name": "car",
                "x_min": 100,
                "y_min": 200,
                "x_max": 300,
                "y_max": 400,
                "confidence": 0.95,
            }
        ]
    }
    res = validator.validate_single_annotation_json(raw)
    assert res.is_valid is True
    assert len(res.valid_boxes) == 1
    b = res.valid_boxes[0]

    # Target image resolution is 1280 x 720
    # x1_px = 100 * 1280 / 1000 = 128.0 px
    # y1_px = 200 * 720 / 1000 = 144.0 px
    # x2_px = 300 * 1280 / 1000 = 384.0 px
    # y2_px = 400 * 720 / 1000 = 288.0 px
    assert math.isclose(b.x1_px, 128.0, abs_tol=1e-3)
    assert math.isclose(b.y1_px, 144.0, abs_tol=1e-3)
    assert math.isclose(b.x2_px, 384.0, abs_tol=1e-3)
    assert math.isclose(b.y2_px, 288.0, abs_tol=1e-3)

    assert b.pixel_width == 256.0
    assert b.pixel_height == 144.0
    assert b.max_dim_px == 256.0
    assert b.recalculated_size == "large"


def test_gemini_native_box_2d_order(validator):
    """Verifies standard Gemini Flash API format [ymin, xmin, ymax, xmax] in [0..1000]."""
    # Deliberately construct an asymmetric rectangle:
    # ymin=100, xmin=200, ymax=300, xmax=600
    # dy = 200 units, dx = 400 units
    raw = {
        "image_id": "test_gemini_native_01",
        "annotations": [
            {
                "class_id": 0,
                "class_name": "truck",
                "box_2d": [100, 200, 300, 600],  # [ymin, xmin, ymax, xmax]
                "confidence": 0.92,
            }
        ]
    }
    res = validator.validate_single_annotation_json(raw)
    assert res.is_valid is True
    assert len(res.valid_boxes) == 1
    b = res.valid_boxes[0]

    # xmin=200 -> x1_px = 200 * 1280 / 1000 = 256.0 px
    # ymin=100 -> y1_px = 100 * 720 / 1000 = 72.0 px
    # xmax=600 -> x2_px = 600 * 1280 / 1000 = 768.0 px
    # ymax=300 -> y2_px = 300 * 720 / 1000 = 216.0 px
    assert math.isclose(b.x1_px, 256.0, abs_tol=1e-3)
    assert math.isclose(b.y1_px, 72.0, abs_tol=1e-3)
    assert math.isclose(b.x2_px, 768.0, abs_tol=1e-3)
    assert math.isclose(b.y2_px, 216.0, abs_tol=1e-3)

    # Width and height must NEVER be swapped:
    assert b.pixel_width == 512.0
    assert b.pixel_height == 144.0
    assert b.pixel_width > b.pixel_height


def test_bbox_pixel_coordinate_format(validator):
    """Verifies direct pixel coordinate inputs [x1, y1, x2, y2]."""
    raw = {
        "image_id": "test_pixel_01",
        "annotations": [
            {
                "class_name": "person",
                "bbox_pixel": [100.0, 150.0, 120.0, 210.0],
                "confidence": 0.88,
            }
        ]
    }
    res = validator.validate_single_annotation_json(raw)
    assert res.is_valid is True
    b = res.valid_boxes[0]
    assert b.class_id == 1  # person is canonical ID 1
    assert math.isclose(b.x1_px, 100.0, abs_tol=1e-3)
    assert math.isclose(b.y1_px, 150.0, abs_tol=1e-3)
    assert math.isclose(b.x2_px, 120.0, abs_tol=1e-3)
    assert math.isclose(b.y2_px, 210.0, abs_tol=1e-3)
    assert b.pixel_width == 20.0
    assert b.pixel_height == 60.0
    assert b.max_dim_px == 60.0
    assert b.recalculated_size == "medium"


# ==============================================================================
# 2. Canonical Class Mapping Order & Rejection of Out-of-Vocabulary Classes
# ==============================================================================

def test_canonical_class_order(validator):
    """Verifies the 6 canonical WeatherGuard-D1 classes and their exact IDs."""
    expected = {
        0: "truck",
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "bus",
    }
    assert VALID_CLASSES == expected
    for cid, cname in expected.items():
        assert CLASS_NAME_TO_ID[cname] == cid


def test_rejection_of_out_of_vocabulary_classes(validator):
    """Ensures non-canonical classes (e.g. dog, traffic_light, airplane) are rejected."""
    raw = {
        "image_id": "test_invalid_class",
        "annotations": [
            {
                "class_name": "traffic_light",
                "x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200,
                "confidence": 0.9,
            },
            {
                "class_id": 99,
                "x_min": 300, "y_min": 300, "x_max": 400, "y_max": 400,
                "confidence": 0.9,
            }
        ]
    }
    res = validator.validate_single_annotation_json(raw)
    assert res.is_valid is False
    assert len(res.valid_boxes) == 0
    assert len(res.rejected_boxes) == 2


# ==============================================================================
# 3. Geometric Non-Inversion, Out-of-Bounds & Sub-pixel Rejection
# ==============================================================================

def test_rejection_of_inverted_and_out_of_bounds_boxes(validator):
    """Verifies rejection of inverted boxes and coordinates exceeding [0, 1000]."""
    raw = {
        "image_id": "test_inverted",
        "annotations": [
            # Inverted x (x_min > x_max)
            {"class_id": 3, "class_name": "car", "x_min": 500, "y_min": 100, "x_max": 400, "y_max": 200},
            # Out of bounds (> 1000)
            {"class_id": 3, "class_name": "car", "x_min": 100, "y_min": 100, "x_max": 1050, "y_max": 200},
            # Sub-pixel dimensions (< 2px)
            {"class_id": 3, "class_name": "car", "x_min": 100, "y_min": 100, "x_max": 100.5, "y_max": 100.5},
        ]
    }
    res = validator.validate_single_annotation_json(raw)
    assert len(res.valid_boxes) == 0
    assert len(res.rejected_boxes) == 3


# ==============================================================================
# 4. YOLO Normalized Coordinate Mathematical Reversibility
# ==============================================================================

def test_yolo_normalization_reversibility(validator):
    """Verifies that YOLO normalized format [xc, yc, w, h] in [0, 1] is perfectly reversible."""
    raw = {
        "image_id": "test_reversibility",
        "annotations": [
            {
                "class_id": 3,
                "class_name": "car",
                "x_min": 250,
                "y_min": 350,
                "x_max": 750,
                "y_max": 650,
                "confidence": 0.95,
            }
        ]
    }
    res = validator.validate_single_annotation_json(raw)
    assert res.is_valid is True
    b = res.valid_boxes[0]

    # Reconstruct pixel coordinates from YOLO normalized values
    W = 1280.0
    H = 720.0
    reconstructed_x1 = (b.yolo_x_center - b.yolo_width / 2.0) * W
    reconstructed_y1 = (b.yolo_y_center - b.yolo_height / 2.0) * H
    reconstructed_x2 = (b.yolo_x_center + b.yolo_width / 2.0) * W
    reconstructed_y2 = (b.yolo_y_center + b.yolo_height / 2.0) * H

    assert math.isclose(reconstructed_x1, b.x1_px, abs_tol=1e-5)
    assert math.isclose(reconstructed_y1, b.y1_px, abs_tol=1e-5)
    assert math.isclose(reconstructed_x2, b.x2_px, abs_tol=1e-5)
    assert math.isclose(reconstructed_y2, b.y2_px, abs_tol=1e-5)


# ==============================================================================
# 5. Object Scale Classification Convention
# ==============================================================================

def test_object_scale_boundaries():
    """Verifies the four scale categories defined by WeatherGuard convention:
    ultra_small (<16px), small (16..32px), medium (32..96px), large (>=96px).
    """
    assert classify_object_size(15.99) == "ultra_small"
    assert classify_object_size(16.00) == "small"
    assert classify_object_size(31.99) == "small"
    assert classify_object_size(32.00) == "medium"
    assert classify_object_size(95.99) == "medium"
    assert classify_object_size(96.00) == "large"
    assert classify_object_size(300.0) == "large"
