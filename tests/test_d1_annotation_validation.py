"""Unit tests for WeatherGuard-D1 Gemini Flash annotation validation and YOLO conversion.

Module: tests/test_d1_annotation_validation.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""
import json
import shutil
import tempfile
from pathlib import Path
import pytest

from scripts.validate_d1_annotations import (
    WeatherGuardD1AnnotationValidator,
    classify_object_size,
    compute_iou_xyxy,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "weatherguard_d1.yaml"


@pytest.fixture
def temp_workspace():
    tmp_dir = Path(tempfile.mkdtemp(prefix="test_d1_annot_"))
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_classify_object_size_convention():
    # Ultra-small: < 16 px
    assert classify_object_size(10.5) == "ultra_small"
    assert classify_object_size(15.9) == "ultra_small"
    # Small: 16 to < 32 px
    assert classify_object_size(16.0) == "small"
    assert classify_object_size(31.9) == "small"
    # Medium: 32 to < 96 px
    assert classify_object_size(32.0) == "medium"
    assert classify_object_size(95.5) == "medium"
    # Large: >= 96 px
    assert classify_object_size(96.0) == "large"
    assert classify_object_size(240.0) == "large"


def test_valid_annotation_and_yolo_conversion(temp_workspace):
    validator = WeatherGuardD1AnnotationValidator(config_path=CONFIG_PATH)

    # 1280x720 reference:
    # x: [500, 600] in 0..1000 scale -> [640, 768] px (width = 128px)
    # y: [300, 400] in 0..1000 scale -> [216, 288] px (height = 72px)
    # max dimension = 128px -> "large"
    sample_json = {
        "image_id": "day_001",
        "filename": "day_001.jpg",
        "condition": "day",
        "annotations": [
            {
                "box_id": "box_01",
                "class_id": 3,
                "class_name": "car",
                "x_min": 500,
                "y_min": 300,
                "x_max": 600,
                "y_max": 400,
                "confidence": 0.95,
                "object_size": "large",
            }
        ]
    }

    res = validator.validate_single_annotation_json(sample_json)
    assert res.is_valid is True
    assert len(res.valid_boxes) == 1
    box = res.valid_boxes[0]

    assert box.class_id == 3
    assert box.class_name == "car"
    assert box.x1_px == 640.0
    assert box.x2_px == 768.0
    assert box.y1_px == 216.0
    assert box.y2_px == 288.0
    assert box.pixel_width == 128.0
    assert box.pixel_height == 72.0
    assert box.recalculated_size == "large"
    assert box.size_disagreement is False

    # YOLO normalized center
    assert pytest.approx(box.yolo_x_center, 0.001) == 0.55
    assert pytest.approx(box.yolo_y_center, 0.001) == 0.35
    assert pytest.approx(box.yolo_width, 0.001) == 0.10
    assert pytest.approx(box.yolo_height, 0.001) == 0.10

    # Test writing YOLO file
    labels_dir = temp_workspace / "labels"
    txt_path = validator.write_yolo_label_file(res, labels_dir)
    assert txt_path.exists()
    content = txt_path.read_text().strip()
    assert content.startswith("3 0.550000 0.350000 0.100000 0.100000")


def test_invalid_inverted_and_out_of_bounds_boxes():
    validator = WeatherGuardD1AnnotationValidator(config_path=CONFIG_PATH)

    sample_json = {
        "image_id": "err_001",
        "annotations": [
            {
                # Inverted x
                "class_id": 3,
                "class_name": "car",
                "x_min": 600,
                "y_min": 200,
                "x_max": 500,
                "y_max": 300,
            },
            {
                # Out of bounds y
                "class_id": 1,
                "class_name": "person",
                "x_min": 100,
                "y_min": 100,
                "x_max": 200,
                "y_max": 1050,
            }
        ]
    }

    res = validator.validate_single_annotation_json(sample_json)
    assert len(res.valid_boxes) == 0
    assert len(res.rejected_boxes) == 2
    assert res.status == "REJECTED_STRUCTURAL_ERROR"


def test_gemini_size_disagreement_detection():
    validator = WeatherGuardD1AnnotationValidator(config_path=CONFIG_PATH)

    # 1280x720: [100, 100, 110, 110] -> width = 12.8px, height = 7.2px -> max dim = 12.8px ("ultra_small")
    # But Gemini incorrectly claims it is "medium"
    sample_json = {
        "image_id": "disagree_001",
        "annotations": [
            {
                "class_id": 3,
                "class_name": "car",
                "x_min": 100,
                "y_min": 100,
                "x_max": 110,
                "y_max": 110,
                "object_size": "medium",  # Gemini claim
            }
        ]
    }

    res = validator.validate_single_annotation_json(sample_json)
    assert len(res.valid_boxes) == 1
    box = res.valid_boxes[0]
    assert box.recalculated_size == "ultra_small"
    assert box.size_disagreement is True
    assert any("Gemini object_size" in fl for fl in box.flags)


def test_class_mismatch_rejection():
    validator = WeatherGuardD1AnnotationValidator(config_path=CONFIG_PATH)

    # class_id 0 is "truck", but JSON claims "person"
    sample_json = {
        "image_id": "mismatch_001",
        "annotations": [
            {
                "class_id": 0,
                "class_name": "person",
                "x_min": 200,
                "y_min": 200,
                "x_max": 300,
                "y_max": 300,
            }
        ]
    }

    res = validator.validate_single_annotation_json(sample_json)
    assert len(res.valid_boxes) == 0
    assert len(res.rejected_boxes) == 1
    assert "Class mismatch" in res.rejected_boxes[0]["rejection_reasons"][0]


def test_intra_image_duplicate_pruning():
    validator = WeatherGuardD1AnnotationValidator(config_path=CONFIG_PATH)

    # Two virtually identical car boxes in the same image
    sample_json = {
        "image_id": "dup_001",
        "annotations": [
            {
                "box_id": "b1",
                "class_id": 3,
                "class_name": "car",
                "x_min": 400,
                "y_min": 300,
                "x_max": 500,
                "y_max": 400,
            },
            {
                "box_id": "b2",
                "class_id": 3,
                "class_name": "car",
                "x_min": 401,
                "y_min": 300,
                "x_max": 501,
                "y_max": 401,
            }
        ]
    }

    res = validator.validate_single_annotation_json(sample_json)
    assert len(res.valid_boxes) == 1
    assert len(res.rejected_boxes) == 1
    assert "Intra-image duplicate" in res.rejected_boxes[0]["rejection_reasons"][0]


def test_empty_scene_handling():
    validator = WeatherGuardD1AnnotationValidator(config_path=CONFIG_PATH)

    sample_json = {
        "image_id": "empty_001",
        "condition": "rain",
        "annotations": []
    }

    res = validator.validate_single_annotation_json(sample_json)
    assert res.is_valid is True
    assert res.is_empty is True
    assert res.status == "EMPTY_ACCEPTED"
    assert len(res.valid_boxes) == 0
