"""Unit tests for Mission 3: Scale-aware object detection and calibration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.model import Detection, Prediction, YOLOv8N, build_detector

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def test_yolo_imgsz_configuration():
    """Verify YOLOv8 detector properly stores and accepts imgsz parameter."""
    detector_640 = build_detector("yolov8n", imgsz=640)
    assert detector_640.imgsz == 640

    detector_832 = build_detector("yolov8n", imgsz=832)
    assert detector_832.imgsz == 832

    detector_1280 = build_detector("yolov8n", imgsz=1280)
    assert detector_1280.imgsz == 1280


def test_scale_bin_classification():
    """Verify scale bins conform strictly to area thresholds:
    small < 32^2 (1024), medium [1024, 9216), large >= 96^2 (9216).
    """
    from scripts.run_scale_adaptive_calibration import get_predicted_scale_bin

    # 30x30 = 900 -> small
    assert get_predicted_scale_bin((0, 0, 30, 30)) == "small"
    # 32x32 = 1024 -> medium
    assert get_predicted_scale_bin((0, 0, 32, 32)) == "medium"
    # 50x50 = 2500 -> medium
    assert get_predicted_scale_bin((0, 0, 50, 50)) == "medium"
    # 96x96 = 9216 -> large
    assert get_predicted_scale_bin((0, 0, 96, 96)) == "large"
    # 150x150 = 22500 -> large
    assert get_predicted_scale_bin((0, 0, 150, 150)) == "large"


def test_nms_suppression():
    """Verify that overlapping detections of the same class are correctly suppressed."""
    from scripts.run_tiled_inference import nms_detections

    # Two overlapping cars at IoU > 0.50
    d1 = Detection(box_xyxy=(10.0, 10.0, 100.0, 100.0), score=0.85, label="car", common_label="Car")
    d2 = Detection(box_xyxy=(12.0, 12.0, 98.0, 98.0), score=0.60, label="car", common_label="Car")
    # One separate car far away
    d3 = Detection(box_xyxy=(300.0, 300.0, 400.0, 400.0), score=0.75, label="car", common_label="Car")
    # One overlapping person (different class should NOT be suppressed)
    d4 = Detection(box_xyxy=(15.0, 15.0, 95.0, 95.0), score=0.70, label="person", common_label="Person")

    merged = nms_detections([d1, d2, d3, d4], iou_threshold=0.50)
    assert len(merged) == 3
    # d1 kept, d2 suppressed, d3 kept, d4 kept
    assert d1 in merged
    assert d2 not in merged
    assert d3 in merged
    assert d4 in merged


def test_split_manifest_disjointness():
    """Verify that Group B (Dev) and Group C (Locked Test) have zero overlap."""
    manifest_path = ARTIFACTS / "mission3_split_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Manifest not found")

    with open(manifest_path) as f:
        manifest = json.load(f)

    dev_ids = set(manifest["groups"]["Group_B_BDD100K_Development"]["image_ids"])
    test_ids = set(manifest["groups"]["Group_C_BDD100K_Locked_Test"]["image_ids"])

    overlap = dev_ids & test_ids
    assert len(overlap) == 0, f"Contamination detected! Overlapping IDs: {overlap}"
    assert len(dev_ids) == 160
    assert len(test_ids) == 203
