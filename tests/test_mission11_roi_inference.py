"""Unit tests for Mission 11: Region-Adaptive High-Resolution Inference & Cross-Domain Generalization.

Tests cover:
1. CropRegion coordinate transforms and exact original-image restoration.
2. Crop boundary detection logic and boundary margin verification.
3. Spatial candidate clustering with context margin and size enforcement.
4. Fixed-grid crop generation along the road horizon line.
5. Duplicate-safe NMS merging and conservative confidence gating.
6. RegionAdaptiveDetector initialization and parameter defaults.
7. Strict metric stream separation (detection stream vs tracking stream).
8. Mission 11 artifact schema and numerical sanity validation across all JSON files.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from src.video_dataset import GroundTruthBox, DetectionBox
from src.roi_inference import (
    CropRegion,
    transform_crop_to_original,
    is_touching_crop_boundary,
    cluster_candidate_boxes,
    generate_fixed_grid_crops,
    merge_detections_nms,
    RegionAdaptiveDetector,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def test_crop_region_creation_and_properties():
    """Verify CropRegion geometry, width, height, and conversion to box tuples."""
    crop = CropRegion(x1=100.0, y1=200.0, x2=500.0, y2=425.0, source_strategy="test")
    assert crop.width == 400.0
    assert crop.height == 225.0
    assert crop.area == 400.0 * 225.0
    assert crop.box_xyxy == (100.0, 200.0, 500.0, 425.0)
    assert crop.source_strategy == "test"


def test_exact_coordinate_restoration():
    """Verify that transform_crop_to_original correctly maps crop coordinates back to native image coordinates."""
    # A crop placed at (200, 100) of size 400x225
    crop = CropRegion(x1=200.0, y1=100.0, x2=600.0, y2=325.0)
    
    # In YOLO crop inference, boxes are in pixel dimensions of the crop image (0..w, 0..h)
    crop_coords = (50.0, 30.0, 150.0, 90.0)

    orig_coords = transform_crop_to_original(crop_coords, crop)
    
    # Expected: x1 = 200 + 50 = 250, y1 = 100 + 30 = 130, x2 = 200 + 150 = 350, y2 = 100 + 90 = 190
    expected_box = (250.0, 130.0, 350.0, 190.0)
    assert orig_coords == expected_box


def test_boundary_detection():
    """Verify is_touching_crop_boundary correctly identifies boxes touching or within margin of crop perimeter."""
    crop_size = (400.0, 300.0)
    
    # Internal box: away from edges
    internal_box = (50.0, 50.0, 150.0, 150.0)
    assert not is_touching_crop_boundary(internal_box, crop_size, boundary_margin_px=4.0)

    # Box touching left boundary (x1 <= 4.0)
    touching_left = (2.0, 50.0, 80.0, 120.0)
    assert is_touching_crop_boundary(touching_left, crop_size, boundary_margin_px=4.0)

    # Box touching top boundary (y1 <= 4.0)
    touching_top = (50.0, 3.0, 120.0, 100.0)
    assert is_touching_crop_boundary(touching_top, crop_size, boundary_margin_px=4.0)

    # Box touching right boundary (x2 >= 400 - 4 = 396.0)
    touching_right = (300.0, 50.0, 398.0, 120.0)
    assert is_touching_crop_boundary(touching_right, crop_size, boundary_margin_px=4.0)

    # Box touching bottom boundary (y2 >= 300 - 4 = 296.0)
    touching_bottom = (50.0, 200.0, 120.0, 297.0)
    assert is_touching_crop_boundary(touching_bottom, crop_size, boundary_margin_px=4.0)


def test_candidate_clustering_and_context_expansion():
    """Verify cluster_candidate_boxes groups nearby candidates, enforces context margin, aspect ratio, and budgets."""
    # Two candidates close to each other
    cand1 = (300.0, 300.0, 340.0, 330.0)
    cand2 = (350.0, 310.0, 390.0, 340.0)
    # One far away candidate
    cand3 = (800.0, 500.0, 850.0, 540.0)

    crops = cluster_candidate_boxes(
        boxes=[cand1, cand2, cand3],
        image_shape=(1280, 720),
        max_distance_px=120.0,
        context_margin=0.20,
        min_crop_size=256.0,
        max_crops=2,
    )

    assert len(crops) <= 2
    for crop in crops:
        assert crop.x1 >= 0.0
        assert crop.y1 >= 0.0
        assert crop.x2 <= 1280.0
        assert crop.y2 <= 720.0
        assert crop.width >= 256.0
        assert crop.height >= 256.0


def test_empty_candidates_returns_empty_crops():
    """Verify cluster_candidate_boxes returns empty list when no candidates exist."""
    crops = cluster_candidate_boxes([], image_shape=(1280, 720))
    assert crops == []


def test_fixed_grid_crop_generation():
    """Verify generate_fixed_grid_crops places crops along the horizon band."""
    crops = generate_fixed_grid_crops(
        image_shape=(1280, 720),
        grid_type="horizon_strip",
    )
    assert len(crops) == 1
    crop = crops[0]
    assert crop.x1 == 0.0
    assert crop.x2 == 1280.0
    assert crop.source_strategy == "horizon_strip"


def test_conservative_nms_merge():
    """Verify merge_detections_nms suppresses duplicates and enforces conservative confidence gating for new crops."""
    # Base detection with high confidence
    base_det = DetectionBox(class_name="Car", confidence=0.70, x1=100.0, y1=100.0, x2=200.0, y2=200.0)
    
    # Crop detection nearly identical to base detection (high IoU)
    crop_dup = DetectionBox(class_name="Car", confidence=0.85, x1=102.0, y1=98.0, x2=202.0, y2=198.0)
    
    # New valid crop discovery with s >= 0.35
    crop_new_valid = DetectionBox(class_name="Car", confidence=0.45, x1=500.0, y1=300.0, x2=550.0, y2=340.0)
    
    # Low-confidence crop detection s < 0.30 (should be rejected under conservative mode)
    crop_low_conf = DetectionBox(class_name="Car", confidence=0.22, x1=700.0, y1=400.0, x2=750.0, y2=440.0)

    merged = merge_detections_nms(
        base_detections=[base_det],
        crop_detections=[crop_dup, crop_new_valid, crop_low_conf],
        iou_threshold=0.45,
        base_conf_threshold=0.20,
        new_crop_conf_threshold=0.30,
        merge_mode="conservative",
    )

    # Should contain: upgraded base_det and crop_new_valid. crop_low_conf must be rejected.
    assert len(merged) == 2
    assert all(d.confidence >= 0.30 for d in merged)


def test_region_adaptive_detector_smoke():
    """Verify RegionAdaptiveDetector initializes properly with default parameters."""
    detector = RegionAdaptiveDetector(
        yolo_model=None,
        base_resolution=640,
        crop_resolution=448,
        strategy="clustered",
        max_crops_per_frame=2,
    )
    assert detector.base_resolution == 640
    assert detector.crop_resolution == 448
    assert detector.strategy == "clustered"
    assert detector.max_crops_per_frame == 2


def test_mission11_ablation_artifacts_schema_and_sanity():
    """Verify structure, completeness, and metric validity of mission11_ablation_results.json."""
    ablation_file = ARTIFACTS_DIR / "mission11_ablation_results.json"
    assert ablation_file.exists(), "mission11_ablation_results.json must exist"

    with open(ablation_file, "r") as f:
        data = json.load(f)

    assert "results" in data
    assert "development" in data["results"]
    assert "locked_test" in data["results"]

    expected_configs = [
        "A_YOLO640_static",
        "B_YOLO832_static",
        "C_HoldTime_H3_FullFrame",
        "D_ROI_Individual_Crops",
        "E_ROI_Clustered_Crops",
        "F_ROI_FixedGrid_Horizon",
        "G_HoldTime_H3_ROI_Hybrid",
        "H_ROI_Clustered_ByteTrack",
        "I_WeatherGuard_ROI_Decoupled",
    ]

    for split in ["development", "locked_test"]:
        for cfg in expected_configs:
            assert cfg in data["results"][split], f"Missing {cfg} in {split}"
            det = data["results"][split][cfg]["detection_stream"]
            assert 0.0 <= det["precision"] <= 1.0
            assert 0.0 <= det["recall"] <= 1.0
            assert 0.0 <= det["f1"] <= 1.0
            assert "scale_breakdown" in det
            for s in ["small", "medium", "large"]:
                assert s in det["scale_breakdown"]
                assert 0.0 <= det["scale_breakdown"][s]["f1"] <= 1.0

    # Key research check: Clustered ROI small F1 must outperform Fixed 832 on locked test
    cfg_e_small_f1 = data["results"]["locked_test"]["E_ROI_Clustered_Crops"]["detection_stream"]["scale_breakdown"]["small"]["f1"]
    cfg_b_small_f1 = data["results"]["locked_test"]["B_YOLO832_static"]["detection_stream"]["scale_breakdown"]["small"]["f1"]
    assert cfg_e_small_f1 > cfg_b_small_f1, f"Expected Clustered ROI ({cfg_e_small_f1}) > Fixed 832 ({cfg_b_small_f1})"


def test_flir_thermal_artifacts_schema_and_sanity():
    """Verify structure and contents of FLIR ADAS Thermal artifacts."""
    manifest_file = ARTIFACTS_DIR / "mission11_flir_thermal_manifest.json"
    results_file = ARTIFACTS_DIR / "mission11_flir_thermal_results.json"

    assert manifest_file.exists()
    assert results_file.exists()

    with open(manifest_file, "r") as f:
        manifest = json.load(f)
    assert manifest["num_sampled_images"] == 150
    assert manifest["total_gt_boxes"] == 1071

    with open(results_file, "r") as f:
        res = json.load(f)
    assert res["num_images"] == 150
    assert "YOLO640_static" in res["results"]
    assert "ROI_Clustered" in res["results"]

    roi_small_f1 = res["results"]["ROI_Clustered"]["small_f1"]
    base_small_f1 = res["results"]["YOLO640_static"]["small_f1"]
    assert roi_small_f1 > base_small_f1, f"Expected Thermal ROI ({roi_small_f1}) > Base ({base_small_f1})"


def test_bootstrap_results_artifacts_schema_and_sanity():
    """Verify circular block bootstrap results artifact."""
    boot_file = ARTIFACTS_DIR / "mission11_bootstrap_results.json"
    assert boot_file.exists()

    with open(boot_file, "r") as f:
        boot = json.load(f)

    roi_vs_832 = boot.get("Clustered ROI vs Fixed 832")
    assert roi_vs_832 is not None
    assert roi_vs_832["mean_delta_small_f1"] > 0.0
    assert roi_vs_832["ci_95_small_f1"][0] > 0.0  # strictly bounded away from zero
    assert roi_vs_832["p_value_small_f1"] == "< 0.001" or (isinstance(roi_vs_832["p_value_small_f1"], float) and roi_vs_832["p_value_small_f1"] < 0.01)


def test_edge_benchmark_artifacts_schema_and_sanity():
    """Verify M1 CPU edge benchmark artifact."""
    bench_file = ARTIFACTS_DIR / "mission11_edge_benchmark.json"
    assert bench_file.exists()

    with open(bench_file, "r") as f:
        bench = json.load(f)

    assert "stages_ms" in bench
    assert "pipelines" in bench
    pipes = bench["pipelines"]
    assert "yolov8n_640_static" in pipes
    assert "yolov8n_832_static" in pipes
    assert "roi_clustered_crops" in pipes

    roi_fps = pipes["roi_clustered_crops"]["fps"]
    fixed832_fps = pipes["yolov8n_832_static"]["fps"]
    assert roi_fps > fixed832_fps, f"Expected ROI FPS ({roi_fps}) > Fixed 832 FPS ({fixed832_fps})"
