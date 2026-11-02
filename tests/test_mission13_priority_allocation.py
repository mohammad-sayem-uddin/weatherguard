"""Unit and integration tests for Mission 13: Priority-Gated Candidate Allocation Controller."""
from __future__ import annotations

import pytest
import numpy as np

from src.video_dataset import DetectionBox
from src.priority_allocation import (
    AllocatedCandidate,
    CropAllocationDecision,
    PriorityAllocationController,
)
from src.roi_inference import (
    CropRegion,
    transform_crop_to_original,
    is_touching_crop_boundary,
)


def test_allocated_candidate_properties():
    """Verify AllocatedCandidate schema, coordinate properties, and dictionary serialization."""
    cand = AllocatedCandidate(
        candidate_id="test_01",
        source="detector_low",
        x1=100.0,
        y1=200.0,
        x2=150.0,
        y2=240.0,
        score=0.22,
        scale_bucket="small",
        frame_idx=5,
        metadata={"class_name": "car"},
        max_detector_overlap_iou=0.0,
        is_new_spatial_coverage=True,
    )

    assert cand.width == 50.0
    assert cand.height == 40.0
    assert cand.area == 2000.0
    assert cand.box_xyxy == (100.0, 200.0, 150.0, 240.0)
    assert cand.center == (125.0, 220.0)

    d = cand.to_dict()
    assert d["candidate_id"] == "test_01"
    assert d["source"] == "detector_low"
    assert d["is_new_spatial_coverage"] is True


def test_detector_first_preservation_policy_a():
    """Verify Policy A guarantees detector candidates are NEVER displaced by independent proposals."""
    ctrl = PriorityAllocationController(policy="policy_a_detector_first", max_crops=2)

    # Frame with 2 detector boxes and 5 high-scoring independent proposals
    det_boxes = [
        DetectionBox("car", 0.30, 200.0, 300.0, 240.0, 330.0),
        DetectionBox("pedestrian", 0.40, 800.0, 310.0, 830.0, 360.0),
    ]

    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 120
    decision = ctrl.allocate_crops(frame, detector_boxes=det_boxes)

    assert decision.policy_name == "policy_a_detector_first"
    assert decision.used_budget <= 2
    # Detector candidates must be retained (0 displaced)
    assert decision.displaced_detector_count == 0
    assert decision.retained_detector_count >= 2


def test_independent_gating_policy_c():
    """Verify Policy C rejects proposals outside the horizon band, with weak evidence, or redundant coverage."""
    ctrl = PriorityAllocationController(
        policy="policy_c_independent_gate",
        max_crops=2,
        min_independent_score=0.50,
        horizon_band=(0.35, 0.58),  # 252 to 417 px on 720p
    )

    # Candidate 1: sky candidate (y=100, far above horizon) -> must be rejected
    c_sky = AllocatedCandidate(
        candidate_id="sky_01",
        source="static_salience",
        x1=500.0,
        y1=80.0,
        x2=550.0,
        y2=120.0,
        score=0.85,
        scale_bucket="small",
        is_new_spatial_coverage=True,
    )

    # Candidate 2: weak evidence (score=0.30 < 0.50) -> must be rejected
    c_weak = AllocatedCandidate(
        candidate_id="weak_01",
        source="frame_diff",
        x1=500.0,
        y1=300.0,
        x2=550.0,
        y2=340.0,
        score=0.30,
        scale_bucket="small",
        is_new_spatial_coverage=True,
    )

    # Candidate 3: redundant (is_new_spatial_coverage = False) -> must be rejected
    c_red = AllocatedCandidate(
        candidate_id="red_01",
        source="frame_diff",
        x1=500.0,
        y1=300.0,
        x2=550.0,
        y2=340.0,
        score=0.75,
        scale_bucket="small",
        is_new_spatial_coverage=False,
    )

    # Candidate 4: valid independent candidate in horizon
    c_valid = AllocatedCandidate(
        candidate_id="valid_01",
        source="static_salience",
        x1=600.0,
        y1=320.0,
        x2=640.0,
        y2=350.0,
        score=0.70,
        scale_bucket="ultra_small",
        is_new_spatial_coverage=True,
    )

    decision = ctrl._allocate_policy_c([c_sky, c_weak, c_red, c_valid], (1280, 720))

    assert c_sky.allocation_status == "rejected"
    assert "outside_horizon" in c_sky.rejection_reason
    assert c_weak.allocation_status == "rejected"
    assert "below_evidence" in c_weak.rejection_reason
    assert c_red.allocation_status == "rejected"
    assert "redundant" in c_red.rejection_reason
    assert c_valid.allocation_status == "allocated"
    assert decision.used_budget == 1


def test_uncertainty_aware_policy_d():
    """Verify Policy D prioritizes low-confidence detector candidates over high-confidence ones."""
    ctrl = PriorityAllocationController(policy="policy_d_uncertainty_aware", max_crops=1)

    c_high = AllocatedCandidate(
        candidate_id="det_high",
        source="detector_high",
        x1=200.0, y1=300.0, x2=350.0, y2=450.0,
        score=0.88,
        scale_bucket="medium",
    )
    c_low = AllocatedCandidate(
        candidate_id="det_low",
        source="detector_low",
        x1=600.0, y1=310.0, x2=630.0, y2=330.0,
        score=0.22,
        scale_bucket="small",
    )

    decision = ctrl._allocate_policy_d([c_high, c_low], (1280, 720))

    # Low-confidence small object has higher uncertainty value -> must be allocated
    assert c_low.priority_score > c_high.priority_score
    assert c_low.allocation_status == "allocated"


def test_adaptive_crop_budget_policy_e():
    """Verify Policy E dynamically allocates 0 crops on empty/clean frames, and 1 crop on single targets."""
    ctrl = PriorityAllocationController(policy="policy_e_adaptive_budget", max_crops=2)

    # 1. Clean frame with 0 candidates
    dec_empty = ctrl._allocate_policy_e([], (1280, 720))
    assert dec_empty.used_budget == 0
    assert len(dec_empty.allocated_crops) == 0

    # 2. Single worthy candidate -> exactly 1 crop (never force 2)
    c_single = AllocatedCandidate(
        candidate_id="det_single",
        source="detector_low",
        x1=600.0, y1=320.0, x2=640.0, y2=350.0,
        score=0.25,
        scale_bucket="small",
    )
    dec_single = ctrl._allocate_policy_e([c_single], (1280, 720))
    assert dec_single.used_budget == 1
    assert len(dec_single.allocated_crops) == 1


def test_policy_j_best_combination_end_to_end():
    """Verify Policy J combines detector priority, admission gating, and adaptive budgeting."""
    ctrl = PriorityAllocationController(
        policy="policy_j_best_combination",
        max_crops=2,
        min_independent_score=0.45,
    )

    # Frame with 1 detector box and 2 independent proposals
    det_box = DetectionBox("car", 0.32, 300.0, 310.0, 340.0, 340.0)
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100

    decision = ctrl.allocate_crops(frame, detector_boxes=[det_box])

    assert decision.policy_name == "policy_j_best_combination"
    assert decision.retained_detector_count >= 1
    assert decision.used_budget >= 1
    assert decision.used_budget <= 2


def test_crop_coordinate_restoration_round_trip():
    """Verify bounding box coordinates in crop space map back to original coordinates without distortion."""
    crop = CropRegion(x1=200.0, y1=150.0, x2=648.0, y2=598.0, candidate_indices=[0], source_strategy="clustered")
    assert crop.width == 448.0
    assert crop.height == 448.0

    # Object at center of 448x448 crop
    det_in_crop = (100.0, 100.0, 200.0, 200.0)
    ox1, oy1, ox2, oy2 = transform_crop_to_original(det_in_crop, crop)

    assert ox1 == pytest.approx(300.0, abs=0.1)
    assert oy1 == pytest.approx(250.0, abs=0.1)
    assert ox2 == pytest.approx(400.0, abs=0.1)
    assert oy2 == pytest.approx(350.0, abs=0.1)


def test_causal_reset_clears_temporal_state():
    """Verify reset clears frame differencing history and resets frame counter."""
    ctrl = PriorityAllocationController()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    ctrl.allocate_crops(frame)

    assert ctrl.frame_counter == 1
    assert ctrl.diff_gen.prev_gray is not None

    ctrl.reset()
    assert ctrl.frame_counter == 0
    assert ctrl.diff_gen.prev_gray is None


def test_decision_serialization_to_dict():
    """Verify CropAllocationDecision and CropRegion serialize cleanly to dict."""
    crop = CropRegion(x1=100.0, y1=100.0, x2=548.0, y2=548.0, candidate_indices=[0], source_strategy="clustered")
    assert crop.to_dict()["x1"] == 100.0

    ctrl = PriorityAllocationController(policy="policy_j_best_combination")
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    det = [DetectionBox("car", 0.45, 100.0, 100.0, 120.0, 120.0)]
    decision = ctrl.allocate_crops(frame, detector_boxes=det)

    d = decision.to_dict()
    assert "frame_idx" in d
    assert "allocated_crops" in d
    assert isinstance(d["allocated_crops"], list)
    assert len(d["allocated_crops"]) > 0
    assert d["policy_name"] == "policy_j_best_combination"
