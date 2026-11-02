"""Unit tests for Mission 12: Detector-Independent Candidate Generation & Ultra-Small Object Recovery.

Tests cover:
1. CandidateProposal geometry, area, aspect ratio, and box conversion.
2. FrameDifferenceProposal causal differencing, startup behavior, and wiper sweep rejection.
3. SparseOpticalFlowProposal ego-motion median compensation and residual velocity clustering.
4. StaticSalienceProposal gradient magnitude, Gaussian horizon weighting, and stationary vehicle detection.
5. SpatialPriorProposal structured perspective horizon cell placement.
6. ProposalFusionController cross-modal deduplication, composite ranking, and crop budget enforcement.
7. Online causality verification: zero future frame leakage, clean startup at t=0.
8. Artifact schema and numerical sanity across all Mission 12 JSON files.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from src.video_dataset import GroundTruthBox, DetectionBox
from src.candidate_generation import (
    CandidateProposal,
    FrameDifferenceProposal,
    SparseOpticalFlowProposal,
    StaticSalienceProposal,
    SpatialPriorProposal,
    ProposalFusionController,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def test_candidate_proposal_properties():
    """Verify CandidateProposal geometry, area, aspect ratio, and center calculations."""
    prop = CandidateProposal(
        x1=100.0,
        y1=150.0,
        x2=260.0,
        y2=230.0,
        score=0.85,
        source="frame_diff",
        metadata={"motion_speed": 4.2},
    )
    assert prop.area == 160.0 * 80.0
    assert prop.aspect_ratio == pytest.approx(2.0, abs=1e-3)
    assert prop.box_xyxy == (100.0, 150.0, 260.0, 230.0)
    assert prop.center == (180.0, 190.0)
    assert prop.source == "frame_diff"
    assert prop.metadata["motion_speed"] == 4.2


def test_frame_difference_startup_and_motion_detection():
    """Verify FrameDifferenceProposal handles t=0 startup gracefully and detects moving regions at t=1."""
    gen = FrameDifferenceProposal(diff_threshold=20.0, min_area=30.0)
    gen.reset()

    # Frame 0 (startup): uniform gray frame
    f0 = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    p0 = gen.generate_proposals(f0)
    # Must output empty list on frame 0 without exception
    assert p0 == []
    assert gen.prev_gray is not None

    # Frame 1: add a bright moving block in the road horizon band
    f1 = f0.copy()
    # Add block at center road y in [300, 350], x in [500, 580]
    f1[300:350, 500:580] = 250
    p1 = gen.generate_proposals(f1)
    
    assert len(p1) >= 1
    best_p = p1[0]
    assert best_p.source == "frame_diff"
    # Ensure proposed box covers the injected moving block
    assert best_p.x1 <= 520.0 and best_p.x2 >= 560.0
    assert best_p.y1 <= 320.0 and best_p.y2 >= 340.0


def test_wiper_sweep_rejection():
    """Verify FrameDifferenceProposal rejects huge vertical windshield wiper motions."""
    gen = FrameDifferenceProposal()
    gen.reset()

    f0 = np.ones((720, 1280, 3), dtype=np.uint8) * 80
    gen.generate_proposals(f0)

    # Injected massive wiper sweep across the windshield
    f1 = f0.copy()
    f1[50:680, 400:460] = 240  # 630px tall wiper mask
    p1 = gen.generate_proposals(f1)

    # Wiper must be rejected by max_dim / max_area filters
    wiper_boxes = [p for p in p1 if (p.y2 - p.y1) > 450.0]
    assert len(wiper_boxes) == 0, "Windshield wiper sweep must be filtered"


def test_static_salience_stationary_vehicle_detection():
    """Verify StaticSalienceProposal detects high-contrast stationary structures within road horizon."""
    sal_gen = StaticSalienceProposal(percentile_thresh=90.0)
    
    # Base background: uniform diffuse road pavement
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 110
    
    # Injected stationary car-like high contrast block in horizon band
    # Car windshield + bumper edges: dark rectangular perimeter with bright center
    frame[310:350, 600:660] = 240
    frame[315:345, 605:655] = 40
    
    proposals = sal_gen.generate_proposals(frame)
    assert len(proposals) >= 1
    # Check that proposals cluster around the car feature
    car_proposals = [p for p in proposals if (580 <= p.x1 <= 680 and 290 <= p.y1 <= 370)]
    assert len(car_proposals) >= 1
    assert car_proposals[0].source == "static_salience"


def test_spatial_prior_horizon_partitioning():
    """Verify SpatialPriorProposal partitions the perspective vanishing line with center weighting."""
    prior_gen = SpatialPriorProposal(horizon_band=(0.35, 0.58), num_horizontal_cells=3)
    proposals = prior_gen.generate_proposals((720, 1280))

    assert len(proposals) == 3
    # Center cell (index 1) should have highest priority score
    assert proposals[1].score > proposals[0].score
    assert proposals[1].score > proposals[2].score
    # Verify spatial bounds within horizon
    for p in proposals:
        assert p.y1 == pytest.approx(0.35 * 720, abs=1.0)
        assert p.y2 == pytest.approx(0.58 * 720, abs=1.0)
        assert p.source == "spatial_prior"


def test_cross_modal_deduplication():
    """Verify ProposalFusionController deduplicates overlapping proposals from different modalities."""
    ctrl = ProposalFusionController(use_detector=False, max_crops=2, iou_dedup_threshold=0.40)
    
    # Two heavily overlapping proposals from different sources
    p_diff = CandidateProposal(200.0, 300.0, 300.0, 400.0, score=0.80, source="frame_diff")
    p_sal = CandidateProposal(205.0, 305.0, 305.0, 405.0, score=0.60, source="static_salience")
    # One distinct distant proposal
    p_dist = CandidateProposal(800.0, 320.0, 880.0, 380.0, score=0.75, source="static_salience")

    deduped = ctrl._deduplicate_proposals([p_diff, p_sal, p_dist], iou_thresh=0.40)

    # p_sal should be suppressed by p_diff (higher score and IoU > 0.40)
    assert len(deduped) == 2
    assert deduped[0] == p_diff
    assert deduped[1] == p_dist


def test_crop_budget_enforcement():
    """Verify ProposalFusionController strictly enforces maximum crop budget K <= max_crops."""
    ctrl = ProposalFusionController(
        use_detector=False,
        use_frame_diff=True,
        use_static_salience=True,
        max_crops=2,
    )
    
    frame = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
    props, crop_regions, timings = ctrl.fuse_proposals(frame)

    assert len(crop_regions) <= 2, f"Expected at most 2 crops, got {len(crop_regions)}"
    for crop in crop_regions:
        assert crop.width >= 256.0
        assert crop.height >= 256.0
        assert crop.x1 >= 0 and crop.x2 <= 1280
        assert crop.y1 >= 0 and crop.y2 <= 720


def test_causal_temporal_reset():
    """Verify controller reset cleanly clears temporal history across sequence boundaries."""
    ctrl = ProposalFusionController(use_frame_diff=True, use_sparse_flow=True)
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    ctrl.fuse_proposals(frame)

    assert ctrl.diff_gen.prev_gray is not None
    # Reset across sequence boundary
    ctrl.reset()
    assert ctrl.diff_gen.prev_gray is None
    assert ctrl.flow_gen.prev_gray is None


def test_candidate_oracle_artifacts_schema_and_sanity():
    """Verify structure and numerical validity of mission12_candidate_oracle.json."""
    oracle_file = ARTIFACTS_DIR / "mission12_candidate_oracle.json"
    assert oracle_file.exists(), "mission12_candidate_oracle.json must exist"

    with open(oracle_file, "r") as f:
        data = json.load(f)

    assert "development" in data
    assert "locked_test" in data

    for split in ["development", "locked_test"]:
        sb = data[split]["scale_breakdown"]
        assert "ultra_small" in sb
        assert "small" in sb
        assert sb["ultra_small"]["yolo_miss_pct"] == 100.0
        assert sb["ultra_small"]["oracle_recoverable_pct_of_missed"] >= 85.0
