"""Comprehensive Regression and Integration Validation Suite for Mission 15.

Module: tests/test_mission15_validation.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-03
Mission: Mission 15 — Temporal Evidence Accumulation & Horizon-Anchored Candidate Generation
"""
import json
from pathlib import Path
import numpy as np
import pytest

from src.mission15.temporal_evidence import (
    CausalTemporalEvidenceAccumulator,
    TemporalEvidenceConfig,
)
from src.mission15.horizon_anchor import (
    AdaptiveHorizonAnchor,
    HorizonAnchorConfig,
)
from src.mission15.policy_k import (
    PolicyKConfig,
    PolicyKController,
)
from src.priority_allocation import DetectionBox

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts" / "mission15"


def test_mission15_manifest_integrity():
    """Verify sequence manifest, deterministic hash, and historical exclusions."""
    manifest_path = ARTIFACTS_DIR / "sequence_manifest.json"
    exclusion_path = ARTIFACTS_DIR / "sequence_exclusion_manifest.json"

    assert manifest_path.exists(), "sequence_manifest.json missing"
    assert exclusion_path.exists(), "sequence_exclusion_manifest.json missing"

    manifest = json.load(open(manifest_path))
    exclusion = json.load(open(exclusion_path))

    # Verify deterministic split configuration hash
    assert manifest["split_configuration_hash"] == "0459bd7fa0fe7ed0"

    # Verify all 22 historical sequences are excluded
    assert exclusion["total_historical_sequences_excluded"] == 22
    excluded_sids = set(e["sequence_id"] for e in exclusion["exclusions"])
    assert len(excluded_sids) == 22

    # Verify that no excluded sequence appears in any Mission 15 cohort
    for cohort_name, c_data in manifest["cohorts"].items():
        for s in c_data["sequences"]:
            assert s["sequence_id"] not in excluded_sids, f"Leakage: {s['sequence_id']} is historical!"
            assert s["frames"] >= 200, f"Incomplete sequence: {s['sequence_id']}"


def test_temporal_evidence_causality_and_bounds():
    """Verify online single-frame operation, proper state clearing, and bounding box constraints."""
    config = TemporalEvidenceConfig(memory_window_T=5, min_evidence_score=0.15)
    acc = CausalTemporalEvidenceAccumulator(config)

    # Frame 1 on seq_1
    f1 = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    p1 = acc.process_frame(f1, sequence_id="seq_1")
    assert acc.frame_index == 1
    assert len(p1) == 0  # Cold start

    # Frame 2 with small salient target
    f2 = f1.copy()
    f2[350:366, 600:616] = 230
    p2 = acc.process_frame(f2, sequence_id="seq_1")
    assert acc.frame_index == 2

    # Switch sequence -> must trigger reset
    p_new = acc.process_frame(f1, sequence_id="seq_2")
    assert acc.current_sequence_id == "seq_2"
    assert acc.frame_index == 1
    assert len(p_new) == 0


def test_horizon_anchor_cone_and_vp():
    """Verify adaptive road vanishing point estimation and trapezoid cone gating."""
    anchor = AdaptiveHorizonAnchor(HorizonAnchorConfig())
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 100

    props = anchor.process_frame(img, sequence_id="seq_test_vp")
    vp_x, vp_y = anchor.smoothed_vp
    # Vanishing point must be in plausible driving horizon zone
    assert 400.0 <= vp_x <= 880.0
    assert 250.0 <= vp_y <= 480.0

    for p in props:
        assert p.source == "horizon_anchor"
        assert p.area <= 46.0 * 46.0  # Ultra-small / small bound


def test_policy_k_priority_and_budget():
    """Verify Policy K enforces crop budget K<=2 and respects detector priority."""
    ctrl = PolicyKController(PolicyKConfig(max_crops=2, crop_size=448))
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100

    det_boxes = [
        DetectionBox(x1=200.0, y1=350.0, x2=230.0, y2=380.0, confidence=0.45, class_name="car"),
        DetectionBox(x1=800.0, y1=350.0, x2=830.0, y2=380.0, confidence=0.40, class_name="car"),
    ]

    crops, cands, timings = ctrl.process_and_allocate(frame, sequence_id="test_alloc", detector_boxes=det_boxes)

    assert len(crops) <= 2
    assert "tier1_detector_ms" in timings
    assert "tier3_temporal_acc_ms" in timings
    assert "tier4_horizon_anchor_ms" in timings
