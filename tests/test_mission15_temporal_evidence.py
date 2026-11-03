"""Tests for Mission 15 Causal Temporal Evidence Accumulator."""
import numpy as np
import pytest

from src.mission15.temporal_evidence import (
    CausalTemporalEvidenceAccumulator,
    TemporalEvidenceConfig,
    estimate_ego_translation,
)
from src.priority_allocation import CandidateProposal


def test_ego_translation_estimation():
    """Verify phase-correlation translation estimation on synthetic shifted patterns."""
    h, w = 360, 640
    img1 = np.zeros((h, w), dtype=np.uint8)
    cv2_box = (100, 100, 50, 50)
    img1[100:150, 100:150] = 200

    # Shift by +4px in x, +2px in y
    img2 = np.zeros((h, w), dtype=np.uint8)
    img2[102:152, 104:154] = 200

    dx, dy = estimate_ego_translation(img1, img2, downscale=1)
    assert abs(dx - 4.0) <= 1.0
    assert abs(dy - 2.0) <= 1.0


def test_temporal_evidence_causality_and_reset():
    """Verify strictly forward processing and proper reset on sequence boundary."""
    config = TemporalEvidenceConfig(memory_window_T=5, min_evidence_score=0.15)
    acc = CausalTemporalEvidenceAccumulator(config)

    # Sequence A
    frame_a1 = np.ones((720, 1280, 3), dtype=np.uint8) * 50
    props_a1 = acc.process_frame(frame_a1, sequence_id="seq_A")
    assert acc.frame_index == 1
    assert len(props_a1) == 0  # Cold start

    frame_a2 = frame_a1.copy()
    frame_a2[350:370, 600:620] = 240  # Add small salient object
    props_a2 = acc.process_frame(frame_a2, sequence_id="seq_A")
    assert acc.frame_index == 2

    # Switch to Sequence B -> Must reset
    frame_b1 = np.ones((720, 1280, 3), dtype=np.uint8) * 80
    props_b1 = acc.process_frame(frame_b1, sequence_id="seq_B")
    assert acc.current_sequence_id == "seq_B"
    assert acc.frame_index == 1
    assert len(props_b1) == 0  # Fresh cold start, no leakage from seq_A


def test_persistent_accumulation_vs_transient_noise():
    """Verify that a multi-frame persistent target accumulates higher score than a 1-frame flash."""
    config = TemporalEvidenceConfig(memory_window_T=5, min_evidence_score=0.10)
    acc = CausalTemporalEvidenceAccumulator(config)

    base_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100

    # Frame 0: background
    acc.process_frame(base_frame, sequence_id="test_acc")

    # Persistent object appears and stays for 4 frames
    for i in range(4):
        f = base_frame.copy()
        # High contrast persistent vehicle patch (16x16 px)
        f[350:366, 600:616] = 230
        props = acc.process_frame(f, sequence_id="test_acc")

    # Should detect the persistent candidate
    assert len(props) > 0
    best_prop = props[0]
    assert isinstance(best_prop, CandidateProposal)
    assert best_prop.source == "temporal_accumulator"
    assert best_prop.score > 0.20
    # Box should enclose the target
    assert best_prop.x1 <= 600 and best_prop.x2 >= 616
    assert best_prop.y1 <= 350 and best_prop.y2 >= 366


def test_ghost_suppression():
    """Verify that disappearing objects decay rapidly under ghost suppression."""
    config = TemporalEvidenceConfig(
        memory_window_T=5,
        min_evidence_score=0.15,
        ghost_suppression=True,
        ghost_decay_multiplier=0.20,
    )
    acc = CausalTemporalEvidenceAccumulator(config)

    base_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    acc.process_frame(base_frame, sequence_id="test_ghost")

    # Object present for 3 frames
    f_obj = base_frame.copy()
    f_obj[350:366, 600:616] = 230
    for _ in range(3):
        acc.process_frame(f_obj, sequence_id="test_ghost")

    # Object suddenly disappears (e.g., drives out or occluded)
    # Next 2 frames are clean background
    props_after1 = acc.process_frame(base_frame, sequence_id="test_ghost")
    props_after2 = acc.process_frame(base_frame, sequence_id="test_ghost")

    # Candidate should decay below threshold or disappear
    matching_props = [p for p in props_after2 if p.x1 <= 605 and p.x2 >= 610 and p.y1 <= 355 and p.y2 >= 360]
    assert len(matching_props) == 0


def test_candidate_size_bounds():
    """Verify that all extracted candidate proposals are within ultra-small/small dimensions."""
    config = TemporalEvidenceConfig(
        min_candidate_size=6,
        max_candidate_size=36,
        min_evidence_score=0.10,
    )
    acc = CausalTemporalEvidenceAccumulator(config)

    base_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    acc.process_frame(base_frame, sequence_id="test_bounds")

    # Add a huge 100x100 patch and a tiny 12x12 patch
    f = base_frame.copy()
    f[300:400, 300:400] = 230  # 100x100 (should be rejected)
    f[350:362, 600:612] = 230  # 12x12 (should be accepted)

    for _ in range(3):
        props = acc.process_frame(f, sequence_id="test_bounds")

    for p in props:
        w = p.x2 - p.x1
        h = p.y2 - p.y1
        # Bounding boxes have 4px padding so w and h can be at most 36 + 8 = 44px
        assert w <= 46.0 and h <= 46.0
        # Check area
        assert p.area <= 46.0 * 46.0
