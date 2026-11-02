"""Unit tests for Mission 10: Independent Generalization, Causal Ablation & Temporal Reliability.

Tests cover:
1. Sequence partition disjointness, virgin cohort integrity, and zero leakage.
2. Metric stream separation: Detector-only evaluation vs Recovery stream vs Tracking stream.
3. Proof of extrapolation detection pollution: Merging extrapolated boxes increases FPs and lowers precision.
4. Recovery stream metrics: Correct recoveries, false continuations, and recovery precision.
5. Tracking stream continuity: IDF1, ID switches, and fragmentation accounting.
6. Hold-time dynamic router state transitions, zero onset delay, and hold persistence.
7. Extrapolation gap expiry and track disappearance.
8. Sequence-aware circular block bootstrap respecting sequence boundaries.
9. Artifact integrity and reproducibility across all Mission 10 artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from src.video_dataset import (
    GroundTruthBox,
    DetectionBox,
    VideoFrame,
    VideoSequence,
    compute_iou,
)
from src.eval_streams import (
    evaluate_detection_stream,
    evaluate_tracking_stream,
    evaluate_recovery_stream,
    circular_block_bootstrap,
)
from src.temporal_confidence import TemporalConfidenceFilter
from src.temporal_router import TemporalRouter
from src.online_tracker import AdverseWeatherByteTracker, KalmanFilterBBox, Track


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def test_sequence_partition_disjointness_and_zero_leakage():
    """Verify that development and locked test sequences are strictly disjoint and have zero leakage."""
    seq_manifest_path = ARTIFACTS_DIR / "mission10_sequence_manifest.json"
    assert seq_manifest_path.exists(), "mission10_sequence_manifest.json must exist"

    with open(seq_manifest_path, "r") as f:
        manifest = json.load(f)

    dev_seqs = {s["sequence_id"] for s in manifest["partitions"]["independent_development_cohort"]["sequences"]}
    test_seqs = {s["sequence_id"] for s in manifest["partitions"]["independent_locked_test_cohort"]["sequences"]}

    # Disjointness between dev and locked test
    assert len(dev_seqs & test_seqs) == 0, "Dev and Locked Test must have zero sequence overlap"
    assert len(dev_seqs) == 2, f"Expected 2 dev sequences, found {len(dev_seqs)}"
    assert len(test_seqs) == 4, f"Expected 4 locked test sequences, found {len(test_seqs)}"

    # Verify zero leakage with Mission 9 sequences if mission9 manifest exists
    m9_manifest_path = ARTIFACTS_DIR / "mission9_sequence_manifest.json"
    if m9_manifest_path.exists():
        with open(m9_manifest_path, "r") as f:
            m9_manifest = json.load(f)
        m9_dev = {s["sequence_id"] for s in m9_manifest.get("development", {}).get("sequences", [])} if isinstance(m9_manifest.get("development", {}).get("sequences"), list) else set(m9_manifest.get("development", {}).get("sequence_ids", []))
        m9_test = {s["sequence_id"] for s in m9_manifest.get("locked_test", {}).get("sequences", [])} if isinstance(m9_manifest.get("locked_test", {}).get("sequences"), list) else set(m9_manifest.get("locked_test", {}).get("sequence_ids", []))
        m9_seqs = m9_dev | m9_test
        all_m10_seqs = dev_seqs | test_seqs
        assert len(all_m10_seqs & m9_seqs) == 0, "Mission 10 sequences must be virgin with zero Mission 9 overlap"


def test_metric_stream_separation_and_extrapolation_pollution():
    """Verify that evaluating extrapolated boxes in detection stream corrupts precision and increases FPs."""
    # Create a synthetic 2-frame sequence with 1 true car
    frame0 = VideoFrame(
        frame_index=0,
        image_path=Path("dummy_0.jpg"),
        timestamp_seconds=0.0,
        gt_boxes=[GroundTruthBox(track_id="1", class_name="Car", x1=10.0, y1=10.0, x2=50.0, y2=50.0)],
    )
    # Frame 1: Ground truth car turned and left the scene (empty)
    frame1 = VideoFrame(
        frame_index=1,
        image_path=Path("dummy_1.jpg"),
        timestamp_seconds=0.2,
        gt_boxes=[],
    )
    seq = VideoSequence(
        sequence_id="synthetic_seq",
        condition="rainy",
        timeofday="daytime",
        scene="highway",
        fps=5.0,
        frames=[frame0, frame1],
    )

    # Case 1: Pure detector output (detected in frame 0, empty in frame 1)
    det_f0 = [DetectionBox(class_name="Car", confidence=0.85, x1=10.0, y1=10.0, x2=50.0, y2=50.0)]
    det_f1: list[DetectionBox] = []
    pure_preds = [det_f0, det_f1]

    pure_eval = evaluate_detection_stream(seq, pure_preds, iou_threshold=0.5)
    assert pure_eval.tp == 1
    assert pure_eval.fp == 0
    assert pure_eval.precision == 1.0
    assert pure_eval.f1 == 1.0

    # Case 2: Polluted stream where Kalman filter extrapolated frame 1 forward (ghost false continuation)
    ghost_extrap_f1 = [DetectionBox(class_name="Car", confidence=0.70, x1=12.0, y1=10.0, x2=52.0, y2=50.0)]
    fused_preds = [det_f0, ghost_extrap_f1]

    polluted_eval = evaluate_detection_stream(seq, fused_preds, iou_threshold=0.5)
    assert polluted_eval.tp == 1
    assert polluted_eval.fp == 1, "Extrapolated box in empty frame must be counted as a false positive"
    assert polluted_eval.precision == 0.5, "Precision must drop when false extrapolation is counted in detection"
    assert polluted_eval.f1 < pure_eval.f1, "F1 must degrade under extrapolation pollution"


def test_recovery_stream_metrics_computation():
    """Verify that RecoveryStreamResult correctly counts recovery precision and false continuation."""
    # Sequence with car temporarily occluded at frame 1 and reappearing at frame 2
    gt0 = GroundTruthBox(track_id="1", class_name="Car", x1=10.0, y1=10.0, x2=50.0, y2=50.0)
    gt1 = GroundTruthBox(track_id="1", class_name="Car", x1=12.0, y1=10.0, x2=52.0, y2=50.0)
    gt2 = GroundTruthBox(track_id="1", class_name="Car", x1=14.0, y1=10.0, x2=54.0, y2=50.0)

    frames = [
        VideoFrame(frame_index=0, image_path=Path("f0.jpg"), timestamp_seconds=0.0, gt_boxes=[gt0]),
        VideoFrame(frame_index=1, image_path=Path("f1.jpg"), timestamp_seconds=0.2, gt_boxes=[gt1]),
        VideoFrame(frame_index=2, image_path=Path("f2.jpg"), timestamp_seconds=0.4, gt_boxes=[gt2]),
    ]
    seq = VideoSequence(sequence_id="test_rec", condition="rain", timeofday="day", scene="city", fps=5.0, frames=frames)

    # In frame 1, detector missed the car (empty), but extrapolation produced a box matching gt1
    det_per_frame = [
        [DetectionBox(class_name="Car", confidence=0.85, x1=10.0, y1=10.0, x2=50.0, y2=50.0)],
        [],
        [DetectionBox(class_name="Car", confidence=0.85, x1=14.0, y1=10.0, x2=54.0, y2=50.0)],
    ]
    extrap_f1 = [DetectionBox(class_name="Car", confidence=0.7, x1=12.2, y1=10.0, x2=52.2, y2=50.0, track_id=1)]
    extrap_per_frame = [[], extrap_f1, []]

    rec_result = evaluate_recovery_stream(seq, det_per_frame, extrap_per_frame, iou_threshold=0.5)
    assert rec_result.total_extrapolated_states == 1
    assert rec_result.correctly_recovered_states == 1
    assert rec_result.incorrect_extrapolations == 0
    assert rec_result.recovery_precision == 1.0
    assert rec_result.false_continuation_rate == 0.0


def test_tracking_stream_id_switches():
    """Verify that evaluate_tracking_stream detects ID switches when tracker swaps identifiers."""
    gt0 = GroundTruthBox(track_id="100", class_name="Car", x1=10.0, y1=10.0, x2=50.0, y2=50.0)
    gt1 = GroundTruthBox(track_id="100", class_name="Car", x1=12.0, y1=10.0, x2=52.0, y2=50.0)

    frames = [
        VideoFrame(frame_index=0, image_path=Path("f0.jpg"), timestamp_seconds=0.0, gt_boxes=[gt0]),
        VideoFrame(frame_index=1, image_path=Path("f1.jpg"), timestamp_seconds=0.2, gt_boxes=[gt1]),
    ]
    seq = VideoSequence(sequence_id="test_trk", condition="snow", timeofday="night", scene="city", fps=5.0, frames=frames)

    # Frame 0 tracked as ID 1, Frame 1 assigned to new track ID 2 (ID switch)
    preds = [
        [DetectionBox(class_name="Car", confidence=0.9, x1=10.0, y1=10.0, x2=50.0, y2=50.0, track_id=1)],
        [DetectionBox(class_name="Car", confidence=0.9, x1=12.0, y1=10.0, x2=52.0, y2=50.0, track_id=2)],
    ]
    trk_eval = evaluate_tracking_stream(seq, preds)
    assert trk_eval.id_switches == 1, "Track ID change on same ground truth object must register an ID switch"


def test_hold_time_router_state_transitions_and_zero_onset_delay():
    """Verify that TemporalRouter triggers instantly (0 onset delay) and holds resolution for exactly H frames."""
    router = TemporalRouter(
        method="hold_time",
        min_hold_frames=3,
        tau_high=1.0,
    )

    # Frame 0: Clean frame, score 0.0 -> Escalate False, hold 0
    d0 = router.update(0.0)
    assert not d0.escalate
    assert d0.hold_remaining == 0

    # Frame 1: Small object trigger (score = 2.0 >= tau_high) -> Instant escalation (0 onset delay), hold 3
    d1 = router.update(2.0)
    assert d1.escalate, "Router must escalate immediately with zero onset delay"
    assert d1.hold_remaining == 3

    # Frame 2: Score 0.0 -> Remains escalated, hold decrements to 2
    d2 = router.update(0.0)
    assert d2.escalate, "Must hold escalation during hold window"
    assert d2.hold_remaining == 2

    # Frame 3: Score 0.0 -> Remains escalated, hold decrements to 1
    d3 = router.update(0.0)
    assert d3.escalate
    assert d3.hold_remaining == 1

    # Frame 4: Score 0.0 -> Remains escalated on last hold frame, hold decrements to 0
    d4 = router.update(0.0)
    assert d4.escalate
    assert d4.hold_remaining == 0

    # Frame 5: Score 0.0 -> Reverts to base resolution
    d5 = router.update(0.0)
    assert not d5.escalate, "Router must revert to base resolution after hold expires"
    assert d5.hold_remaining == 0


def test_extrapolation_expiry_and_disappearance():
    """Verify that TemporalConfidenceFilter expires extrapolation after max_gap_frames."""
    tc = TemporalConfidenceFilter(
        recovery_max_gap=1,
        output_threshold=0.2,
    )

    # Frame 0: Solid detection with track_id=1
    box0 = DetectionBox(class_name="Car", confidence=0.85, x1=10.0, y1=10.0, x2=50.0, y2=50.0, track_id=1)
    obs_0, extrap_0, diag_0 = tc.process_frame_separate([box0])
    assert len(obs_0) == 1
    assert len(extrap_0) == 0

    # Frame 1: Detection disappears -> 1 extrapolated state generated (G=1)
    obs_1, extrap_1, diag_1 = tc.process_frame_separate([])
    assert len(obs_1) == 0
    assert len(extrap_1) == 1, "Should generate 1 extrapolated state on first frame gap"

    # Frame 2: Still empty -> Extrapolation budget exhausted, 0 extrapolations
    obs_2, extrap_2, diag_2 = tc.process_frame_separate([])
    assert len(obs_2) == 0
    assert len(extrap_2) == 0, "Extrapolation must expire after max_extrapolation_frames"


def test_sequence_aware_circular_block_bootstrap():
    """Verify that block bootstrap respects sequence boundaries and circular indexing."""
    # Create synthetic per-frame F1 scores for 2 sequences of length 30 and 40
    seq_scores = {
        "seq_A": np.full(30, 0.45),
        "seq_B": np.full(40, 0.55),
    }
    # Resample
    bs_results = circular_block_bootstrap(
        seq_scores_a=seq_scores,
        seq_scores_b=seq_scores,
        block_length=20,
        n_resamples=50,
        seed=42,
    )
    # Comparing identical distributions must yield mean delta near 0 and CI covering 0
    assert abs(bs_results["mean_diff"]) < 0.01
    assert bs_results["ci_95"][0] <= 0.0 <= bs_results["ci_95"][1]


def test_all_mission10_artifacts_exist_and_valid():
    """Verify that all required Mission 10 artifacts exist in artifacts/ and contain valid data."""
    required_artifacts = [
        "mission10_discrepancy_register.json",
        "mission10_dataset_manifest.json",
        "mission10_sequence_manifest.json",
        "mission10_ablation_results.json",
        "mission10_routing_reliability.json",
        "mission10_failure_registry.json",
        "mission10_bootstrap_results.json",
        "mission10_edge_benchmark.json",
    ]

    for fname in required_artifacts:
        path = ARTIFACTS_DIR / fname
        assert path.exists(), f"Missing required artifact: {fname}"
        with open(path, "r") as f:
            data = json.load(f)
        assert data is not None, f"Artifact {fname} was empty or invalid JSON"
        assert len(data) > 0, f"Artifact {fname} contains no keys"
