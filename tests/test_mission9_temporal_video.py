"""Unit tests for Mission 9 Video Dataset, Online Tracking, Temporal Confidence, and Routing."""
import pytest
import numpy as np
from pathlib import Path

from src.video_dataset import (
    GroundTruthBox,
    DetectionBox,
    VideoFrame,
    VideoSequence,
    VideoDataset,
    compute_iou,
    evaluate_video_sequence,
)
from src.online_tracker import KalmanFilterBBox, Track, AdverseWeatherByteTracker
from src.temporal_confidence import TemporalConfidenceFilter
from src.temporal_router import TemporalRouter
from src.temporal_false_alarm import TemporalFalseAlarmSuppressor


def test_iou_computation():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (5.0, 0.0, 15.0, 10.0)
    iou = compute_iou(box1, box2)
    # intersection: 5 * 10 = 50, union: 100 + 100 - 50 = 150
    assert pytest.approx(iou, 0.01) == 50.0 / 150.0

    # Non-overlapping
    box3 = (20.0, 20.0, 30.0, 30.0)
    assert compute_iou(box1, box3) == 0.0


def test_kalman_filter_bbox():
    init_box = (10.0, 10.0, 30.0, 40.0)  # w=20, h=30, xc=20, yc=25
    kf = KalmanFilterBBox(init_box)

    # Predict
    pred_box = kf.predict()
    assert len(pred_box) == 4
    assert pred_box[2] > pred_box[0]
    assert pred_box[3] > pred_box[1]

    # Update with shifted measurement (moving right)
    meas_box = (12.0, 10.0, 32.0, 40.0)
    kf.update(meas_box)
    updated_box = kf.get_bbox()
    assert updated_box[0] > 10.0  # shifted towards measurement


def test_track_lifecycle():
    init_box = (100.0, 100.0, 150.0, 160.0)
    kf = KalmanFilterBBox(init_box)
    t = Track(
        track_id=1,
        class_name="Car",
        kf=kf,
        state="tentative",
        bbox=init_box,
    )
    assert t.state == "tentative"
    assert t.hits == 1

    # First update: hits=2, becomes confirmed
    det = DetectionBox(class_name="Car", confidence=0.85, x1=102.0, y1=100.0, x2=152.0, y2=160.0)
    t.update(det)
    assert t.hits == 2
    assert t.state == "confirmed"
    assert t.confidence == 0.85


def test_adverse_weather_bytetracker():
    tracker = AdverseWeatherByteTracker(
        high_thresh=0.40,
        low_thresh=0.15,
        confirm_frames=1,
        max_age=3,
    )

    # Frame 1: High confidence detection
    dets_f1 = [
        DetectionBox(class_name="Car", confidence=0.80, x1=50.0, y1=50.0, x2=100.0, y2=100.0),
    ]
    out_f1 = tracker.update(dets_f1)
    assert len(out_f1) == 1
    assert out_f1[0].track_id is not None
    tid = out_f1[0].track_id

    # Frame 2: Low-confidence detection (weather degradation: e.g. wiper sweep c=0.25)
    dets_f2 = [
        DetectionBox(class_name="Car", confidence=0.25, x1=52.0, y1=50.0, x2=102.0, y2=100.0),
    ]
    out_f2 = tracker.update(dets_f2)
    assert len(out_f2) == 1
    # Check that ByteTrack matched the low-confidence detection to the same track!
    assert out_f2[0].track_id == tid


def test_temporal_confidence_recovery():
    tc_filter = TemporalConfidenceFilter(
        alpha=0.60,
        recovery_max_gap=1,
        decay_factor=0.85,
        output_threshold=0.20,
    )

    # Frame 1: Detection present
    d1 = [DetectionBox(class_name="Car", confidence=0.80, x1=10.0, y1=10.0, x2=50.0, y2=50.0, track_id="1")]
    out1, diag1 = tc_filter.process_frame(d1, enable_recovery=True)
    assert len(out1) == 1
    assert diag1["recovered_count"] == 0

    # Frame 2: Object missed by detector (empty input)
    d2 = []
    out2, diag2 = tc_filter.process_frame(d2, enable_recovery=True)
    assert len(out2) == 1
    assert diag2["recovered_count"] == 1
    assert out2[0].confidence == pytest.approx(0.80 * 0.85, 0.01)

    # Frame 3: Object missed again (exceeds recovery_max_gap=1)
    d3 = []
    out3, diag3 = tc_filter.process_frame(d3, enable_recovery=True)
    assert len(out3) == 0  # expired


def test_temporal_router_hold_time():
    router = TemporalRouter(method="hold_time", tau_high=1.0, min_hold_frames=3)
    router.reset()

    # Frame 1: k_small = 1 (triggers escalation)
    dec1 = router.update(1.0)
    assert dec1.escalate is True

    # Frame 2: k_small = 0 (held by hold_time)
    dec2 = router.update(0.0)
    assert dec2.escalate is True

    # Frame 3: k_small = 0 (held by hold_time)
    dec3 = router.update(0.0)
    assert dec3.escalate is True

    # Frame 4: k_small = 0 (final hold frame)
    dec4 = router.update(0.0)
    assert dec4.escalate is True

    # Frame 5: k_small = 0 (hold_time expired)
    dec5 = router.update(0.0)
    assert dec5.escalate is False


def test_temporal_false_alarm_suppression():
    suppressor = TemporalFalseAlarmSuppressor(min_persistence_frames=2, stationary_iou_thresh=0.85)

    # Frame 1: Fleeting artifact (first time seen)
    d1 = [DetectionBox(class_name="Car", confidence=0.75, x1=20.0, y1=20.0, x2=40.0, y2=40.0, track_id="artifact_1")]
    out1, diag1 = suppressor.filter_frame(d1)
    assert len(out1) == 0  # suppressed (age=1 < 2)
    assert diag1["suppressed_ephemeral_boxes"] == 1

    # Frame 2: Persistent object (second time seen)
    d2 = [DetectionBox(class_name="Car", confidence=0.75, x1=21.0, y1=20.0, x2=41.0, y2=40.0, track_id="artifact_1")]
    out2, diag2 = suppressor.filter_frame(d2)
    assert len(out2) == 1  # passed (age=2 >= 2)


def test_evaluate_video_sequence():
    gt_box = GroundTruthBox(track_id="gt1", class_name="Car", x1=10.0, y1=10.0, x2=50.0, y2=50.0)
    frame0 = VideoFrame(frame_index=0, image_path=Path("dummy0.jpg"), timestamp_seconds=0.0, gt_boxes=[gt_box])
    frame1 = VideoFrame(frame_index=1, image_path=Path("dummy1.jpg"), timestamp_seconds=0.2, gt_boxes=[gt_box])
    seq = VideoSequence(sequence_id="test_seq", condition="rainy", timeofday="day", scene="city", fps=5.0, frames=[frame0, frame1])

    preds = [
        [DetectionBox(class_name="Car", confidence=0.90, x1=10.0, y1=10.0, x2=50.0, y2=50.0)],
        [],  # missed on frame 1
    ]

    res = evaluate_video_sequence(seq, preds, iou_threshold=0.50)
    assert res["tp"] == 1
    assert res["fn"] == 1
    assert res["fp"] == 0
    assert pytest.approx(res["f1"], 0.01) == 2.0 / 3.0  # 0.6667
    assert res["temporal_stability"]["mean_track_coverage"] == 0.50
