"""
Unit tests for WeatherGuard Mission 8 Temporal Stability & Consistency
"""

import pytest
import numpy as np
from src.temporal_router import TemporalRouter, TemporalDecision
from src.temporal_consistency import TemporalConsistencyTracker


def test_moving_average_router():
    router = TemporalRouter(method="moving_average", window_size=3, tau_high=0.5)
    # Stream of inputs: [0, 0, 1, 1, 1]
    d1 = router.update(0.0)
    assert d1.escalate == False

    d2 = router.update(0.0)
    assert d2.escalate == False

    d3 = router.update(1.0)
    # avg = (0 + 0 + 1)/3 = 0.333 (< 0.5)
    assert d3.escalate == False

    d4 = router.update(1.0)
    # avg = (0 + 1 + 1)/3 = 0.667 (>= 0.5)
    assert d4.escalate == True


def test_exponential_moving_average_router():
    router = TemporalRouter(method="ema", ema_alpha=0.5, tau_high=0.5)
    d1 = router.update(0.0)
    assert d1.escalate == False

    d2 = router.update(1.0)
    # ema = 0.5 * 1.0 + 0.5 * 0.0 = 0.5 >= 0.5
    assert d2.escalate == True


def test_hysteresis_router_asymmetric_switching():
    router = TemporalRouter(method="hysteresis", tau_high=1.0, tau_low=0.3)
    assert router.current_escalated == False

    # Input 0.5 (between 0.3 and 1.0) -> stays False
    d1 = router.update(0.5)
    assert d1.escalate == False

    # Input 1.0 (>= 1.0) -> escalates to 832px (True)
    d2 = router.update(1.0)
    assert d2.escalate == True
    assert router.current_escalated == True

    # Input 0.5 (between 0.3 and 1.0) -> stays True! (Hysteresis memory)
    d3 = router.update(0.5)
    assert d3.escalate == True

    # Input 0.2 (<= 0.3) -> de-escalates to 640px (False)
    d4 = router.update(0.2)
    assert d4.escalate == False
    assert router.current_escalated == False


def test_hold_time_router():
    router = TemporalRouter(method="hold_time", min_hold_frames=3, tau_high=0.5)
    # Input 1.0 triggers escalation
    d1 = router.update(1.0)
    assert d1.escalate == True
    assert d1.state == "switched_on"

    # Subsequent inputs < threshold are held
    d2 = router.update(0.0)
    assert d2.escalate == True
    assert d2.state == "hold"

    d3 = router.update(0.0)
    assert d3.escalate == True
    assert d3.state == "hold"

    d4 = router.update(0.0)
    assert d4.escalate == True  # 3rd held frame

    d5 = router.update(0.0)
    assert d5.escalate == False # hold expired
    assert d5.state == "switched_off"


def test_temporal_consistency_tracker():
    tracker = TemporalConsistencyTracker(iou_threshold=0.30, max_age=2, min_hits=1)

    # Frame 1: Car at [100, 100, 200, 200]
    dets_f1 = [
        {"class_name": "Car", "score": 0.85, "box_xyxy": [100.0, 100.0, 200.0, 200.0]}
    ]
    out_f1 = tracker.update(dets_f1)
    assert len(out_f1) == 1
    assert "track_id" in out_f1[0]
    t_id = out_f1[0]["track_id"]

    # Frame 2: Car slightly moved to [102, 101, 202, 201]
    dets_f2 = [
        {"class_name": "Car", "score": 0.88, "box_xyxy": [102.0, 101.0, 202.0, 201.0]}
    ]
    out_f2 = tracker.update(dets_f2)
    assert len(out_f2) == 1
    assert out_f2[0]["track_id"] == t_id  # successfully associated!
