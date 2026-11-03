"""Tests for Mission 15 Adaptive Vanishing-Point and Horizon Anchor Candidate Generator."""
import cv2
import numpy as np
import pytest

from src.mission15.horizon_anchor import (
    AdaptiveHorizonAnchor,
    HorizonAnchorConfig,
)
from src.priority_allocation import CandidateProposal


def create_synthetic_road_scene(w=1280, h=720, vp_x=640, vp_y=360):
    """Generates a synthetic road frame with left and right lane markings converging at (vp_x, vp_y)."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 80  # Dark gray road

    # Sky above horizon
    img[:vp_y, :] = 160

    # Draw converging lane markings
    # Left lane line from (200, 720) to (vp_x, vp_y)
    cv2.line(img, (200, 720), (vp_x, vp_y), (255, 255, 255), 4)
    # Right lane line from (1080, 720) to (vp_x, vp_y)
    cv2.line(img, (1080, 720), (vp_x, vp_y), (255, 255, 255), 4)

    return img


def test_vanishing_point_estimation():
    """Verify that Hough line intersections recover the geometric road vanishing point."""
    w, h = 1280, 720
    true_vp_x, true_vp_y = 640, 360
    img = create_synthetic_road_scene(w, h, true_vp_x, true_vp_y)

    config = HorizonAnchorConfig(min_line_length=30)
    anchor = AdaptiveHorizonAnchor(config)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    est_vp = anchor._estimate_vanishing_point(gray, w, h)

    assert est_vp is not None
    # Estimation should be close to true intersection within ±25px
    assert abs(est_vp[0] - true_vp_x) < 25.0
    assert abs(est_vp[1] - true_vp_y) < 25.0


def test_temporal_smoothing_and_reset():
    """Verify EMA smoothing across frames and state reset across sequence boundaries."""
    config = HorizonAnchorConfig(ema_beta=0.80)
    anchor = AdaptiveHorizonAnchor(config)

    img1 = create_synthetic_road_scene(vp_x=640, vp_y=360)
    anchor.process_frame(img1, sequence_id="seq_1")
    assert anchor.frame_index == 1
    vp1 = anchor.smoothed_vp

    # Next frame with slightly shifted camera pitch (vp_y = 370)
    img2 = create_synthetic_road_scene(vp_x=640, vp_y=370)
    anchor.process_frame(img2, sequence_id="seq_1")
    assert anchor.frame_index == 2
    vp2 = anchor.smoothed_vp

    # Smoothed VP should move smoothly between 360 and 370
    assert 360.0 <= vp2[1] <= 370.0

    # Reset on new sequence
    anchor.process_frame(img1, sequence_id="seq_2")
    assert anchor.current_sequence_id == "seq_2"
    assert anchor.frame_index == 1


def test_horizon_candidate_extraction():
    """Verify that distant targets inside the road convergence cone generate proposals."""
    img = create_synthetic_road_scene(vp_x=640, vp_y=360)

    # Insert small vehicle patch (14x14px) near vanishing point in the road lane
    img[385:399, 633:647] = 230

    config = HorizonAnchorConfig(min_candidate_size=6, max_candidate_size=32)
    anchor = AdaptiveHorizonAnchor(config)

    props = anchor.process_frame(img, sequence_id="seq_horizon")

    assert len(props) > 0
    # Should identify a candidate near the target patch
    matched = False
    for p in props:
        assert isinstance(p, CandidateProposal)
        assert p.source == "horizon_anchor"
        assert p.metadata["vp_x"] == pytest.approx(640, abs=30)
        if p.x1 <= 633 and p.x2 >= 647 and p.y1 <= 385 and p.y2 >= 399:
            matched = True
            assert 0.15 <= p.score <= 1.0

    assert matched, "Target patch near horizon was not proposed"


def test_sky_and_lateral_background_rejection():
    """Verify that high-contrast blobs in the sky or lateral background are excluded."""
    img = create_synthetic_road_scene(vp_x=640, vp_y=360)

    # Insert noise patch high up in sky (y = 150)
    img[140:154, 630:644] = 240
    # Insert noise patch far to the side outside road cone (x = 50, y = 370)
    img[370:384, 50:64] = 240

    config = HorizonAnchorConfig()
    anchor = AdaptiveHorizonAnchor(config)

    props = anchor.process_frame(img, sequence_id="seq_reject")

    # Sky and side patches should not be proposed
    for p in props:
        assert p.y1 >= 330.0  # Must be near horizon or road
        assert p.x1 >= 250.0  # Must be inside road cone
