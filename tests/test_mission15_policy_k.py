"""Tests for Mission 15 Policy K — Temporal-Horizon Priority Allocation."""
import numpy as np
import pytest

from src.mission15.policy_k import (
    CandidateAdmissionRecord,
    PolicyKConfig,
    PolicyKController,
)
from src.priority_allocation import (
    AllocatedCandidate,
    DetectionBox,
)


def test_policy_k_detector_capacity_reservation():
    """Verify that Tier 1 detector proposals receive highest priority and are not displaced."""
    controller = PolicyKController(PolicyKConfig(max_crops=2, crop_size=448))

    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100

    # Supply 2 valid small detector boxes in separate quadrants
    det_boxes = [
        DetectionBox(x1=100.0, y1=350.0, x2=130.0, y2=380.0, confidence=0.45, class_name="car"),
        DetectionBox(x1=900.0, y1=350.0, x2=930.0, y2=380.0, confidence=0.40, class_name="car"),
    ]

    crops, cands, timings = controller.process_and_allocate(frame, sequence_id="test_alloc", detector_boxes=det_boxes)

    assert len(crops) == 2
    # Both crops should cover the 2 detector boxes
    c1_encloses_d1 = (crops[0][0] <= 100 and crops[0][1] <= 350 and crops[0][2] >= 130 and crops[0][3] >= 380) or \
                     (crops[1][0] <= 100 and crops[1][1] <= 350 and crops[1][2] >= 130 and crops[1][3] >= 380)
    c2_encloses_d2 = (crops[0][0] <= 900 and crops[0][1] <= 350 and crops[0][2] >= 930 and crops[0][3] >= 380) or \
                     (crops[1][0] <= 900 and crops[1][1] <= 350 and crops[1][2] >= 930 and crops[1][3] >= 380)

    assert c1_encloses_d1
    assert c2_encloses_d2


def test_policy_k_spatial_redundancy_pruning():
    """Verify that independent proposals overlapping existing detector boxes are pruned."""
    controller = PolicyKController(PolicyKConfig(redundancy_iou_thresh=0.20))

    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100

    # Detector candidate at (600, 350, 630, 380)
    det_boxes = [
        DetectionBox(x1=600.0, y1=350.0, x2=630.0, y2=380.0, confidence=0.50, class_name="car")
    ]

    crops, cands, timings = controller.process_and_allocate(frame, sequence_id="test_redundancy", detector_boxes=det_boxes)

    # All admitted candidates should have max_detector_overlap_iou <= 0.20 or be from detector
    for cand in cands:
        if cand.source != "detector_high" and cand.source != "detector_low":
            assert cand.max_detector_overlap_iou <= 0.20


def test_policy_k_crop_budget_bound():
    """Verify that allocated crops never exceed configured max_crops."""
    for k_budget in [1, 2, 3]:
        controller = PolicyKController(PolicyKConfig(max_crops=k_budget))
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100

        # Create 5 distant detector proposals scattered across frame
        det_boxes = [
            DetectionBox(x1=float(100 + i * 220), y1=360.0, x2=float(120 + i * 220), y2=380.0, confidence=0.45, class_name="car")
            for i in range(5)
        ]

        crops, cands, timings = controller.process_and_allocate(frame, sequence_id=f"test_k_{k_budget}", detector_boxes=det_boxes)
        assert len(crops) <= k_budget


def test_policy_k_admission_logging(tmp_path):
    """Verify that admission and rejection records are faithfully captured and exportable."""
    controller = PolicyKController(PolicyKConfig())
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100

    det_boxes = [
        DetectionBox(x1=500.0, y1=350.0, x2=520.0, y2=370.0, confidence=0.40, class_name="car")
    ]

    crops, cands, timings = controller.process_and_allocate(frame, sequence_id="test_logging", detector_boxes=det_boxes)

    assert len(controller.admission_logs) > 0
    # Export to JSONL
    out_file = str(tmp_path / "test_admission_log.jsonl")
    count = controller.export_admission_logs(out_file)
    assert count == len(controller.admission_logs)
