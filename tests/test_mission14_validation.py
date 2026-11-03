"""Unit and integration test suite for Mission 14 validation and reproducibility."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.video_dataset import (
    CANONICAL_CLASSES,
    CANONICAL_MAPPING,
    DetectionBox,
    GroundTruthBox,
)
from src.priority_allocation import PriorityAllocationController
from src.roi_inference import merge_detections_nms
from src.temporal_router import TemporalRouter
from scripts.run_mission13_suite import evaluate_ultra_small_scale_metrics

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts" / "mission14"
DATA_DIR = ROOT / "data" / "mission14_independent"


class TestMission14Validation:
    """Verifies sequence exclusion, cohort partitioning, manifests, and core algorithms."""

    def test_sequence_exclusion_integrity(self):
        """Verifies that all 14 historical sequences are strictly excluded."""
        exclusion_file = ARTIFACTS_DIR / "sequence_exclusion_manifest.json"
        assert exclusion_file.exists(), f"Exclusion manifest missing at {exclusion_file}"

        with open(exclusion_file) as f:
            data = json.load(f)

        assert data["total_historical_sequences_excluded"] == 14
        excluded_sids = {e["sequence_id"] for e in data["exclusions"]}
        assert len(excluded_sids) == 14

        # Historical lists
        m9_sids = {
            "b1e9ee0e-67e26f2e", "b1d0a191-65deaeef", "b24071b8-b3ee1196", "b2208b0f-2796a692",
            "b2064e61-2beadd45", "b1d0a191-03dcecc2", "b1d968b9-ce42734f", "b1ca2e5d-84cf9134",
        }
        m10_13_sids = {
            "b24380ab-6dbeb908", "b255cd6c-2f889586", "b1e1a7b8-65ec7612", "b1d0a191-06deb55d",
            "b24d283f-33783d1b", "b23c9e00-b425de1b",
        }
        all_hist = m9_sids | m10_13_sids
        assert excluded_sids == all_hist, "Exclusion manifest does not match historical set exactly"

    def test_no_train_test_sequence_overlap(self):
        """Verifies zero overlap between Development cohort, Locked Test cohort, and Historical exclusions."""
        manifest_file = ARTIFACTS_DIR / "sequence_manifest.json"
        assert manifest_file.exists(), f"Sequence manifest missing at {manifest_file}"

        with open(manifest_file) as f:
            manifest = json.load(f)

        dev_sids = {s["sequence_id"] for s in manifest["cohorts"]["development_cohort"]["sequences"]}
        test_sids = {s["sequence_id"] for s in manifest["cohorts"]["locked_test_cohort"]["sequences"]}

        assert len(dev_sids) == 2, "Development cohort must have exactly 2 sequences"
        assert len(test_sids) == 6, "Locked test cohort must have exactly 6 sequences"

        # Disjoint cohorts
        overlap = dev_sids & test_sids
        assert len(overlap) == 0, f"Data leakage: Dev and Test overlap on {overlap}"

        # Exclusion check
        exclusion_file = ARTIFACTS_DIR / "sequence_exclusion_manifest.json"
        with open(exclusion_file) as f:
            excl_data = json.load(f)
        excluded_sids = {e["sequence_id"] for e in excl_data["exclusions"]}

        assert len(dev_sids & excluded_sids) == 0, "Dev cohort contains historical sequence!"
        assert len(test_sids & excluded_sids) == 0, "Test cohort contains historical sequence!"

    def test_sequence_frames_and_labels_intact(self):
        """Verifies that all 8 sequences have their continuous images and JSON labels intact on disk."""
        manifest_file = ARTIFACTS_DIR / "sequence_manifest.json"
        with open(manifest_file) as f:
            manifest = json.load(f)

        all_sids = [s["sequence_id"] for s in manifest["cohorts"]["development_cohort"]["sequences"]] + \
                   [s["sequence_id"] for s in manifest["cohorts"]["locked_test_cohort"]["sequences"]]

        for sid in all_sids:
            seq_dir = DATA_DIR / "sequences" / sid
            lbl_file = DATA_DIR / "labels" / f"{sid}.json"
            assert seq_dir.exists(), f"Missing sequence directory: {seq_dir}"
            assert lbl_file.exists(), f"Missing label file: {lbl_file}"

            frames = list(seq_dir.glob("*.jpg"))
            assert len(frames) >= 200, f"Sequence {sid} has only {len(frames)} frames (<200)"

            with open(lbl_file) as f:
                labels_data = json.load(f)
            assert len(labels_data) >= 200, f"Label {lbl_file} has only {len(labels_data)} frames"

    def test_object_size_assignment_boundaries(self):
        """Tests that object sizes are categorized according to strict area thresholds."""
        ultra_box = GroundTruthBox(track_id="1", class_name="car", x1=10, y1=10, x2=22, y2=22)  # 12x12 = 144 < 256
        small_box = GroundTruthBox(track_id="2", class_name="car", x1=10, y1=10, x2=34, y2=34)  # 24x24 = 576 (256-1024)
        med_box = GroundTruthBox(track_id="3", class_name="car", x1=10, y1=10, x2=60, y2=60)    # 50x50 = 2500 (1024-9216)
        large_box = GroundTruthBox(track_id="4", class_name="car", x1=10, y1=10, x2=150, y2=150) # 140x140 = 19600 >= 9216

        assert ultra_box.area < 256.0
        assert 256.0 <= small_box.area < 1024.0
        assert 1024.0 <= med_box.area < 9216.0
        assert large_box.area >= 9216.0

    def test_iou_matching_and_metric_computation(self):
        """Tests IoU matching logic and scale-separated metrics."""
        gt = [
            GroundTruthBox(track_id="1", class_name="car", x1=10, y1=10, x2=24, y2=24),  # 14x14 = 196 (ultra-small)
            GroundTruthBox(track_id="2", class_name="car", x1=100, y1=100, x2=125, y2=125), # 25x25 = 625 (small)
        ]
        # Pred 1 matches GT 1 exactly
        # Pred 2 matches GT 2 with IoU > 0.5
        preds = [
            DetectionBox(class_name="car", confidence=0.85, x1=10, y1=10, x2=24, y2=24),
            DetectionBox(class_name="car", confidence=0.75, x1=102, y1=102, x2=125, y2=125),
        ]

        m = evaluate_ultra_small_scale_metrics([preds], [[gt[0], gt[1]]], iou_thresh=0.50)
        assert m["tp"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["f1"] == 1.0
        assert m["scale_breakdown"]["ultra_small"]["tp"] == 1
        assert m["scale_breakdown"]["small"]["tp"] == 1

    def test_paired_bootstrap_math_consistency(self):
        """Verifies that paired sequence bootstrap preserves pairing and bounds."""
        rng = np.random.RandomState(42)
        # Synthetic sequence scores
        scores_a = np.array([0.45, 0.48, 0.42, 0.46, 0.44, 0.47])
        scores_b = scores_a + 0.03  # deterministic +0.03 gain
        diffs = scores_b - scores_a

        boot_diffs = []
        for _ in range(500):
            idx = rng.choice(len(diffs), size=len(diffs), replace=True)
            boot_diffs.append(np.mean(diffs[idx]))

        ci_low = np.percentile(boot_diffs, 2.5)
        ci_high = np.percentile(boot_diffs, 97.5)

        assert abs(np.mean(boot_diffs) - 0.03) < 0.005
        assert ci_low > 0.0, "Bootstrap CI should strictly exclude 0 when true effect is positive"

    def test_priority_allocation_controller_policy_j_modes(self):
        """Verifies that Policy J allocates budget within bounds and preserves detector candidates."""
        ctrl = PriorityAllocationController(
            policy="policy_j_best_combination",
            max_crops=2,
            cluster_distance_px=120.0,
            min_independent_score=0.45,
        )

        dummy_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100

        # Frame with 1 detector box
        det_box = DetectionBox("car", 0.32, 300.0, 310.0, 340.0, 340.0)
        dec = ctrl.allocate_crops(dummy_frame, detector_boxes=[det_box])

        assert dec.policy_name == "policy_j_best_combination"
        assert dec.retained_detector_count >= 1
        assert 1 <= dec.used_budget <= 2
        assert len(dec.allocated_crops) == dec.used_budget

    def test_two_tier_nms_merging_deduplication(self):
        """Verifies that two-tier NMS deduplicates overlapping crop and base predictions properly."""
        base_dets = [
            DetectionBox(class_name="car", confidence=0.30, x1=100, y1=100, x2=130, y2=130),
        ]
        # Higher-acuity crop detection covering the exact same object
        crop_dets = [
            DetectionBox(class_name="car", confidence=0.65, x1=101, y1=101, x2=129, y2=129),
        ]

        merged = merge_detections_nms(
            base_detections=base_dets,
            crop_detections=crop_dets,
            iou_threshold=0.45,
            base_conf_threshold=0.20,
            new_crop_conf_threshold=0.35,
            merge_mode="conservative",
        )

        assert len(merged) == 1, f"Expected exactly 1 merged detection, got {len(merged)}"
        # Crop detection with higher confidence should survive
        assert merged[0].confidence == 0.65
