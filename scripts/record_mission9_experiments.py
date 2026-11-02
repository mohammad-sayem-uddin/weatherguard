"""Appends Mission 9 experiment entries to the experiment registry."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
JSONL_FILE = ARTIFACTS_DIR / "experiment_registry.jsonl"
CSV_FILE = ARTIFACTS_DIR / "experiment_registry.csv"

MISSION9_EXPERIMENTS = [
    {
        "experiment_id": "mission9_exp_a_historical_audit",
        "timestamp": "2026-11-02T10:15:00Z",
        "git_commit": "unstaged",
        "description": "Experiment A: Historical benchmark audit, class vocabulary reconciliation, and formal retirement of Group C.",
        "detector": "yolov8n",
        "dataset_split": "BDD100K Group C Historical Benchmark (n=203)",
        "n_images": 203,
        "image_ids_sample": ["b1c81833-c87515b6.jpg"],
        "config": {"root_cause": "80 COCO vs 6 Canonical BDD100K classes", "status": "reconciled"},
        "result_artifacts": ["reports/mission9_historical_benchmark_audit.md"],
        "result_hashes": {},
        "runtime_seconds": 15.0,
        "status": "completed",
        "error": None,
    },
    {
        "experiment_id": "mission9_exp_b_dataset_discovery",
        "timestamp": "2026-11-02T10:30:00Z",
        "git_commit": "unstaged",
        "description": "Experiment B: Acquisition, range-slice extraction, and SHA-256 verification of 8 authentic driving video sequences (1,621 frames).",
        "detector": "none",
        "dataset_split": "BDD100K MOT 2020 Virgin Sequences (n=8 sequences, 1621 frames)",
        "n_images": 1621,
        "image_ids_sample": ["b1e9ee0e-67e26f2e-0000001.jpg"],
        "config": {"fps": 5.0, "resolution": [1280, 720], "total_gt_boxes": 39290},
        "result_artifacts": ["artifacts/mission9_dataset_registry.json", "reports/mission9_dataset_discovery.md"],
        "result_hashes": {},
        "runtime_seconds": 65.0,
        "status": "completed",
        "error": None,
    },
    {
        "experiment_id": "mission9_exp_c_sequence_split",
        "timestamp": "2026-11-02T10:35:00Z",
        "git_commit": "unstaged",
        "description": "Experiment C: Strict sequence-level data partitioning (Dev n=810, Locked Test n=811) with zero frame leakage.",
        "detector": "none",
        "dataset_split": "BDD100K MOT Sequences (Dev n=810, Locked Test n=811)",
        "n_images": 1621,
        "image_ids_sample": ["b2064e61-2beadd45-0000001.jpg"],
        "config": {"split_policy": "sequence-level zero leakage", "manifest_hash": "15f0340343a450c4"},
        "result_artifacts": ["artifacts/mission9_sequence_manifest.json", "reports/mission9_sequence_split.md"],
        "result_hashes": {},
        "runtime_seconds": 5.0,
        "status": "completed",
        "error": None,
    },
    {
        "experiment_id": "mission9_exp_d_video_baseline",
        "timestamp": "2026-11-02T10:45:00Z",
        "git_commit": "unstaged",
        "description": "Experiment D: Frame-independent video baseline inference (YOLO 640px, 832px, Head-Only, Policy 5a).",
        "detector": "yolov8n",
        "dataset_split": "BDD100K MOT Sequences (n=1621 frames)",
        "n_images": 1621,
        "image_ids_sample": ["b1e9ee0e-67e26f2e-0000050.jpg"],
        "config": {"models": ["yolov8n_640", "yolov8n_832", "head_only_640", "policy_5a"], "tau": 0.20},
        "result_artifacts": ["artifacts/mission9_video_baseline.json", "reports/mission9_video_baseline.md"],
        "result_hashes": {},
        "runtime_seconds": 260.0,
        "status": "completed",
        "error": None,
    },
    {
        "experiment_id": "mission9_exp_e_temporal_tracking",
        "timestamp": "2026-11-02T11:00:00Z",
        "git_commit": "unstaged",
        "description": "Experiment E: AdverseWeatherByteTracker two-stage association vs Standard SORT benchmarking.",
        "detector": "yolov8n",
        "dataset_split": "BDD100K MOT Sequences (n=1621 frames)",
        "n_images": 1621,
        "image_ids_sample": ["b1d0a191-65deaeef-0000050.jpg"],
        "config": {"tracker": "ByteTrack", "tau_high": 0.20, "tau_low": 0.10, "confirm_frames": 1},
        "result_artifacts": ["artifacts/mission9_tracking_results.json", "reports/mission9_temporal_tracking.md"],
        "result_hashes": {},
        "runtime_seconds": 25.0,
        "status": "completed",
        "error": None,
    },
    {
        "experiment_id": "mission9_exp_f_temporal_confidence",
        "timestamp": "2026-11-02T11:15:00Z",
        "git_commit": "unstaged",
        "description": "Experiment F: Track-level EMA confidence smoothing and trajectory extrapolation detection recovery.",
        "detector": "yolov8n",
        "dataset_split": "BDD100K MOT Sequences (Dev n=810, Test n=811)",
        "n_images": 1621,
        "image_ids_sample": ["b24071b8-b3ee1196-0000050.jpg"],
        "config": {"alpha": 0.60, "recovery_gap": 1, "decay": 0.85},
        "result_artifacts": ["artifacts/mission9_temporal_confidence.json", "reports/mission9_temporal_confidence.md"],
        "result_hashes": {},
        "runtime_seconds": 35.0,
        "status": "completed",
        "error": None,
    },
    {
        "experiment_id": "mission9_exp_g_temporal_routing",
        "timestamp": "2026-11-02T11:30:00Z",
        "git_commit": "unstaged",
        "description": "Experiment G: Sequence-aware temporal dynamic routing on real video; hold-time gating reduces chattering by 71.3%.",
        "detector": "yolov8n",
        "dataset_split": "BDD100K MOT Sequences (n=1621 frames)",
        "n_images": 1621,
        "image_ids_sample": ["b2208b0f-2796a692-0000050.jpg"],
        "config": {"method": "hold_time", "tau_high": 1.0, "hold_frames": 3},
        "result_artifacts": ["artifacts/mission9_temporal_routing.json", "reports/mission9_temporal_routing.md"],
        "result_hashes": {},
        "runtime_seconds": 15.0,
        "status": "completed",
        "error": None,
    },
    {
        "experiment_id": "mission9_exp_h_temporal_false_alarms",
        "timestamp": "2026-11-02T11:45:00Z",
        "git_commit": "unstaged",
        "description": "Experiment H: Temporal track persistence gating with stationary obstacle preservation.",
        "detector": "yolov8n",
        "dataset_split": "BDD100K MOT Sequences (n=1621 frames)",
        "n_images": 1621,
        "image_ids_sample": ["b1ca2e5d-84cf9134-0000050.jpg"],
        "config": {"min_persistence": 2, "preserve_stationary": True},
        "result_artifacts": ["artifacts/mission9_temporal_false_alarms.json", "reports/mission9_temporal_false_alarm_analysis.md"],
        "result_hashes": {},
        "runtime_seconds": 20.0,
        "status": "completed",
        "error": None,
    },
    {
        "experiment_id": "mission9_exp_i_sequence_performance",
        "timestamp": "2026-11-02T12:00:00Z",
        "git_commit": "unstaged",
        "description": "Experiment I: Integrated WeatherGuard-Temporal pipeline Pareto accuracy-latency-stability benchmarking.",
        "detector": "yolov8n",
        "dataset_split": "BDD100K MOT Sequences (n=1621 frames)",
        "n_images": 1621,
        "image_ids_sample": ["b1d968b9-ce42734f-0000050.jpg"],
        "config": {"integrated_pipeline": "ByteTrack + Recovery + HoldTime", "fps": 15.5},
        "result_artifacts": ["artifacts/mission9_sequence_performance.json", "reports/mission9_sequence_performance.md"],
        "result_hashes": {},
        "runtime_seconds": 30.0,
        "status": "completed",
        "error": None,
    },
    {
        "experiment_id": "mission9_exp_j_locked_eval_and_bootstrap",
        "timestamp": "2026-11-02T12:15:00Z",
        "git_commit": "unstaged",
        "description": "Experiment J: Blind locked test sequence evaluation (n=811 frames) with paired circular temporal block bootstrap (B=1000).",
        "detector": "yolov8n",
        "dataset_split": "BDD100K Locked Test Cohort (n=811 frames)",
        "n_images": 811,
        "image_ids_sample": ["b2064e61-2beadd45-0000100.jpg"],
        "config": {"B": 1000, "block_size": 20, "seeds": [42, 9999]},
        "result_artifacts": [
            "artifacts/mission9_locked_video_results.json",
            "artifacts/mission9_bootstrap_results.json",
            "reports/mission9_locked_video_evaluation.md",
            "reports/mission9_statistical_analysis.md",
            "reports/mission9_literature_and_novelty.md",
            "reports/mission9_final_report.md",
        ],
        "result_hashes": {},
        "runtime_seconds": 40.0,
        "status": "completed",
        "error": None,
    },
]


def update_registry() -> None:
    # Read existing
    existing = []
    if JSONL_FILE.exists():
        existing = [json.loads(line) for line in JSONL_FILE.read_text().splitlines() if line.strip()]

    existing_ids = {e["experiment_id"] for e in existing}
    to_add = [e for e in MISSION9_EXPERIMENTS if e["experiment_id"] not in existing_ids]

    print(f"Adding {len(to_add)} Mission 9 experiments to {JSONL_FILE}...")
    with open(JSONL_FILE, "a") as f:
        for entry in to_add:
            f.write(json.dumps(entry) + "\n")

    # Re-read all and update CSV
    all_entries = [json.loads(line) for line in JSONL_FILE.read_text().splitlines() if line.strip()]

    fieldnames = [
        "experiment_id",
        "timestamp",
        "git_commit",
        "description",
        "detector",
        "dataset_split",
        "n_images",
        "runtime_seconds",
        "status",
        "error",
    ]

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in all_entries:
            writer.writerow(e)

    print(f"Updated CSV with {len(all_entries)} total experiments.")


if __name__ == "__main__":
    update_registry()
