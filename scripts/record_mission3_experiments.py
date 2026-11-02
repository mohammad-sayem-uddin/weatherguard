"""Appends Mission 3 experimental records to artifacts/experiment_registry.jsonl and reports/experiment_log.md."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiment_registry import append_entry, make_entry

entries = [
    {
        "id": "mission3_split_prep_and_manifest",
        "desc": "Sampling of disjoint Group B development cohort (160 images) and generation of 3-cohort split manifest.",
        "detector": "none",
        "split": "BDD100K Development (n=160)",
        "images": ["b1e1a7b8-b397c445.jpg"],
        "config": {"seed": 101, "disjoint_check": "passed", "cohort_size": 160},
        "artifacts": ["artifacts/mission3_split_manifest.json", "artifacts/mission3_development_summary.json"],
        "runtime": 35.0,
    },
    {
        "id": "mission3_baseline_evaluation",
        "desc": "Baseline evaluation of YOLOv8n (640px, tau=0.20) and Faster R-CNN on Group B Development cohort.",
        "detector": "both",
        "split": "BDD100K Development (n=160)",
        "images": ["b1e1a7b8-b397c445.jpg"],
        "config": {"yolo_tau": 0.20, "frcnn_tau": 0.60},
        "artifacts": ["artifacts/mission3_baseline.json", "artifacts/mission3_baseline.csv", "reports/mission3_baseline.md"],
        "runtime": 194.0,
    },
    {
        "id": "mission3_exp_a_multiscale",
        "desc": "Experiment A: Multi-Scale input sweep (640, 832, 960, 1280) on Group B Development cohort.",
        "detector": "yolov8n",
        "split": "BDD100K Development (n=160)",
        "images": ["b1e1a7b8-b397c445.jpg"],
        "config": {"resolutions": [640, 832, 960, 1280]},
        "artifacts": ["artifacts/mission3_multiscale.json", "artifacts/mission3_multiscale.csv", "artifacts/plots/mission3_multiscale.png"],
        "runtime": 113.0,
    },
    {
        "id": "mission3_exp_b_tiling",
        "desc": "Experiment B: Tiled sliced inference (2x2 grid with 20% overlap, 5 passes/img) on Group B.",
        "detector": "yolov8n",
        "split": "BDD100K Development (n=160)",
        "images": ["b1e1a7b8-b397c445.jpg"],
        "config": {"grid": "2x2", "overlap": 0.20, "nms_iou": 0.50},
        "artifacts": ["artifacts/mission3_tiling.json", "artifacts/mission3_tiling.csv", "artifacts/plots/mission3_tiling.png"],
        "runtime": 52.0,
    },
    {
        "id": "mission3_exp_c_calibration",
        "desc": "Experiment C: Scale-adaptive confidence calibration grid search (729 combinations) on Group B.",
        "detector": "yolov8n",
        "split": "BDD100K Development (n=160)",
        "images": ["b1e1a7b8-b397c445.jpg"],
        "config": {"grid_size": 729, "bins": ["small<1024", "med[1024,9216)", "large>=9216"]},
        "artifacts": ["artifacts/mission3_scale_calibration.json", "artifacts/mission3_scale_calibration.csv", "artifacts/plots/mission3_scale_calibration.png"],
        "runtime": 83.0,
    },
    {
        "id": "mission3_exp_d_e_combined_and_external_test",
        "desc": "Experiments D & E: Model selection on Group B and final external validation on Locked Test Cohort (Group C, n=203).",
        "detector": "both",
        "split": "BDD100K Locked Test (n=203)",
        "images": ["b1c81833-c87515b6.jpg"],
        "config": {"winning_res": 1280, "frozen_tau": 0.20},
        "artifacts": ["artifacts/mission3_dev_candidates.json", "artifacts/mission3_final_results.json", "artifacts/mission3_efficiency.json", "artifacts/plots/mission3_accuracy_latency.png"],
        "runtime": 78.0,
    },
    {
        "id": "mission3_paired_bootstrap",
        "desc": "Part 12: Paired image-level bootstrap (1,000 resamples, seeds 20261101 and 20261102) on locked test cohort.",
        "detector": "both",
        "split": "BDD100K Locked Test (n=203)",
        "images": ["b1c81833-c87515b6.jpg"],
        "config": {"B": 1000, "primary_seed": 20261101, "secondary_seed": 20261102},
        "artifacts": ["artifacts/mission3_bootstrap_results.json", "artifacts/mission3_bootstrap_results.csv", "artifacts/plots/mission3_bootstrap_distributions.png"],
        "runtime": 0.15,
    },
    {
        "id": "mission3_weather_scale_analysis",
        "desc": "Part 10: Weather-scale interaction analysis across rain, snow, overcast, clear, and fog.",
        "detector": "both",
        "split": "BDD100K Locked Test (n=203)",
        "images": ["b1c81833-c87515b6.jpg"],
        "config": {"conditions": ["rainy", "snowy", "overcast", "clear", "foggy"]},
        "artifacts": ["artifacts/mission3_weather_scale.json", "artifacts/mission3_weather_scale.csv", "artifacts/plots/mission3_weather_scale.png", "reports/mission3_weather_scale_interaction.md"],
        "runtime": 0.5,
    },
    {
        "id": "mission3_qualitative_case_studies",
        "desc": "Part 11: Generation of 10 composite quad-panel qualitative driving scenarios.",
        "detector": "both",
        "split": "BDD100K Locked Test (n=203)",
        "images": ["b1c81833-c87515b6.jpg"],
        "config": {"cases": 10},
        "artifacts": ["reports/mission3_qualitative_analysis.md"],
        "runtime": 2.0,
    },
]

for e in entries:
    ent = make_entry(
        experiment_id=e["id"],
        description=e["desc"],
        detector=e["detector"],
        dataset_split=e["split"],
        image_ids=e["images"],
        config=e["config"],
        result_artifacts=e["artifacts"],
        status="completed",
        runtime_seconds=e["runtime"],
    )
    append_entry(ent)
    print(f"Logged {e['id']}")
