"""Run external evaluation and threshold transfer study on BDD100K.

Evaluates Faster R-CNN ResNet-50 FPN and YOLOv8n on the 203-image external
cohort, evaluates locked DAWN thresholds (0.60 and 0.20), default threshold (0.50),
and diagnostic oracle thresholds, computing all transfer metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.external_dataset import CANONICAL_CLASSES, Bdd100kDataset
from src.inference import predictions_to_eval_records, run_inference
from src.metrics import evaluate_detections
from src.model import build_detector

ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data" / "bdd100k"
IMAGES_DIR = DATA_DIR / "images"
SAMPLES_JSON = DATA_DIR / "samples.json"

# Locked thresholds from DAWN development study
DAWN_SELECTED_THRESHOLDS = {
    "fasterrcnn_resnet50_fpn": 0.60,
    "yolov8n": 0.20,
}

# DAWN heldout baseline results (202 images) from threshold_transfer.json
DAWN_HELDOUT_BASELINES = {
    "fasterrcnn_resnet50_fpn": {
        "dev_thresh": 0.60,
        "f1": 0.6738,
        "precision": 0.7027,
        "recall": 0.6472,
        "mAP_50": 0.6866,
        "default_f1": 0.6580,
        "oracle_f1": 0.6800,
    },
    "yolov8n": {
        "dev_thresh": 0.20,
        "f1": 0.5781,
        "precision": 0.5190,
        "recall": 0.6525,
        "mAP_50": 0.5694,
        "default_f1": 0.4830,
        "oracle_f1": 0.5800,
    },
}


def sweep_thresholds(eval_records: list[dict], min_th: float = 0.05, max_th: float = 0.95, step: float = 0.05) -> list[dict]:
    thresholds = [round(th, 2) for th in np.arange(min_th, max_th + 1e-6, step)]
    sweep_results = []
    for th in thresholds:
        ev = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=th)
        sweep_results.append({
            "threshold": th,
            "f1": ev.micro["f1"],
            "precision": ev.micro["precision"],
            "recall": ev.micro["recall"],
            "tp": ev.micro["tp"],
            "fp": ev.micro["fp"],
            "fn": ev.micro["fn"],
            "mAP_50": ev.macro["mAP_50"],
        })
    return sweep_results


def evaluate_cohort_by_condition(eval_records: list[dict], score_threshold: float) -> dict[str, dict]:
    by_cond = defaultdict(list)
    for rec in eval_records:
        by_cond[rec["condition"]].append(rec)
    res = {}
    for cond, recs in by_cond.items():
        ev = evaluate_detections(recs, iou_threshold=0.5, score_threshold=score_threshold)
        res[cond] = {
            "n_images": len(recs),
            "f1": ev.micro["f1"],
            "precision": ev.micro["precision"],
            "recall": ev.micro["recall"],
            "tp": ev.micro["tp"],
            "fp": ev.micro["fp"],
            "fn": ev.micro["fn"],
            "mAP_50": ev.macro["mAP_50"],
        }
    return res


def evaluate_cohort_by_class(eval_records: list[dict], score_threshold: float) -> dict[str, dict]:
    ev = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=score_threshold)
    res = {}
    for item in ev.per_class:
        res[item.class_name] = {
            "n_gt": item.n_gt,
            "n_pred": item.n_pred,
            "tp": item.tp,
            "fp": item.fp,
            "fn": item.fn,
            "precision": item.precision,
            "recall": item.recall,
            "f1": item.f1,
            "ap": item.ap_50,
        }
    return res


def run_evaluation(detector_key: str, records: list[Any], recompute: bool = False) -> tuple[dict, list[dict]]:
    cache_path = ARTIFACTS / f"eval_external_{detector_key}.json"
    if cache_path.exists() and not recompute:
        print(f"Loading cached predictions for {detector_key} from {cache_path}...")
        with open(cache_path, "r") as f:
            data = json.load(f)
        return data, data["per_image"]

    detector_name_map = {
        "fasterrcnn_resnet50_fpn": "fasterrcnn_resnet50_fpn",
        "yolov8n": "yolov8n",
    }
    det_id = detector_name_map[detector_key]
    print(f"\nBuilding and loading detector: {det_id}...")
    detector = build_detector(det_id, score_threshold=0.0)
    t_load_start = time.perf_counter()
    detector.load()
    load_time = time.perf_counter() - t_load_start
    print(f"Loaded {detector.name} in {load_time:.2f}s on {detector.device}")

    print(f"Running inference on {len(records)} external images (warmup=True)...")
    run = run_inference(detector, records, score_threshold=0.0, warmup=True)
    print(f"Inference completed in {run.total_inference_seconds:.2f}s "
          f"(mean: {run.mean_inference_seconds*1000:.1f}ms/img, "
          f"median: {run.median_inference_seconds*1000:.1f}ms/img)")

    eval_records = predictions_to_eval_records(records, run.predictions)

    eval50 = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=0.5)

    payload = {
        "tag": f"external_{detector_key}",
        "dataset": "BDD100K (External Validation Cohort)",
        "detector_name": detector.name,
        "detector_key": detector_key,
        "detector_load_seconds": load_time,
        "inference_total_seconds": run.total_inference_seconds,
        "inference_mean_seconds": run.mean_inference_seconds,
        "inference_median_seconds": run.median_inference_seconds,
        "device": detector.device,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "image_dimensions": "1280x720",
        "n_images": len(records),
        "image_ids": [r.image_id for r in records],
        "eval_iou0.5_score0.5": eval50.to_dict(),
        "per_image": eval_records,
    }

    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"Cached predictions saved to {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")

    return payload, eval_records


def main() -> None:
    print("=" * 70)
    print("WEATHERGUARD MISSION 2 — EXTERNAL EVALUATION & THRESHOLD TRANSFER")
    print("=" * 70)

    # 1. Load dataset & cohort
    dataset = Bdd100kDataset.from_fiftyone_json(SAMPLES_JSON, images_dir=IMAGES_DIR)
    cohort = dataset.select_weather_stratified_cohort(seed=42)
    print(f"External evaluation cohort: {len(cohort)} images.")

    detectors = ["fasterrcnn_resnet50_fpn", "yolov8n"]
    results_by_detector = {}
    transfer_results = {}
    csv_rows = []

    for det_key in detectors:
        payload, eval_records = run_evaluation(det_key, cohort)

        # 1. Sweep thresholds
        print(f"\n--- Threshold sweep for {det_key} ---")
        sweep = sweep_thresholds(eval_records)
        oracle_entry = max(sweep, key=lambda x: x["f1"])
        oracle_th = oracle_entry["threshold"]
        oracle_f1 = oracle_entry["f1"]

        dawn_th = DAWN_SELECTED_THRESHOLDS[det_key]
        default_th = 0.50

        ev_dawn = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=dawn_th)
        ev_default = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=default_th)
        ev_oracle = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=oracle_th)

        # Baseline comparison
        dawn_base = DAWN_HELDOUT_BASELINES[det_key]
        dawn_heldout_f1 = dawn_base["f1"]
        ext_f1 = ev_dawn.micro["f1"]
        abs_f1_deg = ext_f1 - dawn_heldout_f1
        rel_f1_deg = (abs_f1_deg / dawn_heldout_f1) * 100.0

        p_change = ev_dawn.micro["precision"] - dawn_base["precision"]
        r_change = ev_dawn.micro["recall"] - dawn_base["recall"]
        thresh_gap = oracle_f1 - ext_f1

        # Condition breakdown at DAWN threshold
        cond_breakdown = evaluate_cohort_by_condition(eval_records, score_threshold=dawn_th)
        class_breakdown = evaluate_cohort_by_class(eval_records, score_threshold=dawn_th)

        transfer_results[det_key] = {
            "detector_name": payload["detector_name"],
            "dawn_selected_threshold": dawn_th,
            "default_threshold": default_th,
            "oracle_threshold": oracle_th,
            "dawn_heldout_baseline": dawn_base,
            "external_at_dawn_threshold": {
                "f1": ev_dawn.micro["f1"],
                "precision": ev_dawn.micro["precision"],
                "recall": ev_dawn.micro["recall"],
                "tp": ev_dawn.micro["tp"],
                "fp": ev_dawn.micro["fp"],
                "fn": ev_dawn.micro["fn"],
                "mAP_50": ev_dawn.macro["mAP_50"],
            },
            "external_at_default_threshold": {
                "f1": ev_default.micro["f1"],
                "precision": ev_default.micro["precision"],
                "recall": ev_default.micro["recall"],
                "tp": ev_default.micro["tp"],
                "fp": ev_default.micro["fp"],
                "fn": ev_default.micro["fn"],
                "mAP_50": ev_default.macro["mAP_50"],
            },
            "external_at_oracle_threshold": {
                "threshold": oracle_th,
                "f1": oracle_f1,
                "precision": oracle_entry["precision"],
                "recall": oracle_entry["recall"],
                "tp": oracle_entry["tp"],
                "fp": oracle_entry["fp"],
                "fn": oracle_entry["fn"],
                "mAP_50": oracle_entry["mAP_50"],
            },
            "degradation": {
                "absolute_f1_degradation": abs_f1_deg,
                "relative_f1_degradation_pct": rel_f1_deg,
                "precision_change": p_change,
                "recall_change": r_change,
                "threshold_transfer_gap": thresh_gap,
            },
            "condition_breakdown": cond_breakdown,
            "class_breakdown": class_breakdown,
            "latency": {
                "load_seconds": payload["detector_load_seconds"],
                "mean_inference_seconds": payload["inference_mean_seconds"],
                "median_inference_seconds": payload["inference_median_seconds"],
                "total_inference_seconds": payload["inference_total_seconds"],
            },
            "threshold_sweep": sweep,
        }

        # CSV row
        csv_rows.append({
            "detector": det_key,
            "operating_point": "DAWN_selected",
            "threshold": dawn_th,
            "f1": f"{ev_dawn.micro['f1']:.4f}",
            "precision": f"{ev_dawn.micro['precision']:.4f}",
            "recall": f"{ev_dawn.micro['recall']:.4f}",
            "mAP_50": f"{ev_dawn.macro['mAP_50']:.4f}",
            "tp": ev_dawn.micro["tp"],
            "fp": ev_dawn.micro["fp"],
            "fn": ev_dawn.micro["fn"],
            "abs_f1_shift": f"{abs_f1_deg:.4f}",
            "rel_f1_shift_pct": f"{rel_f1_deg:.2f}%",
        })
        csv_rows.append({
            "detector": det_key,
            "operating_point": "Default_0.50",
            "threshold": default_th,
            "f1": f"{ev_default.micro['f1']:.4f}",
            "precision": f"{ev_default.micro['precision']:.4f}",
            "recall": f"{ev_default.micro['recall']:.4f}",
            "mAP_50": f"{ev_default.macro['mAP_50']:.4f}",
            "tp": ev_default.micro["tp"],
            "fp": ev_default.micro["fp"],
            "fn": ev_default.micro["fn"],
            "abs_f1_shift": "N/A",
            "rel_f1_shift_pct": "N/A",
        })
        csv_rows.append({
            "detector": det_key,
            "operating_point": "Diagnostic_Oracle",
            "threshold": oracle_th,
            "f1": f"{oracle_f1:.4f}",
            "precision": f"{oracle_entry['precision']:.4f}",
            "recall": f"{oracle_entry['recall']:.4f}",
            "mAP_50": f"{oracle_entry['mAP_50']:.4f}",
            "tp": oracle_entry["tp"],
            "fp": oracle_entry["fp"],
            "fn": oracle_entry["fn"],
            "abs_f1_shift": "N/A",
            "rel_f1_shift_pct": "N/A",
        })

    # Save outputs
    results_json = ARTIFACTS / "cross_dataset_results.json"
    with open(results_json, "w") as f:
        json.dump(transfer_results, f, indent=2)
    print(f"\nSaved cross_dataset_results.json to {results_json}")

    transfer_json = ARTIFACTS / "cross_dataset_threshold_transfer.json"
    with open(transfer_json, "w") as f:
        json.dump(transfer_results, f, indent=2)
    print(f"Saved cross_dataset_threshold_transfer.json to {transfer_json}")

    results_csv = ARTIFACTS / "cross_dataset_results.csv"
    with open(results_csv, "w", newline="") as f:
        fieldnames = ["detector", "operating_point", "threshold", "f1", "precision", "recall", "mAP_50", "tp", "fp", "fn", "abs_f1_shift", "rel_f1_shift_pct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved cross_dataset_results.csv to {results_csv}")


if __name__ == "__main__":
    main()
