"""Runs Experiments D & E: Combined Scale-Aware Inference & External Test Evaluation.

1. Evaluates candidate configurations on Group B Development Cohort:
   - Config 1: Baseline YOLOv8n (640, tau=0.20)
   - Config 2: Multi-Scale YOLOv8n (832, fixed tau=0.20 & optimal single tau)
   - Config 3: Multi-Scale YOLOv8n (960, fixed tau=0.20 & optimal single tau)
   - Config 4: Multi-Scale YOLOv8n (1280, fixed tau=0.20 & optimal single tau)
   - Config 5: Tiled YOLOv8n (2x2 hybrid, fixed tau=0.20)
   - Config 6: Baseline YOLOv8n (640) + Scale-Adaptive Calibration (tau_s, tau_m, tau_l)
   - Config 7: Multi-Scale YOLOv8n (832) + Scale-Adaptive Calibration
   - Config 8: Multi-Scale YOLOv8n (960) + Scale-Adaptive Calibration
   - Config 9: Multi-Scale YOLOv8n (1280) + Scale-Adaptive Calibration
   - Reference: Faster R-CNN (tau=0.60)

2. Selects Pareto-optimal winning configuration strictly on Group B.
3. Freezes parameters and evaluates winning configuration on LOCKED Test Cohort (Group C, n=203).
4. Produces artifacts:
   - artifacts/mission3_dev_candidates.json
   - artifacts/mission3_dev_candidates.csv
   - artifacts/mission3_efficiency.json
   - artifacts/mission3_efficiency.csv
   - artifacts/plots/mission3_accuracy_latency.png
   - artifacts/mission3_final_results.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.external_dataset import CANONICAL_CLASSES, Bdd100kDataset
from src.inference import predictions_to_eval_records, run_inference
from src.metrics import evaluate_detections, greedy_match
from src.model import Detection, Prediction, build_detector

ARTIFACTS = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "bdd100k"
IMAGES_DIR = DATA_DIR / "images"
SAMPLES_JSON = DATA_DIR / "samples.json"
MANIFEST_JSON = ARTIFACTS / "mission3_split_manifest.json"


def get_predicted_scale_bin(box: list[float] | tuple[float, float, float, float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area < 1024.0:
        return "small"
    elif area < 9216.0:
        return "medium"
    else:
        return "large"


def evaluate_scale_breakdown_with_thresholds(
    eval_records: list[dict],
    thresholds: dict[str, float],
) -> dict[str, dict]:
    scales = {
        "small": {"gt": 0, "tp": 0, "fn": 0},
        "medium": {"gt": 0, "tp": 0, "fn": 0},
        "large": {"gt": 0, "tp": 0, "fn": 0},
    }

    for r in eval_records:
        gt_boxes = [g["box_xyxy"] for g in r["ground_truth"]]
        gt_labels = [g["class_name"] for g in r["ground_truth"]]

        preds = [
            p for p in r["predictions"]
            if p["score"] >= thresholds[get_predicted_scale_bin(p["box_xyxy"])]
            and p["common_label"] is not None
        ]
        pred_boxes = [p["box_xyxy"] for p in preds]
        pred_scores = [p["score"] for p in preds]
        pred_labels = [p["common_label"] for p in preds]

        matched_gt = set()
        for c in CANONICAL_CLASSES:
            c_p_idx = [i for i, l in enumerate(pred_labels) if l == c]
            c_g_idx = [j for j, l in enumerate(gt_labels) if l == c]
            if not c_p_idx or not c_g_idx:
                continue
            sub_p = [pred_boxes[i] for i in c_p_idx]
            sub_s = [pred_scores[i] for i in c_p_idx]
            sub_g = [gt_boxes[j] for j in c_g_idx]
            matches = greedy_match(sub_p, sub_s, sub_g, iou_threshold=0.5)
            for p_sub, g_sub, _ in matches:
                matched_gt.add(c_g_idx[g_sub])

        for j, g in enumerate(r["ground_truth"]):
            b = g["box_xyxy"]
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area < 1024:
                s_key = "small"
            elif area < 9216:
                s_key = "medium"
            else:
                s_key = "large"

            scales[s_key]["gt"] += 1
            if j in matched_gt:
                scales[s_key]["tp"] += 1
            else:
                scales[s_key]["fn"] += 1

    for s_key in ["small", "medium", "large"]:
        gt = scales[s_key]["gt"]
        tp = scales[s_key]["tp"]
        scales[s_key]["recall"] = (tp / gt) if gt > 0 else 0.0
        scales[s_key]["miss_rate"] = (scales[s_key]["fn"] / gt) if gt > 0 else 0.0

    return scales


def filter_predictions_scale_adaptive(
    eval_records: list[dict],
    thresholds: dict[str, float],
) -> list[dict]:
    out = []
    for r in eval_records:
        filt = []
        for p in r["predictions"]:
            sbin = get_predicted_scale_bin(p["box_xyxy"])
            if p["score"] >= thresholds[sbin]:
                filt.append(p)
        out.append({
            "image_id": r["image_id"],
            "condition": r.get("condition", "unknown"),
            "ground_truth": r["ground_truth"],
            "predictions": filt,
        })
    return out


def evaluate_condition_breakdown_adaptive(
    eval_records: list[dict],
    thresholds: dict[str, float],
) -> dict[str, dict]:
    by_cond = defaultdict(list)
    for r in eval_records:
        by_cond[r.get("condition", "unknown")].append(r)

    out = {}
    for cond, recs in by_cond.items():
        filt_recs = filter_predictions_scale_adaptive(recs, thresholds)
        ev = evaluate_detections(filt_recs, iou_threshold=0.5, score_threshold=0.0)
        scale_stats = evaluate_scale_breakdown_with_thresholds(recs, thresholds)
        out[cond] = {
            "n_images": len(recs),
            "f1": ev.micro["f1"],
            "precision": ev.micro["precision"],
            "recall": ev.micro["recall"],
            "tp": ev.micro["tp"],
            "fp": ev.micro["fp"],
            "fn": ev.micro["fn"],
            "mAP_50": ev.macro["mAP_50"],
            "small_recall": scale_stats["small"]["recall"],
            "medium_recall": scale_stats["medium"]["recall"],
            "large_recall": scale_stats["large"]["recall"],
        }
    return out


def run_yolo_inference_cached(
    detector_name: str,
    imgsz: int,
    records: list,
    cache_tag: str,
) -> tuple[dict, list[dict]]:
    cache_path = ARTIFACTS / f"eval_{cache_tag}.json"
    if cache_path.exists():
        print(f"Loading cached predictions from {cache_path}...")
        with open(cache_path) as f:
            data = json.load(f)
        return data, data["per_image"]

    print(f"\nBuilding and loading {detector_name} with imgsz={imgsz}...")
    detector = build_detector("yolov8n", score_threshold=0.0, imgsz=imgsz)
    t0 = time.perf_counter()
    detector.load()
    load_time = time.perf_counter() - t0

    print(f"Running inference on {len(records)} images (imgsz={imgsz}, warmup=True)...")
    run = run_inference(detector, records, score_threshold=0.0, warmup=True)
    print(f"Inference completed in {run.total_inference_seconds:.2f}s "
          f"(mean: {run.mean_inference_seconds*1000:.1f}ms, median: {run.median_inference_seconds*1000:.1f}ms)")

    eval_records = predictions_to_eval_records(records, run.predictions)
    ev50 = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=0.5)
    latencies = [p.inference_seconds for p in run.predictions]

    payload = {
        "tag": cache_tag,
        "detector_name": f"yolov8n_{imgsz}",
        "resolution": imgsz,
        "detector_load_seconds": load_time,
        "inference_total_seconds": run.total_inference_seconds,
        "inference_mean_seconds": run.mean_inference_seconds,
        "inference_median_seconds": run.median_inference_seconds,
        "inference_std_seconds": float(np.std(latencies)) if latencies else 0.0,
        "per_image_seconds": latencies,
        "device": detector.device,
        "n_images": len(records),
        "image_ids": [r.image_id for r in records],
        "eval_iou0.5_score0.5": ev50.to_dict(),
        "per_image": eval_records,
    }

    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"Saved cached predictions to {cache_path}")

    return payload, eval_records


def main() -> None:
    print("=" * 70)
    print("WEATHERGUARD MISSION 3 — EXPERIMENTS D & E: COMBINED INFERENCE & EXTERNAL TEST")
    print("=" * 70)

    # 1. Load Manifest
    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)

    dev_ids = set(manifest["groups"]["Group_B_BDD100K_Development"]["image_ids"])
    test_ids = set(manifest["groups"]["Group_C_BDD100K_Locked_Test"]["image_ids"])

    full_ds = Bdd100kDataset.from_fiftyone_json(SAMPLES_JSON, images_dir=IMAGES_DIR)
    dev_records = [r for r in full_ds if r.image_id in dev_ids]
    dev_records.sort(key=lambda r: r.image_id)
    test_records = [r for r in full_ds if r.image_id in test_ids]
    test_records.sort(key=lambda r: r.image_id)

    print(f"Loaded {len(dev_records)} Dev records and {len(test_records)} Locked Test records.")

    # 2. Load Dev Results from Experiments A, B, and C
    with open(ARTIFACTS / "mission3_baseline.json") as f:
        baseline_data = json.load(f)
    with open(ARTIFACTS / "mission3_multiscale.json") as f:
        multiscale_data = json.load(f)
    with open(ARTIFACTS / "mission3_tiling.json") as f:
        tiling_data = json.load(f)
    with open(ARTIFACTS / "mission3_scale_calibration.json") as f:
        calib_data = json.load(f)

    # 3. Assemble Dev Candidate Comparison Table
    candidates = []

    # Config 1: YOLO Baseline (640, tau=0.20)
    y_base = baseline_data["yolov8n_0.2"]
    candidates.append({
        "id": "C1_yolo_base_640",
        "name": "YOLOv8n Baseline (640, tau=0.20)",
        "resolution": 640,
        "calibration": "Fixed (0.20, 0.20, 0.20)",
        "latency_ms": y_base["latency"]["mean_ms"],
        "f1": y_base["overall"]["f1"],
        "precision": y_base["overall"]["precision"],
        "recall": y_base["overall"]["recall"],
        "mAP_50": y_base["overall"]["mAP_50"],
        "small_recall": y_base["scale_breakdown"]["small"]["recall"],
        "med_recall": y_base["scale_breakdown"]["medium"]["recall"],
        "large_recall": y_base["scale_breakdown"]["large"]["recall"],
    })

    # Config 2: YOLO 832 (tau=0.20)
    m832 = multiscale_data["832"]["eval_tau_0.20"]
    candidates.append({
        "id": "C2_yolo_ms_832",
        "name": "YOLOv8n Multi-Scale 832 (tau=0.20)",
        "resolution": 832,
        "calibration": "Fixed (0.20, 0.20, 0.20)",
        "latency_ms": multiscale_data["832"]["latency"]["mean_ms"],
        "f1": m832["f1"],
        "precision": m832["precision"],
        "recall": m832["recall"],
        "mAP_50": m832["mAP_50"],
        "small_recall": m832["scale_breakdown"]["small"]["recall"],
        "med_recall": m832["scale_breakdown"]["medium"]["recall"],
        "large_recall": m832["scale_breakdown"]["large"]["recall"],
    })

    # Config 3: YOLO 960 (tau=0.20)
    m960 = multiscale_data["960"]["eval_tau_0.20"]
    candidates.append({
        "id": "C3_yolo_ms_960",
        "name": "YOLOv8n Multi-Scale 960 (tau=0.20)",
        "resolution": 960,
        "calibration": "Fixed (0.20, 0.20, 0.20)",
        "latency_ms": multiscale_data["960"]["latency"]["mean_ms"],
        "f1": m960["f1"],
        "precision": m960["precision"],
        "recall": m960["recall"],
        "mAP_50": m960["mAP_50"],
        "small_recall": m960["scale_breakdown"]["small"]["recall"],
        "med_recall": m960["scale_breakdown"]["medium"]["recall"],
        "large_recall": m960["scale_breakdown"]["large"]["recall"],
    })

    # Config 4: YOLO 1280 (tau=0.20)
    m1280 = multiscale_data["1280"]["eval_tau_0.20"]
    candidates.append({
        "id": "C4_yolo_ms_1280",
        "name": "YOLOv8n Multi-Scale 1280 (tau=0.20)",
        "resolution": 1280,
        "calibration": "Fixed (0.20, 0.20, 0.20)",
        "latency_ms": multiscale_data["1280"]["latency"]["mean_ms"],
        "f1": m1280["f1"],
        "precision": m1280["precision"],
        "recall": m1280["recall"],
        "mAP_50": m1280["mAP_50"],
        "small_recall": m1280["scale_breakdown"]["small"]["recall"],
        "med_recall": m1280["scale_breakdown"]["medium"]["recall"],
        "large_recall": m1280["scale_breakdown"]["large"]["recall"],
    })

    # Config 5: Tiled 2x2 Hybrid
    t2x2 = tiling_data["threshold_eval"]["tau_0.20"]
    candidates.append({
        "id": "C5_yolo_tiled_2x2",
        "name": "YOLOv8n Tiled 2x2 Hybrid (tau=0.20)",
        "resolution": 640,
        "calibration": "Fixed (0.20, 0.20, 0.20)",
        "latency_ms": tiling_data["metadata"]["mean_latency_ms"],
        "f1": t2x2["f1"],
        "precision": t2x2["precision"],
        "recall": t2x2["recall"],
        "mAP_50": t2x2["mAP_50"],
        "small_recall": t2x2["scale_breakdown"]["small"]["recall"],
        "med_recall": t2x2["scale_breakdown"]["medium"]["recall"],
        "large_recall": t2x2["scale_breakdown"]["large"]["recall"],
    })

    # Config 6: YOLO 640 + Calibrated
    c640 = calib_data["yolov8n_640"]["scale_adaptive_calibration"]
    candidates.append({
        "id": "C6_yolo_calib_640",
        "name": f"YOLOv8n 640 + Calibrated ({c640['tau_small']:.2f}, {c640['tau_medium']:.2f}, {c640['tau_large']:.2f})",
        "resolution": 640,
        "calibration": f"Adaptive ({c640['tau_small']:.2f}, {c640['tau_medium']:.2f}, {c640['tau_large']:.2f})",
        "latency_ms": y_base["latency"]["mean_ms"],
        "f1": c640["f1"],
        "precision": c640["precision"],
        "recall": c640["recall"],
        "mAP_50": c640["mAP_50"],
        "small_recall": c640["scale_breakdown"]["small"]["recall"],
        "med_recall": c640["scale_breakdown"]["medium"]["recall"],
        "large_recall": c640["scale_breakdown"]["large"]["recall"],
    })

    # Config 7: YOLO 832 + Calibrated
    c832 = calib_data["yolov8n_832"]["scale_adaptive_calibration"]
    candidates.append({
        "id": "C7_yolo_calib_832",
        "name": f"YOLOv8n 832 + Calibrated ({c832['tau_small']:.2f}, {c832['tau_medium']:.2f}, {c832['tau_large']:.2f})",
        "resolution": 832,
        "calibration": f"Adaptive ({c832['tau_small']:.2f}, {c832['tau_medium']:.2f}, {c832['tau_large']:.2f})",
        "latency_ms": multiscale_data["832"]["latency"]["mean_ms"],
        "f1": c832["f1"],
        "precision": c832["precision"],
        "recall": c832["recall"],
        "mAP_50": c832["mAP_50"],
        "small_recall": c832["scale_breakdown"]["small"]["recall"],
        "med_recall": c832["scale_breakdown"]["medium"]["recall"],
        "large_recall": c832["scale_breakdown"]["large"]["recall"],
    })

    # Config 8: YOLO 960 + Calibrated
    c960 = calib_data["yolov8n_960"]["scale_adaptive_calibration"]
    candidates.append({
        "id": "C8_yolo_calib_960",
        "name": f"YOLOv8n 960 + Calibrated ({c960['tau_small']:.2f}, {c960['tau_medium']:.2f}, {c960['tau_large']:.2f})",
        "resolution": 960,
        "calibration": f"Adaptive ({c960['tau_small']:.2f}, {c960['tau_medium']:.2f}, {c960['tau_large']:.2f})",
        "latency_ms": multiscale_data["960"]["latency"]["mean_ms"],
        "f1": c960["f1"],
        "precision": c960["precision"],
        "recall": c960["recall"],
        "mAP_50": c960["mAP_50"],
        "small_recall": c960["scale_breakdown"]["small"]["recall"],
        "med_recall": c960["scale_breakdown"]["medium"]["recall"],
        "large_recall": c960["scale_breakdown"]["large"]["recall"],
    })

    # Reference: Faster R-CNN
    fr_base = baseline_data["fasterrcnn_resnet50_fpn_0.6"]
    candidates.append({
        "id": "REF_faster_rcnn_fpn",
        "name": "Faster R-CNN ResNet-50 FPN (tau=0.60)",
        "resolution": 1333,
        "calibration": "Fixed (0.60, 0.60, 0.60)",
        "latency_ms": fr_base["latency"]["mean_ms"],
        "f1": fr_base["overall"]["f1"],
        "precision": fr_base["overall"]["precision"],
        "recall": fr_base["overall"]["recall"],
        "mAP_50": fr_base["overall"]["mAP_50"],
        "small_recall": fr_base["scale_breakdown"]["small"]["recall"],
        "med_recall": fr_base["scale_breakdown"]["medium"]["recall"],
        "large_recall": fr_base["scale_breakdown"]["large"]["recall"],
    })

    # Save candidates artifacts
    with open(ARTIFACTS / "mission3_dev_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)
    with open(ARTIFACTS / "mission3_dev_candidates.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)
    print(f"Saved {ARTIFACTS / 'mission3_dev_candidates.json'} and CSV")

    # 4. Selection of Winner on Group B:
    # Filter only YOLO candidates and rank by F1
    yolo_candidates = [c for c in candidates if c["id"] != "REF_faster_rcnn_fpn"]
    yolo_candidates.sort(key=lambda c: c["f1"], reverse=True)
    winner = yolo_candidates[0]
    print("\n" + "=" * 70)
    print(f"WINNING CONFIGURATION SELECTED ON GROUP B DEV COHORT: {winner['name']}")
    print(f"  Development F1: {winner['f1']:.4f} (vs Baseline: {y_base['overall']['f1']:.4f}, Gain: +{(winner['f1'] - y_base['overall']['f1'])*100:.2f} pp)")
    print(f"  Small Recall:   {winner['small_recall']*100:.2f}% (vs Baseline: {y_base['scale_breakdown']['small']['recall']*100:.2f}%)")
    print(f"  Mean Latency:   {winner['latency_ms']:.1f} ms ({1000/winner['latency_ms']:.1f} FPS)")
    print("=" * 70)

    # Parse winning parameters
    winning_resolution = winner["resolution"]
    if "Adaptive" in winner["calibration"]:
        # Extract triplet from winner id / calib_data
        calib_entry = calib_data[f"yolov8n_{winning_resolution}"]["scale_adaptive_calibration"]
        winning_thresholds = {
            "small": calib_entry["tau_small"],
            "medium": calib_entry["tau_medium"],
            "large": calib_entry["tau_large"],
        }
    else:
        winning_thresholds = {"small": 0.20, "medium": 0.20, "large": 0.20}

    print(f"\nFROZEN PARAMETERS FOR LOCKED TEST EVALUATION:")
    print(f"  Resolution: {winning_resolution}")
    print(f"  Thresholds: {winning_thresholds}")

    # 5. Evaluate WINNING CONFIGURATION on LOCKED Test Cohort (Group C, n=203)
    # Run or load YOLO inference at winning_resolution on test_records
    tag_test = f"mission3_test_yolov8n_{winning_resolution}"
    test_payload, test_eval_recs = run_yolo_inference_cached(
        detector_name=f"yolov8n_{winning_resolution}",
        imgsz=winning_resolution,
        records=test_records,
        cache_tag=tag_test,
    )

    # Filter with frozen scale-adaptive thresholds
    filt_test_recs = filter_predictions_scale_adaptive(test_eval_recs, winning_thresholds)
    test_ev = evaluate_detections(filt_test_recs, iou_threshold=0.5, score_threshold=0.0)
    test_scale_stats = evaluate_scale_breakdown_with_thresholds(test_eval_recs, winning_thresholds)
    test_cond_stats = evaluate_condition_breakdown_adaptive(test_eval_recs, winning_thresholds)

    # Load baseline YOLO test results (from Mission 2)
    with open(ARTIFACTS / "eval_external_yolov8n.json") as f:
        m2_yolo_test = json.load(f)
    ev_m2_yolo = evaluate_detections(m2_yolo_test["per_image"], iou_threshold=0.5, score_threshold=0.20)
    scale_m2_yolo = evaluate_scale_breakdown_with_thresholds(m2_yolo_test["per_image"], {"small": 0.20, "medium": 0.20, "large": 0.20})
    cond_m2_yolo = evaluate_condition_breakdown_adaptive(m2_yolo_test["per_image"], {"small": 0.20, "medium": 0.20, "large": 0.20})

    # Load baseline Faster R-CNN test results (from Mission 2)
    with open(ARTIFACTS / "eval_external_fasterrcnn_resnet50_fpn.json") as f:
        m2_frcnn_test = json.load(f)
    ev_m2_frcnn = evaluate_detections(m2_frcnn_test["per_image"], iou_threshold=0.5, score_threshold=0.60)
    scale_m2_frcnn = evaluate_scale_breakdown_with_thresholds(m2_frcnn_test["per_image"], {"small": 0.60, "medium": 0.60, "large": 0.60})
    cond_m2_frcnn = evaluate_condition_breakdown_adaptive(m2_frcnn_test["per_image"], {"small": 0.60, "medium": 0.60, "large": 0.60})

    # Save final results comparison on Locked Test Set
    final_test_results = {
        "locked_test_cohort": {
            "name": "BDD100K Locked Evaluation Cohort (Group C)",
            "n_images": 203,
            "provenance": "Locked Mission 2 test split",
        },
        "models": {
            "yolov8n_baseline": {
                "name": "YOLOv8n Baseline (640, tau=0.20)",
                "resolution": 640,
                "latency_mean_ms": m2_yolo_test["inference_mean_seconds"] * 1000.0,
                "latency_median_ms": m2_yolo_test["inference_median_seconds"] * 1000.0,
                "f1": ev_m2_yolo.micro["f1"],
                "precision": ev_m2_yolo.micro["precision"],
                "recall": ev_m2_yolo.micro["recall"],
                "mAP_50": ev_m2_yolo.macro["mAP_50"],
                "tp": ev_m2_yolo.micro["tp"],
                "fp": ev_m2_yolo.micro["fp"],
                "fn": ev_m2_yolo.micro["fn"],
                "scale_breakdown": scale_m2_yolo,
                "condition_breakdown": cond_m2_yolo,
            },
            "yolov8n_scale_aware_optimized": {
                "name": f"Scale-Aware YOLOv8n ({winning_resolution}px + Adaptive Calibration)",
                "resolution": winning_resolution,
                "thresholds": winning_thresholds,
                "latency_mean_ms": test_payload["inference_mean_seconds"] * 1000.0,
                "latency_median_ms": test_payload["inference_median_seconds"] * 1000.0,
                "f1": test_ev.micro["f1"],
                "precision": test_ev.micro["precision"],
                "recall": test_ev.micro["recall"],
                "mAP_50": test_ev.macro["mAP_50"],
                "tp": test_ev.micro["tp"],
                "fp": test_ev.micro["fp"],
                "fn": test_ev.micro["fn"],
                "scale_breakdown": test_scale_stats,
                "condition_breakdown": test_cond_stats,
            },
            "faster_rcnn_reference": {
                "name": "Faster R-CNN ResNet-50 FPN (tau=0.60)",
                "resolution": 1333,
                "latency_mean_ms": m2_frcnn_test["inference_mean_seconds"] * 1000.0,
                "latency_median_ms": m2_frcnn_test["inference_median_seconds"] * 1000.0,
                "f1": ev_m2_frcnn.micro["f1"],
                "precision": ev_m2_frcnn.micro["precision"],
                "recall": ev_m2_frcnn.micro["recall"],
                "mAP_50": ev_m2_frcnn.macro["mAP_50"],
                "tp": ev_m2_frcnn.micro["tp"],
                "fp": ev_m2_frcnn.micro["fp"],
                "fn": ev_m2_frcnn.micro["fn"],
                "scale_breakdown": scale_m2_frcnn,
                "condition_breakdown": cond_m2_frcnn,
            },
        },
        "gains_vs_yolo_baseline": {
            "f1_delta": test_ev.micro["f1"] - ev_m2_yolo.micro["f1"],
            "recall_delta": test_ev.micro["recall"] - ev_m2_yolo.micro["recall"],
            "small_recall_delta": test_scale_stats["small"]["recall"] - scale_m2_yolo["small"]["recall"],
            "mAP_50_delta": test_ev.macro["mAP_50"] - ev_m2_yolo.macro["mAP_50"],
            "latency_slowdown_factor": test_payload["inference_mean_seconds"] / m2_yolo_test["inference_mean_seconds"],
            "speedup_vs_faster_rcnn": m2_frcnn_test["inference_mean_seconds"] / test_payload["inference_mean_seconds"],
        },
    }

    with open(ARTIFACTS / "mission3_final_results.json", "w") as f:
        json.dump(final_test_results, f, indent=2)
    print(f"Saved {ARTIFACTS / 'mission3_final_results.json'}")

    # Save Efficiency Artifacts
    eff_rows = [
        {
            "model": "YOLOv8n Baseline (640)",
            "resolution": 640,
            "mean_latency_ms": f"{m2_yolo_test['inference_mean_seconds']*1000.0:.1f}",
            "median_latency_ms": f"{m2_yolo_test['inference_median_seconds']*1000.0:.1f}",
            "fps": f"{1.0/m2_yolo_test['inference_mean_seconds']:.1f}",
            "f1": f"{ev_m2_yolo.micro['f1']:.4f}",
            "small_recall": f"{scale_m2_yolo['small']['recall']*100:.2f}%",
            "speedup_vs_frcnn": f"{m2_frcnn_test['inference_mean_seconds']/m2_yolo_test['inference_mean_seconds']:.1f}x",
        },
        {
            "model": f"Scale-Aware YOLOv8n ({winning_resolution})",
            "resolution": winning_resolution,
            "mean_latency_ms": f"{test_payload['inference_mean_seconds']*1000.0:.1f}",
            "median_latency_ms": f"{test_payload['inference_median_seconds']*1000.0:.1f}",
            "fps": f"{1.0/test_payload['inference_mean_seconds']:.1f}",
            "f1": f"{test_ev.micro['f1']:.4f}",
            "small_recall": f"{test_scale_stats['small']['recall']*100:.2f}%",
            "speedup_vs_frcnn": f"{m2_frcnn_test['inference_mean_seconds']/test_payload['inference_mean_seconds']:.1f}x",
        },
        {
            "model": "Faster R-CNN ResNet-50 FPN",
            "resolution": 1333,
            "mean_latency_ms": f"{m2_frcnn_test['inference_mean_seconds']*1000.0:.1f}",
            "median_latency_ms": f"{m2_frcnn_test['inference_median_seconds']*1000.0:.1f}",
            "fps": f"{1.0/m2_frcnn_test['inference_mean_seconds']:.1f}",
            "f1": f"{ev_m2_frcnn.micro['f1']:.4f}",
            "small_recall": f"{scale_m2_frcnn['small']['recall']*100:.2f}%",
            "speedup_vs_frcnn": "1.0x",
        },
    ]

    with open(ARTIFACTS / "mission3_efficiency.json", "w") as f:
        json.dump(eff_rows, f, indent=2)
    with open(ARTIFACTS / "mission3_efficiency.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(eff_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eff_rows)
    print(f"Saved {ARTIFACTS / 'mission3_efficiency.json'} and CSV")

    # Plot Accuracy vs Latency Pareto Frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    lat_pts = [float(r["mean_latency_ms"]) for r in eff_rows]
    f1_pts = [float(r["f1"]) for r in eff_rows]
    names = [r["model"] for r in eff_rows]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for x_p, y_p, name, col in zip(lat_pts, f1_pts, names, colors):
        ax.scatter([x_p], [y_p], s=140, color=col, edgecolors="black", zorder=5)
        ax.annotate(
            f" {name}\n (Lat: {x_p:.0f}ms, F1: {y_p:.3f})",
            (x_p, y_p),
            xytext=(10, -5),
            textcoords="offset points",
            fontweight="bold" if "Scale-Aware" in name else "normal",
            fontsize=10,
        )

    ax.plot(lat_pts[:2], f1_pts[:2], "--", color="#7f8c8d", alpha=0.7)
    ax.set_title("Accuracy vs. Latency Pareto Comparison on Locked BDD100K Test Cohort", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean CPU Latency per Image (ms, Apple M1)")
    ax.set_ylabel("Micro F1 Score")
    ax.set_xlim(0, max(lat_pts) * 1.15)
    ax.set_ylim(0.45, 0.75)
    ax.grid(True, alpha=0.3)

    plot_path = PLOTS_DIR / "mission3_accuracy_latency.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot {plot_path}")


if __name__ == "__main__":
    main()

