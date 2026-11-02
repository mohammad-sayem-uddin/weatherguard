"""Runs Experiment C: Scale-Adaptive Confidence Calibration on Group B Development Cohort.

Investigates whether scale-dependent confidence thresholds based strictly on
PREDICTED bounding-box area can improve overall F1 and small-object recall
compared to a single global confidence threshold.

Scale Bins:
  - Small:  area < 32^2  (1,024 px^2)
  - Medium: 32^2 <= area < 96^2  (1,024 to 9,216 px^2)
  - Large:  area >= 96^2 (9,216 px^2)

Evaluates on:
  1. Baseline YOLOv8n (640x640)
  2. Multi-Scale YOLOv8n (at various resolutions)

Performs grid search on Group B Dev cohort over (tau_small, tau_med, tau_large) in [0.05, 0.60]
to find optimal triplet that maximizes micro F1.

Saves artifacts to:
  - artifacts/mission3_scale_calibration.json
  - artifacts/mission3_scale_calibration.csv
  - artifacts/plots/mission3_scale_calibration.png
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.external_dataset import CANONICAL_CLASSES
from src.metrics import evaluate_detections, greedy_match

ARTIFACTS = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_predicted_scale_bin(box: list[float] | tuple[float, float, float, float]) -> str:
    """Classifies box scale based strictly on PREDICTED box area."""
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area < 1024.0:
        return "small"
    elif area < 9216.0:
        return "medium"
    else:
        return "large"


def filter_by_scale_adaptive_thresholds(
    eval_records: list[dict],
    tau_small: float,
    tau_med: float,
    tau_large: float,
) -> list[dict]:
    """Applies scale-dependent confidence thresholds based strictly on PREDICTED box area."""
    thresholds = {"small": tau_small, "medium": tau_med, "large": tau_large}
    filtered_records = []

    for r in eval_records:
        filtered_preds = []
        for p in r["predictions"]:
            s_bin = get_predicted_scale_bin(p["box_xyxy"])
            th = thresholds[s_bin]
            if p["score"] >= th:
                filtered_preds.append(p)

        filtered_records.append({
            "image_id": r["image_id"],
            "condition": r.get("condition", "unknown"),
            "ground_truth": r["ground_truth"],
            "predictions": filtered_preds,
        })
    return filtered_records


def evaluate_scale_breakdown_adaptive(
    eval_records: list[dict],
    tau_small: float,
    tau_med: float,
    tau_large: float,
) -> dict[str, dict]:
    """Measures GT scale breakdown under scale-adaptive filtering."""
    thresholds = {"small": tau_small, "medium": tau_med, "large": tau_large}
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


def grid_search_scale_thresholds(eval_records: list[dict]) -> tuple[tuple[float, float, float], float, dict]:
    """Searches over (tau_small, tau_med, tau_large) on Group B development cohort."""
    tau_candidates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    best_triplet = (0.20, 0.20, 0.20)
    best_f1 = -1.0
    best_stats = {}

    print(f"Grid searching {len(tau_candidates)**3} scale-adaptive threshold combinations...")
    t0 = time.perf_counter()

    for ts, tm, tl in product(tau_candidates, tau_candidates, tau_candidates):
        filt_recs = filter_by_scale_adaptive_thresholds(eval_records, ts, tm, tl)
        # Using score_threshold=0.0 because filtering has already been applied based on predicted scale
        ev = evaluate_detections(filt_recs, iou_threshold=0.5, score_threshold=0.0)
        f1 = ev.micro["f1"]

        if f1 > best_f1:
            best_f1 = f1
            best_triplet = (ts, tm, tl)
            best_stats = {
                "f1": f1,
                "precision": ev.micro["precision"],
                "recall": ev.micro["recall"],
                "tp": ev.micro["tp"],
                "fp": ev.micro["fp"],
                "fn": ev.micro["fn"],
                "mAP_50": ev.macro["mAP_50"],
            }

    elapsed = time.perf_counter() - t0
    print(f"Grid search completed in {elapsed:.2f}s. Best triplet: {best_triplet} with F1={best_f1:.4f}")
    return best_triplet, best_f1, best_stats


def main() -> None:
    print("=" * 70)
    print("WEATHERGUARD MISSION 3 — EXPERIMENT C: SCALE-ADAPTIVE CALIBRATION")
    print("=" * 70)

    # 1. Load dev predictions for baseline 640 and resolutions
    dev_files = {
        "yolov8n_640": ARTIFACTS / "eval_mission3_dev_yolov8n_baseline.json",
        "yolov8n_832": ARTIFACTS / "eval_mission3_dev_yolov8n_832.json",
        "yolov8n_960": ARTIFACTS / "eval_mission3_dev_yolov8n_960.json",
        "yolov8n_1280": ARTIFACTS / "eval_mission3_dev_yolov8n_1280.json",
    }

    results = {}
    csv_rows = []

    for name, path in dev_files.items():
        if not path.exists():
            print(f"Prediction cache {path} not found. Skipping {name}...")
            continue

        print(f"\nAnalyzing {name} from {path}...")
        with open(path) as f:
            data = json.load(f)
        eval_recs = data["per_image"]

        # Baseline single threshold (tau=0.20)
        ev_fixed = evaluate_detections(eval_recs, iou_threshold=0.5, score_threshold=0.20)
        scale_fixed = evaluate_scale_breakdown_adaptive(eval_recs, 0.20, 0.20, 0.20)

        # Baseline single optimal threshold
        opt_th = 0.20
        best_single_f1 = -1.0
        for th_int in range(5, 75, 5):
            th = th_int / 100.0
            ev = evaluate_detections(eval_recs, iou_threshold=0.5, score_threshold=th)
            if ev.micro["f1"] > best_single_f1:
                best_single_f1 = ev.micro["f1"]
                opt_th = th
        scale_single_opt = evaluate_scale_breakdown_adaptive(eval_recs, opt_th, opt_th, opt_th)

        # Scale-adaptive grid search
        best_triplet, best_f1, best_stats = grid_search_scale_thresholds(eval_recs)
        ts, tm, tl = best_triplet
        scale_adaptive = evaluate_scale_breakdown_adaptive(eval_recs, ts, tm, tl)

        entry = {
            "model_variant": name,
            "baseline_fixed_tau_0.20": {
                "threshold": 0.20,
                "f1": ev_fixed.micro["f1"],
                "precision": ev_fixed.micro["precision"],
                "recall": ev_fixed.micro["recall"],
                "mAP_50": ev_fixed.macro["mAP_50"],
                "scale_breakdown": scale_fixed,
            },
            "best_single_global_tau": {
                "threshold": opt_th,
                "f1": best_single_f1,
                "scale_breakdown": scale_single_opt,
            },
            "scale_adaptive_calibration": {
                "tau_small": ts,
                "tau_medium": tm,
                "tau_large": tl,
                "f1": best_f1,
                "precision": best_stats["precision"],
                "recall": best_stats["recall"],
                "tp": best_stats["tp"],
                "fp": best_stats["fp"],
                "fn": best_stats["fn"],
                "mAP_50": best_stats["mAP_50"],
                "f1_gain_vs_fixed": best_f1 - ev_fixed.micro["f1"],
                "f1_gain_vs_single_opt": best_f1 - best_single_f1,
                "scale_breakdown": scale_adaptive,
            },
        }
        results[name] = entry

        csv_rows.append({
            "model_variant": name,
            "fixed_tau": 0.20,
            "fixed_f1": f"{ev_fixed.micro['f1']:.4f}",
            "fixed_small_rec": f"{scale_fixed['small']['recall']*100:.2f}%",
            "single_opt_tau": f"{opt_th:.2f}",
            "single_opt_f1": f"{best_single_f1:.4f}",
            "adaptive_tau_s": f"{ts:.2f}",
            "adaptive_tau_m": f"{tm:.2f}",
            "adaptive_tau_l": f"{tl:.2f}",
            "adaptive_f1": f"{best_f1:.4f}",
            "adaptive_precision": f"{best_stats['precision']:.4f}",
            "adaptive_recall": f"{best_stats['recall']:.4f}",
            "adaptive_small_rec": f"{scale_adaptive['small']['recall']*100:.2f}%",
            "adaptive_med_rec": f"{scale_adaptive['medium']['recall']*100:.2f}%",
            "adaptive_large_rec": f"{scale_adaptive['large']['recall']*100:.2f}%",
            "f1_gain_vs_fixed": f"{(best_f1 - ev_fixed.micro['f1']):+.4f}",
        })

    # Save artifacts
    out_json = ARTIFACTS / "mission3_scale_calibration.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")

    out_csv = ARTIFACTS / "mission3_scale_calibration.csv"
    with open(out_csv, "w", newline="") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"Saved {out_csv}")

    # Generate plot comparing Fixed vs Single Optimal vs Scale-Adaptive F1 and Small Recall
    if csv_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        variants = [r["model_variant"].replace("yolov8n_", "YOLO ") + "px" for r in csv_rows]
        x = np.arange(len(variants))
        width = 0.25

        f1_fixed = [float(r["fixed_f1"]) for r in csv_rows]
        f1_single = [float(r["single_opt_f1"]) for r in csv_rows]
        f1_adapt = [float(r["adaptive_f1"]) for r in csv_rows]

        # Subplot 1: F1 score comparison
        ax1 = axes[0]
        ax1.bar(x - width, f1_fixed, width, label=r"Fixed $\tau=0.20$", color="#95a5a6")
        ax1.bar(x, f1_single, width, label=r"Single Optimal $\tau^*$", color="#3498db")
        ax1.bar(x + width, f1_adapt, width, label=r"Scale-Adaptive $(\tau_s, \tau_m, \tau_l)$", color="#e74c3c")
        ax1.set_title("F1 Comparison: Fixed vs Single Optimal vs Scale-Adaptive", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(variants)
        ax1.set_ylabel("Micro F1 Score")
        ax1.set_ylim(0.40, max(f1_adapt) * 1.15)
        ax1.legend()
        ax1.grid(True, axis="y", alpha=0.3)

        # Subplot 2: Small Object Recall Comparison
        s_rec_fixed = [float(r["fixed_small_rec"].replace("%", "")) for r in csv_rows]
        s_rec_adapt = [float(r["adaptive_small_rec"].replace("%", "")) for r in csv_rows]

        ax2 = axes[1]
        ax2.bar(x - width/2, s_rec_fixed, width, label=r"Fixed $\tau=0.20$", color="#95a5a6")
        ax2.bar(x + width/2, s_rec_adapt, width, label=r"Scale-Adaptive", color="#e74c3c")
        ax2.set_title("Small-Object Recall Comparison (<32² px)", fontsize=11, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(variants)
        ax2.set_ylabel("Small Object Recall (%)")
        ax2.set_ylim(0, max(s_rec_adapt) * 1.25)
        ax2.legend()
        ax2.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = PLOTS_DIR / "mission3_scale_calibration.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved plot {plot_path}")


if __name__ == "__main__":
    main()
