"""Runs Experiment A: Multi-Scale Inference on Group B Development Cohort.

Evaluates YOLOv8n across input resolutions: 640, 832, 960, and 1280.
Measures detection accuracy (F1, mAP@0.5, Precision, Recall),
scale-stratified recall (Small, Medium, Large), condition-stratified recall,
and CPU latency (mean, median, total).
Saves artifacts to:
  - artifacts/eval_mission3_dev_yolov8n_{scale}.json
  - artifacts/mission3_multiscale.json
  - artifacts/mission3_multiscale.csv
  - artifacts/plots/mission3_multiscale.png
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.external_dataset import CANONICAL_CLASSES, Bdd100kDataset
from src.inference import predictions_to_eval_records, run_inference
from src.metrics import evaluate_detections, greedy_match
from src.model import build_detector

ARTIFACTS = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "bdd100k"
IMAGES_DIR = DATA_DIR / "images"
SAMPLES_JSON = DATA_DIR / "samples.json"
MANIFEST_JSON = ARTIFACTS / "mission3_split_manifest.json"

RESOLUTIONS = [640, 832, 960, 1280]


def evaluate_scale_breakdown(eval_records: list[dict], threshold: float) -> dict[str, dict]:
    scales = {
        "small": {"gt": 0, "tp": 0, "fn": 0},
        "medium": {"gt": 0, "tp": 0, "fn": 0},
        "large": {"gt": 0, "tp": 0, "fn": 0},
    }

    for r in eval_records:
        gt_boxes = [g["box_xyxy"] for g in r["ground_truth"]]
        gt_labels = [g["class_name"] for g in r["ground_truth"]]

        preds = [p for p in r["predictions"] if p["score"] >= threshold and p["common_label"] is not None]
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


def evaluate_condition_breakdown(eval_records: list[dict], threshold: float) -> dict[str, dict]:
    by_cond = defaultdict(list)
    for r in eval_records:
        by_cond[r["condition"]].append(r)

    out = {}
    for cond, recs in by_cond.items():
        ev = evaluate_detections(recs, iou_threshold=0.5, score_threshold=threshold)
        scale_stats = evaluate_scale_breakdown(recs, threshold)
        out[cond] = {
            "n_images": len(recs),
            "f1": ev.micro["f1"],
            "precision": ev.micro["precision"],
            "recall": ev.micro["recall"],
            "mAP_50": ev.macro["mAP_50"],
            "small_recall": scale_stats["small"]["recall"],
            "medium_recall": scale_stats["medium"]["recall"],
            "large_recall": scale_stats["large"]["recall"],
        }
    return out


def run_yolo_at_scale(scale: int, records: list) -> tuple[dict, list[dict]]:
    cache_path = ARTIFACTS / f"eval_mission3_dev_yolov8n_{scale}.json"
    if cache_path.exists():
        print(f"Loading cached predictions from {cache_path}...")
        with open(cache_path) as f:
            data = json.load(f)
        return data, data["per_image"]

    print(f"\nBuilding and loading YOLOv8n with imgsz={scale}...")
    detector = build_detector("yolov8n", score_threshold=0.0, imgsz=scale)
    t0 = time.perf_counter()
    detector.load()
    load_time = time.perf_counter() - t0

    print(f"Running inference on {len(records)} Group B dev images (imgsz={scale}, warmup=True)...")
    run = run_inference(detector, records, score_threshold=0.0, warmup=True)
    print(f"Scale {scale} completed in {run.total_inference_seconds:.2f}s "
          f"(mean: {run.mean_inference_seconds*1000:.1f}ms, median: {run.median_inference_seconds*1000:.1f}ms)")

    eval_records = predictions_to_eval_records(records, run.predictions)
    ev50 = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=0.5)
    latencies = [p.inference_seconds for p in run.predictions]

    payload = {
        "tag": f"mission3_dev_yolov8n_{scale}",
        "dataset": "BDD100K Development Cohort (Group B, n=160)",
        "detector_name": f"yolov8n_{scale}",
        "resolution": scale,
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


def sweep_threshold(recs: list[dict]) -> tuple[float, float, dict]:
    """Find threshold that maximizes micro F1."""
    best_th = 0.20
    best_f1 = -1.0
    best_ev = None
    for th_int in range(5, 75, 5):
        th = th_int / 100.0
        ev = evaluate_detections(recs, iou_threshold=0.5, score_threshold=th)
        if ev.micro["f1"] > best_f1:
            best_f1 = ev.micro["f1"]
            best_th = th
            best_ev = ev
    return best_th, best_f1, best_ev


def main() -> None:
    print("=" * 70)
    print("WEATHERGUARD MISSION 3 — EXPERIMENT A: MULTI-SCALE INFERENCE")
    print("=" * 70)

    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)
    dev_ids = set(manifest["groups"]["Group_B_BDD100K_Development"]["image_ids"])
    full_ds = Bdd100kDataset.from_fiftyone_json(SAMPLES_JSON, images_dir=IMAGES_DIR)
    dev_records = [r for r in full_ds if r.image_id in dev_ids]
    dev_records.sort(key=lambda r: r.image_id)
    print(f"Loaded {len(dev_records)} Group B dev records.")

    results = {}
    csv_rows = []

    for scale in RESOLUTIONS:
        payload, eval_recs = run_yolo_at_scale(scale, dev_records)

        # Baseline threshold (0.20)
        ev_02 = evaluate_detections(eval_recs, iou_threshold=0.5, score_threshold=0.20)
        scale_02 = evaluate_scale_breakdown(eval_recs, 0.20)
        cond_02 = evaluate_condition_breakdown(eval_recs, 0.20)

        # Optimal threshold sweep
        opt_th, opt_f1, opt_ev = sweep_threshold(eval_recs)
        scale_opt = evaluate_scale_breakdown(eval_recs, opt_th)
        cond_opt = evaluate_condition_breakdown(eval_recs, opt_th)

        latency_mean = payload["inference_mean_seconds"] * 1000.0
        latency_med = payload["inference_median_seconds"] * 1000.0

        scale_res = {
            "resolution": scale,
            "latency": {
                "mean_ms": latency_mean,
                "median_ms": latency_med,
                "total_seconds": payload["inference_total_seconds"],
                "fps": 1.0 / payload["inference_mean_seconds"] if payload["inference_mean_seconds"] > 0 else 0.0,
            },
            "eval_tau_0.20": {
                "threshold": 0.20,
                "f1": ev_02.micro["f1"],
                "precision": ev_02.micro["precision"],
                "recall": ev_02.micro["recall"],
                "tp": ev_02.micro["tp"],
                "fp": ev_02.micro["fp"],
                "fn": ev_02.micro["fn"],
                "mAP_50": ev_02.macro["mAP_50"],
                "scale_breakdown": scale_02,
                "condition_breakdown": cond_02,
            },
            "eval_optimal_tau": {
                "threshold": opt_th,
                "f1": opt_f1,
                "precision": opt_ev.micro["precision"],
                "recall": opt_ev.micro["recall"],
                "tp": opt_ev.micro["tp"],
                "fp": opt_ev.micro["fp"],
                "fn": opt_ev.micro["fn"],
                "mAP_50": opt_ev.macro["mAP_50"],
                "scale_breakdown": scale_opt,
                "condition_breakdown": cond_opt,
            },
        }
        results[str(scale)] = scale_res

        csv_rows.append({
            "resolution": scale,
            "latency_mean_ms": f"{latency_mean:.1f}",
            "fps": f"{scale_res['latency']['fps']:.1f}",
            "tau_fixed": 0.20,
            "f1_tau0.20": f"{ev_02.micro['f1']:.4f}",
            "precision_tau0.20": f"{ev_02.micro['precision']:.4f}",
            "recall_tau0.20": f"{ev_02.micro['recall']:.4f}",
            "mAP50_tau0.20": f"{ev_02.macro['mAP_50']:.4f}",
            "small_recall_tau0.20": f"{scale_02['small']['recall']*100:.2f}%",
            "med_recall_tau0.20": f"{scale_02['medium']['recall']*100:.2f}%",
            "large_recall_tau0.20": f"{scale_02['large']['recall']*100:.2f}%",
            "opt_tau": f"{opt_th:.2f}",
            "opt_f1": f"{opt_f1:.4f}",
            "opt_small_recall": f"{scale_opt['small']['recall']*100:.2f}%",
        })

    # Save artifacts
    out_json = ARTIFACTS / "mission3_multiscale.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")

    out_csv = ARTIFACTS / "mission3_multiscale.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {out_csv}")

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    scales_num = [r["resolution"] for r in csv_rows]
    latencies = [float(r["latency_mean_ms"]) for r in csv_rows]
    f1s_02 = [float(r["f1_tau0.20"]) for r in csv_rows]
    f1s_opt = [float(r["opt_f1"]) for r in csv_rows]
    mAPs = [float(r["mAP50_tau0.20"]) for r in csv_rows]

    s_recs = [float(r["small_recall_tau0.20"].replace("%", "")) for r in csv_rows]
    m_recs = [float(r["med_recall_tau0.20"].replace("%", "")) for r in csv_rows]
    l_recs = [float(r["large_recall_tau0.20"].replace("%", "")) for r in csv_rows]

    # Subplot 1: Accuracy metrics vs Scale
    ax1 = axes[0, 0]
    ax1.plot(scales_num, f1s_02, "o-", color="#1f77b4", label=r"F1 ($\tau=0.20$)", linewidth=2)
    ax1.plot(scales_num, f1s_opt, "s--", color="#ff7f0e", label=r"F1 ($\tau^*$, tuned on dev)", linewidth=2)
    ax1.plot(scales_num, mAPs, "^-.", color="#2ca02c", label=r"mAP@0.5 ($\tau=0.20$)", linewidth=2)
    ax1.set_title("Detection Accuracy vs Input Resolution (YOLOv8n)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Input Resolution (px)")
    ax1.set_ylabel("Metric Value")
    ax1.set_xticks(scales_num)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    # Subplot 2: Scale-Stratified Recall vs Resolution
    ax2 = axes[0, 1]
    ax2.plot(scales_num, s_recs, "o-", color="#d62728", label="Small Objects (<32²)", linewidth=2)
    ax2.plot(scales_num, m_recs, "s-", color="#9467bd", label="Medium Objects (32²-96²)", linewidth=2)
    ax2.plot(scales_num, l_recs, "^-", color="#8c564b", label="Large Objects (≥96²)", linewidth=2)
    ax2.set_title("Scale-Stratified Recall vs Input Resolution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Input Resolution (px)")
    ax2.set_ylabel("Recall (%)")
    ax2.set_xticks(scales_num)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="center right")

    # Subplot 3: Latency vs Resolution
    ax3 = axes[1, 0]
    ax3.bar([str(s) for s in scales_num], latencies, color="#34495e", width=0.5, edgecolor="black")
    for i, v in enumerate(latencies):
        ax3.text(i, v + 2, f"{v:.1f} ms\n({1000/v:.1f} fps)", ha="center", fontsize=9)
    ax3.set_title("CPU Latency vs Input Resolution (Apple M1)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Input Resolution")
    ax3.set_ylabel("Mean Latency (ms/image)")
    ax3.set_ylim(0, max(latencies) * 1.25)
    ax3.grid(True, axis="y", alpha=0.3)

    # Subplot 4: Pareto Curve (F1 vs Latency)
    ax4 = axes[1, 1]
    ax4.plot(latencies, f1s_02, "o-", color="#16a085", linewidth=2, markersize=8)
    for i, s in enumerate(scales_num):
        ax4.annotate(f" {s}px\n (F1={f1s_02[i]:.3f})", (latencies[i], f1s_02[i]), fontsize=9, textcoords="offset points", xytext=(5, -5))
    ax4.set_title("Accuracy vs Latency Trade-Off (Pareto Curve)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Mean CPU Latency (ms/image)")
    ax4.set_ylabel(r"F1 Score ($\tau=0.20$)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "mission3_multiscale.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot {plot_path}")


if __name__ == "__main__":
    main()
