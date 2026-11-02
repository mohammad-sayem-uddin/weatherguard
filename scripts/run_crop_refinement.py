"""Run Targeted Region Refinement Experiments for WeatherGuard Mission 4.

Evaluates crop proposal, local upsampling, and coordinate-mapped NMS merging
on Group B (BDD100K Development Cohort, N=160).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.dataset import COCO_TO_DAWN
from src.model import build_detector
from src.crop_refinement import (
    CropRegion,
    generate_candidate_crops,
    transform_crop_to_original,
    merge_and_nms,
)
from src.metrics import evaluate_detections
from scripts.run_combined_scale_inference import evaluate_scale_breakdown_with_thresholds

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = ROOT / "data" / "bdd100k" / "images"


def run_crop_refinement_experiment(
    data_640: dict[str, Any],
    detector: Any,
    max_crops: int = 1,
    min_score: float = 0.15,
    context_margin: float = 0.20,
    nms_iou_threshold: float = 0.50,
    tau: float = 0.20,
) -> dict[str, Any]:
    """Execute crop refinement across the cohort."""
    per_img_640 = {item["image_id"]: item for item in data_640["per_image"]}
    latencies_640 = dict(zip(data_640["image_ids"], data_640["per_image_seconds"]))
    image_ids = data_640["image_ids"]
    n_images = len(image_ids)

    refined_records: list[dict] = []
    seq_latencies: list[float] = []
    crops_per_image: list[int] = []

    print(f"\nRunning Crop Refinement (max_crops={max_crops}, min_score={min_score})...")
    t_start = time.perf_counter()

    for i, img_id in enumerate(image_ids):
        item_640 = per_img_640[img_id]
        t_base = latencies_640[img_id]
        base_preds = item_640["predictions"]

        # 1. Propose candidate crops
        crops = generate_candidate_crops(
            base_preds,
            image_shape=(1280, 720),
            max_crops=max_crops,
            context_margin=context_margin,
            min_score=min_score,
        )
        crops_per_image.append(len(crops))

        t_crop_total = 0.0
        all_crop_preds: list[dict[str, Any]] = []

        if crops:
            img_path = IMAGES_DIR / img_id
            pil_img = Image.open(img_path).convert("RGB")

            for crop in crops:
                crop_pil = pil_img.crop((crop.x1, crop.y1, crop.x2, crop.y2))
                pred = detector.predict(crop_pil)
                t_crop_total += pred.inference_seconds

                # Map detections back to native coordinates
                for det in pred.detections:
                    mapped_box = transform_crop_to_original(
                        det.box_xyxy, crop, crop_input_size=(640, 640)
                    )
                    common_lbl = COCO_TO_DAWN.get(det.label)
                    all_crop_preds.append({
                        "label": det.label,
                        "common_label": common_lbl,
                        "score": det.score,
                        "box_xyxy": mapped_box,
                    })

        # 2. Merge with base predictions using NMS
        merged_preds = merge_and_nms(
            base_preds, all_crop_preds, nms_iou_threshold=nms_iou_threshold
        )

        t_seq = t_base + t_crop_total
        seq_latencies.append(t_seq)

        refined_records.append({
            "image_id": img_id,
            "condition": item_640["condition"],
            "predictions": merged_preds,
            "ground_truth": item_640["ground_truth"],
        })

        if (i + 1) % 40 == 0 or (i + 1) == n_images:
            elapsed = time.perf_counter() - t_start
            print(f"  Processed {i+1}/{n_images} images in {elapsed:.1f}s (Crops: {sum(crops_per_image)})")

    # Evaluate detection metrics
    eval_res = evaluate_detections(refined_records, score_threshold=tau, iou_threshold=0.5)
    scale_res = evaluate_scale_breakdown_with_thresholds(
        refined_records, {"small": tau, "medium": tau, "large": tau}
    )

    mean_seq_ms = float(np.mean(seq_latencies) * 1000.0)
    median_seq_ms = float(np.median(seq_latencies) * 1000.0)
    p95_seq_ms = float(np.percentile(seq_latencies, 95) * 1000.0)
    fps_seq = 1000.0 / mean_seq_ms if mean_seq_ms > 0 else 0.0

    return {
        "max_crops": max_crops,
        "min_score": min_score,
        "n_images": n_images,
        "total_crops_generated": sum(crops_per_image),
        "pct_images_with_crops": round(sum(1 for c in crops_per_image if c > 0) / n_images * 100.0, 2),
        "mean_crops_per_image": round(float(np.mean(crops_per_image)), 2),
        "detection_metrics": {
            "f1": round(eval_res.micro["f1"], 5),
            "precision": round(eval_res.micro["precision"], 5),
            "recall": round(eval_res.micro["recall"], 5),
            "tp": eval_res.micro["tp"],
            "fp": eval_res.micro["fp"],
            "fn": eval_res.micro["fn"],
            "fp_per_image": round(eval_res.micro["fp"] / n_images, 3),
        },
        "scale_recall": {
            "small": round(scale_res["small"]["recall"] * 100.0, 2),
            "medium": round(scale_res["medium"]["recall"] * 100.0, 2),
            "large": round(scale_res["large"]["recall"] * 100.0, 2),
            "small_tp": scale_res["small"]["tp"],
            "small_gt": scale_res["small"]["gt"],
        },
        "latency_sequential": {
            "mean_ms": round(mean_seq_ms, 2),
            "median_ms": round(median_seq_ms, 2),
            "p95_ms": round(p95_seq_ms, 2),
            "fps": round(fps_seq, 2),
        },
        "records": refined_records,
    }


def main() -> None:
    print("=" * 75)
    print("WeatherGuard Mission 4 — Experiment B: Targeted Region Refinement")
    print("Evaluating on Group B Development Cohort (N = 160)")
    print("=" * 75)

    path_640 = ARTIFACTS / "eval_mission3_dev_yolov8n_640.json"
    data_640 = json.load(open(path_640))

    # Initialize YOLOv8n detector
    print("Loading YOLOv8n detector for crop inference...")
    detector = build_detector("yolov8n", imgsz=640)
    detector.load()
    detector.warmup()

    configs = [
        ("crop_refine_k1_s15", 1, 0.15),
        ("crop_refine_k2_s15", 2, 0.15),
        ("crop_refine_k1_s20", 1, 0.20),
    ]

    results = {}
    summary_rows = []

    for name, k, s in configs:
        res = run_crop_refinement_experiment(
            data_640=data_640,
            detector=detector,
            max_crops=k,
            min_score=s,
            context_margin=0.20,
            nms_iou_threshold=0.50,
            tau=0.20,
        )
        # Store clean result without huge records payload in summary
        cleaned_res = {k_res: v_res for k_res, v_res in res.items() if k_res != "records"}
        results[name] = cleaned_res

        summary_rows.append({
            "Method": name,
            "Max_Crops": k,
            "Min_Score": s,
            "Crops_%": res["pct_images_with_crops"],
            "Total_Crops": res["total_crops_generated"],
            "F1": res["detection_metrics"]["f1"],
            "Precision": res["detection_metrics"]["precision"],
            "Recall": res["detection_metrics"]["recall"],
            "Small_Recall_%": res["scale_recall"]["small"],
            "Med_Recall_%": res["scale_recall"]["medium"],
            "Large_Recall_%": res["scale_recall"]["large"],
            "FP/Img": res["detection_metrics"]["fp_per_image"],
            "Mean_Latency_ms": res["latency_sequential"]["mean_ms"],
            "P95_Latency_ms": res["latency_sequential"]["p95_ms"],
            "FPS": res["latency_sequential"]["fps"],
        })
        print(f"[{name:20s}] F1: {res['detection_metrics']['f1']:.4f} | Small R: {res['scale_recall']['small']:5.2f}% | Large R: {res['scale_recall']['large']:5.2f}% | Lat: {res['latency_sequential']['mean_ms']:5.1f} ms | FPS: {res['latency_sequential']['fps']:4.1f} | Crops%: {res['pct_images_with_crops']:4.1f}%")

    # Add reference baselines to CSV
    df = pd.DataFrame(summary_rows)
    csv_path = ARTIFACTS / "mission4_crop_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary to: {csv_path}")

    # Save JSON artifact
    json_path = ARTIFACTS / "mission4_crop_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "cohort": "Group_B_BDD100K_Development",
            "n_images": 160,
            "experiments": results,
        }, f, indent=2)
    print(f"Saved JSON artifact to: {json_path}")

    # Generate Plot comparing Crop Refinement with Global Baselines & Routing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Baseline points
    ax1.scatter(52.4, 0.4780, color="#6c757d", s=100, marker="o", label="Fixed 640px")
    ax1.scatter(95.7, 0.5344, color="#6c757d", s=100, marker="s", label="Fixed 832px")
    ax1.scatter(394.5, 0.5518, color="#6c757d", s=100, marker="^", label="Fixed 1280px")
    ax1.scatter(315.7, 0.5462, color="#1d3557", s=100, marker="D", label="Adaptive Routing (k1)")

    for row in summary_rows:
        ax1.scatter(row["Mean_Latency_ms"], row["F1"], color="#e63946", s=100, marker="X")
        ax1.annotate(row["Method"], (row["Mean_Latency_ms"] + 4, row["F1"] - 0.003), fontsize=8)

    ax1.set_xlabel("Mean Sequential Latency (ms) [Apple M1 CPU]", fontsize=11)
    ax1.set_ylabel("Micro F1 Score (IoU=0.50, tau=0.20)", fontsize=11)
    ax1.set_title("Targeted Crop Refinement vs. Global Scaling (F1)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9)

    # Small recall
    ax2.scatter(52.4, 8.09, color="#6c757d", s=100, marker="o", label="Fixed 640px")
    ax2.scatter(95.7, 16.18, color="#6c757d", s=100, marker="s", label="Fixed 832px")
    ax2.scatter(394.5, 24.16, color="#6c757d", s=100, marker="^", label="Fixed 1280px")
    ax2.scatter(315.7, 22.06, color="#1d3557", s=100, marker="D", label="Adaptive Routing (k1)")

    for row in summary_rows:
        ax2.scatter(row["Mean_Latency_ms"], row["Small_Recall_%"], color="#e63946", s=100, marker="X")
        ax2.annotate(row["Method"], (row["Mean_Latency_ms"] + 4, row["Small_Recall_%"] - 0.5), fontsize=8)

    ax2.set_xlabel("Mean Sequential Latency (ms) [Apple M1 CPU]", fontsize=11)
    ax2.set_ylabel("Small-Object Recall (%)", fontsize=11)
    ax2.set_title("Small-Object Recall vs. Latency", fontsize=12, fontweight="bold")
    ax2.axvline(100.0, color="#f4a261", linestyle="--", linewidth=1.5, label="10 FPS Boundary")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "mission4_crop_refinement.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
