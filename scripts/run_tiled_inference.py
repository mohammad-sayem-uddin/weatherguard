"""Runs Experiment B: Tiled / Sliced Inference on Group B Development Cohort.

Evaluates 2x2 sliced image tiling (with ~20% overlap) for distant/small object detection.
Maps tile detections back to native 1280x720 coordinates, merges using class-wise NMS (IoU=0.50),
and evaluates precision, recall, false-positive inflation, and CPU latency.

Saves artifacts to:
  - artifacts/eval_mission3_dev_yolov8n_tiled.json
  - artifacts/mission3_tiling.json
  - artifacts/mission3_tiling.csv
  - artifacts/plots/mission3_tiling.png
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
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.external_dataset import CANONICAL_CLASSES, Bdd100kDataset
from src.inference import predictions_to_eval_records
from src.metrics import evaluate_detections, greedy_match
from src.model import Detection, Prediction, build_detector

ARTIFACTS = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "bdd100k"
IMAGES_DIR = DATA_DIR / "images"
SAMPLES_JSON = DATA_DIR / "samples.json"
MANIFEST_JSON = ARTIFACTS / "mission3_split_manifest.json"

# Define 2x2 grid windows for 1280x720 frames with overlap
TILE_WINDOWS = [
    (0, 0, 704, 416),      # Top-left
    (576, 0, 1280, 416),   # Top-right (128px overlap)
    (0, 304, 704, 720),    # Bottom-left (112px overlap)
    (576, 304, 1280, 720), # Bottom-right
]


def nms_detections(detections: list[Detection], iou_threshold: float = 0.50) -> list[Detection]:
    """Greedy Non-Maximum Suppression grouped by common class label."""
    kept: list[Detection] = []
    by_class: dict[str, list[Detection]] = defaultdict(list)
    for d in detections:
        if d.common_label is not None:
            by_class[d.common_label].append(d)

    for label, class_dets in by_class.items():
        class_dets.sort(key=lambda d: d.score, reverse=True)
        suppressed = [False] * len(class_dets)
        for i in range(len(class_dets)):
            if suppressed[i]:
                continue
            kept.append(class_dets[i])
            b1 = class_dets[i].box_xyxy
            for j in range(i + 1, len(class_dets)):
                if suppressed[j]:
                    continue
                b2 = class_dets[j].box_xyxy
                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = area1 + area2 - inter
                iou = (inter / union) if union > 0 else 0.0
                if iou >= iou_threshold:
                    suppressed[j] = True
    return kept


def predict_tiled(
    detector,
    image_path: Path,
    image_id: str,
    include_full_image: bool = True,
) -> Prediction:
    """Performs sliced inference over image tiles and merges detections."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    t0 = time.perf_counter()
    all_raw_dets: list[Detection] = []

    # 1. Full image inference (to catch large objects spanning across tile boundaries)
    if include_full_image:
        pred_full = detector.predict(img, image_id=image_id)
        all_raw_dets.extend(pred_full.detections)

    # 2. Tile inferences
    for (x0, y0, x1, y1) in TILE_WINDOWS:
        tile_img = img.crop((x0, y0, x1, y1))
        pred_tile = detector.predict(tile_img, image_id=f"{image_id}_tile")
        for d in pred_tile.detections:
            # Shift box coordinates to original image space
            tb = d.box_xyxy
            gb = (tb[0] + x0, tb[1] + y0, tb[2] + x0, tb[3] + y0)
            all_raw_dets.append(
                Detection(
                    box_xyxy=gb,
                    score=d.score,
                    label=d.label,
                    common_label=d.common_label,
                )
            )

    # 3. Class-wise NMS merging
    merged_dets = nms_detections(all_raw_dets, iou_threshold=0.50)
    elapsed = time.perf_counter() - t0

    return Prediction(
        image_id=image_id,
        width=w,
        height=h,
        detections=merged_dets,
        inference_seconds=elapsed,
    )


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


def main() -> None:
    print("=" * 70)
    print("WEATHERGUARD MISSION 3 — EXPERIMENT B: TILED INFERENCE")
    print("=" * 70)

    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)
    dev_ids = set(manifest["groups"]["Group_B_BDD100K_Development"]["image_ids"])
    full_ds = Bdd100kDataset.from_fiftyone_json(SAMPLES_JSON, images_dir=IMAGES_DIR)
    dev_records = [r for r in full_ds if r.image_id in dev_ids]
    dev_records.sort(key=lambda r: r.image_id)
    print(f"Loaded {len(dev_records)} Group B dev records.")

    cache_path = ARTIFACTS / "eval_mission3_dev_yolov8n_tiled.json"
    if cache_path.exists():
        print(f"Loading cached predictions from {cache_path}...")
        with open(cache_path) as f:
            payload = json.load(f)
        eval_records = payload["per_image"]
    else:
        print("\nLoading YOLOv8n detector for tiled inference...")
        detector = build_detector("yolov8n", score_threshold=0.0, imgsz=640)
        detector.load()
        detector.warmup()

        print("Running tiled inference (full + 4 tiles = 5 passes/img) on 160 dev images...")
        t_start = time.perf_counter()
        predictions = []
        latencies = []

        for idx, r in enumerate(dev_records):
            img_path = IMAGES_DIR / r.image_id
            pred = predict_tiled(detector, img_path, r.image_id, include_full_image=True)
            predictions.append(pred)
            latencies.append(pred.inference_seconds)
            if (idx + 1) % 40 == 0 or (idx + 1) == len(dev_records):
                print(f"  Processed [{idx+1}/{len(dev_records)}] images (current latency: {pred.inference_seconds*1000:.1f}ms)...")

        total_sec = time.perf_counter() - t_start
        mean_sec = float(np.mean(latencies))
        med_sec = float(np.median(latencies))
        print(f"Tiled inference finished in {total_sec:.2f}s (mean: {mean_sec*1000:.1f}ms, median: {med_sec*1000:.1f}ms)")

        eval_records = predictions_to_eval_records(dev_records, predictions)

        payload = {
            "tag": "mission3_dev_yolov8n_tiled",
            "dataset": "BDD100K Development Cohort (Group B, n=160)",
            "detector_name": "yolov8n_tiled_hybrid",
            "tiling_scheme": "2x2 grid (704x416 tiles, ~20% overlap) + full image (640)",
            "inference_total_seconds": total_sec,
            "inference_mean_seconds": mean_sec,
            "inference_median_seconds": med_sec,
            "per_image_seconds": latencies,
            "n_images": len(dev_records),
            "image_ids": [r.image_id for r in dev_records],
            "per_image": eval_records,
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"Saved {cache_path}")

    # Evaluate at multiple thresholds
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    csv_rows = []
    tiled_results = {
        "metadata": {
            "mean_latency_ms": payload["inference_mean_seconds"] * 1000.0,
            "median_latency_ms": payload["inference_median_seconds"] * 1000.0,
            "fps": 1.0 / payload["inference_mean_seconds"] if payload["inference_mean_seconds"] > 0 else 0.0,
            "scheme": payload.get("tiling_scheme", "2x2 hybrid"),
        },
        "threshold_eval": {},
    }

    for th in thresholds:
        ev = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=th)
        scale_stats = evaluate_scale_breakdown(eval_records, th)
        tiled_results["threshold_eval"][f"tau_{th:.2f}"] = {
            "threshold": th,
            "f1": ev.micro["f1"],
            "precision": ev.micro["precision"],
            "recall": ev.micro["recall"],
            "tp": ev.micro["tp"],
            "fp": ev.micro["fp"],
            "fn": ev.micro["fn"],
            "mAP_50": ev.macro["mAP_50"],
            "scale_breakdown": scale_stats,
        }
        csv_rows.append({
            "threshold": f"{th:.2f}",
            "f1": f"{ev.micro['f1']:.4f}",
            "precision": f"{ev.micro['precision']:.4f}",
            "recall": f"{ev.micro['recall']:.4f}",
            "tp": ev.micro["tp"],
            "fp": ev.micro["fp"],
            "fn": ev.micro["fn"],
            "mAP_50": f"{ev.macro['mAP_50']:.4f}",
            "small_recall": f"{scale_stats['small']['recall']*100:.2f}%",
            "med_recall": f"{scale_stats['medium']['recall']*100:.2f}%",
            "large_recall": f"{scale_stats['large']['recall']*100:.2f}%",
            "latency_ms": f"{payload['inference_mean_seconds']*1000.0:.1f}",
        })

    out_json = ARTIFACTS / "mission3_tiling.json"
    with open(out_json, "w") as f:
        json.dump(tiled_results, f, indent=2)
    print(f"\nSaved {out_json}")

    out_csv = ARTIFACTS / "mission3_tiling.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {out_csv}")

    # Generate plot comparing baseline vs tiling
    # Load baseline for comparison
    baseline_json = ARTIFACTS / "mission3_baseline.json"
    if baseline_json.exists():
        with open(baseline_json) as f:
            base_data = json.load(f)
        yolo_base = base_data.get("yolov8n_0.2", {})
    else:
        yolo_base = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Precision vs Recall vs Threshold
    ax1 = axes[0]
    th_vals = [float(r["threshold"]) for r in csv_rows]
    precs = [float(r["precision"]) for r in csv_rows]
    recs = [float(r["recall"]) for r in csv_rows]
    f1s = [float(r["f1"]) for r in csv_rows]

    ax1.plot(th_vals, precs, "s--", label="Precision", color="#2980b9", linewidth=2)
    ax1.plot(th_vals, recs, "^--", label="Recall", color="#27ae60", linewidth=2)
    ax1.plot(th_vals, f1s, "o-", label="F1 Score", color="#e74c3c", linewidth=2.5)
    ax1.set_title("Tiled YOLOv8n Performance across Confidence Thresholds", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Metric Value")
    ax1.set_xticks(th_vals)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Subplot 2: Scale Recall Comparison (Baseline vs Tiled at tau=0.20)
    ax2 = axes[1]
    tiled_02 = tiled_results["threshold_eval"]["tau_0.20"]["scale_breakdown"]
    labels = ["Small (<32²)", "Medium (32²-96²)", "Large (≥96²)"]
    x = np.arange(len(labels))
    width = 0.35

    base_scales = [
        yolo_base.get("scale_breakdown", {}).get("small", {}).get("recall", 0.0) * 100,
        yolo_base.get("scale_breakdown", {}).get("medium", {}).get("recall", 0.0) * 100,
        yolo_base.get("scale_breakdown", {}).get("large", {}).get("recall", 0.0) * 100,
    ]
    tiled_scales = [
        tiled_02["small"]["recall"] * 100,
        tiled_02["medium"]["recall"] * 100,
        tiled_02["large"]["recall"] * 100,
    ]

    ax2.bar(x - width/2, base_scales, width, label="YOLOv8n Baseline (640)", color="#7f8c8d")
    ax2.bar(x + width/2, tiled_scales, width, label="YOLOv8n Tiled (2x2)", color="#8e44ad")
    ax2.set_title("Scale-Stratified Recall: Baseline vs Tiled (tau=0.20)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Recall (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = PLOTS_DIR / "mission3_tiling.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot {plot_path}")


if __name__ == "__main__":
    main()
