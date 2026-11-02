"""Error taxonomy and qualitative error analysis for external BDD100K evaluation.

Analyzes:
1. Weather-specific false-positive and missed-detection patterns.
2. Class confusion and misclassification tendencies.
3. Scale-dependent failures (small vs. medium vs. large objects).
4. Score-range error dynamics (operating point impact).
5. Generates qualitative failure visualizations in artifacts/qualitative/external/
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import BBox
from src.external_dataset import CANONICAL_CLASSES, Bdd100kDataset
from src.metrics import evaluate_detections, greedy_match, iou_xyxy
from src.visualization import draw_boxes

ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data" / "bdd100k"
IMAGES_DIR = DATA_DIR / "images"
SAMPLES_JSON = DATA_DIR / "samples.json"
QUALITATIVE_DIR = ARTIFACTS / "qualitative" / "external"


def run_error_analysis() -> dict[str, any]:
    QUALITATIVE_DIR.mkdir(parents=True, exist_ok=True)

    frcnn_file = ARTIFACTS / "eval_external_fasterrcnn_resnet50_fpn.json"
    yolo_file = ARTIFACTS / "eval_external_yolov8n.json"

    with open(frcnn_file) as f:
        frcnn_data = json.load(f)
    with open(yolo_file) as f:
        yolo_data = json.load(f)

    frcnn_recs = frcnn_data["per_image"]
    yolo_recs = yolo_data["per_image"]

    # We evaluate FRCNN at 0.60 and YOLO at 0.20
    models = {
        "fasterrcnn_resnet50_fpn": {"recs": frcnn_recs, "th": 0.60},
        "yolov8n": {"recs": yolo_recs, "th": 0.20},
    }

    report = {}

    for model_name, cfg in models.items():
        recs = cfg["recs"]
        th = cfg["th"]

        weather_fp = Counter()
        weather_fn = Counter()
        weather_tp = Counter()

        scale_failures = {
            "small": {"gt": 0, "tp": 0, "fn": 0},
            "medium": {"gt": 0, "tp": 0, "fn": 0},
            "large": {"gt": 0, "tp": 0, "fn": 0},
        }

        class_fp = Counter()
        class_fn = Counter()
        class_tp = Counter()

        # Detailed failure inspection
        failure_gallery_candidates = []

        for r in recs:
            cond = r["condition"]
            gt_boxes = [g["box_xyxy"] for g in r["ground_truth"]]
            gt_labels = [g["class_name"] for g in r["ground_truth"]]

            # Filter predictions by threshold
            preds = [p for p in r["predictions"] if p["score"] >= th and p["common_label"] is not None]
            pred_boxes = [p["box_xyxy"] for p in preds]
            pred_scores = [p["score"] for p in preds]
            pred_labels = [p["common_label"] for p in preds]

            # Greedy match by class
            matched_gt_indices = set()
            matched_pred_indices = set()

            for c in CANONICAL_CLASSES:
                c_pred_idx = [i for i, lbl in enumerate(pred_labels) if lbl == c]
                c_gt_idx = [j for j, lbl in enumerate(gt_labels) if lbl == c]

                if not c_pred_idx or not c_gt_idx:
                    continue

                sub_pred_boxes = [pred_boxes[i] for i in c_pred_idx]
                sub_pred_scores = [pred_scores[i] for i in c_pred_idx]
                sub_gt_boxes = [gt_boxes[j] for j in c_gt_idx]

                matches = greedy_match(sub_pred_boxes, sub_pred_scores, sub_gt_boxes, iou_threshold=0.5)
                for p_sub, g_sub, _ in matches:
                    orig_p = c_pred_idx[p_sub]
                    orig_g = c_gt_idx[g_sub]
                    matched_pred_indices.add(orig_p)
                    matched_gt_indices.add(orig_g)
                    class_tp[c] += 1
                    weather_tp[cond] += 1

            # Tally FP and FN
            for i, p in enumerate(preds):
                if i not in matched_pred_indices:
                    class_fp[p["common_label"]] += 1
                    weather_fp[cond] += 1

            for j, g in enumerate(r["ground_truth"]):
                c = g["class_name"]
                box = g["box_xyxy"]
                area = (box[2] - box[0]) * (box[3] - box[1])

                if area < 1024:
                    s_bin = "small"
                elif area < 9216:
                    s_bin = "medium"
                else:
                    s_bin = "large"

                scale_failures[s_bin]["gt"] += 1
                if j in matched_gt_indices:
                    scale_failures[s_bin]["tp"] += 1
                else:
                    scale_failures[s_bin]["fn"] += 1
                    class_fn[c] += 1
                    weather_fn[cond] += 1

            n_fp = len(preds) - len(matched_pred_indices)
            n_fn = len(r["ground_truth"]) - len(matched_gt_indices)
            if n_fp >= 3 or n_fn >= 3:
                failure_gallery_candidates.append({
                    "image_id": r["image_id"],
                    "condition": cond,
                    "n_fp": n_fp,
                    "n_fn": n_fn,
                    "n_gt": len(r["ground_truth"]),
                    "n_pred": len(preds),
                })

        # Scale recall calculation
        for s_bin in ["small", "medium", "large"]:
            gt_cnt = scale_failures[s_bin]["gt"]
            tp_cnt = scale_failures[s_bin]["tp"]
            scale_failures[s_bin]["recall"] = (tp_cnt / gt_cnt) if gt_cnt > 0 else 0.0

        report[model_name] = {
            "operating_threshold": th,
            "weather_fp": dict(weather_fp),
            "weather_fn": dict(weather_fn),
            "weather_tp": dict(weather_tp),
            "scale_breakdown": scale_failures,
            "class_fp": dict(class_fp),
            "class_fn": dict(class_fn),
            "class_tp": dict(class_tp),
            "top_failure_images": sorted(failure_gallery_candidates, key=lambda x: (x["n_fp"] + x["n_fn"]), reverse=True)[:10],
        }

    # Render a few failure visualization images for local review
    # Select top 4 distinct condition failure images
    rendered_images = []
    top_failures = report["fasterrcnn_resnet50_fpn"]["top_failure_images"]
    seen_conditions = set()

    for item in top_failures:
        cond = item["condition"]
        if cond in seen_conditions:
            continue
        img_id = item["image_id"]
        img_path = IMAGES_DIR / img_id
        if not img_path.exists():
            continue

        # Find matching eval records
        f_rec = next((r for r in frcnn_recs if r["image_id"] == img_id), None)
        if not f_rec:
            continue

        gt_boxes = [tuple(g["box_xyxy"]) for g in f_rec["ground_truth"]]
        gt_labels = [g["class_name"] for g in f_rec["ground_truth"]]

        pred_boxes = [tuple(p["box_xyxy"]) for p in f_rec["predictions"] if p["score"] >= 0.60 and p["common_label"] is not None]
        pred_labels = [f"{p['common_label']} {p['score']:.2f}" for p in f_rec["predictions"] if p["score"] >= 0.60 and p["common_label"] is not None]

        # Render composite
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        img = Image.open(img_path).convert("RGB")
        img_gt = draw_boxes(img.copy(), gt_boxes, gt_labels, color=(0, 255, 0), width=2)
        img_pred = draw_boxes(img.copy(), pred_boxes, pred_labels, color=(255, 50, 50), width=2)

        axes[0].imshow(img_gt)
        axes[0].set_title(f"Ground Truth ({cond.capitalize()}) - {len(gt_boxes)} targets", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(img_pred)
        axes[1].set_title(f"FRCNN Predictions (thresh=0.60) - {len(pred_boxes)} detections", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        out_render = QUALITATIVE_DIR / f"failure_{cond}_{Path(img_id).stem}.png"
        plt.tight_layout()
        plt.savefig(out_render, dpi=180)
        plt.close()
        rendered_images.append(str(out_render))
        seen_conditions.add(cond)
        if len(seen_conditions) >= 4:
            break

    report["rendered_qualitative_examples"] = rendered_images
    return report


def main() -> None:
    print("Running cross-dataset error analysis...")
    report = run_error_analysis()
    out_path = ARTIFACTS / "cross_dataset_error_taxonomy.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved error taxonomy JSON to {out_path}")


if __name__ == "__main__":
    main()
