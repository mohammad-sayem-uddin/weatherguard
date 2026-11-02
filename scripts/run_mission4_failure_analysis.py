"""Run Failure-Cause Analysis for WeatherGuard Mission 4.

Investigates why 71% of small objects remain undetected even under scale-aware inference,
and deconstructs false negatives and false positives into visual, geometric, and environmental causes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from src.external_dataset import CANONICAL_CLASSES
from src.metrics import greedy_match, iou_xyxy

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def is_near_boundary(box: list[float], w: int = 1280, h: int = 720, margin: float = 10.0) -> bool:
    """Check if bounding box touches or is near image edge."""
    return box[0] <= margin or box[1] <= margin or box[2] >= (w - margin) or box[3] >= (h - margin)


def compute_aspect_ratio(box: list[float]) -> float:
    bw = max(1e-3, box[2] - box[0])
    bh = max(1e-3, box[3] - box[1])
    return bw / bh


def analyze_cohort_failures(
    eval_records: list[dict[str, Any]],
    tau: float = 0.20,
    iou_thresh: float = 0.50,
) -> dict[str, Any]:
    """Perform detailed geometric and environmental attribution of FNs and FPs."""
    total_gt = 0
    fn_count = 0
    tp_count = 0
    fp_count = 0

    fn_by_scale = {"sub16": 0, "small16_32": 0, "medium": 0, "large": 0}
    gt_by_scale = {"sub16": 0, "small16_32": 0, "medium": 0, "large": 0}

    fn_causes = {
        "extreme_scale_sub16": 0,
        "boundary_truncation": 0,
        "crowding_occlusion": 0,
        "extreme_aspect_ratio": 0,
        "optical_weather_degraded": 0,
        "standard_adverse_miss": 0,
    }

    fp_causes = {
        "class_confusion": 0,
        "low_confidence_boundary": 0,
        "background_structure": 0,
    }

    class_breakdown: dict[str, dict[str, int]] = {
        c: {"gt": 0, "tp": 0, "fn": 0, "fp": 0} for c in CANONICAL_CLASSES
    }

    weather_fn: dict[str, dict[str, int]] = {}

    for r in eval_records:
        cond = r.get("condition", "unknown")
        if cond not in weather_fn:
            weather_fn[cond] = {"gt": 0, "fn": 0, "small_gt": 0, "small_fn": 0}

        gt_boxes = [g["box_xyxy"] for g in r["ground_truth"]]
        gt_labels = [g["class_name"] for g in r["ground_truth"]]

        preds = [
            p for p in r["predictions"]
            if p.get("score", 0.0) >= tau and p.get("common_label") is not None
        ]
        pred_boxes = [p["box_xyxy"] for p in preds]
        pred_scores = [p["score"] for p in preds]
        pred_labels = [p["common_label"] for p in preds]

        matched_gt_indices = set()
        matched_pred_indices = set()

        for c in CANONICAL_CLASSES:
            c_p_idx = [i for i, l in enumerate(pred_labels) if l == c]
            c_g_idx = [j for j, l in enumerate(gt_labels) if l == c]
            if not c_p_idx or not c_g_idx:
                continue
            sub_p = [pred_boxes[i] for i in c_p_idx]
            sub_s = [pred_scores[i] for i in c_p_idx]
            sub_g = [gt_boxes[j] for j in c_g_idx]
            matches = greedy_match(sub_p, sub_s, sub_g, iou_threshold=iou_thresh)
            for p_sub, g_sub, _ in matches:
                matched_pred_indices.add(c_p_idx[p_sub])
                matched_gt_indices.add(c_g_idx[g_sub])

        # Track GT & FNs
        for j, g in enumerate(r["ground_truth"]):
            total_gt += 1
            box = g["box_xyxy"]
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            c_name = g["class_name"]
            if c_name in class_breakdown:
                class_breakdown[c_name]["gt"] += 1
            weather_fn[cond]["gt"] += 1

            if area < 256.0:  # < 16^2
                scale_cat = "sub16"
            elif area < 1024.0:  # 16^2 - 32^2
                scale_cat = "small16_32"
            elif area < 9216.0:
                scale_cat = "medium"
            else:
                scale_cat = "large"

            gt_by_scale[scale_cat] += 1
            if scale_cat in ["sub16", "small16_32"]:
                weather_fn[cond]["small_gt"] += 1

            if j in matched_gt_indices:
                tp_count += 1
                if c_name in class_breakdown:
                    class_breakdown[c_name]["tp"] += 1
            else:
                fn_count += 1
                fn_by_scale[scale_cat] += 1
                if c_name in class_breakdown:
                    class_breakdown[c_name]["fn"] += 1
                weather_fn[cond]["fn"] += 1
                if scale_cat in ["sub16", "small16_32"]:
                    weather_fn[cond]["small_fn"] += 1

                # Attribution hierarchy for FN
                ar = compute_aspect_ratio(box)
                is_boundary = is_near_boundary(box)
                # Check crowding (overlaps with another GT box)
                crowded = any(
                    k != j and iou_xyxy(box, gt_boxes[k]) > 0.15 for k in range(len(gt_boxes))
                )

                if scale_cat == "sub16":
                    fn_causes["extreme_scale_sub16"] += 1
                elif is_boundary:
                    fn_causes["boundary_truncation"] += 1
                elif crowded:
                    fn_causes["crowding_occlusion"] += 1
                elif ar > 3.0 or ar < 0.33:
                    fn_causes["extreme_aspect_ratio"] += 1
                elif cond in ["foggy", "snowy", "rainy"]:
                    fn_causes["optical_weather_degraded"] += 1
                else:
                    fn_causes["standard_adverse_miss"] += 1

        # Track FPs
        for i, p in enumerate(preds):
            c_name = p["common_label"]
            if i in matched_pred_indices:
                pass
            else:
                fp_count += 1
                if c_name in class_breakdown:
                    class_breakdown[c_name]["fp"] += 1

                # Check if box overlapped with ANY ground truth (class confusion)
                box = p["box_xyxy"]
                has_any_gt_overlap = any(iou_xyxy(box, gb) >= 0.50 for gb in gt_boxes)
                if has_any_gt_overlap:
                    fp_causes["class_confusion"] += 1
                elif p["score"] < 0.30:
                    fp_causes["low_confidence_boundary"] += 1
                else:
                    fp_causes["background_structure"] += 1

    return {
        "total_gt": total_gt,
        "tp": tp_count,
        "fn": fn_count,
        "fp": fp_count,
        "miss_rate_overall_%": round(fn_count / total_gt * 100.0, 2),
        "gt_by_scale": gt_by_scale,
        "fn_by_scale": fn_by_scale,
        "scale_miss_rates_%": {
            k: round(fn_by_scale[k] / gt_by_scale[k] * 100.0, 2) if gt_by_scale[k] > 0 else 0.0
            for k in gt_by_scale
        },
        "fn_attribution": {
            k: {
                "count": v,
                "pct_of_fns": round(v / fn_count * 100.0, 2) if fn_count > 0 else 0.0,
            }
            for k, v in fn_causes.items()
        },
        "fp_attribution": {
            k: {
                "count": v,
                "pct_of_fps": round(v / fp_count * 100.0, 2) if fp_count > 0 else 0.0,
            }
            for k, v in fp_causes.items()
        },
        "class_breakdown": class_breakdown,
        "weather_breakdown": weather_fn,
    }


def main() -> None:
    print("=" * 75)
    print("WeatherGuard Mission 4 — Experiment C: Failure-Cause Analysis")
    print("Evaluating on Group B Development Cohort (N = 160)")
    print("=" * 75)

    path_640 = ARTIFACTS / "eval_mission3_dev_yolov8n_640.json"
    path_1280 = ARTIFACTS / "eval_mission3_dev_yolov8n_1280.json"

    data_640 = json.load(open(path_640))
    data_1280 = json.load(open(path_1280))

    analysis_640 = analyze_cohort_failures(data_640["per_image"], tau=0.20)
    analysis_1280 = analyze_cohort_failures(data_1280["per_image"], tau=0.20)

    print("\n--- Scale-Stratified Miss Rates ---")
    print(f"Sub-16px (<256 px^2):  640px = {analysis_640['scale_miss_rates_%']['sub16']}%  |  1280px = {analysis_1280['scale_miss_rates_%']['sub16']}%")
    print(f"Small (16-32px):       640px = {analysis_640['scale_miss_rates_%']['small16_32']}%  |  1280px = {analysis_1280['scale_miss_rates_%']['small16_32']}%")
    print(f"Medium (32-96px):      640px = {analysis_640['scale_miss_rates_%']['medium']}%  |  1280px = {analysis_1280['scale_miss_rates_%']['medium']}%")
    print(f"Large (>=96px):        640px = {analysis_640['scale_miss_rates_%']['large']}%  |  1280px = {analysis_1280['scale_miss_rates_%']['large']}%")

    print("\n--- Failure Attribution (1280px High-Resolution Pass) ---")
    for k, v in analysis_1280["fn_attribution"].items():
        print(f"  {k:28s}: {v['count']:4d} ({v['pct_of_fns']:5.1f}%)")

    print("\n--- False Positive Attribution (1280px) ---")
    for k, v in analysis_1280["fp_attribution"].items():
        print(f"  {k:28s}: {v['count']:4d} ({v['pct_of_fps']:5.1f}%)")

    # Save JSON artifact
    json_path = ARTIFACTS / "mission4_failure_analysis.json"
    with open(json_path, "w") as f:
        json.dump({
            "cohort": "Group_B_BDD100K_Development",
            "n_images": 160,
            "analysis_640": analysis_640,
            "analysis_1280": analysis_1280,
        }, f, indent=2)
    print(f"\nSaved JSON artifact to: {json_path}")

    # Save summary CSV
    rows = []
    for k, v in analysis_1280["fn_attribution"].items():
        rows.append({"Type": "False Negative", "Cause": k, "Count": v["count"], "Percentage_%": v["pct_of_fns"]})
    for k, v in analysis_1280["fp_attribution"].items():
        rows.append({"Type": "False Positive", "Cause": k, "Count": v["count"], "Percentage_%": v["pct_of_fps"]})
    df = pd.DataFrame(rows)
    csv_path = ARTIFACTS / "mission4_failure_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV artifact to: {csv_path}")


if __name__ == "__main__":
    main()
