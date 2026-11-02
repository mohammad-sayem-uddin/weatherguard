"""Run Weather x Scale x Inference Interaction Analysis for WeatherGuard Mission 4.

Investigates whether adaptive resolution routing benefits all weather categories equally
across Clear, Overcast, Rainy, Snowy (and Foggy where present).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.adaptive_routing import (
    extract_routing_features,
    policy_small_box_trigger_k1,
    policy_two_stage_hierarchical,
)
from src.metrics import evaluate_detections
from scripts.run_combined_scale_inference import evaluate_scale_breakdown_with_thresholds

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_weather_strata(
    records: list[dict[str, Any]],
    tau: float = 0.20,
    model_name: str = "",
) -> dict[str, dict[str, Any]]:
    """Compute performance metrics stratified by weather condition."""
    conditions = sorted(list(set(r.get("condition", "unknown") for r in records)))
    strata: dict[str, dict[str, Any]] = {}

    for cond in conditions:
        cond_records = [r for r in records if r.get("condition") == cond]
        n_imgs = len(cond_records)
        eval_res = evaluate_detections(cond_records, score_threshold=tau, iou_threshold=0.5)
        scale_res = evaluate_scale_breakdown_with_thresholds(
            cond_records, {"small": tau, "medium": tau, "large": tau}
        )

        strata[cond] = {
            "n_images": n_imgs,
            "f1": round(eval_res.micro["f1"], 5),
            "precision": round(eval_res.micro["precision"], 5),
            "recall": round(eval_res.micro["recall"], 5),
            "tp": eval_res.micro["tp"],
            "fp": eval_res.micro["fp"],
            "fn": eval_res.micro["fn"],
            "fp_per_image": round(eval_res.micro["fp"] / n_imgs, 3) if n_imgs > 0 else 0.0,
            "small_recall_%": round(scale_res["small"]["recall"] * 100.0, 2),
            "medium_recall_%": round(scale_res["medium"]["recall"] * 100.0, 2),
            "large_recall_%": round(scale_res["large"]["recall"] * 100.0, 2),
            "small_gt": scale_res["small"]["gt"],
            "small_tp": scale_res["small"]["tp"],
        }
    return strata


def main() -> None:
    print("=" * 75)
    print("WeatherGuard Mission 4 — Experiment D: Weather x Scale Interaction")
    print("Evaluating on Group B Development Cohort (N = 160)")
    print("=" * 75)

    path_640 = ARTIFACTS / "eval_mission3_dev_yolov8n_640.json"
    path_832 = ARTIFACTS / "eval_mission3_dev_yolov8n_832.json"
    path_1280 = ARTIFACTS / "eval_mission3_dev_yolov8n_1280.json"
    path_frcnn = ARTIFACTS / "eval_mission3_dev_fasterrcnn_baseline.json"

    d640 = json.load(open(path_640))
    d832 = json.load(open(path_832))
    d1280 = json.load(open(path_1280))
    dfrcnn = json.load(open(path_frcnn))

    per_img_640 = {x["image_id"]: x for x in d640["per_image"]}
    per_img_832 = {x["image_id"]: x for x in d832["per_image"]}
    per_img_1280 = {x["image_id"]: x for x in d1280["per_image"]}

    # Simulate adaptive routing records for weather breakdown
    adaptive_records_k1: list[dict] = []
    adaptive_records_hier: list[dict] = []

    escalation_by_weather_k1: dict[str, list[int]] = {}
    escalation_by_weather_hier: dict[str, list[int]] = {}

    for img_id in d640["image_ids"]:
        item640 = per_img_640[img_id]
        cond = item640["condition"]
        feats = extract_routing_features(item640["predictions"], min_score=0.15)

        # Policy k1
        res_k1 = policy_small_box_trigger_k1(feats)
        if cond not in escalation_by_weather_k1:
            escalation_by_weather_k1[cond] = []
        escalation_by_weather_k1[cond].append(res_k1)

        if res_k1 == 1280:
            adaptive_records_k1.append(per_img_1280[img_id])
        else:
            adaptive_records_k1.append(item640)

        # Policy hierarchical
        res_hier = policy_two_stage_hierarchical(feats)
        if cond not in escalation_by_weather_hier:
            escalation_by_weather_hier[cond] = []
        escalation_by_weather_hier[cond].append(res_hier)

        if res_hier == 1280:
            adaptive_records_hier.append(per_img_1280[img_id])
        elif res_hier == 832:
            adaptive_records_hier.append(per_img_832[img_id])
        else:
            adaptive_records_hier.append(item640)

    # Evaluate strata
    strata_640 = evaluate_weather_strata(d640["per_image"], tau=0.20, model_name="YOLOv8n-640")
    strata_832 = evaluate_weather_strata(d832["per_image"], tau=0.20, model_name="YOLOv8n-832")
    strata_1280 = evaluate_weather_strata(d1280["per_image"], tau=0.20, model_name="YOLOv8n-1280")
    strata_frcnn = evaluate_weather_strata(dfrcnn["per_image"], tau=0.60, model_name="Faster R-CNN")
    strata_adapt_k1 = evaluate_weather_strata(adaptive_records_k1, tau=0.20, model_name="Adaptive-k1")
    strata_adapt_hier = evaluate_weather_strata(adaptive_records_hier, tau=0.20, model_name="Adaptive-Hier")

    models = [
        ("YOLOv8n-640", strata_640),
        ("YOLOv8n-832", strata_832),
        ("YOLOv8n-1280", strata_1280),
        ("Adaptive-k1", strata_adapt_k1),
        ("Adaptive-Hier", strata_adapt_hier),
        ("Faster R-CNN", strata_frcnn),
    ]

    summary_rows = []
    print(f"\n| Weather | Model | N_img | F1 | Prec | Rec | Small_Rec_% | Med_Rec_% | Large_Rec_% | FP/Img |")
    print(f"|---|---|---|---|---|---|---|---|---|---|")

    for cond in sorted(strata_640.keys()):
        for m_name, m_strata in models:
            res = m_strata[cond]
            esc_info = ""
            if m_name == "Adaptive-k1":
                esc_pct = sum(1 for r in escalation_by_weather_k1[cond] if r == 1280) / len(escalation_by_weather_k1[cond]) * 100
                esc_info = f" (Esc1280={esc_pct:.0f}%)"
            elif m_name == "Adaptive-Hier":
                esc_pct = sum(1 for r in escalation_by_weather_hier[cond] if r == 1280) / len(escalation_by_weather_hier[cond]) * 100
                esc_info = f" (Esc1280={esc_pct:.0f}%)"

            summary_rows.append({
                "Weather": cond,
                "Model": m_name,
                "N_Images": res["n_images"],
                "F1": res["f1"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "Small_Recall_%": res["small_recall_%"],
                "Med_Recall_%": res["medium_recall_%"],
                "Large_Recall_%": res["large_recall_%"],
                "FP_per_Image": res["fp_per_image"],
            })
            print(f"| {cond:8s} | {m_name + esc_info:20s} | {res['n_images']:3d} | {res['f1']:.4f} | {res['precision']:.4f} | {res['recall']:.4f} | {res['small_recall_%']:5.2f}% | {res['medium_recall_%']:5.2f}% | {res['large_recall_%']:5.2f}% | {res['fp_per_image']:4.2f} |")

    # Save summary CSV
    df = pd.DataFrame(summary_rows)
    csv_path = ARTIFACTS / "mission4_weather_scale.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV artifact to: {csv_path}")

    # Save JSON artifact
    json_path = ARTIFACTS / "mission4_weather_scale.json"
    with open(json_path, "w") as f:
        json.dump({
            "cohort": "Group_B_BDD100K_Development",
            "n_images": 160,
            "weather_strata": {
                "640px": strata_640,
                "832px": strata_832,
                "1280px": strata_1280,
                "adaptive_k1": strata_adapt_k1,
                "adaptive_hier": strata_adapt_hier,
                "faster_rcnn": strata_frcnn,
            },
            "escalation_rates_k1_%": {
                c: round(sum(1 for r in v if r == 1280) / len(v) * 100.0, 2)
                for c, v in escalation_by_weather_k1.items()
            },
            "escalation_rates_hier_%": {
                c: round(sum(1 for r in v if r == 1280) / len(v) * 100.0, 2)
                for c, v in escalation_by_weather_hier.items()
            },
        }, f, indent=2)
    print(f"Saved JSON artifact to: {json_path}")

    # Generate Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    conditions = sorted(strata_640.keys())
    x = np.arange(len(conditions))
    width = 0.18

    # Ax1: F1 by Weather
    ax1.bar(x - 1.5 * width, [strata_640[c]["f1"] for c in conditions], width, label="YOLOv8n 640px", color="#6c757d")
    ax1.bar(x - 0.5 * width, [strata_832[c]["f1"] for c in conditions], width, label="YOLOv8n 832px", color="#457b9d")
    ax1.bar(x + 0.5 * width, [strata_adapt_k1[c]["f1"] for c in conditions], width, label="Adaptive (k1)", color="#1d3557")
    ax1.bar(x + 1.5 * width, [strata_1280[c]["f1"] for c in conditions], width, label="YOLOv8n 1280px", color="#e63946")

    ax1.set_xticks(x)
    ax1.set_xticklabels([c.capitalize() for c in conditions], fontsize=11)
    ax1.set_ylabel("Micro F1 Score", fontsize=11)
    ax1.set_title("Detection F1 Across Weather Strata (Group B)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(loc="lower right", fontsize=9)

    # Ax2: Small-Object Recall by Weather
    ax2.bar(x - 1.5 * width, [strata_640[c]["small_recall_%"] for c in conditions], width, label="YOLOv8n 640px", color="#6c757d")
    ax2.bar(x - 0.5 * width, [strata_832[c]["small_recall_%"] for c in conditions], width, label="YOLOv8n 832px", color="#457b9d")
    ax2.bar(x + 0.5 * width, [strata_adapt_k1[c]["small_recall_%"] for c in conditions], width, label="Adaptive (k1)", color="#1d3557")
    ax2.bar(x + 1.5 * width, [strata_1280[c]["small_recall_%"] for c in conditions], width, label="YOLOv8n 1280px", color="#e63946")

    ax2.set_xticks(x)
    ax2.set_xticklabels([c.capitalize() for c in conditions], fontsize=11)
    ax2.set_ylabel("Small-Object Recall (%)", fontsize=11)
    ax2.set_title("Small-Object Recall Across Weather Strata (Group B)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "mission4_weather_scale.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
