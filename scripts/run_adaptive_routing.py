"""Run Image-Level Adaptive Resolution Routing Experiments for WeatherGuard Mission 4.

Evaluates deterministic routing policies on Group B (BDD100K Development Cohort, N=160).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.adaptive_routing import (
    ImageRoutingFeatures,
    extract_routing_features,
    policy_fixed_640,
    policy_fixed_832,
    policy_fixed_1280,
    policy_small_box_trigger_k1,
    policy_small_box_trigger_k2,
    policy_low_conf_trigger_k2,
    policy_density_trigger_k8,
    policy_combined_trigger,
    policy_two_stage_hierarchical,
    policy_hierarchical_fast_832,
)
from src.metrics import evaluate_detections
from scripts.run_combined_scale_inference import evaluate_scale_breakdown_with_thresholds

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_cohort_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def simulate_routing_policy(
    policy_name: str,
    policy_fn: Any,
    data_640: dict[str, Any],
    data_832: dict[str, Any],
    data_1280: dict[str, Any],
    tau: float = 0.20,
    feat_min_score: float = 0.15,
) -> dict[str, Any]:
    """Simulate a routing policy across the cohort."""
    per_img_640 = {item["image_id"]: item for item in data_640["per_image"]}
    per_img_832 = {item["image_id"]: item for item in data_832["per_image"]}
    per_img_1280 = {item["image_id"]: item for item in data_1280["per_image"]}

    latencies_640 = dict(zip(data_640["image_ids"], data_640["per_image_seconds"]))
    latencies_832 = dict(zip(data_832["image_ids"], data_832["per_image_seconds"]))
    latencies_1280 = dict(zip(data_1280["image_ids"], data_1280["per_image_seconds"]))

    image_ids = data_640["image_ids"]
    n_images = len(image_ids)

    routed_records: list[dict] = []
    seq_latencies: list[float] = []
    standalone_latencies: list[float] = []
    decisions: list[dict[str, Any]] = []

    res_counts = {640: 0, 832: 0, 1280: 0}

    for img_id in image_ids:
        item_640 = per_img_640[img_id]
        t_base = latencies_640[img_id]

        feats = extract_routing_features(
            item_640["predictions"],
            image_id=img_id,
            min_score=feat_min_score,
        )

        chosen_res = policy_fn(feats)
        res_counts[chosen_res] += 1

        if chosen_res == 640:
            chosen_item = item_640
            t_seq = t_base
            t_standalone = t_base
        elif chosen_res == 832:
            chosen_item = per_img_832[img_id]
            t_escalated = latencies_832[img_id]
            t_seq = t_base + t_escalated
            t_standalone = t_escalated
        elif chosen_res == 1280:
            chosen_item = per_img_1280[img_id]
            t_escalated = latencies_1280[img_id]
            t_seq = t_base + t_escalated
            t_standalone = t_escalated
        else:
            raise ValueError(f"Unknown resolution: {chosen_res}")

        # For static baselines, sequential latency equals standalone latency
        if policy_name in ["fixed_640", "fixed_832", "fixed_1280"]:
            t_seq = t_standalone

        seq_latencies.append(t_seq)
        standalone_latencies.append(t_standalone)

        routed_records.append({
            "image_id": img_id,
            "condition": item_640["condition"],
            "predictions": chosen_item["predictions"],
            "ground_truth": item_640["ground_truth"],
        })

        decisions.append({
            "image_id": img_id,
            "condition": item_640["condition"],
            "routed_res": chosen_res,
            "n_small_pred": feats.n_small_boxes,
            "n_low_conf": feats.n_low_conf,
            "total_candidates": feats.n_total_candidates,
            "t_seq_ms": round(t_seq * 1000.0, 2),
        })

    # Evaluate detection metrics
    eval_res = evaluate_detections(routed_records, score_threshold=tau, iou_threshold=0.5)
    scale_res = evaluate_scale_breakdown_with_thresholds(
        routed_records, {"small": tau, "medium": tau, "large": tau}
    )

    mean_seq_ms = float(np.mean(seq_latencies) * 1000.0)
    median_seq_ms = float(np.median(seq_latencies) * 1000.0)
    p95_seq_ms = float(np.percentile(seq_latencies, 95) * 1000.0)
    fps_seq = 1000.0 / mean_seq_ms if mean_seq_ms > 0 else 0.0

    mean_standalone_ms = float(np.mean(standalone_latencies) * 1000.0)
    fps_standalone = 1000.0 / mean_standalone_ms if mean_standalone_ms > 0 else 0.0

    fixed_1280_latency = float(np.mean(list(latencies_1280.values())) * 1000.0)
    compute_savings_pct = round((fixed_1280_latency - mean_seq_ms) / fixed_1280_latency * 100.0, 2)

    return {
        "policy": policy_name,
        "n_images": n_images,
        "tau": tau,
        "feat_min_score": feat_min_score,
        "resolution_distribution": {
            "pct_640": round(res_counts[640] / n_images * 100.0, 2),
            "pct_832": round(res_counts[832] / n_images * 100.0, 2),
            "pct_1280": round(res_counts[1280] / n_images * 100.0, 2),
            "counts": res_counts,
        },
        "detection_metrics": {
            "f1": round(eval_res.micro["f1"], 5),
            "precision": round(eval_res.micro["precision"], 5),
            "recall": round(eval_res.micro["recall"], 5),
            "tp": eval_res.micro["tp"],
            "fp": eval_res.micro["fp"],
            "fn": eval_res.micro["fn"],
            "fp_per_image": round(eval_res.micro["fp"] / n_images, 3),
            "fn_per_image": round(eval_res.micro["fn"] / n_images, 3),
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
            "compute_savings_vs_1280_%": compute_savings_pct,
        },
        "latency_standalone": {
            "mean_ms": round(mean_standalone_ms, 2),
            "fps": round(fps_standalone, 2),
        },
        "decisions": decisions,
    }


def main() -> None:
    print("=" * 75)
    print("WeatherGuard Mission 4 — Experiment A: Adaptive Resolution Routing")
    print("Evaluating on Group B Development Cohort (N = 160)")
    print("=" * 75)

    path_640 = ARTIFACTS / "eval_mission3_dev_yolov8n_640.json"
    path_832 = ARTIFACTS / "eval_mission3_dev_yolov8n_832.json"
    path_1280 = ARTIFACTS / "eval_mission3_dev_yolov8n_1280.json"

    data_640 = load_cohort_json(path_640)
    data_832 = load_cohort_json(path_832)
    data_1280 = load_cohort_json(path_1280)

    policies = [
        ("fixed_640", policy_fixed_640, 0.15),
        ("fixed_832", policy_fixed_832, 0.15),
        ("fixed_1280", policy_fixed_1280, 0.15),
        ("small_box_trigger_k1", policy_small_box_trigger_k1, 0.15),
        ("small_box_trigger_k2", policy_small_box_trigger_k2, 0.15),
        ("small_box_trigger_op", policy_small_box_trigger_k1, 0.20),
        ("low_conf_trigger_k2", policy_low_conf_trigger_k2, 0.15),
        ("density_trigger_k8", policy_density_trigger_k8, 0.15),
        ("combined_trigger", policy_combined_trigger, 0.15),
        ("two_stage_hierarchical", policy_two_stage_hierarchical, 0.15),
        ("hierarchical_fast_832", policy_hierarchical_fast_832, 0.15),
    ]

    results = []
    summary_rows = []

    for name, fn, f_min_score in policies:
        res = simulate_routing_policy(
            name, fn, data_640, data_832, data_1280, tau=0.20, feat_min_score=f_min_score
        )
        results.append(res)

        summary_rows.append({
            "Policy": name,
            "F1": res["detection_metrics"]["f1"],
            "Precision": res["detection_metrics"]["precision"],
            "Recall": res["detection_metrics"]["recall"],
            "Small_Recall_%": res["scale_recall"]["small"],
            "Med_Recall_%": res["scale_recall"]["medium"],
            "Large_Recall_%": res["scale_recall"]["large"],
            "FP/Img": res["detection_metrics"]["fp_per_image"],
            "Pct_640": res["resolution_distribution"]["pct_640"],
            "Pct_832": res["resolution_distribution"]["pct_832"],
            "Pct_1280": res["resolution_distribution"]["pct_1280"],
            "Mean_Latency_ms": res["latency_sequential"]["mean_ms"],
            "P95_Latency_ms": res["latency_sequential"]["p95_ms"],
            "FPS": res["latency_sequential"]["fps"],
            "Compute_Savings_%": res["latency_sequential"]["compute_savings_vs_1280_%"],
        })
        print(f"[{name:24s}] F1: {res['detection_metrics']['f1']:.4f} | Small R: {res['scale_recall']['small']:5.2f}% | Large R: {res['scale_recall']['large']:5.2f}% | Lat: {res['latency_sequential']['mean_ms']:5.1f} ms | FPS: {res['latency_sequential']['fps']:4.1f} | Esc1280: {res['resolution_distribution']['pct_1280']:4.1f}%")

    # Save summary CSV
    df = pd.DataFrame(summary_rows)
    csv_path = ARTIFACTS / "mission4_routing_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary to: {csv_path}")

    # Save JSON artifact
    json_path = ARTIFACTS / "mission4_routing_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "cohort": "Group_B_BDD100K_Development",
            "n_images": 160,
            "policies": results,
        }, f, indent=2)
    print(f"Saved JSON artifact to: {json_path}")

    # Generate Publication-Quality Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Ax1: F1 vs Latency
    for row in summary_rows:
        is_fixed = "fixed" in row["Policy"]
        marker = "o" if is_fixed else "D"
        color = "#e63946" if is_fixed else "#1d3557"
        ax1.scatter(row["Mean_Latency_ms"], row["F1"], color=color, s=90, marker=marker, zorder=4)
        offset_y = 0.003 if "k1" in row["Policy"] or "op" in row["Policy"] else -0.004
        ax1.annotate(row["Policy"], (row["Mean_Latency_ms"] + 4, row["F1"] + offset_y), fontsize=8)

    ax1.set_xlabel("Mean Sequential Pipeline Latency (ms) [Apple M1 CPU]", fontsize=11)
    ax1.set_ylabel("Micro F1 Score (IoU=0.50, tau=0.20)", fontsize=11)
    ax1.set_title("Detection Accuracy vs. Latency Trade-Off (Group B Dev)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Ax2: Small-Object Recall vs Latency
    for row in summary_rows:
        is_fixed = "fixed" in row["Policy"]
        marker = "o" if is_fixed else "D"
        color = "#e63946" if is_fixed else "#2a9d8f"
        ax2.scatter(row["Mean_Latency_ms"], row["Small_Recall_%"], color=color, s=90, marker=marker, zorder=4)
        ax2.annotate(row["Policy"], (row["Mean_Latency_ms"] + 4, row["Small_Recall_%"] - 0.5), fontsize=8)

    ax2.set_xlabel("Mean Sequential Pipeline Latency (ms) [Apple M1 CPU]", fontsize=11)
    ax2.set_ylabel("Small-Object Recall (%)", fontsize=11)
    ax2.set_title("Small-Object Recovery vs. Latency (Group B Dev)", fontsize=12, fontweight="bold")
    ax2.axhline(24.16, color="gray", linestyle=":", label="Fixed 1280px Ceiling (24.16%)")
    ax2.axhline(8.09, color="darkgray", linestyle="--", label="Fixed 640px Floor (8.09%)")
    ax2.axvline(100.0, color="#f4a261", linestyle="--", linewidth=1.5, label="10 FPS Real-Time Boundary (100 ms)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "mission4_routing.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
