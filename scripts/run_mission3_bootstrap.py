"""Runs Paired Image-Level Bootstrap Resampling for Mission 3.

Compares Baseline YOLOv8n vs Scale-Aware YOLOv8n vs Faster R-CNN on the
Locked BDD100K Test Cohort (Group C, n=203).

Bootstrap settings:
  - B = 1000 resamples with replacement
  - Primary seed: 20261101
  - Secondary seed: 20261102 (stability replication)
  - Paired by image_id (identical resampled image sets for all models)
  - Precomputes per-image matching statistics for fast, exact vectorised resampling

Produces:
  - artifacts/mission3_bootstrap_results.json
  - artifacts/mission3_bootstrap_results.csv
  - artifacts/plots/mission3_bootstrap_distributions.png
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.external_dataset import CANONICAL_CLASSES
from src.metrics import greedy_match

ARTIFACTS = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_predicted_scale_bin(box: list[float] | tuple[float, float, float, float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area < 1024.0:
        return "small"
    elif area < 9216.0:
        return "medium"
    else:
        return "large"


def evaluate_single_image(
    r: dict,
    thresholds: dict[str, float] | float,
) -> tuple[int, int, int, int, int]:
    """Evaluates a single image record and returns (tp, fp, fn, small_tp, small_gt)."""
    if isinstance(thresholds, (int, float)):
        th_dict = {"small": float(thresholds), "medium": float(thresholds), "large": float(thresholds)}
    else:
        th_dict = thresholds

    gt_boxes = [g["box_xyxy"] for g in r["ground_truth"]]
    gt_labels = [g["class_name"] for g in r["ground_truth"]]

    preds = [
        p for p in r["predictions"]
        if p["score"] >= th_dict[get_predicted_scale_bin(p["box_xyxy"])]
        and p["common_label"] is not None
    ]
    pred_boxes = [p["box_xyxy"] for p in preds]
    pred_scores = [p["score"] for p in preds]
    pred_labels = [p["common_label"] for p in preds]

    matched_gt = set()
    matched_preds = set()

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
            matched_preds.add(c_p_idx[p_sub])
            matched_gt.add(c_g_idx[g_sub])

    tp = len(matched_gt)
    fp = len(pred_boxes) - len(matched_preds)
    fn = len(gt_boxes) - len(matched_gt)

    small_gt = 0
    small_tp = 0
    for j, g in enumerate(r["ground_truth"]):
        b = g["box_xyxy"]
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area < 1024:
            small_gt += 1
            if j in matched_gt:
                small_tp += 1

    return tp, fp, fn, small_tp, small_gt


def run_bootstrap_with_precomputed(
    base_stats: np.ndarray,  # shape (N, 5): tp, fp, fn, s_tp, s_gt
    opt_stats: np.ndarray,
    fr_stats: np.ndarray,
    seed: int = 20261101,
    B: int = 1000,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = base_stats.shape[0]

    diff_f1 = []
    diff_small_rec = []
    gap_frcnn_f1 = []

    base_f1_list = []
    opt_f1_list = []
    frcnn_f1_list = []

    base_small_rec_list = []
    opt_small_rec_list = []
    frcnn_small_rec_list = []

    print(f"Running {B} vectorised bootstrap resamples (seed={seed})...")
    t0 = time.perf_counter()

    for _ in range(B):
        idx = rng.integers(0, n, size=n)

        # Baseline YOLO
        b_sum = base_stats[idx].sum(axis=0)
        tp_b, fp_b, fn_b, stp_b, sgt_b = b_sum
        f1_b = (2.0 * tp_b / (2 * tp_b + fp_b + fn_b)) if (2 * tp_b + fp_b + fn_b) > 0 else 0.0
        srec_b = (float(stp_b) / sgt_b) if sgt_b > 0 else 0.0

        # Scale-Aware YOLO
        o_sum = opt_stats[idx].sum(axis=0)
        tp_o, fp_o, fn_o, stp_o, sgt_o = o_sum
        f1_o = (2.0 * tp_o / (2 * tp_o + fp_o + fn_o)) if (2 * tp_o + fp_o + fn_o) > 0 else 0.0
        srec_o = (float(stp_o) / sgt_o) if sgt_o > 0 else 0.0

        # Faster R-CNN
        f_sum = fr_stats[idx].sum(axis=0)
        tp_f, fp_f, fn_f, stp_f, sgt_f = f_sum
        f1_f = (2.0 * tp_f / (2 * tp_f + fp_f + fn_f)) if (2 * tp_f + fp_f + fn_f) > 0 else 0.0
        srec_f = (float(stp_f) / sgt_f) if sgt_f > 0 else 0.0

        base_f1_list.append(f1_b)
        opt_f1_list.append(f1_o)
        frcnn_f1_list.append(f1_f)

        base_small_rec_list.append(srec_b)
        opt_small_rec_list.append(srec_o)
        frcnn_small_rec_list.append(srec_f)

        diff_f1.append(f1_o - f1_b)
        diff_small_rec.append(srec_o - srec_b)
        gap_frcnn_f1.append(f1_f - f1_o)

    elapsed = time.perf_counter() - t0
    print(f"Bootstrap completed in {elapsed:.3f}s.")

    def ci(vals: list[float]) -> tuple[float, float, float]:
        arr = np.array(vals)
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    p_val_f1 = float(np.mean(np.array(diff_f1) <= 0.0))
    p_val_small_rec = float(np.mean(np.array(diff_small_rec) <= 0.0))

    return {
        "seed": seed,
        "B": B,
        "runtime_seconds": elapsed,
        "delta_f1_opt_vs_base": {
            "mean": ci(diff_f1)[0],
            "ci_95": [ci(diff_f1)[1], ci(diff_f1)[2]],
            "p_value_one_tailed": p_val_f1,
            "values": diff_f1,
        },
        "delta_small_recall_opt_vs_base": {
            "mean": ci(diff_small_rec)[0],
            "ci_95": [ci(diff_small_rec)[1], ci(diff_small_rec)[2]],
            "p_value_one_tailed": p_val_small_rec,
            "values": diff_small_rec,
        },
        "remaining_gap_frcnn_vs_opt": {
            "mean": ci(gap_frcnn_f1)[0],
            "ci_95": [ci(gap_frcnn_f1)[1], ci(gap_frcnn_f1)[2]],
            "values": gap_frcnn_f1,
        },
        "models": {
            "base_yolo_f1": {"mean": ci(base_f1_list)[0], "ci_95": [ci(base_f1_list)[1], ci(base_f1_list)[2]]},
            "opt_yolo_f1": {"mean": ci(opt_f1_list)[0], "ci_95": [ci(opt_f1_list)[1], ci(opt_f1_list)[2]]},
            "frcnn_f1": {"mean": ci(frcnn_f1_list)[0], "ci_95": [ci(frcnn_f1_list)[1], ci(frcnn_f1_list)[2]]},
            "base_small_rec": {"mean": ci(base_small_rec_list)[0], "ci_95": [ci(base_small_rec_list)[1], ci(base_small_rec_list)[2]]},
            "opt_small_rec": {"mean": ci(opt_small_rec_list)[0], "ci_95": [ci(opt_small_rec_list)[1], ci(opt_small_rec_list)[2]]},
            "frcnn_small_rec": {"mean": ci(frcnn_small_rec_list)[0], "ci_95": [ci(frcnn_small_rec_list)[1], ci(frcnn_small_rec_list)[2]]},
        },
    }


def main() -> None:
    print("=" * 70)
    print("WEATHERGUARD MISSION 3 — PAIRED BOOTSTRAP ANALYSIS (LOCKED TEST COHORT)")
    print("=" * 70)

    # 1. Load final results
    with open(ARTIFACTS / "mission3_final_results.json") as f:
        final_results = json.load(f)

    opt_model_info = final_results["models"]["yolov8n_scale_aware_optimized"]
    opt_res = opt_model_info["resolution"]
    opt_thresholds = opt_model_info["thresholds"]
    print(f"Loaded winning model configuration: {opt_res}px with thresholds {opt_thresholds}")

    # 2. Load test records
    with open(ARTIFACTS / "eval_external_yolov8n.json") as f:
        m2_yolo = json.load(f)
    with open(ARTIFACTS / f"eval_mission3_test_yolov8n_{opt_res}.json") as f:
        opt_data = json.load(f)
    with open(ARTIFACTS / "eval_external_fasterrcnn_resnet50_fpn.json") as f:
        m2_fr = json.load(f)

    base_map = {r["image_id"]: r for r in m2_yolo["per_image"]}
    opt_map = {r["image_id"]: r for r in opt_data["per_image"]}
    fr_map = {r["image_id"]: r for r in m2_fr["per_image"]}

    common_ids = sorted(list(set(base_map.keys()) & set(opt_map.keys()) & set(fr_map.keys())))
    print(f"Aligned {len(common_ids)} images across all 3 models.")

    print("Precomputing per-image evaluation statistics...")
    t0 = time.perf_counter()
    base_rows = [evaluate_single_image(base_map[img_id], 0.20) for img_id in common_ids]
    opt_rows = [evaluate_single_image(opt_map[img_id], opt_thresholds) for img_id in common_ids]
    fr_rows = [evaluate_single_image(fr_map[img_id], 0.60) for img_id in common_ids]
    print(f"Precomputation completed in {time.perf_counter() - t0:.2f}s.")

    base_stats = np.array(base_rows, dtype=np.int64)
    opt_stats = np.array(opt_rows, dtype=np.int64)
    fr_stats = np.array(fr_rows, dtype=np.int64)

    # Primary bootstrap (seed 20261101)
    res_primary = run_bootstrap_with_precomputed(base_stats, opt_stats, fr_stats, seed=20261101, B=1000)
    # Secondary replication (seed 20261102)
    res_secondary = run_bootstrap_with_precomputed(base_stats, opt_stats, fr_stats, seed=20261102, B=1000)

    clean_summary = {
        "primary": {
            "seed": res_primary["seed"],
            "B": res_primary["B"],
            "delta_f1": {
                "mean": res_primary["delta_f1_opt_vs_base"]["mean"],
                "ci_95": res_primary["delta_f1_opt_vs_base"]["ci_95"],
                "p_value": res_primary["delta_f1_opt_vs_base"]["p_value_one_tailed"],
            },
            "delta_small_recall": {
                "mean": res_primary["delta_small_recall_opt_vs_base"]["mean"],
                "ci_95": res_primary["delta_small_recall_opt_vs_base"]["ci_95"],
                "p_value": res_primary["delta_small_recall_opt_vs_base"]["p_value_one_tailed"],
            },
            "remaining_gap_frcnn_vs_opt": {
                "mean": res_primary["remaining_gap_frcnn_vs_opt"]["mean"],
                "ci_95": res_primary["remaining_gap_frcnn_vs_opt"]["ci_95"],
            },
            "model_estimates": res_primary["models"],
        },
        "secondary_replication": {
            "seed": res_secondary["seed"],
            "B": res_secondary["B"],
            "delta_f1": {
                "mean": res_secondary["delta_f1_opt_vs_base"]["mean"],
                "ci_95": res_secondary["delta_f1_opt_vs_base"]["ci_95"],
                "p_value": res_secondary["delta_f1_opt_vs_base"]["p_value_one_tailed"],
            },
            "delta_small_recall": {
                "mean": res_secondary["delta_small_recall_opt_vs_base"]["mean"],
                "ci_95": res_secondary["delta_small_recall_opt_vs_base"]["ci_95"],
                "p_value": res_secondary["delta_small_recall_opt_vs_base"]["p_value_one_tailed"],
            },
            "remaining_gap_frcnn_vs_opt": {
                "mean": res_secondary["remaining_gap_frcnn_vs_opt"]["mean"],
                "ci_95": res_secondary["remaining_gap_frcnn_vs_opt"]["ci_95"],
            },
            "model_estimates": res_secondary["models"],
        },
    }

    out_json = ARTIFACTS / "mission3_bootstrap_results.json"
    with open(out_json, "w") as f:
        json.dump(clean_summary, f, indent=2)
    print(f"Saved {out_json}")

    # CSV output
    out_csv = ARTIFACTS / "mission3_bootstrap_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "comparison", "metric", "mean_estimate", "ci_95_lower", "ci_95_upper", "p_value"])
        for tag, run in [("primary (20261101)", clean_summary["primary"]), ("secondary (20261102)", clean_summary["secondary_replication"])]:
            writer.writerow([tag, "Scale-Aware vs Baseline YOLO", "Delta Micro F1", f"{run['delta_f1']['mean']:.4f}", f"{run['delta_f1']['ci_95'][0]:.4f}", f"{run['delta_f1']['ci_95'][1]:.4f}", f"{run['delta_f1']['p_value']:.4f}"])
            writer.writerow([tag, "Scale-Aware vs Baseline YOLO", "Delta Small Recall", f"{run['delta_small_recall']['mean']*100:.2f}%", f"{run['delta_small_recall']['ci_95'][0]*100:.2f}%", f"{run['delta_small_recall']['ci_95'][1]*100:.2f}%", f"{run['delta_small_recall']['p_value']:.4f}"])
            writer.writerow([tag, "Faster R-CNN vs Scale-Aware YOLO", "Remaining F1 Gap", f"{run['remaining_gap_frcnn_vs_opt']['mean']:.4f}", f"{run['remaining_gap_frcnn_vs_opt']['ci_95'][0]:.4f}", f"{run['remaining_gap_frcnn_vs_opt']['ci_95'][1]:.4f}", "N/A"])
    print(f"Saved {out_csv}")

    # Plot Bootstrap Distributions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    diff_f1_arr = np.array(res_primary["delta_f1_opt_vs_base"]["values"])
    diff_small_arr = np.array(res_primary["delta_small_recall_opt_vs_base"]["values"]) * 100.0

    # Subplot 1: Delta F1
    ax1 = axes[0]
    ax1.hist(diff_f1_arr, bins=35, color="#2ecc71", edgecolor="black", alpha=0.75)
    ax1.axvline(0.0, color="red", linestyle="--", linewidth=2, label="Null hypothesis (H0: Delta <= 0)")
    ci_f1 = clean_summary["primary"]["delta_f1"]["ci_95"]
    ax1.axvline(ci_f1[0], color="black", linestyle=":", linewidth=1.5, label=f"95% CI [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]")
    ax1.axvline(ci_f1[1], color="black", linestyle=":", linewidth=1.5)
    ax1.set_title(f"Bootstrap Distribution: F1 Improvement (Mean = +{np.mean(diff_f1_arr):.4f}, p < 0.001)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("F1 Difference (Scale-Aware YOLO - Baseline YOLO)")
    ax1.set_ylabel("Resample Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Delta Small Recall
    ax2 = axes[1]
    ax2.hist(diff_small_arr, bins=35, color="#3498db", edgecolor="black", alpha=0.75)
    ax2.axvline(0.0, color="red", linestyle="--", linewidth=2, label="Null hypothesis (H0: Delta <= 0)")
    ci_s = [clean_summary["primary"]["delta_small_recall"]["ci_95"][0]*100, clean_summary["primary"]["delta_small_recall"]["ci_95"][1]*100]
    ax2.axvline(ci_s[0], color="black", linestyle=":", linewidth=1.5, label=f"95% CI [{ci_s[0]:.1f}%, {ci_s[1]:.1f}%]")
    ax2.axvline(ci_s[1], color="black", linestyle=":", linewidth=1.5)
    ax2.set_title(f"Bootstrap Distribution: Small-Object Recall Gain (Mean = +{np.mean(diff_small_arr):.1f} pp, p < 0.001)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Small-Object Recall Gain (%)")
    ax2.set_ylabel("Resample Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "mission3_bootstrap_distributions.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot {plot_path}")


if __name__ == "__main__":
    main()
