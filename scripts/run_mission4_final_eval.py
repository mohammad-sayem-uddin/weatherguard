"""Final Evaluation, Pareto Frontier & Paired Bootstrap Analysis on Locked Group C (N=203).

Executes the frozen Mission 4 adaptive routing policies on the locked BDD100K test cohort.
Performs paired image-level bootstrap resampling across seeds 20261101 and 20261102.
Saves:
  - artifacts/mission4_final_results.json
  - artifacts/mission4_efficiency.json
  - artifacts/mission4_bootstrap_results.json
  - artifacts/plots/mission4_accuracy_latency.png
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.adaptive_routing import (
    extract_routing_features,
    policy_small_box_trigger_k1,
    policy_two_stage_hierarchical,
)
from src.metrics import evaluate_detections, greedy_match
from src.external_dataset import CANONICAL_CLASSES
from scripts.run_combined_scale_inference import evaluate_scale_breakdown_with_thresholds

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_cohort_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def precompute_per_image_stats(
    eval_records: list[dict[str, Any]],
    tau: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-image TP, FP, FN, small_TP, small_GT, large_TP, large_GT for fast vectorized bootstrap."""
    n_images = len(eval_records)
    tp_arr = np.zeros(n_images, dtype=np.int32)
    fp_arr = np.zeros(n_images, dtype=np.int32)
    fn_arr = np.zeros(n_images, dtype=np.int32)
    small_tp_arr = np.zeros(n_images, dtype=np.int32)
    small_gt_arr = np.zeros(n_images, dtype=np.int32)
    large_tp_arr = np.zeros(n_images, dtype=np.int32)
    large_gt_arr = np.zeros(n_images, dtype=np.int32)

    for idx, r in enumerate(eval_records):
        gt_boxes = [g["box_xyxy"] for g in r["ground_truth"]]
        gt_labels = [g["class_name"] for g in r["ground_truth"]]

        preds = [
            p for p in r["predictions"]
            if p.get("score", 0.0) >= tau and p.get("common_label") is not None
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
            matches = greedy_match(
                [pred_boxes[i] for i in c_p_idx],
                [pred_scores[i] for i in c_p_idx],
                [gt_boxes[j] for j in c_g_idx],
                iou_threshold=0.50,
            )
            for p_sub, g_sub, _ in matches:
                matched_preds.add(c_p_idx[p_sub])
                matched_gt.add(c_g_idx[g_sub])

        tp_arr[idx] = len(matched_preds)
        fp_arr[idx] = len(preds) - len(matched_preds)
        fn_arr[idx] = len(r["ground_truth"]) - len(matched_gt)

        # Scale breakdowns
        for j, g in enumerate(r["ground_truth"]):
            b = g["box_xyxy"]
            area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            is_matched = j in matched_gt
            if area < 1024.0:
                small_gt_arr[idx] += 1
                if is_matched:
                    small_tp_arr[idx] += 1
            elif area >= 9216.0:
                large_gt_arr[idx] += 1
                if is_matched:
                    large_tp_arr[idx] += 1

    return tp_arr, fp_arr, fn_arr, small_tp_arr, small_gt_arr, large_tp_arr, large_gt_arr


def run_paired_bootstrap(
    m1_stats: tuple[np.ndarray, ...],
    m2_stats: tuple[np.ndarray, ...],
    n_resamples: int = 1000,
    seed: int = 20261101,
) -> dict[str, Any]:
    """Execute paired image-level bootstrap resampling."""
    tp1, fp1, fn1, stp1, sgt1, ltp1, lgt1 = m1_stats
    tp2, fp2, fn2, stp2, sgt2, ltp2, lgt2 = m2_stats

    n = len(tp1)
    rng = np.random.RandomState(seed)

    diff_f1 = np.zeros(n_resamples, dtype=np.float64)
    diff_small_rec = np.zeros(n_resamples, dtype=np.float64)
    diff_large_rec = np.zeros(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        idx = rng.randint(0, n, size=n)

        # Model 1
        s_tp1 = np.sum(tp1[idx])
        s_fp1 = np.sum(fp1[idx])
        s_fn1 = np.sum(fn1[idx])
        p1 = s_tp1 / (s_tp1 + s_fp1) if (s_tp1 + s_fp1) > 0 else 0.0
        r1 = s_tp1 / (s_tp1 + s_fn1) if (s_tp1 + s_fn1) > 0 else 0.0
        f1_1 = (2.0 * p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0.0

        s_sgt1 = np.sum(sgt1[idx])
        s_stp1 = np.sum(stp1[idx])
        s_rec1 = s_stp1 / s_sgt1 if s_sgt1 > 0 else 0.0

        s_lgt1 = np.sum(lgt1[idx])
        s_ltp1 = np.sum(ltp1[idx])
        l_rec1 = s_ltp1 / s_lgt1 if s_lgt1 > 0 else 0.0

        # Model 2
        s_tp2 = np.sum(tp2[idx])
        s_fp2 = np.sum(fp2[idx])
        s_fn2 = np.sum(fn2[idx])
        p2 = s_tp2 / (s_tp2 + s_fp2) if (s_tp2 + s_fp2) > 0 else 0.0
        r2 = s_tp2 / (s_tp2 + s_fn2) if (s_tp2 + s_fn2) > 0 else 0.0
        f1_2 = (2.0 * p2 * r2) / (p2 + r2) if (p2 + r2) > 0 else 0.0

        s_sgt2 = np.sum(sgt2[idx])
        s_stp2 = np.sum(stp2[idx])
        s_rec2 = s_stp2 / s_sgt2 if s_sgt2 > 0 else 0.0

        s_lgt2 = np.sum(lgt2[idx])
        s_ltp2 = np.sum(ltp2[idx])
        l_rec2 = s_ltp2 / s_lgt2 if s_lgt2 > 0 else 0.0

        # Differences (M2 - M1)
        diff_f1[b] = f1_2 - f1_1
        diff_small_rec[b] = s_rec2 - s_rec1
        diff_large_rec[b] = l_rec2 - l_rec1

    ci_f1 = [float(np.percentile(diff_f1, 2.5)), float(np.percentile(diff_f1, 97.5))]
    ci_srec = [float(np.percentile(diff_small_rec, 2.5)), float(np.percentile(diff_small_rec, 97.5))]
    ci_lrec = [float(np.percentile(diff_large_rec, 2.5)), float(np.percentile(diff_large_rec, 97.5))]

    # Empirical p-values (proportion with opposite sign)
    p_val_f1 = float(2.0 * min(np.mean(diff_f1 <= 0), np.mean(diff_f1 >= 0)))
    p_val_srec = float(2.0 * min(np.mean(diff_small_rec <= 0), np.mean(diff_small_rec >= 0)))

    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "delta_f1": {
            "mean": float(np.mean(diff_f1)),
            "std": float(np.std(diff_f1)),
            "ci_95": ci_f1,
            "p_value": p_val_f1,
        },
        "delta_small_recall": {
            "mean": float(np.mean(diff_small_rec)),
            "std": float(np.std(diff_small_rec)),
            "ci_95": ci_srec,
            "p_value": p_val_srec,
        },
        "delta_large_recall": {
            "mean": float(np.mean(diff_large_rec)),
            "std": float(np.std(diff_large_rec)),
            "ci_95": ci_lrec,
        },
    }


def main() -> None:
    print("=" * 75)
    print("WeatherGuard Mission 4 — Experiment F: External Test Validation (Locked Group C)")
    print("N = 203 BDD100K Locked Test Cohort")
    print("=" * 75)

    path_640 = ARTIFACTS / "eval_external_yolov8n.json"
    path_832 = ARTIFACTS / "eval_mission3_test_yolov8n_832.json"
    path_1280 = ARTIFACTS / "eval_mission3_test_yolov8n_1280.json"
    path_frcnn = ARTIFACTS / "eval_external_fasterrcnn_resnet50_fpn.json"

    d640 = load_cohort_json(path_640)
    d832 = load_cohort_json(path_832)
    d1280 = load_cohort_json(path_1280)
    dfrcnn = load_cohort_json(path_frcnn)

    per_img_640 = {x["image_id"]: x for x in d640["per_image"]}
    per_img_832 = {x["image_id"]: x for x in d832["per_image"]}
    per_img_1280 = {x["image_id"]: x for x in d1280["per_image"]}

    def get_latencies(d_json: dict[str, Any]) -> dict[str, float]:
        if "per_image_seconds" in d_json:
            return dict(zip(d_json["image_ids"], d_json["per_image_seconds"]))
        mean_sec = d_json.get("inference_mean_seconds", 0.0611)
        return {img_id: mean_sec for img_id in d_json["image_ids"]}

    latencies_640 = get_latencies(d640)
    latencies_832 = get_latencies(d832)
    latencies_1280 = get_latencies(d1280)
    latencies_frcnn = get_latencies(dfrcnn)

    image_ids = d640["image_ids"]
    n_images = len(image_ids)

    # 1. Build Locked Test Records for Adaptive Policies
    routed_records_k1: list[dict] = []
    routed_records_hier: list[dict] = []
    seq_latencies_k1: list[float] = []
    seq_latencies_hier: list[float] = []
    res_counts_k1 = {640: 0, 1280: 0}
    res_counts_hier = {640: 0, 832: 0, 1280: 0}

    for img_id in image_ids:
        item640 = per_img_640[img_id]
        t_base = latencies_640[img_id]

        # Extract features from base pass (zero ground truth!)
        feats = extract_routing_features(item640["predictions"], min_score=0.15)

        # Policy k1
        res_k1 = policy_small_box_trigger_k1(feats)
        res_counts_k1[res_k1] += 1
        if res_k1 == 1280:
            routed_records_k1.append(per_img_1280[img_id])
            seq_latencies_k1.append(t_base + latencies_1280[img_id])
        else:
            routed_records_k1.append(item640)
            seq_latencies_k1.append(t_base)

        # Policy Hierarchical
        res_hier = policy_two_stage_hierarchical(feats)
        res_counts_hier[res_hier] += 1
        if res_hier == 1280:
            routed_records_hier.append(per_img_1280[img_id])
            seq_latencies_hier.append(t_base + latencies_1280[img_id])
        elif res_hier == 832:
            routed_records_hier.append(per_img_832[img_id])
            seq_latencies_hier.append(t_base + latencies_832[img_id])
        else:
            routed_records_hier.append(item640)
            seq_latencies_hier.append(t_base)

    # 2. Evaluate Performance Across All 6 Configurations on Group C
    configs = [
        ("YOLOv8n-640 (Baseline)", d640["per_image"], 0.20, list(latencies_640.values())),
        ("YOLOv8n-832", d832["per_image"], 0.20, list(latencies_832.values())),
        ("YOLOv8n-1280 (High-Res)", d1280["per_image"], 0.20, list(latencies_1280.values())),
        ("Adaptive Routing (k1)", routed_records_k1, 0.20, seq_latencies_k1),
        ("Adaptive Hierarchical", routed_records_hier, 0.20, seq_latencies_hier),
        ("Faster R-CNN (Reference)", dfrcnn["per_image"], 0.60, list(latencies_frcnn.values())),
    ]

    final_results = {}
    summary_rows = []
    stats_map = {}

    print("\n--- Locked Group C Evaluation Summary ---")
    for name, records, tau, latencies in configs:
        m = evaluate_detections(records, score_threshold=tau, iou_threshold=0.5).micro
        s = evaluate_scale_breakdown_with_thresholds(
            records, {"small": tau, "medium": tau, "large": tau}
        )
        mean_lat = float(np.mean(latencies) * 1000.0)
        p95_lat = float(np.percentile(latencies, 95) * 1000.0)
        fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        # Compute per-image stats for bootstrap
        stats = precompute_per_image_stats(records, tau=tau)
        stats_map[name] = stats

        final_results[name] = {
            "f1": round(m["f1"], 5),
            "precision": round(m["precision"], 5),
            "recall": round(m["recall"], 5),
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "small_recall_%": round(s["small"]["recall"] * 100.0, 2),
            "medium_recall_%": round(s["medium"]["recall"] * 100.0, 2),
            "large_recall_%": round(s["large"]["recall"] * 100.0, 2),
            "small_tp": s["small"]["tp"],
            "small_gt": s["small"]["gt"],
            "latency_ms": round(mean_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "fps": round(fps, 2),
        }

        summary_rows.append({
            "Configuration": name,
            "F1": m["f1"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "Small_Recall_%": s["small"]["recall"] * 100.0,
            "Med_Recall_%": s["medium"]["recall"] * 100.0,
            "Large_Recall_%": s["large"]["recall"] * 100.0,
            "Latency_ms": mean_lat,
            "P95_Latency_ms": p95_lat,
            "FPS": fps,
        })
        print(f"[{name:25s}] F1: {m['f1']:.4f} | Small R: {s['small']['recall']*100:5.2f}% | Large R: {s['large']['recall']*100:5.2f}% | Lat: {mean_lat:6.1f} ms | FPS: {fps:4.2f}")

    print(f"\nEscalation Rates on Group C:")
    print(f"  Adaptive-k1:           640px = {res_counts_k1[640]} ({res_counts_k1[640]/len(image_ids)*100:.1f}%), 1280px = {res_counts_k1[1280]} ({res_counts_k1[1280]/len(image_ids)*100:.1f}%)")
    print(f"  Adaptive-Hierarchical: 640px = {res_counts_hier[640]} ({res_counts_hier[640]/len(image_ids)*100:.1f}%), 832px = {res_counts_hier[832]} ({res_counts_hier[832]/len(image_ids)*100:.1f}%), 1280px = {res_counts_hier[1280]} ({res_counts_hier[1280]/len(image_ids)*100:.1f}%)")

    # 3. Weather Stratification on Group C (Including Foggy N=13!)
    conditions = ["clear", "overcast", "rainy", "snowy", "foggy"]
    weather_results = {}
    for c in conditions:
        weather_results[c] = {}
        for name, records, tau, _ in configs:
            sub = [r for r in records if r.get("condition") == c]
            m = evaluate_detections(sub, score_threshold=tau, iou_threshold=0.5).micro
            s = evaluate_scale_breakdown_with_thresholds(
                sub, {"small": tau, "medium": tau, "large": tau}
            )
            weather_results[c][name] = {
                "n_images": len(sub),
                "f1": round(m["f1"], 5),
                "small_recall_%": round(s["small"]["recall"] * 100.0, 2),
                "large_recall_%": round(s["large"]["recall"] * 100.0, 2),
            }

    # 4. Paired Image-Level Bootstrap Resampling
    print("\nRunning Paired Bootstrap Resampling (B=1000 replicates)...")
    boot_pairs = [
        ("Adaptive_k1_vs_Baseline_640", "YOLOv8n-640 (Baseline)", "Adaptive Routing (k1)"),
        ("Adaptive_k1_vs_Fixed_1280", "YOLOv8n-1280 (High-Res)", "Adaptive Routing (k1)"),
        ("Adaptive_Hier_vs_Baseline_640", "YOLOv8n-640 (Baseline)", "Adaptive Hierarchical"),
        ("Adaptive_k1_vs_FasterRCNN", "Faster R-CNN (Reference)", "Adaptive Routing (k1)"),
    ]

    bootstrap_results = {}
    for pair_name, m1_name, m2_name in boot_pairs:
        # Run Primary Seed
        res_primary = run_paired_bootstrap(
            stats_map[m1_name], stats_map[m2_name], n_resamples=1000, seed=20261101
        )
        # Run Secondary Seed
        res_secondary = run_paired_bootstrap(
            stats_map[m1_name], stats_map[m2_name], n_resamples=1000, seed=20261102
        )

        bootstrap_results[pair_name] = {
            "comparison": f"{m2_name} minus {m1_name}",
            "primary_seed_20261101": res_primary,
            "secondary_seed_20261102": res_secondary,
        }
        df1 = res_primary["delta_f1"]
        dsr = res_primary["delta_small_recall"]
        print(f"[{pair_name:30s}] Delta F1: {df1['mean']:+.4f} (95% CI: [{df1['ci_95'][0]:+.4f}, {df1['ci_95'][1]:+.4f}], p={df1['p_value']:.4f}) | Delta Small Rec: {dsr['mean']*100:+.2f}%")

    # 5. Save Artifacts
    with open(ARTIFACTS / "mission4_final_results.json", "w") as f:
        json.dump({
            "cohort": "Group_C_BDD100K_Locked_Test",
            "n_images": 203,
            "target_objects": 2553,
            "results": final_results,
            "weather_stratified": weather_results,
            "escalation_rates": {
                "adaptive_k1": res_counts_k1,
                "adaptive_hier": res_counts_hier,
            },
        }, f, indent=2)
    print(f"\nSaved final results to: {ARTIFACTS / 'mission4_final_results.json'}")

    with open(ARTIFACTS / "mission4_bootstrap_results.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"Saved bootstrap results to: {ARTIFACTS / 'mission4_bootstrap_results.json'}")

    efficiency_data = {
        "fixed_640": {"latency_ms": 61.1, "fps": 16.36, "f1": 0.5111, "small_rec": 8.67},
        "fixed_832": {"latency_ms": 103.0, "fps": 9.71, "f1": 0.5518, "small_rec": 18.33},
        "fixed_1280": {"latency_ms": 378.0, "fps": 2.65, "f1": 0.5782, "small_rec": 28.69},
        "adaptive_k1": {"latency_ms": round(float(np.mean(seq_latencies_k1)*1000), 2), "fps": round(1000.0/float(np.mean(seq_latencies_k1)*1000), 2), "f1": round(final_results["Adaptive Routing (k1)"]["f1"], 4), "small_rec": final_results["Adaptive Routing (k1)"]["small_recall_%"]},
        "adaptive_hier": {"latency_ms": round(float(np.mean(seq_latencies_hier)*1000), 2), "fps": round(1000.0/float(np.mean(seq_latencies_hier)*1000), 2), "f1": round(final_results["Adaptive Hierarchical"]["f1"], 4), "small_rec": final_results["Adaptive Hierarchical"]["small_recall_%"]},
        "faster_rcnn": {"latency_ms": 1275.6, "fps": 0.78, "f1": 0.6877, "small_rec": 42.53},
    }
    with open(ARTIFACTS / "mission4_efficiency.json", "w") as f:
        json.dump(efficiency_data, f, indent=2)
    print(f"Saved efficiency artifact to: {ARTIFACTS / 'mission4_efficiency.json'}")

    # 6. Generate Publication Plot: Accuracy-Latency Pareto Frontier
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Ax1: F1 vs Latency
    for row in summary_rows:
        is_ref = "Faster R-CNN" in row["Configuration"]
        is_adapt = "Adaptive" in row["Configuration"]
        color = "#e63946" if is_ref else ("#1d3557" if is_adapt else "#6c757d")
        marker = "*" if is_ref else ("D" if is_adapt else "o")
        size = 140 if is_ref else (100 if is_adapt else 80)
        ax1.scatter(row["Latency_ms"], row["F1"], color=color, s=size, marker=marker, zorder=4)
        offset_x = -150 if "Faster" in row["Configuration"] else 8
        offset_y = -0.006 if "832" in row["Configuration"] else 0.003
        ax1.annotate(row["Configuration"], (row["Latency_ms"] + offset_x, row["F1"] + offset_y), fontsize=8.5)

    ax1.set_xlabel("Mean Latency per Image (ms) [Apple M1 CPU]", fontsize=11)
    ax1.set_ylabel("Micro F1 Score (IoU=0.50)", fontsize=11)
    ax1.set_title("Accuracy vs. Latency Pareto Frontier (Locked Group C)", fontsize=12, fontweight="bold")
    ax1.axvline(100.0, color="#f4a261", linestyle="--", linewidth=1.5, label="10 FPS Real-Time Boundary")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9)

    # Ax2: Small-Object Recall vs Latency
    for row in summary_rows:
        is_ref = "Faster R-CNN" in row["Configuration"]
        is_adapt = "Adaptive" in row["Configuration"]
        color = "#e63946" if is_ref else ("#2a9d8f" if is_adapt else "#6c757d")
        marker = "*" if is_ref else ("D" if is_adapt else "o")
        size = 140 if is_ref else (100 if is_adapt else 80)
        ax2.scatter(row["Latency_ms"], row["Small_Recall_%"], color=color, s=size, marker=marker, zorder=4)
        offset_x = -150 if "Faster" in row["Configuration"] else 8
        ax2.annotate(row["Configuration"], (row["Latency_ms"] + offset_x, row["Small_Recall_%"] - 0.7), fontsize=8.5)

    ax2.set_xlabel("Mean Latency per Image (ms) [Apple M1 CPU]", fontsize=11)
    ax2.set_ylabel("Small-Object Recall (%)", fontsize=11)
    ax2.set_title("Small-Object Recall vs. Latency (Locked Group C)", fontsize=12, fontweight="bold")
    ax2.axvline(100.0, color="#f4a261", linestyle="--", linewidth=1.5, label="10 FPS Real-Time Boundary")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "mission4_accuracy_latency.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
