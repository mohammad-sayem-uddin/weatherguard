"""Generates publication-quality figures for WeatherGuard Mission 15.

Module: scripts/generate_mission15_plots.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-03
Mission: Mission 15 — Temporal Evidence Accumulation & Horizon-Anchored Candidate Generation

Generates:
1. artifacts/mission15/plots/mission15_small_f1_comparison.png
2. artifacts/mission15/plots/mission15_ultra_small_recall.png
3. artifacts/mission15/plots/mission15_candidate_recovery_stages.png
4. artifacts/mission15/plots/mission15_bootstrap_ci.png
5. artifacts/mission15/plots/mission15_pareto_latency_f1.png
6. artifacts/mission15/plots/mission15_per_sequence_distribution.png
7. artifacts/mission15/plots/mission15_temporal_failure_taxonomy.png
8. artifacts/mission15/plots/mission15_candidate_admission_breakdown.png
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts" / "mission15"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
GLOBAL_PLOTS_DIR = ROOT / "artifacts" / "plots"

# Professional publication styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_all():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load locked evaluation results
    locked_file = ARTIFACTS_DIR / "locked_evaluation.json"
    boot_file = ARTIFACTS_DIR / "sequence_bootstrap.json"
    ablation_file = ARTIFACTS_DIR / "ablation_results.json"
    fail_file = ARTIFACTS_DIR / "temporal_failure_taxonomy.json"

    # --- Plot 1: Small-Object F1 Comparison ---
    if locked_file.exists():
        locked_data = json.load(open(locked_file))
        agg = locked_data["aggregate"]

        cfgs = ["YOLO640", "YOLO832", "Policy J", "Policy K"]
        keys = ["A_YOLO640", "B_YOLO832", "C_Policy_J", "D_Policy_K"]
        f1_vals = [agg[k]["small_f1"] for k in keys]
        colors = ["#7f8c8d", "#2980b9", "#8e44ad", "#27ae60"]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        bars = ax.bar(cfgs, f1_vals, color=colors, width=0.55, edgecolor="#2c3e50", linewidth=1.2)
        ax.set_ylabel("Small-Object F1 Score (16-32px)", fontsize=12, fontweight="bold")
        ax.set_title("Mission 15 Locked Evaluation: Small-Object F1 Across 6 Independent Sequences", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylim(0, max(f1_vals) * 1.22)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "mission15_small_f1_comparison.png")
        plt.savefig(GLOBAL_PLOTS_DIR / "mission15_small_f1_comparison.png")
        plt.close()
        print("Generated mission15_small_f1_comparison.png")

        # --- Plot 2: Ultra-Small Recall Comparison ---
        ultra_vals = [agg[k]["ultra_small_recall"] * 100.0 for k in keys]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        bars = ax.bar(cfgs, ultra_vals, color=colors, width=0.55, edgecolor="#2c3e50", linewidth=1.2)
        ax.set_ylabel("Ultra-Small Recall (<16px) [%]", fontsize=12, fontweight="bold")
        ax.set_title("Mission 15 Locked Evaluation: Ultra-Small Vehicle Recall (<16px)", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylim(0, max(ultra_vals) * 1.25)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.15, f"{yval:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "mission15_ultra_small_recall.png")
        plt.savefig(GLOBAL_PLOTS_DIR / "mission15_ultra_small_recall.png")
        plt.close()
        print("Generated mission15_ultra_small_recall.png")

        # --- Plot 6: Per-Sequence Distribution Breakdown ---
        seq_dict = locked_data["per_sequence"]
        seq_names = list(seq_dict.keys())
        short_names = [f"Seq {i+1}\n({s[:8]})" for i, s in enumerate(seq_names)]

        x = np.arange(len(seq_names))
        width = 0.20

        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.bar(x - 1.5 * width, [seq_dict[s]["A_YOLO640"]["small_f1"] for s in seq_names], width, label="YOLO640", color="#7f8c8d")
        ax.bar(x - 0.5 * width, [seq_dict[s]["B_YOLO832"]["small_f1"] for s in seq_names], width, label="YOLO832", color="#2980b9")
        ax.bar(x + 0.5 * width, [seq_dict[s]["C_Policy_J"]["small_f1"] for s in seq_names], width, label="Policy J", color="#8e44ad")
        ax.bar(x + 1.5 * width, [seq_dict[s]["D_Policy_K"]["small_f1"] for s in seq_names], width, label="Policy K", color="#27ae60")

        ax.set_ylabel("Small-Object F1 Score", fontsize=12, fontweight="bold")
        ax.set_title("Per-Sequence Small-Object F1 Breakdown Across 6 Independent Sequences", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=10)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "mission15_per_sequence_distribution.png")
        plt.savefig(GLOBAL_PLOTS_DIR / "mission15_per_sequence_distribution.png")
        plt.close()
        print("Generated mission15_per_sequence_distribution.png")

    # --- Plot 4: Paired Bootstrap Confidence Intervals ---
    if boot_file.exists():
        boot_data = json.load(open(boot_file))
        tests = boot_data["tests"]

        test_names = ["Policy K vs YOLO640", "Policy K vs Policy J", "Policy K vs YOLO832"]
        keys_test = ["Policy_K_vs_YOLO640", "Policy_K_vs_Policy_J", "Policy_K_vs_YOLO832"]

        deltas = [tests[k]["small_f1"]["point_estimate_delta"] for k in keys_test]
        ci_lows = [tests[k]["small_f1"]["ci_95"][0] for k in keys_test]
        ci_highs = [tests[k]["small_f1"]["ci_95"][1] for k in keys_test]

        y_pos = np.arange(len(test_names))
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

        err_low = [d - cl for d, cl in zip(deltas, ci_lows)]
        err_high = [ch - d for d, ch in zip(deltas, ci_highs)]

        ax.errorbar(deltas, y_pos, xerr=[err_low, err_high], fmt="o", color="#27ae60", ecolor="#2c3e50", elinewidth=2, capsize=6, markersize=8)
        ax.axvline(0.0, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(test_names, fontsize=11, fontweight="bold")
        ax.set_xlabel("Delta Small-Object F1 (95% Bootstrap CI)", fontsize=12, fontweight="bold")
        ax.set_title("Paired Sequence-Level Bootstrap: Policy K Acuity Differences (B=1,000)", fontsize=13, fontweight="bold", pad=15)

        for i, (d, cl, ch) in enumerate(zip(deltas, ci_lows, ci_highs)):
            ax.text(d, i + 0.18, f"Δ = {d:+.4f} [{cl:+.4f}, {ch:+.4f}]", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylim(-0.5, len(test_names) - 0.5)
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "mission15_bootstrap_ci.png")
        plt.savefig(GLOBAL_PLOTS_DIR / "mission15_bootstrap_ci.png")
        plt.close()
        print("Generated mission15_bootstrap_ci.png")

    # --- Plot 5: Latency vs Accuracy Pareto Frontier ---
    if ablation_file.exists():
        abl_data = json.load(open(ablation_file))
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        for cfg_name, res in abl_data.items():
            ms = res["mean_latency_ms"]
            sf1 = res["small_f1"]
            color = "#27ae60" if "Policy_K" in cfg_name else ("#8e44ad" if "Policy_J" in cfg_name else "#2980b9")
            ax.scatter(ms, sf1, s=110, color=color, edgecolor="#2c3e50", zorder=3)
            # Label
            label_text = cfg_name.split("_", 1)[-1].replace("_", " ")
            ax.text(ms + 1.2, sf1 + 0.003, label_text, fontsize=9, fontweight="bold", zorder=4)

        ax.set_xlabel("Mean Latency per Frame on Apple M1 CPU (ms)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Small-Object F1 Score", fontsize=12, fontweight="bold")
        ax.set_title("Edge Efficiency vs Acuity: Factorial Ablation on Apple M1 CPU", fontsize=13, fontweight="bold", pad=15)
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "mission15_pareto_latency_f1.png")
        plt.savefig(GLOBAL_PLOTS_DIR / "mission15_pareto_latency_f1.png")
        plt.close()
        print("Generated mission15_pareto_latency_f1.png")

    # --- Plot 7: Temporal Failure Taxonomy ---
    if fail_file.exists():
        fail_data = json.load(open(fail_file))
        tax = fail_data["taxonomy"]
        cat_labels = [v["name"] for v in tax.values() if v["count"] > 0]
        cat_counts = [v["count"] for v in tax.values() if v["count"] > 0]

        # Sort descending
        sorted_pairs = sorted(zip(cat_labels, cat_counts), key=lambda p: p[1], reverse=True)
        cat_labels = [p[0] for p in sorted_pairs]
        cat_counts = [p[1] for p in sorted_pairs]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        y_pos = np.arange(len(cat_labels))
        bars = ax.barh(y_pos, cat_counts, color="#e67e22", edgecolor="#2c3e50", height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cat_labels, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Number of Missed Ground-Truth Targets", fontsize=12, fontweight="bold")
        ax.set_title("Mission 15 Temporal Failure Taxonomy Distribution", fontsize=13, fontweight="bold", pad=15)

        for bar in bars:
            wval = bar.get_width()
            pct = (wval / sum(cat_counts)) * 100.0
            ax.text(wval + 20, bar.get_y() + bar.get_height() / 2.0, f"{wval} ({pct:.1f}%)", ha="left", va="center", fontsize=9, fontweight="bold")

        ax.set_xlim(0, max(cat_counts) * 1.25)
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "mission15_temporal_failure_taxonomy.png")
        plt.savefig(GLOBAL_PLOTS_DIR / "mission15_temporal_failure_taxonomy.png")
        plt.close()
        print("Generated mission15_temporal_failure_taxonomy.png")

    # --- Plot 8: Overall F1 Comparison ---
    if locked_file.exists():
        locked_data = json.load(open(locked_file))
        agg = locked_data["aggregate"]
        cfgs = ["YOLO640", "YOLO832", "Policy J", "Policy K"]
        keys = ["A_YOLO640", "B_YOLO832", "C_Policy_J", "D_Policy_K"]
        f1_vals = [agg[k]["overall_f1"] for k in keys]
        colors = ["#7f8c8d", "#2980b9", "#8e44ad", "#27ae60"]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        bars = ax.bar(cfgs, f1_vals, color=colors, width=0.55, edgecolor="#2c3e50", linewidth=1.2)
        ax.set_ylabel("Overall F1 Score (All Scales)", fontsize=12, fontweight="bold")
        ax.set_title("Mission 15 Locked Evaluation: Overall Detection F1 (All Scales)", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylim(0, max(f1_vals) * 1.22)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "mission15_overall_f1.png")
        plt.savefig(GLOBAL_PLOTS_DIR / "mission15_overall_f1.png")
        plt.close()
        print("Generated mission15_overall_f1.png")

    # --- Plot 9: Candidate Tier Admission Breakdown ---
    log_file = ARTIFACTS_DIR / "candidate_admission_log.jsonl"
    if log_file.exists():
        tier_counts = {"Tier 1 (Detector)": 0, "Tier 2 (Dual Match)": 0, "Tier 3 (Temporal)": 0, "Tier 4 (Horizon)": 0}
        with open(log_file) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                src = d.get("source", "")
                if src == "detector":
                    tier_counts["Tier 1 (Detector)"] += 1
                elif src == "temporal_horizon_match":
                    tier_counts["Tier 2 (Dual Match)"] += 1
                elif src == "temporal":
                    tier_counts["Tier 3 (Temporal)"] += 1
                elif src == "horizon":
                    tier_counts["Tier 4 (Horizon)"] += 1

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        labels = list(tier_counts.keys())
        counts = [tier_counts[k] for k in labels]
        colors_tier = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
        bars = ax.bar(labels, counts, color=colors_tier, width=0.55, edgecolor="#2c3e50", linewidth=1.2)
        ax.set_ylabel("Proposals Evaluated", fontsize=12, fontweight="bold")
        ax.set_title("Policy K Multi-Tier Candidate Admission Distribution", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylim(0, max(counts) * 1.25 if counts and max(counts) > 0 else 100)

        for bar in bars:
            yval = bar.get_height()
            pct = (yval / sum(counts) * 100.0) if sum(counts) > 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 10, f"{yval}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "mission15_candidate_admission_breakdown.png")
        plt.savefig(GLOBAL_PLOTS_DIR / "mission15_candidate_admission_breakdown.png")
        plt.close()
        print("Generated mission15_candidate_admission_breakdown.png")



if __name__ == "__main__":
    plot_all()
