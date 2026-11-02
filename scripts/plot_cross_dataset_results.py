"""Generates publication-quality figures for the cross-dataset evaluation.

Outputs:
1. artifacts/plots/cross_dataset_performance.png
2. artifacts/plots/cross_dataset_threshold_transfer.png
3. artifacts/plots/cross_dataset_weather_breakdown.png
4. artifacts/plots/cross_dataset_class_breakdown.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"


def plot_all() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    results_path = ARTIFACTS / "cross_dataset_results.json"
    if not results_path.exists():
        print(f"Cannot plot: {results_path} not found.")
        return

    with open(results_path) as f:
        res = json.load(f)

    frcnn = res["fasterrcnn_resnet50_fpn"]
    yolo = res["yolov8n"]

    # -------------------------------------------------------------
    # 1. Performance Overview (DAWN Held-out vs BDD100K External)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    metrics = ["mAP@0.5", "Micro F1 (DAWN Thresh)"]
    x = np.arange(len(metrics))
    width = 0.35

    # Panel A: Faster R-CNN
    frcnn_dawn = [frcnn["dawn_heldout_baseline"]["mAP_50"], frcnn["dawn_heldout_baseline"]["f1"]]
    frcnn_ext = [frcnn["external_at_dawn_threshold"]["mAP_50"], frcnn["external_at_dawn_threshold"]["f1"]]

    axes[0].bar(x - width/2, frcnn_dawn, width, label="DAWN Held-out (n=202)", color="#2b5c8f", edgecolor="black")
    axes[0].bar(x + width/2, frcnn_ext, width, label="BDD100K External (n=203)", color="#e27c38", edgecolor="black")
    axes[0].set_title("A. Faster R-CNN ResNet-50 FPN Performance", fontweight="bold")
    axes[0].set_ylabel("Metric Score")
    axes[0].set_ylim(0, 0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    for i in range(len(metrics)):
        axes[0].text(x[i] - width/2, frcnn_dawn[i] + 0.015, f"{frcnn_dawn[i]:.3f}", ha="center", fontsize=10)
        axes[0].text(x[i] + width/2, frcnn_ext[i] + 0.015, f"{frcnn_ext[i]:.3f}", ha="center", fontsize=10)

    # Panel B: YOLOv8n
    yolo_dawn = [yolo["dawn_heldout_baseline"]["mAP_50"], yolo["dawn_heldout_baseline"]["f1"]]
    yolo_ext = [yolo["external_at_dawn_threshold"]["mAP_50"], yolo["external_at_dawn_threshold"]["f1"]]

    axes[1].bar(x - width/2, yolo_dawn, width, label="DAWN Held-out (n=202)", color="#2b5c8f", edgecolor="black")
    axes[1].bar(x + width/2, yolo_ext, width, label="BDD100K External (n=203)", color="#e27c38", edgecolor="black")
    axes[1].set_title("B. YOLOv8n Performance", fontweight="bold")
    axes[1].set_ylabel("Metric Score")
    axes[1].set_ylim(0, 0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    for i in range(len(metrics)):
        axes[1].text(x[i] - width/2, yolo_dawn[i] + 0.015, f"{yolo_dawn[i]:.3f}", ha="center", fontsize=10)
        axes[1].text(x[i] + width/2, yolo_ext[i] + 0.015, f"{yolo_ext[i]:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    p1 = PLOTS / "cross_dataset_performance.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved {p1}")

    # -------------------------------------------------------------
    # 2. Threshold Transfer Curves
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Faster R-CNN Sweep
    f_th = [item["threshold"] for item in frcnn["threshold_sweep"]]
    f_f1 = [item["f1"] for item in frcnn["threshold_sweep"]]
    axes[0].plot(f_th, f_f1, marker="o", color="#2b5c8f", linewidth=2, label="External F1 Curve")
    # Mark DAWN threshold
    d_th = frcnn["dawn_selected_threshold"]
    d_f1 = frcnn["external_at_dawn_threshold"]["f1"]
    axes[0].scatter([d_th], [d_f1], color="red", s=120, zorder=5, label=f"DAWN-Selected (τ={d_th:.2f}, F1={d_f1:.3f})")
    # Mark Default threshold
    def_th = 0.50
    def_f1 = frcnn["external_at_default_threshold"]["f1"]
    axes[0].scatter([def_th], [def_f1], color="green", s=100, marker="s", zorder=5, label=f"Default (τ=0.50, F1={def_f1:.3f})")
    # Mark Oracle threshold
    o_th = frcnn["oracle_threshold"]
    o_f1 = frcnn["external_at_oracle_threshold"]["f1"]
    axes[0].scatter([o_th], [o_f1], color="purple", s=110, marker="^", zorder=5, label=f"Oracle (τ={o_th:.2f}, F1={o_f1:.3f})")
    axes[0].set_title("A. Faster R-CNN ResNet-50 FPN Threshold Transfer", fontweight="bold")
    axes[0].set_xlabel("Confidence Score Threshold")
    axes[0].set_ylabel("Micro F1 Score")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Panel B: YOLOv8n Sweep
    y_th = [item["threshold"] for item in yolo["threshold_sweep"]]
    y_f1 = [item["f1"] for item in yolo["threshold_sweep"]]
    axes[1].plot(y_th, y_f1, marker="o", color="#e27c38", linewidth=2, label="External F1 Curve")
    # Mark DAWN threshold
    yd_th = yolo["dawn_selected_threshold"]
    yd_f1 = yolo["external_at_dawn_threshold"]["f1"]
    axes[1].scatter([yd_th], [yd_f1], color="red", s=120, zorder=5, label=f"DAWN-Selected (τ={yd_th:.2f}, F1={yd_f1:.3f})")
    # Mark Default threshold
    ydef_th = 0.50
    ydef_f1 = yolo["external_at_default_threshold"]["f1"]
    axes[1].scatter([ydef_th], [ydef_f1], color="green", s=100, marker="s", zorder=5, label=f"Default (τ=0.50, F1={ydef_f1:.3f})")
    # Mark Oracle threshold
    yo_th = yolo["oracle_threshold"]
    yo_f1 = yolo["external_at_oracle_threshold"]["f1"]
    axes[1].scatter([yo_th], [yo_f1], color="purple", s=110, marker="^", zorder=5, label=f"Oracle (τ={yo_th:.2f}, F1={yo_f1:.3f})")
    axes[1].set_title("B. YOLOv8n Threshold Transfer", fontweight="bold")
    axes[1].set_xlabel("Confidence Score Threshold")
    axes[1].set_ylabel("Micro F1 Score")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    p2 = PLOTS / "cross_dataset_threshold_transfer.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved {p2}")

    # -------------------------------------------------------------
    # 3. Weather Breakdown Figure
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    conds = ["rainy", "snowy", "foggy", "overcast", "clear"]
    cond_names = ["Rainy", "Snowy", "Foggy (n=13)", "Overcast", "Clear (Ctrl)"]

    f_cond_f1 = [frcnn["condition_breakdown"].get(c, {}).get("f1", 0.0) for c in conds]
    y_cond_f1 = [yolo["condition_breakdown"].get(c, {}).get("f1", 0.0) for c in conds]

    x_c = np.arange(len(conds))
    width = 0.35
    ax.bar(x_c - width/2, f_cond_f1, width, label="Faster R-CNN (τ=0.60)", color="#2b5c8f", edgecolor="black")
    ax.bar(x_c + width/2, y_cond_f1, width, label="YOLOv8n (τ=0.20)", color="#e27c38", edgecolor="black")
    ax.set_title("External Performance by Weather Condition (Micro F1)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Micro F1 Score")
    ax.set_xticks(x_c)
    ax.set_xticklabels(cond_names)
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend()
    for i in range(len(conds)):
        ax.text(x_c[i] - width/2, f_cond_f1[i] + 0.015, f"{f_cond_f1[i]:.3f}", ha="center", fontsize=9)
        ax.text(x_c[i] + width/2, y_cond_f1[i] + 0.015, f"{y_cond_f1[i]:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    p3 = PLOTS / "cross_dataset_weather_breakdown.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved {p3}")

    # -------------------------------------------------------------
    # 4. Class Breakdown Figure
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    classes = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]
    f_cls_ap = [frcnn["class_breakdown"].get(c, {}).get("ap", 0.0) for c in classes]
    y_cls_ap = [yolo["class_breakdown"].get(c, {}).get("ap", 0.0) for c in classes]

    x_cl = np.arange(len(classes))
    ax.bar(x_cl - width/2, f_cls_ap, width, label="Faster R-CNN AP@0.5", color="#2b5c8f", edgecolor="black")
    ax.bar(x_cl + width/2, y_cls_ap, width, label="YOLOv8n AP@0.5", color="#e27c38", edgecolor="black")
    ax.set_title("External Detection Performance by Target Class (AP@0.5)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Average Precision (AP@0.5)")
    ax.set_xticks(x_cl)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend()
    for i in range(len(classes)):
        ax.text(x_cl[i] - width/2, f_cls_ap[i] + 0.015, f"{f_cls_ap[i]:.2f}", ha="center", fontsize=9)
        ax.text(x_cl[i] + width/2, y_cls_ap[i] + 0.015, f"{y_cls_ap[i]:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    p4 = PLOTS / "cross_dataset_class_breakdown.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved {p4}")


if __name__ == "__main__":
    plot_all()
