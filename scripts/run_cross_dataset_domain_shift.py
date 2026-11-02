"""Analyzes cross-dataset domain shift between DAWN held-out and BDD100K external.

Computes low-level image statistics (brightness, contrast, color saturation),
geometric object distributions (sizes, aspect ratios, scale bins),
and confidence score shifts. Generates domain shift artifacts and summary figure.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import DawnDataset
from src.external_dataset import CANONICAL_CLASSES, Bdd100kDataset

ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"
DATA_DIR = ROOT / "data" / "bdd100k"
IMAGES_DIR = DATA_DIR / "images"
SAMPLES_JSON = DATA_DIR / "samples.json"
VAL_PARQUET = ROOT / "val-00000-of-00001.parquet"
CLEAN_HELDOUT_JSON = ARTIFACTS / "clean_heldout_image_ids.json"


def compute_image_low_level_stats(img: Image.Image) -> dict[str, float]:
    stat = ImageStat.Stat(img)
    # RGB means
    r_mean, g_mean, b_mean = stat.mean[:3]
    # Brightness (perceived luminance)
    brightness = 0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean
    # RMS contrast = std dev of grayscale intensities
    gray = img.convert("L")
    gray_stat = ImageStat.Stat(gray)
    rms_contrast = gray_stat.stddev[0]

    # Saturation (HSV)
    hsv = img.convert("HSV")
    hsv_stat = ImageStat.Stat(hsv)
    saturation = hsv_stat.mean[1]  # S channel mean in [0, 255]

    return {
        "brightness": brightness,
        "rms_contrast": rms_contrast,
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "saturation": saturation,
    }


def analyze_dataset_stats(records: list[any], name: str) -> dict[str, any]:
    brightnesses = []
    contrasts = []
    saturations = []
    widths = []
    heights = []
    box_areas_abs = []
    box_areas_rel = []
    aspect_ratios = []
    class_counts = Counter()
    condition_counts = Counter()
    size_bins = {"small": 0, "medium": 0, "large": 0}

    for rec in records:
        widths.append(rec.width)
        heights.append(rec.height)
        condition_counts[rec.condition] += 1
        img = rec.to_pil()
        stats = compute_image_low_level_stats(img)
        brightnesses.append(stats["brightness"])
        contrasts.append(stats["rms_contrast"])
        saturations.append(stats["saturation"])

        for b in rec.objects:
            class_counts[b.class_name] += 1
            area_abs = b.width * b.height
            box_areas_abs.append(area_abs)
            area_rel = area_abs / (rec.width * rec.height)
            box_areas_rel.append(area_rel)
            ar = b.width / max(1.0, b.height)
            aspect_ratios.append(ar)

            # COCO scale bins: small < 32^2 (1024), medium [1024, 9216), large >= 9216
            if area_abs < 1024:
                size_bins["small"] += 1
            elif area_abs < 9216:
                size_bins["medium"] += 1
            else:
                size_bins["large"] += 1

    total_boxes = len(box_areas_abs)
    return {
        "dataset_name": name,
        "n_images": len(records),
        "total_boxes": total_boxes,
        "boxes_per_image": {
            "mean": float(np.mean([len(r.objects) for r in records])),
            "median": float(np.median([len(r.objects) for r in records])),
        },
        "dimensions": {
            "mean_width": float(np.mean(widths)),
            "mean_height": float(np.mean(heights)),
        },
        "low_level_image_stats": {
            "brightness_mean": float(np.mean(brightnesses)),
            "brightness_std": float(np.std(brightnesses)),
            "rms_contrast_mean": float(np.mean(contrasts)),
            "rms_contrast_std": float(np.std(contrasts)),
            "saturation_mean": float(np.mean(saturations)),
            "saturation_std": float(np.std(saturations)),
        },
        "object_size_stats": {
            "abs_area_mean": float(np.mean(box_areas_abs)) if box_areas_abs else 0.0,
            "abs_area_median": float(np.median(box_areas_abs)) if box_areas_abs else 0.0,
            "rel_area_mean": float(np.mean(box_areas_rel)) if box_areas_rel else 0.0,
            "rel_area_median": float(np.median(box_areas_rel)) if box_areas_rel else 0.0,
            "size_bins_count": size_bins,
            "size_bins_pct": {
                k: float(v / total_boxes * 100) if total_boxes else 0.0
                for k, v in size_bins.items()
            },
        },
        "aspect_ratio_stats": {
            "mean": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
            "median": float(np.median(aspect_ratios)) if aspect_ratios else 0.0,
        },
        "class_distribution": dict(class_counts),
        "weather_distribution": dict(condition_counts),
        "_raw_distributions": {
            "brightness": [round(x, 2) for x in brightnesses],
            "contrast": [round(x, 2) for x in contrasts],
            "saturation": [round(x, 2) for x in saturations],
            "box_areas_rel": [round(x, 5) for x in box_areas_rel],
            "aspect_ratios": [round(x, 3) for x in aspect_ratios],
        },
    }


def main() -> None:
    print("=" * 70)
    print("WEATHERGUARD MISSION 2 — CROSS-DATASET DOMAIN SHIFT ANALYSIS")
    print("=" * 70)

    # 1. Load DAWN clean held-out records
    print("Loading DAWN held-out dataset...")
    dawn_full = DawnDataset.from_parquet(VAL_PARQUET)
    clean_info = json.loads(CLEAN_HELDOUT_JSON.read_text())
    heldout_ids = set(clean_info["heldout_image_ids"])
    dawn_heldout = [r for r in dawn_full if r.image_id in heldout_ids]
    print(f"DAWN clean heldout: {len(dawn_heldout)} images.")

    # 2. Load BDD100K external records
    print("Loading BDD100K external cohort...")
    bdd_dataset = Bdd100kDataset.from_fiftyone_json(SAMPLES_JSON, images_dir=IMAGES_DIR)
    bdd_cohort = bdd_dataset.select_weather_stratified_cohort(seed=42)
    print(f"BDD100K external cohort: {len(bdd_cohort)} images.")

    # 3. Analyze both
    print("Computing domain statistics for DAWN held-out...")
    dawn_stats = analyze_dataset_stats(dawn_heldout, "DAWN Held-out (202)")
    print("Computing domain statistics for BDD100K external...")
    bdd_stats = analyze_dataset_stats(bdd_cohort, "BDD100K External (203)")

    # 4. Save JSON artifact
    payload = {
        "analysis_name": "Cross-Dataset Domain Shift Analysis",
        "datasets": {
            "dawn_heldout": dawn_stats,
            "bdd100k_external": bdd_stats,
        },
        "comparisons": {
            "brightness_difference": bdd_stats["low_level_image_stats"]["brightness_mean"] - dawn_stats["low_level_image_stats"]["brightness_mean"],
            "contrast_difference": bdd_stats["low_level_image_stats"]["rms_contrast_mean"] - dawn_stats["low_level_image_stats"]["rms_contrast_mean"],
            "saturation_difference": bdd_stats["low_level_image_stats"]["saturation_mean"] - dawn_stats["low_level_image_stats"]["saturation_mean"],
            "density_difference": bdd_stats["boxes_per_image"]["mean"] - dawn_stats["boxes_per_image"]["mean"],
            "small_object_pct_difference": bdd_stats["object_size_stats"]["size_bins_pct"]["small"] - dawn_stats["object_size_stats"]["size_bins_pct"]["small"],
            "medium_object_pct_difference": bdd_stats["object_size_stats"]["size_bins_pct"]["medium"] - dawn_stats["object_size_stats"]["size_bins_pct"]["medium"],
            "large_object_pct_difference": bdd_stats["object_size_stats"]["size_bins_pct"]["large"] - dawn_stats["object_size_stats"]["size_bins_pct"]["large"],
        },
    }

    out_json = ARTIFACTS / "cross_dataset_domain_shift.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved domain shift JSON to {out_json}")

    # 5. Generate Publication-Quality Domain Shift Figure
    plot_path = PLOTS / "cross_dataset_domain_shift.png"
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel A: Image Brightness & Contrast
    labels = ["Brightness", "RMS Contrast", "Saturation"]
    dawn_means = [
        dawn_stats["low_level_image_stats"]["brightness_mean"],
        dawn_stats["low_level_image_stats"]["rms_contrast_mean"],
        dawn_stats["low_level_image_stats"]["saturation_mean"],
    ]
    bdd_means = [
        bdd_stats["low_level_image_stats"]["brightness_mean"],
        bdd_stats["low_level_image_stats"]["rms_contrast_mean"],
        bdd_stats["low_level_image_stats"]["saturation_mean"],
    ]
    x = np.arange(len(labels))
    width = 0.35
    axes[0, 0].bar(x - width/2, dawn_means, width, label="DAWN Held-out", color="#2b5c8f", edgecolor="black")
    axes[0, 0].bar(x + width/2, bdd_means, width, label="BDD100K External", color="#e27c38", edgecolor="black")
    axes[0, 0].set_ylabel("Pixel Metric Intensity (0-255)")
    axes[0, 0].set_title("A. Image Photometric Distribution", fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.6)

    # Panel B: Object Scale Distribution (Small / Med / Large)
    scale_labels = ["Small (<32²)", "Medium [32², 96²)", "Large (≥96²)"]
    dawn_scale_pct = [dawn_stats["object_size_stats"]["size_bins_pct"][k] for k in ["small", "medium", "large"]]
    bdd_scale_pct = [bdd_stats["object_size_stats"]["size_bins_pct"][k] for k in ["small", "medium", "large"]]
    x_sc = np.arange(len(scale_labels))
    axes[0, 1].bar(x_sc - width/2, dawn_scale_pct, width, label="DAWN Held-out", color="#2b5c8f", edgecolor="black")
    axes[0, 1].bar(x_sc + width/2, bdd_scale_pct, width, label="BDD100K External", color="#e27c38", edgecolor="black")
    axes[0, 1].set_ylabel("Percentage of Total Boxes (%)")
    axes[0, 1].set_title("B. Object Scale Distribution (COCO Bins)", fontweight="bold")
    axes[0, 1].set_xticks(x_sc)
    axes[0, 1].set_xticklabels(scale_labels)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.6)

    # Panel C: Object Aspect Ratio Distribution
    d_ar = dawn_stats["_raw_distributions"]["aspect_ratios"]
    b_ar = bdd_stats["_raw_distributions"]["aspect_ratios"]
    bins_ar = np.linspace(0.1, 3.0, 30)
    axes[1, 0].hist(d_ar, bins=bins_ar, alpha=0.55, label="DAWN Held-out", color="#2b5c8f", density=True, edgecolor="none")
    axes[1, 0].hist(b_ar, bins=bins_ar, alpha=0.55, label="BDD100K External", color="#e27c38", density=True, edgecolor="none")
    axes[1, 0].set_xlabel("Aspect Ratio (Width / Height)")
    axes[1, 0].set_ylabel("Probability Density")
    axes[1, 0].set_title("C. Bounding Box Aspect Ratio Distribution", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.6)

    # Panel D: Relative Class Proportions
    cls_labels = CANONICAL_CLASSES
    d_total = sum(dawn_stats["class_distribution"].values())
    b_total = sum(bdd_stats["class_distribution"].values())
    d_cls_pct = [(dawn_stats["class_distribution"].get(c, 0) / d_total * 100) for c in cls_labels]
    b_cls_pct = [(bdd_stats["class_distribution"].get(c, 0) / b_total * 100) for c in cls_labels]
    x_cls = np.arange(len(cls_labels))
    axes[1, 1].bar(x_cls - width/2, d_cls_pct, width, label="DAWN Held-out", color="#2b5c8f", edgecolor="black")
    axes[1, 1].bar(x_cls + width/2, b_cls_pct, width, label="BDD100K External", color="#e27c38", edgecolor="black")
    axes[1, 1].set_ylabel("Class Share (%)")
    axes[1, 1].set_title("D. Class Representation in Target Space", fontweight="bold")
    axes[1, 1].set_xticks(x_cls)
    axes[1, 1].set_xticklabels(cls_labels, rotation=25)
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved domain shift figure to {plot_path}")


if __name__ == "__main__":
    main()
