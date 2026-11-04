"""WeatherGuard-D1 Dataset Statistical Analysis and Characterization.

Module: scripts/analyze_weatherguard_d1.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
Mission: Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

Calculates comprehensive dataset statistics (class balance, condition distributions,
object-size scales, confidence percentiles, empty image ratios).
When dataset is not yet ingested, emits explicit NOT_AVAILABLE status without fabricating data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "weatherguard_d1.yaml"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports" / "mission16_preparation"


def compute_gini_coefficient(counts: List[int]) -> float:
    """Computes Gini coefficient of inequality for class frequencies (0.0=equal, 1.0=skewed)."""
    if not counts or sum(counts) == 0:
        return 0.0
    arr = np.sort(np.array(counts, dtype=np.float64))
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))


def compute_shannon_entropy(counts: List[int]) -> float:
    """Computes Shannon entropy (in nats) across categorical distributions."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return float(-sum(p * math.log(p) for p in probs))


class D1DatasetAnalyzer:
    """Analyzes ingested WeatherGuard-D1 images and validated annotations."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.target_counts = {k: v.get("target_count", 0) for k, v in self.config.get("conditions", {}).items()}
        self.class_names = [self.config["classes"][i] for i in sorted(self.config.get("classes", {}).keys())]

    def analyze_dataset(
        self,
        ingestion_manifest_path: Path,
        validation_manifest_path: Path,
    ) -> Dict[str, Any]:
        """Performs exhaustive analysis on ingestion and annotation manifests."""
        # Check availability
        if not ingestion_manifest_path.exists() or not validation_manifest_path.exists():
            missing = []
            if not ingestion_manifest_path.exists():
                missing.append(str(ingestion_manifest_path.name))
            if not validation_manifest_path.exists():
                missing.append(str(validation_manifest_path.name))

            payload = {
                "dataset_name": self.config.get("dataset_name", "WeatherGuard-D1"),
                "status": "NOT_AVAILABLE",
                "message": f"Dataset manifests not found: {', '.join(missing)}. Awaiting raw synthetic data from Google Flow.",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "target_specification": {
                    "target_images": self.config.get("target_total_images", 2000),
                    "target_conditions": self.target_counts,
                    "target_classes": self.class_names,
                    "target_resolution": f"{self.config.get('image_properties', {}).get('target_width')}x{self.config.get('image_properties', {}).get('target_height')}",
                }
            }
            return payload

        # Load manifests
        with open(ingestion_manifest_path, "r") as f:
            ingestion_data = json.load(f)
        with open(validation_manifest_path, "r") as f:
            validation_data = json.load(f)

        img_records = ingestion_data.get("records", [])
        annot_results = validation_data.get("results", [])

        total_images = len(img_records)
        accepted_images = sum(1 for r in img_records if r.get("status") in ("ACCEPTED", "NEAR_DUPLICATE_FLAGGED"))

        # Condition breakdown
        cond_counts = {c: 0 for c in self.target_counts}
        for r in img_records:
            if r.get("status") in ("ACCEPTED", "NEAR_DUPLICATE_FLAGGED"):
                c = r.get("condition")
                if c in cond_counts:
                    cond_counts[c] += 1

        # Annotation aggregations
        all_boxes: List[Dict[str, Any]] = []
        empty_images_count = 0
        cond_class_matrix = {c: {cls: 0 for cls in self.class_names} for c in self.target_counts}
        class_counts = {cls: 0 for cls in self.class_names}
        size_counts = {"ultra_small": 0, "small": 0, "medium": 0, "large": 0}
        confidences: List[float] = []

        for a_res in annot_results:
            v_boxes = a_res.get("valid_boxes", [])
            cond = a_res.get("condition", "day")
            if a_res.get("is_empty") or len(v_boxes) == 0:
                empty_images_count += 1

            for b in v_boxes:
                all_boxes.append(b)
                cls_name = b.get("class_name")
                sz = b.get("recalculated_size")
                conf = float(b.get("confidence", 1.0))

                if cls_name in class_counts:
                    class_counts[cls_name] += 1
                if cond in cond_class_matrix and cls_name in cond_class_matrix[cond]:
                    cond_class_matrix[cond][cls_name] += 1
                if sz in size_counts:
                    size_counts[sz] += 1
                confidences.append(conf)

        total_objects = len(all_boxes)
        class_vals = [class_counts[c] for c in self.class_names]
        gini = compute_gini_coefficient(class_vals)
        entropy = compute_shannon_entropy(class_vals)

        conf_arr = np.array(confidences) if confidences else np.array([0.0])
        conf_stats = {
            "mean": round(float(np.mean(conf_arr)), 4),
            "std": round(float(np.std(conf_arr)), 4),
            "median": round(float(np.median(conf_arr)), 4),
            "p10": round(float(np.percentile(conf_arr, 10)), 4),
            "p90": round(float(np.percentile(conf_arr, 90)), 4),
        }

        analysis = {
            "dataset_name": self.config.get("dataset_name", "WeatherGuard-D1"),
            "dataset_version": self.config.get("dataset_version", "1.0.0-synthetic"),
            "status": "AVAILABLE",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image_summary": {
                "total_images_processed": total_images,
                "accepted_images": accepted_images,
                "empty_images_count": empty_images_count,
                "empty_image_ratio": round(empty_images_count / max(accepted_images, 1), 4),
            },
            "condition_distribution": {
                "counts": cond_counts,
                "targets": self.target_counts,
                "completion_pct": {
                    c: round((cond_counts[c] / self.target_counts[c]) * 100.0, 2)
                    if self.target_counts.get(c, 0) > 0 else 0.0
                    for c in self.target_counts
                }
            },
            "object_summary": {
                "total_valid_objects": total_objects,
                "average_objects_per_image": round(total_objects / max(accepted_images, 1), 2),
                "class_counts": class_counts,
                "class_balance_gini": round(gini, 4),
                "class_entropy_nats": round(entropy, 4),
                "object_size_distribution": size_counts,
                "size_percentages": {
                    sz: round((size_counts[sz] / max(total_objects, 1)) * 100.0, 2)
                    for sz in size_counts
                },
                "condition_class_matrix": cond_class_matrix,
            },
            "annotation_confidence_profile": conf_stats,
        }

        return analysis

    def export_reports(self, analysis: Dict[str, Any], output_json: Path, output_csv: Path, output_report: Path):
        """Writes JSON, CSV, and markdown reports."""
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_report.parent.mkdir(parents=True, exist_ok=True)

        # JSON Export
        with open(output_json, "w") as f:
            json.dump(analysis, f, indent=2)

        if analysis.get("status") == "NOT_AVAILABLE":
            # Write placeholder report explaining uningested status
            with open(output_report, "w") as f:
                f.write(
                    "# WeatherGuard-D1 Dataset Statistical Status\n"
                    "**Status:** NOT_AVAILABLE (Pending Ingestion)  \n\n"
                    f"{analysis.get('message')}\n\n"
                    "### Target Specifications:\n"
                    f"- **Target Images:** {analysis['target_specification']['target_images']:,}\n"
                    f"- **Target Resolution:** {analysis['target_specification']['target_resolution']}\n"
                    f"- **Conditions:** {', '.join(analysis['target_specification']['target_conditions'].keys())}\n"
                    f"- **Classes:** {', '.join(analysis['target_specification']['target_classes'])}\n"
                )
            return

        # CSV Export for Class Distribution
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_name", "count", "percentage"])
            tot = analysis["object_summary"]["total_valid_objects"]
            for cls_name, c_cnt in analysis["object_summary"]["class_counts"].items():
                pct = (c_cnt / max(tot, 1)) * 100.0
                writer.writerow([cls_name, c_cnt, f"{pct:.2f}"])

        # Markdown Report
        lines = [
            "# WeatherGuard-D1 Dataset Characterization & Statistical Audit",
            f"**Generated:** {analysis.get('timestamp')}  ",
            f"**Dataset Version:** {analysis.get('dataset_version')}  ",
            "",
            "## 1. Image & Condition Allocation",
            f"- **Total Accepted Images:** {analysis['image_summary']['accepted_images']:,}",
            f"- **Empty Images:** {analysis['image_summary']['empty_images_count']} ({analysis['image_summary']['empty_image_ratio']*100:.1f}%)",
            "",
            "| Condition | Target Quota | Ingested Count | Fulfillment % |",
            "| :--- | :---: | :---: | :---: |",
        ]
        for c, cnt in analysis["condition_distribution"]["counts"].items():
            tgt = analysis["condition_distribution"]["targets"].get(c, 0)
            pct = analysis["condition_distribution"]["completion_pct"].get(c, 0.0)
            lines.append(f"| **{c.title()}** | {tgt:,} | {cnt:,} | {pct:.1f}% |")

        lines.extend([
            "",
            "## 2. Object Scale Distribution (WeatherGuard Convention)",
            f"- **Total Bounding Boxes:** {analysis['object_summary']['total_valid_objects']:,}",
            f"- **Average Boxes / Image:** {analysis['object_summary']['average_objects_per_image']:.2f}",
            "",
            "| Scale Category | Definition (Max Dim) | Object Count | % Share |",
            "| :--- | :--- | :---: | :---: |",
            f"| **Ultra-Small** | < 16 px | {analysis['object_summary']['object_size_distribution']['ultra_small']:,} | {analysis['object_summary']['size_percentages']['ultra_small']:.2f}% |",
            f"| **Small** | 16 to < 32 px | {analysis['object_summary']['object_size_distribution']['small']:,} | {analysis['object_summary']['size_percentages']['small']:.2f}% |",
            f"| **Medium** | 32 to < 96 px | {analysis['object_summary']['object_size_distribution']['medium']:,} | {analysis['object_summary']['size_percentages']['medium']:.2f}% |",
            f"| **Large** | >= 96 px | {analysis['object_summary']['object_size_distribution']['large']:,} | {analysis['object_summary']['size_percentages']['large']:.2f}% |",
            "",
            "## 3. Class Balance & Diversity",
            f"- **Class Gini Coefficient:** {analysis['object_summary']['class_balance_gini']:.4f} (0=uniform, 1=skewed)",
            f"- **Shannon Entropy:** {analysis['object_summary']['class_entropy_nats']:.4f} nats",
            "",
            "| Class | Count | % Share |",
            "| :--- | :---: | :---: |",
        ])
        for cls_name, c_cnt in analysis["object_summary"]["class_counts"].items():
            pct = (c_cnt / max(tot, 1)) * 100.0
            lines.append(f"| **{cls_name}** | {c_cnt:,} | {pct:.2f}% |")

        lines.extend([
            "",
            "## 4. Annotation Confidence Profile",
            f"- **Mean Confidence:** {analysis['annotation_confidence_profile']['mean']:.4f}",
            f"- **Median Confidence:** {analysis['annotation_confidence_profile']['median']:.4f}",
            f"- **10th Percentile:** {analysis['annotation_confidence_profile']['p10']:.4f}",
            f"- **90th Percentile:** {analysis['annotation_confidence_profile']['p90']:.4f}",
        ])

        with open(output_report, "w") as f:
            f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze WeatherGuard-D1 dataset statistics.")
    parser.add_argument("--ingest-manifest", type=str, default="data/WeatherGuard-D1/manifests/ingestion_manifest.json")
    parser.add_argument("--validation-manifest", type=str, default="data/WeatherGuard-D1/manifests/annotation_validation_manifest.json")
    parser.add_argument("--output-json", type=str, default="artifacts/d1_dataset_analysis.json")
    parser.add_argument("--output-csv", type=str, default="artifacts/d1_class_distribution.csv")
    parser.add_argument("--output-report", type=str, default="reports/mission16_preparation/D1_DATASET_ANALYSIS.md")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    analyzer = D1DatasetAnalyzer(config_path=Path(args.config))
    analysis = analyzer.analyze_dataset(
        ingestion_manifest_path=Path(args.ingest-manifest if hasattr(args, "ingest-manifest") else args.ingest_manifest),
        validation_manifest_path=Path(args.validation_manifest),
    )
    analyzer.export_reports(
        analysis=analysis,
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
        output_report=Path(args.output_report),
    )
    print(f"Analysis status: {analysis.get('status')} - {analysis.get('message', 'Completed successfully.')}")


if __name__ == "__main__":
    main()
