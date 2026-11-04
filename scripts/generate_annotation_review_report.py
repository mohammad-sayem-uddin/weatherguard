"""Generates human-readable markdown reports for WeatherGuard-D1 visual review queues.

Module: scripts/generate_annotation_review_report.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
Mission: Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


def generate_markdown_report(queue_data: Dict[str, Any], output_path: Path) -> Path:
    summary = queue_data.get("summary", {})
    review_items = queue_data.get("needs_visual_review", [])
    rejected_items = queue_data.get("rejected", [])

    lines = [
        "# WeatherGuard-D1 Annotation Quality Control & Visual Review Report",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        "**Protocol:** Gemini Flash AI-assisted annotations must be independently validated before model training.",
        "",
        "## 1. Triage Summary",
        f"- **Accepted (Structurally Sound & High Confidence):** {summary.get('total_accepted_structural', 0)}",
        f"- **Needs Visual Review (Flagged for Inspection):** {summary.get('total_needs_visual_review', 0)}",
        f"- **Rejected (Geometric / Coordinate Failures):** {summary.get('total_rejected', 0)}",
        "",
        "## 2. Records Requiring Visual Inspection",
        "The following annotations passed structural JSON parsing but exhibit borderline properties (e.g. low confidence, size disagreements, or ultra-small dimensions):",
        "",
        "| Image ID | Condition | Class | Pixel Size | Max Dim | Confidence | Review Triggers |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
    ]

    for item in review_items[:100]:  # Cap at 100 in table
        reasons = "; ".join(item.get("review_reasons", []))
        lines.append(
            f"| `{item.get('image_id')}` | {item.get('condition')} | {item.get('class_name')} | "
            f"{item.get('pixel_dimensions')} | {item.get('max_dim_px')}px | {item.get('confidence'):.2f} | {reasons} |"
        )

    if len(review_items) > 100:
        lines.append(f"\n*(Showing 100 of {len(review_items)} total items requiring review. See machine JSON for full queue.)*")

    if rejected_items:
        lines.extend([
            "",
            "## 3. Rejected Records (Structural Violations)",
            "| Image ID | Condition | Failure Reason | Details |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for rej in rejected_items[:50]:
            box_info = rej.get("box", {})
            reasons = "; ".join(box_info.get("rejection_reasons", ["Invalid"]))
            lines.append(f"| `{rej.get('image_id')}` | {rej.get('condition')} | {rej.get('reason')} | {reasons} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate human-readable annotation review markdown report.")
    parser.add_argument("--queue", type=str, default="data/WeatherGuard-D1/quality_control/uncertainty_queue.json")
    parser.add_argument("--report", type=str, default="data/WeatherGuard-D1/reports/annotation_review_report.md")
    args = parser.parse_args()

    q_path = Path(args.queue)
    if not q_path.exists():
        print(f"Queue file {q_path} not found. Exiting.")
        return

    with open(q_path) as f:
        data = json.load(f)

    r_path = generate_markdown_report(data, Path(args.report))
    print(f"Review report generated at: {r_path}")


if __name__ == "__main__":
    main()
