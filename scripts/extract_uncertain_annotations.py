"""WeatherGuard-D1 Annotation Uncertainty Extraction & Quality Control Review Queue.

Module: scripts/extract_uncertain_annotations.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
Mission: Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

Tiers annotations into three separate review queues:
1. ACCEPTED_STRUCTURAL: High-confidence, validated geometries.
2. NEEDS_VISUAL_REVIEW: Ambiguous, ultra-small, borderline confidence, size disagreements.
3. REJECTED: Structural, coordinate, or non-positive geometric violations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "weatherguard_d1.yaml"


class UncertaintyExtractor:
    """Audits validated annotation results and triages them into 3 review queues."""

    def __init__(
        self,
        low_confidence_threshold: float = 0.50,
        tiny_ambiguous_px_threshold: float = 8.0,
        suspicious_large_ratio: float = 0.85,
        max_boxes_per_image: int = 30,
    ):
        self.low_conf_thresh = low_confidence_threshold
        self.tiny_px_thresh = tiny_ambiguous_px_threshold
        self.suspicious_large_ratio = suspicious_large_ratio
        self.max_boxes_per_image = max_boxes_per_image

    def triage_validation_manifest(self, manifest_data: Dict[str, Any]) -> Dict[str, Any]:
        """Triages a validated manifest dictionary into structured review queues."""
        accepted_structural: List[Dict[str, Any]] = []
        needs_visual_review: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        results = manifest_data.get("results", [])

        for img_res in results:
            img_id = img_res.get("image_id", "unknown")
            cond = img_res.get("condition", "unknown")
            valid_boxes = img_res.get("valid_boxes", [])
            rejected_boxes = img_res.get("rejected_boxes", [])
            is_empty = img_res.get("is_empty", False)

            # Record any already-rejected boxes
            for r_box in rejected_boxes:
                rejected.append({
                    "image_id": img_id,
                    "condition": cond,
                    "box": r_box,
                    "reason": "Structural or geometric validation failure",
                })

            # Check for image-level anomalies
            img_review_flags: List[str] = []
            if len(valid_boxes) > self.max_boxes_per_image:
                img_review_flags.append(f"High box density: {len(valid_boxes)} boxes > limit {self.max_boxes_per_image}")
            if is_empty:
                img_review_flags.append("Empty image (0 annotations)")

            # Check individual valid boxes
            for b in valid_boxes:
                b_flags: List[str] = list(img_review_flags)
                conf = float(b.get("confidence", 1.0))
                max_dim = float(b.get("max_dim_px", 0.0))
                w_px = float(b.get("pixel_width", 0.0))
                h_px = float(b.get("pixel_height", 0.0))
                disagree = bool(b.get("size_disagreement", False))

                if conf < self.low_conf_thresh:
                    b_flags.append(f"Borderline confidence: {conf:.2f} < {self.low_conf_thresh:.2f}")

                if max_dim < self.tiny_px_thresh:
                    b_flags.append(f"Extremely small object: max dim {max_dim:.1f}px < {self.tiny_px_thresh}px")

                if (w_px / 1280.0 > self.suspicious_large_ratio) or (h_px / 720.0 > self.suspicious_large_ratio):
                    b_flags.append(f"Suspiciously large bounding box: {w_px:.0f}x{h_px:.0f} spans >{self.suspicious_large_ratio*100:.0f}% of frame")

                if disagree:
                    b_flags.append(f"Size category disagreement: Gemini claimed '{b.get('gemini_supplied_size')}' vs WeatherGuard '{b.get('recalculated_size')}'")

                if b.get("flags"):
                    for fl in b["flags"]:
                        if fl not in b_flags:
                            b_flags.append(fl)

                entry = {
                    "image_id": img_id,
                    "condition": cond,
                    "box_id": b.get("box_id"),
                    "class_name": b.get("class_name"),
                    "confidence": conf,
                    "pixel_dimensions": f"{w_px:.1f}x{h_px:.1f}",
                    "max_dim_px": max_dim,
                    "recalculated_size": b.get("recalculated_size"),
                    "review_reasons": b_flags,
                }

                if b_flags:
                    needs_visual_review.append(entry)
                else:
                    accepted_structural.append(entry)

        return {
            "summary": {
                "total_accepted_structural": len(accepted_structural),
                "total_needs_visual_review": len(needs_visual_review),
                "total_rejected": len(rejected),
            },
            "accepted_structural": accepted_structural,
            "needs_visual_review": needs_visual_review,
            "rejected": rejected,
        }


def main():
    parser = argparse.ArgumentParser(description="Extract uncertain annotations into review queues.")
    parser.add_argument("--manifest", "--validation-report", "--validation_report", dest="manifest", type=str, default="data/WeatherGuard-D1/manifests/annotation_validation_manifest.json", help="Path to validation report/manifest")
    parser.add_argument("--output", "--output-dir", "--output_dir", dest="output", type=str, default="data/WeatherGuard-D1/quality_control/uncertainty_queue.json", help="Output path or directory for review queues")
    args = parser.parse_args()

    m_path = Path(args.manifest)
    if not m_path.exists():
        # Fallback check under artifacts
        alt_m = Path("artifacts/d1_annotation_validation_report.json")
        if alt_m.exists():
            m_path = alt_m
        else:
            print(f"Validation manifest not found: {m_path}. Exiting.")
            return

    with open(m_path) as f:
        data = json.load(f)

    extractor = UncertaintyExtractor()
    triage = extractor.triage_validation_manifest(data)

    out_p = Path(args.output)
    if out_p.is_dir() or not out_p.suffix:
        out_p.mkdir(parents=True, exist_ok=True)
        out_p = out_p / "uncertainty_queue.json"
    else:
        out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(triage, f, indent=2)

    print(f"Triage complete: {triage['summary']['total_accepted_structural']} accepted, "
          f"{triage['summary']['total_needs_visual_review']} need visual review, "
          f"{triage['summary']['total_rejected']} rejected.")


if __name__ == "__main__":
    main()
