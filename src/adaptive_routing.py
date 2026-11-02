"""Adaptive inference routing module for WeatherGuard Mission 4.

Provides training-free, deterministic routing policies that dynamically decide
whether a driving frame can be resolved at standard resolution (640px),
intermediate resolution (832px), or high resolution (1280px) based on lightweight
signals extracted from an initial 640px base pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import numpy as np


@dataclass
class ImageRoutingFeatures:
    """Lightweight signals extracted from a 640px base detection pass."""
    image_id: str
    n_small_boxes: int
    n_medium_boxes: int
    n_large_boxes: int
    n_low_conf: int
    n_total_candidates: int
    min_box_area: float
    mean_box_area: float
    has_small_box: bool
    uncertainty_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "n_small_boxes": self.n_small_boxes,
            "n_medium_boxes": self.n_medium_boxes,
            "n_large_boxes": self.n_large_boxes,
            "n_low_conf": self.n_low_conf,
            "n_total_candidates": self.n_total_candidates,
            "min_box_area": self.min_box_area,
            "mean_box_area": self.mean_box_area,
            "has_small_box": self.has_small_box,
            "uncertainty_score": self.uncertainty_score,
        }


def extract_routing_features(
    predictions: Sequence[dict[str, Any]],
    image_id: str = "",
    min_score: float = 0.15,
    small_area_limit: float = 1024.0,
    medium_area_limit: float = 9216.0,
    low_conf_lower: float = 0.05,
    low_conf_upper: float = 0.25,
) -> ImageRoutingFeatures:
    """Extract heuristic routing signals from raw 640px detections.
    
    Zero access to ground truth. Uses only detector predictions.
    By default, box counts use min_score=0.15 to filter out extreme background noise,
    while low_conf counts boxes in [0.05, 0.25).
    """
    valid_preds = [
        p for p in predictions
        if p.get("score", 0.0) >= min_score and p.get("common_label") is not None
    ]

    all_candidates = [
        p for p in predictions
        if p.get("score", 0.0) >= low_conf_lower and p.get("common_label") is not None
    ]

    small_count = 0
    med_count = 0
    large_count = 0
    areas: list[float] = []

    for p in valid_preds:
        box = p["box_xyxy"]
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        areas.append(area)

        if area < small_area_limit:
            small_count += 1
        elif area < medium_area_limit:
            med_count += 1
        else:
            large_count += 1

    low_conf_count = sum(
        1 for p in all_candidates if low_conf_lower <= p.get("score", 0.0) < low_conf_upper
    )

    min_area = float(np.min(areas)) if areas else 0.0
    mean_area = float(np.mean(areas)) if areas else 0.0
    uncertainty = (low_conf_count / len(all_candidates)) if all_candidates else 0.0

    return ImageRoutingFeatures(
        image_id=image_id,
        n_small_boxes=small_count,
        n_medium_boxes=med_count,
        n_large_boxes=large_count,
        n_low_conf=low_conf_count,
        n_total_candidates=len(valid_preds),
        min_box_area=min_area,
        mean_box_area=mean_area,
        has_small_box=(small_count > 0),
        uncertainty_score=uncertainty,
    )


# Policy Definitions
RoutingPolicyFn = Callable[[ImageRoutingFeatures], int]


def policy_fixed_640(feats: ImageRoutingFeatures) -> int:
    return 640


def policy_fixed_832(feats: ImageRoutingFeatures) -> int:
    return 832


def policy_fixed_1280(feats: ImageRoutingFeatures) -> int:
    return 1280


def policy_small_box_trigger_k1(feats: ImageRoutingFeatures) -> int:
    """Escalate to 1280px if base 640px detects >= 1 small box (at score >= 0.15)."""
    return 1280 if feats.n_small_boxes >= 1 else 640


def policy_small_box_trigger_k2(feats: ImageRoutingFeatures) -> int:
    """Escalate to 1280px if base 640px detects >= 2 small boxes (dense small objects)."""
    return 1280 if feats.n_small_boxes >= 2 else 640


def policy_small_box_trigger_op(feats: ImageRoutingFeatures) -> int:
    """Escalate to 1280px if base 640px detects small box at operational score >= 0.20."""
    return 1280 if feats.n_small_boxes >= 1 else 640


def policy_low_conf_trigger_k2(feats: ImageRoutingFeatures) -> int:
    """Escalate to 1280px if candidate pool has >= 2 low-confidence detections."""
    return 1280 if feats.n_low_conf >= 2 else 640


def policy_density_trigger_k8(feats: ImageRoutingFeatures) -> int:
    """Escalate to 1280px if total candidate count exceeds 8."""
    return 1280 if feats.n_total_candidates >= 8 else 640


def policy_combined_trigger(feats: ImageRoutingFeatures) -> int:
    """Escalate to 1280px if small box detected OR >= 3 low-confidence candidates."""
    if feats.n_small_boxes >= 1 or feats.n_low_conf >= 3:
        return 1280
    return 640


def policy_two_stage_hierarchical(feats: ImageRoutingFeatures) -> int:
    """Three-tier hierarchical routing: 640 -> 832 -> 1280.
    
    Tier 1 (640px): Frame contains only large/sparse objects (n_small == 0).
    Tier 2 (832px): Frame has single small box (n_small == 1).
    Tier 3 (1280px): Frame has dense small boxes (n_small >= 2).
    """
    if feats.n_small_boxes >= 2:
        return 1280
    elif feats.n_small_boxes >= 1:
        return 832
    return 640


def policy_hierarchical_fast_832(feats: ImageRoutingFeatures) -> int:
    """Capped hierarchical routing: never exceeds 832px to guarantee edge real-time throughput."""
    if feats.n_small_boxes >= 1:
        return 832
    return 640
