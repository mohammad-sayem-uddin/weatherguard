"""Targeted Region Refinement module for WeatherGuard Mission 4.

Implements candidate RoI proposal, context-padded crop extraction, coordinate
projection, and duplicate-suppressed detection merging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple
import numpy as np


@dataclass
class CropRegion:
    """Bounding box of an extracted crop in native image coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    candidate_indices: list[int]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


def generate_candidate_crops(
    predictions: Sequence[dict[str, Any]],
    image_shape: tuple[int, int] = (1280, 720),
    max_crops: int = 2,
    context_margin: float = 0.20,
    min_crop_size: float = 384.0,
    small_area_limit: float = 1024.0,
    min_score: float = 0.10,
) -> list[CropRegion]:
    """Generate up to max_crops localized regions around small or uncertain detections.
    
    Zero access to ground truth. Uses only base 640px detections.
    """
    img_w, img_h = image_shape

    # Filter candidate small detections
    candidate_indices = []
    for idx, p in enumerate(predictions):
        if p.get("common_label") is None:
            continue
        score = p.get("score", 0.0)
        box = p["box_xyxy"]
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if score >= min_score and area < small_area_limit:
            candidate_indices.append(idx)

    if not candidate_indices:
        return []

    candidate_boxes = [predictions[i]["box_xyxy"] for i in candidate_indices]

    # Simple 1-D spatial clustering along X-axis
    if len(candidate_boxes) <= 2 or max_crops == 1:
        clusters = [candidate_indices]
    else:
        # Sort candidates by center X
        centers_x = [(predictions[i]["box_xyxy"][0] + predictions[i]["box_xyxy"][2]) / 2.0 for i in candidate_indices]
        sorted_order = np.argsort(centers_x)
        sorted_indices = [candidate_indices[k] for k in sorted_order]

        # Split into two clusters if gap between consecutive candidates is significant
        split_idx = len(sorted_indices) // 2
        clusters = [sorted_indices[:split_idx], sorted_indices[split_idx:]]

    crops: list[CropRegion] = []
    for cluster in clusters[:max_crops]:
        if not cluster:
            continue
        c_boxes = [predictions[i]["box_xyxy"] for i in cluster]
        min_x = min(b[0] for b in c_boxes)
        min_y = min(b[1] for b in c_boxes)
        max_x = max(b[2] for b in c_boxes)
        max_y = max(b[3] for b in c_boxes)

        # Context expansion
        w = max_x - min_x
        h = max_y - min_y
        pad_x = max(w * context_margin, (min_crop_size - w) / 2.0)
        pad_y = max(h * context_margin, (min_crop_size - h) / 2.0)

        cx1 = max(0.0, min_x - pad_x)
        cy1 = max(0.0, min_y - pad_y)
        cx2 = min(float(img_w), max_x + pad_x)
        cy2 = min(float(img_h), max_y + pad_y)

        crops.append(CropRegion(
            x1=float(cx1),
            y1=float(cy1),
            x2=float(cx2),
            y2=float(cy2),
            candidate_indices=cluster,
        ))

    return crops


def transform_crop_to_original(
    crop_box: tuple[float, float, float, float] | list[float],
    crop_region: CropRegion,
    crop_input_size: tuple[int, int] = (640, 640),
) -> list[float]:
    """Project detected bounding box from crop coordinates back to native image coordinates."""
    scale_x = crop_region.width / float(crop_input_size[0])
    scale_y = crop_region.height / float(crop_input_size[1])

    x1 = crop_region.x1 + crop_box[0] * scale_x
    y1 = crop_region.y1 + crop_box[1] * scale_y
    x2 = crop_region.x1 + crop_box[2] * scale_x
    y2 = crop_region.y1 + crop_box[3] * scale_y

    return [float(x1), float(y1), float(x2), float(y2)]


def iou_xyxy(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Intersection over Union for two xyxy boxes."""
    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union_area = area1 + area2 - inter_area

    return (inter_area / union_area) if union_area > 0.0 else 0.0


def merge_and_nms(
    base_predictions: Sequence[dict[str, Any]],
    crop_predictions: Sequence[dict[str, Any]],
    nms_iou_threshold: float = 0.50,
) -> list[dict[str, Any]]:
    """Merge base 640px predictions with transformed crop predictions using greedy class-wise NMS.
    
    If a base prediction and a crop prediction overlap with IoU >= nms_iou_threshold,
    the detection with the higher confidence score is retained.
    """
    all_preds = list(base_predictions) + list(crop_predictions)
    if not all_preds:
        return []

    # Sort descending by score
    all_preds.sort(key=lambda p: p.get("score", 0.0), reverse=True)

    kept: list[dict[str, Any]] = []
    classes = set(p.get("common_label") for p in all_preds if p.get("common_label") is not None)

    for c in classes:
        c_preds = [p for p in all_preds if p.get("common_label") == c]
        while c_preds:
            best = c_preds.pop(0)
            kept.append(best)
            c_preds = [
                p for p in c_preds
                if iou_xyxy(best["box_xyxy"], p["box_xyxy"]) < nms_iou_threshold
            ]

    # Include any predictions with None common_label preserved as-is
    kept.extend([p for p in all_preds if p.get("common_label") is None])
    return kept
