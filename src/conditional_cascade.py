"""
WeatherGuard Conditional Model Cascade Module
Implements Full-Frame and Region-Based Crop Escalation architectures with
coordinate transformations, padding controls, context preservation, and duplicate suppression.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Sequence, Optional
import numpy as np

from src.metrics import iou_xyxy

@dataclass
class CropRegion:
    """Bounding box crop in original image coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    orig_pred_score: float
    orig_label: str

def construct_candidate_crops(
    detections: Sequence[Dict[str, Any]],
    img_width: int = 1280,
    img_height: int = 720,
    padding_factor: float = 0.5,
    min_crop_size: int = 128,
    max_crops_per_image: int = 4,
) -> List[CropRegion]:
    """Generates padded square/rectangular crops around small or low-confidence candidates."""
    candidate_boxes = []
    for d in detections:
        s = d.get("score", 0.0)
        b = d.get("box_xyxy", [0, 0, 0, 0])
        w = max(0.0, b[2] - b[0])
        h = max(0.0, b[3] - b[1])
        area = w * h
        # Escalate on small objects or ambiguous scores
        if area < 2048 or (0.15 <= s <= 0.35):
            candidate_boxes.append((b, s, d.get("common_label") or d.get("class_name") or d.get("label", "Unknown")))

    if not candidate_boxes:
        return []

    # Sort candidates by area (smallest first) then score
    candidate_boxes.sort(key=lambda item: (max(0.0, item[0][2] - item[0][0]) * max(0.0, item[0][3] - item[0][1]), item[1]))
    candidate_boxes = candidate_boxes[:max_crops_per_image]

    crops = []
    for b, s, lbl in candidate_boxes:
        w = max(1.0, b[2] - b[0])
        h = max(1.0, b[3] - b[1])
        pad_w = w * padding_factor
        pad_h = h * padding_factor

        cx1 = max(0.0, b[0] - pad_w)
        cy1 = max(0.0, b[1] - pad_h)
        cx2 = min(float(img_width), b[2] + pad_w)
        cy2 = min(float(img_height), b[3] + pad_h)

        # Ensure minimum size
        cw = cx2 - cx1
        ch = cy2 - cy1
        if cw < min_crop_size:
            diff = (min_crop_size - cw) / 2.0
            cx1 = max(0.0, cx1 - diff)
            cx2 = min(float(img_width), cx2 + diff)
        if ch < min_crop_size:
            diff = (min_crop_size - ch) / 2.0
            cy1 = max(0.0, cy1 - diff)
            cy2 = min(float(img_height), cy2 + diff)

        crops.append(CropRegion(cx1, cy1, cx2, cy2, s, lbl))

    return crops

def map_crop_to_image_coords(
    crop_box_xyxy: Sequence[float],
    crop_region: CropRegion,
    crop_img_size: int = 640,
) -> List[float]:
    """Maps normalized or pixel crop coordinates back to full-frame image pixels."""
    bx1, by1, bx2, by2 = crop_box_xyxy
    crop_w = crop_region.x2 - crop_region.x1
    crop_h = crop_region.y2 - crop_region.y1

    # Map relative to crop image size
    scale_x = crop_w / float(crop_img_size)
    scale_y = crop_h / float(crop_img_size)

    img_x1 = crop_region.x1 + (bx1 * scale_x)
    img_y1 = crop_region.y1 + (by1 * scale_y)
    img_x2 = crop_region.x1 + (bx2 * scale_x)
    img_y2 = crop_region.y1 + (by2 * scale_y)

    return [float(img_x1), float(img_y1), float(img_x2), float(img_y2)]

def merge_cascade_predictions(
    base_preds: List[Dict[str, Any]],
    escalated_preds: List[Dict[str, Any]],
    nms_iou_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """Merges base and escalated detections, suppressing duplicates with priority to escalated boxes."""
    if not escalated_preds:
        return base_preds
    if not base_preds:
        return escalated_preds

    # Escalated predictions take precedence
    final_preds = list(escalated_preds)

    def get_lbl(p):
        return p.get("common_label") or p.get("class_name") or p.get("label")

    for bp in base_preds:
        b_box = bp.get("box_xyxy", [0, 0, 0, 0])
        b_lbl = get_lbl(bp)
        duplicate = False
        for ep in final_preds:
            if get_lbl(ep) == b_lbl:
                iou = iou_xyxy(b_box, ep.get("box_xyxy", [0, 0, 0, 0]))
                if iou >= nms_iou_threshold:
                    duplicate = True
                    break
        if not duplicate:
            final_preds.append(bp)

    return final_preds
