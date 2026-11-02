"""WeatherGuard Mission 6 — Model Fusion & Complementary Prediction Integration.

Provides lightweight ensembling and fusion strategies combining
Pretrained YOLOv8n and Fine-Tuned / Adapted YOLOv8n models:
  1. Simple Union + Class-Wise NMS:
       Combines prediction sets and suppresses spatial overlap with IoU threshold.
  2. Confidence-Weighted Box Fusion (WBF):
       Clusters overlapping predictions for the same class; fuses coordinates as
       score-weighted averages; penalizes unconfirmed single-model predictions.
  3. Selective Fallback / Hybrid Routing:
       Routes small (< 1024 px^2) and low-contrast objects from the Pretrained detector,
       and high-confidence salient weather foreground from the Fine-Tuned detector.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from src.external_dataset import CANONICAL_CLASSES
from src.metrics import greedy_match, iou_xyxy


def nms_boxes(
    boxes: List[List[float]],
    scores: List[float],
    iou_thresh: float = 0.50,
) -> List[int]:
    """Applies Non-Maximum Suppression and returns indices of kept boxes."""
    if not boxes:
        return []

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while order:
        i = order.pop(0)
        keep.append(i)
        order = [
            j for j in order
            if iou_xyxy(boxes[i], boxes[j]) < iou_thresh
        ]

    return keep


def fuse_predictions_simple_ensemble(
    preds_pre: List[Dict[str, Any]],
    preds_ft: List[Dict[str, Any]],
    threshold: float = 0.20,
    nms_iou: float = 0.50,
) -> List[Dict[str, Any]]:
    """Simple union of predictions filtered by score and merged with class-wise NMS."""
    merged_raw = [
        p for p in preds_pre
        if p.get("score", 0.0) >= threshold and p.get("common_label") in CANONICAL_CLASSES
    ] + [
        p for p in preds_ft
        if p.get("score", 0.0) >= threshold and p.get("common_label") in CANONICAL_CLASSES
    ]

    fused_preds = []
    for c in CANONICAL_CLASSES:
        c_preds = [p for p in merged_raw if p["common_label"] == c]
        if not c_preds:
            continue
        boxes = [p["box_xyxy"] for p in c_preds]
        scores = [p["score"] for p in c_preds]
        kept_indices = nms_boxes(boxes, scores, iou_thresh=nms_iou)
        for idx in kept_indices:
            fused_preds.append(c_preds[idx])

    return fused_preds


def fuse_predictions_wbf(
    preds_pre: List[Dict[str, Any]],
    preds_ft: List[Dict[str, Any]],
    threshold: float = 0.20,
    iou_thresh: float = 0.55,
    unconfirmed_penalty: float = 0.70,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Confidence-Weighted Box Fusion with unconfirmed hallucination suppression."""
    pre_filt = [
        p for p in preds_pre
        if p.get("score", 0.0) >= threshold and p.get("common_label") in CANONICAL_CLASSES
    ]
    ft_filt = [
        p for p in preds_ft
        if p.get("score", 0.0) >= threshold and p.get("common_label") in CANONICAL_CLASSES
    ]

    fused_preds = []
    stats = {"pre_only": 0, "ft_only": 0, "agreed": 0}

    for c in CANONICAL_CLASSES:
        p_list = [(p, "pre") for p in pre_filt if p["common_label"] == c]
        f_list = [(p, "ft") for p in ft_filt if p["common_label"] == c]
        all_c = p_list + f_list
        if not all_c:
            continue

        # Sort descending by score
        all_c.sort(key=lambda x: x[0]["score"], reverse=True)
        clusters: List[List[Tuple[Dict[str, Any], str]]] = []

        for p_item, src in all_c:
            box = p_item["box_xyxy"]
            matched_cluster = None
            for cl in clusters:
                # Compare against cluster mean box
                cl_boxes = [item[0]["box_xyxy"] for item in cl]
                mean_b = np.mean(cl_boxes, axis=0).tolist()
                if iou_xyxy(box, mean_b) >= iou_thresh:
                    matched_cluster = cl
                    break

            if matched_cluster is not None:
                matched_cluster.append((p_item, src))
            else:
                clusters.append([(p_item, src)])

        for cl in clusters:
            sources = {item[1] for item in cl}
            weights = np.array([item[0]["score"] for item in cl], dtype=np.float32)
            total_w = np.sum(weights)
            boxes = np.array([item[0]["box_xyxy"] for item in cl], dtype=np.float32)

            fused_box = (np.sum(boxes * weights[:, None], axis=0) / total_w).tolist()

            if len(sources) > 1:
                # Both models agree on detection!
                stats["agreed"] += 1
                fused_score = float(np.mean(weights))
            elif "pre" in sources:
                stats["pre_only"] += 1
                fused_score = float(np.mean(weights))
            else:
                stats["ft_only"] += 1
                # Penalize unconfirmed fine-tuned hallucination
                fused_score = float(np.mean(weights) * unconfirmed_penalty)

            if fused_score >= threshold:
                fused_preds.append({
                    "common_label": c,
                    "score": fused_score,
                    "box_xyxy": fused_box,
                    "models_agreed": len(sources) > 1,
                })

    return fused_preds, stats


def fuse_predictions_selective_fallback(
    preds_pre: List[Dict[str, Any]],
    preds_ft: List[Dict[str, Any]],
    threshold: float = 0.20,
    small_area_limit: float = 1024.0,
    ft_high_conf_limit: float = 0.40,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Selective fallback: Pretrained handles small objects; Fine-Tuned handles salient large objects."""
    pre_filt = [
        p for p in preds_pre
        if p.get("score", 0.0) >= threshold and p.get("common_label") in CANONICAL_CLASSES
    ]
    ft_filt = [
        p for p in preds_ft
        if p.get("score", 0.0) >= threshold and p.get("common_label") in CANONICAL_CLASSES
    ]

    fused_preds = []
    stats = {"pre_small": 0, "ft_salient": 0, "pre_default": 0}

    # 1. Add all small-object predictions from Pretrained YOLOv8n (preserving tiny object recall)
    for p in pre_filt:
        b = p["box_xyxy"]
        area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        if area < small_area_limit:
            fused_preds.append(p)
            stats["pre_small"] += 1

    # 2. Add high-confidence predictions from Fine-Tuned model (capturing adverse-weather targets)
    for p in ft_filt:
        b = p["box_xyxy"]
        area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        if area >= small_area_limit and p["score"] >= ft_high_conf_limit:
            # Check overlap with existing fused boxes
            overlap = any(
                iou_xyxy(p["box_xyxy"], fp["box_xyxy"]) >= 0.50 and p["common_label"] == fp["common_label"]
                for fp in fused_preds
            )
            if not overlap:
                fused_preds.append(p)
                stats["ft_salient"] += 1

    # 3. Add medium/large pretrained predictions where fine-tuned did not make a high-confidence call
    for p in pre_filt:
        b = p["box_xyxy"]
        area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        if area >= small_area_limit:
            overlap = any(
                iou_xyxy(p["box_xyxy"], fp["box_xyxy"]) >= 0.50 and p["common_label"] == fp["common_label"]
                for fp in fused_preds
            )
            if not overlap:
                fused_preds.append(p)
                stats["pre_default"] += 1

    return fused_preds, stats
