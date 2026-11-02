"""
WeatherGuard False-Alarm Suppression Module
Implements domain-aware geometric, reflection-gating, and cross-resolution
verification filters to suppress hallucinated false alarms under adverse weather.
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Sequence
import numpy as np

from src.metrics import iou_xyxy

def filter_aspect_ratio_anomalies(
    predictions: Sequence[Dict[str, Any]],
    min_aspect_ratio: float = 0.25,
    max_aspect_ratio: float = 4.5,
    high_conf_override: float = 0.50,
) -> List[Dict[str, Any]]:
    """Suppresses extreme-aspect-ratio artifacts (guardrails, overhead poles) unless very high confidence."""
    filtered = []
    for p in predictions:
        s = p.get("score", 0.0)
        b = p.get("box_xyxy", [0, 0, 0, 0])
        w = max(1.0, b[2] - b[0])
        h = max(1.0, b[3] - b[1])
        aspect = w / h

        if (aspect < min_aspect_ratio or aspect > max_aspect_ratio) and s < high_conf_override:
            # Suppress extreme slender box
            continue
        filtered.append(p)
    return filtered

def filter_road_surface_reflections(
    predictions: Sequence[Dict[str, Any]],
    img_height: int = 720,
    bottom_zone_frac: float = 0.20,
    min_score_in_zone: float = 0.28,
) -> List[Dict[str, Any]]:
    """Suppresses weak vehicle detections directly on wet asphalt reflection zones."""
    zone_top = float(img_height) * (1.0 - bottom_zone_frac)
    filtered = []

    def get_lbl(p):
        return p.get("common_label") or p.get("class_name") or p.get("label", "")

    for p in predictions:
        s = p.get("score", 0.0)
        b = p.get("box_xyxy", [0, 0, 0, 0])
        lbl = get_lbl(p)

        # Bottom zone reflection suppression for Car
        if lbl == "Car" and b[1] >= zone_top and s < min_score_in_zone:
            continue
        filtered.append(p)
    return filtered

def filter_cross_resolution_unconfirmed(
    primary_preds: Sequence[Dict[str, Any]],
    secondary_preds: Sequence[Dict[str, Any]],
    uncertain_band: Tuple[float, float] = (0.20, 0.28),
    min_confirm_iou: float = 0.25,
) -> List[Dict[str, Any]]:
    """Requires marginal primary predictions to be confirmed by secondary pass or higher confidence."""
    if not secondary_preds:
        return list(primary_preds)

    def get_lbl(p):
        return p.get("common_label") or p.get("class_name") or p.get("label", "")

    filtered = []
    for pp in primary_preds:
        s = pp.get("score", 0.0)
        # If score is solid, retain unconditionally
        if s > uncertain_band[1]:
            filtered.append(pp)
            continue
        # If in uncertain band, check for cross-resolution confirmation
        p_box = pp.get("box_xyxy", [0, 0, 0, 0])
        p_lbl = get_lbl(pp)
        confirmed = False
        for sp in secondary_preds:
            if get_lbl(sp) == p_lbl:
                iou = iou_xyxy(p_box, sp.get("box_xyxy", [0, 0, 0, 0]))
                if iou >= min_confirm_iou:
                    confirmed = True
                    break
        if confirmed:
            filtered.append(pp)
        # Else: unconfirmed marginal box is suppressed

    return filtered

def apply_adaptive_suppression(
    primary_preds: Sequence[Dict[str, Any]],
    secondary_preds: Optional[Sequence[Dict[str, Any]]] = None,
    use_aspect_filter: bool = True,
    use_road_filter: bool = True,
    use_cross_confirm: bool = True,
    img_height: int = 720,
) -> List[Dict[str, Any]]:
    """Combines all false-alarm suppression filters."""
    preds = list(primary_preds)
    if use_aspect_filter:
        preds = filter_aspect_ratio_anomalies(preds)
    if use_road_filter:
        preds = filter_road_surface_reflections(preds, img_height=img_height)
    if use_cross_confirm and secondary_preds is not None:
        preds = filter_cross_resolution_unconfirmed(preds, secondary_preds)
    return preds
