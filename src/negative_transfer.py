"""WeatherGuard Mission 6 — Prediction-Level Forensics & Negative Transfer Analysis.

Provides modular analysis tools to inspect prediction transitions between
a baseline detector (e.g. Pretrained YOLOv8n) and an adapted detector (e.g. Fine-Tuned Champion):
  1. Detailed instance-level matching to ground truth.
  2. Construction of the 7-state detection transition matrix:
       - State 1: Correct -> Correct (Pretrained TP -> Adapted TP)
       - State 2: Correct -> Missed (Pretrained TP -> Adapted FN) [Negative Transfer Loss]
       - State 3: Missed -> Recovered (Pretrained FN -> Adapted TP) [Adaptation Gain]
       - State 4: Correct -> Wrong Class (Pretrained TP -> Adapted FP with class confusion)
       - State 5: Additional FP (Adapted introduces new hallucination)
       - State 6: FP -> Corrected (Pretrained FP suppressed by adaptation)
       - State 7: Retained FP (Both models hallucinate same background clutter)
  3. Stratification by canonical class, object scale (Small, Medium, Large), and weather.
  4. Confidence shift distributions across TP, FP, and FN candidate detections.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.external_dataset import CANONICAL_CLASSES
from src.metrics import iou_xyxy, greedy_match


@dataclass
class InstanceTransition:
    image_id: str
    weather: str
    scale_category: str
    class_name: str
    transition_type: str  # 1_correct_to_correct, 2_correct_to_missed, etc.
    pretrained_conf: Optional[float] = None
    adapted_conf: Optional[float] = None
    gt_box: Optional[List[float]] = None
    pretrained_box: Optional[List[float]] = None
    adapted_box: Optional[List[float]] = None
    iou_pre_gt: Optional[float] = None
    iou_adapt_gt: Optional[float] = None


def get_scale_category(box_xyxy: List[float]) -> str:
    w = max(0.0, box_xyxy[2] - box_xyxy[0])
    h = max(0.0, box_xyxy[3] - box_xyxy[1])
    area = w * h
    if area < 1024:
        return "small"
    elif area < 9216:
        return "medium"
    return "large"


def match_image_predictions(
    gt_objects: List[Dict[str, Any]],
    pre_preds: List[Dict[str, Any]],
    adapt_preds: List[Dict[str, Any]],
    threshold: float = 0.20,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Matches predictions of both models to GT and each other on a single image."""
    filtered_pre = [
        p for p in pre_preds
        if p.get("score", 0.0) >= threshold and p.get("common_label") in CANONICAL_CLASSES
    ]
    filtered_adapt = [
        p for p in adapt_preds
        if p.get("score", 0.0) >= threshold and p.get("common_label") in CANONICAL_CLASSES
    ]

    # Pretrained matching per class
    pre_matched_gt = {}  # gt_idx -> (pred_idx, iou)
    pre_matched_preds = {}  # pred_idx -> (gt_idx, iou)
    for c in CANONICAL_CLASSES:
        c_preds = [(i, p) for i, p in enumerate(filtered_pre) if p["common_label"] == c]
        c_gts = [(j, g) for j, g in enumerate(gt_objects) if g.get("class_name") == c]
        if not c_preds or not c_gts:
            continue
        p_boxes = [p["box_xyxy"] for _, p in c_preds]
        p_scores = [p["score"] for _, p in c_preds]
        g_boxes = [g["box_xyxy"] for _, g in c_gts]
        matches = greedy_match(p_boxes, p_scores, g_boxes, iou_threshold=iou_thresh)
        for sub_p, sub_g, iou_val in matches:
            p_idx = c_preds[sub_p][0]
            g_idx = c_gts[sub_g][0]
            pre_matched_gt[g_idx] = (p_idx, float(iou_val))
            pre_matched_preds[p_idx] = (g_idx, float(iou_val))

    # Adapted matching per class
    adapt_matched_gt = {}
    adapt_matched_preds = {}
    for c in CANONICAL_CLASSES:
        c_preds = [(i, p) for i, p in enumerate(filtered_adapt) if p["common_label"] == c]
        c_gts = [(j, g) for j, g in enumerate(gt_objects) if g.get("class_name") == c]
        if not c_preds or not c_gts:
            continue
        p_boxes = [p["box_xyxy"] for _, p in c_preds]
        p_scores = [p["score"] for _, p in c_preds]
        g_boxes = [g["box_xyxy"] for _, g in c_gts]
        matches = greedy_match(p_boxes, p_scores, g_boxes, iou_threshold=iou_thresh)
        for sub_p, sub_g, iou_val in matches:
            p_idx = c_preds[sub_p][0]
            g_idx = c_gts[sub_g][0]
            adapt_matched_gt[g_idx] = (p_idx, float(iou_val))
            adapt_matched_preds[p_idx] = (g_idx, float(iou_val))

    return {
        "filtered_pre": filtered_pre,
        "filtered_adapt": filtered_adapt,
        "pre_matched_gt": pre_matched_gt,
        "pre_matched_preds": pre_matched_preds,
        "adapt_matched_gt": adapt_matched_gt,
        "adapt_matched_preds": adapt_matched_preds,
    }


def analyze_image_transitions(
    image_id: str,
    weather: str,
    gt_objects: List[Dict[str, Any]],
    pre_preds: List[Dict[str, Any]],
    adapt_preds: List[Dict[str, Any]],
    threshold: float = 0.20,
    iou_thresh: float = 0.50,
) -> List[InstanceTransition]:
    """Generates the granular instance transitions for a single image."""
    m = match_image_predictions(
        gt_objects, pre_preds, adapt_preds, threshold=threshold, iou_thresh=iou_thresh
    )
    filtered_pre = m["filtered_pre"]
    filtered_adapt = m["filtered_adapt"]
    pre_matched_gt = m["pre_matched_gt"]
    pre_matched_preds = m["pre_matched_preds"]
    adapt_matched_gt = m["adapt_matched_gt"]
    adapt_matched_preds = m["adapt_matched_preds"]

    transitions: List[InstanceTransition] = []

    # 1. Ground truth centered transitions (States 1, 2, 3, 4)
    for g_idx, g in enumerate(gt_objects):
        c_name = g.get("class_name", "unknown")
        if c_name not in CANONICAL_CLASSES:
            continue
        g_box = g["box_xyxy"]
        scale = get_scale_category(g_box)

        pre_hit = g_idx in pre_matched_gt
        adapt_hit = g_idx in adapt_matched_gt

        if pre_hit and adapt_hit:
            p_idx, p_iou = pre_matched_gt[g_idx]
            a_idx, a_iou = adapt_matched_gt[g_idx]
            transitions.append(
                InstanceTransition(
                    image_id=image_id,
                    weather=weather,
                    scale_category=scale,
                    class_name=c_name,
                    transition_type="1_correct_to_correct",
                    pretrained_conf=float(filtered_pre[p_idx]["score"]),
                    adapted_conf=float(filtered_adapt[a_idx]["score"]),
                    gt_box=g_box,
                    pretrained_box=filtered_pre[p_idx]["box_xyxy"],
                    adapted_box=filtered_adapt[a_idx]["box_xyxy"],
                    iou_pre_gt=p_iou,
                    iou_adapt_gt=a_iou,
                )
            )
        elif pre_hit and not adapt_hit:
            # Pretrained was correct, adapted missed. Check if adapted had wrong class!
            p_idx, p_iou = pre_matched_gt[g_idx]
            # Check if any adapt prediction overlapped this GT with iou >= 0.5 but different class
            wrong_class_candidate = None
            for a_idx, a_p in enumerate(filtered_adapt):
                if a_idx in adapt_matched_preds:
                    continue
                iou_val = iou_xyxy(a_p["box_xyxy"], g_box)
                if iou_val >= iou_thresh:
                    wrong_class_candidate = (a_idx, a_p, float(iou_val))
                    break

            if wrong_class_candidate is not None:
                a_idx, a_p, a_iou = wrong_class_candidate
                transitions.append(
                    InstanceTransition(
                        image_id=image_id,
                        weather=weather,
                        scale_category=scale,
                        class_name=c_name,
                        transition_type="4_correct_to_wrong_class",
                        pretrained_conf=float(filtered_pre[p_idx]["score"]),
                        adapted_conf=float(a_p["score"]),
                        gt_box=g_box,
                        pretrained_box=filtered_pre[p_idx]["box_xyxy"],
                        adapted_box=a_p["box_xyxy"],
                        iou_pre_gt=p_iou,
                        iou_adapt_gt=a_iou,
                    )
                )
            else:
                # Find candidate adapted prediction below threshold or best spatial overlap for conf comparison
                best_below_thresh_conf = None
                for a_p in adapt_preds:
                    if a_p.get("common_label") == c_name:
                        iou_val = iou_xyxy(a_p["box_xyxy"], g_box)
                        if iou_val >= iou_thresh:
                            best_below_thresh_conf = max(
                                best_below_thresh_conf or 0.0, float(a_p.get("score", 0.0))
                            )

                transitions.append(
                    InstanceTransition(
                        image_id=image_id,
                        weather=weather,
                        scale_category=scale,
                        class_name=c_name,
                        transition_type="2_correct_to_missed",
                        pretrained_conf=float(filtered_pre[p_idx]["score"]),
                        adapted_conf=best_below_thresh_conf,
                        gt_box=g_box,
                        pretrained_box=filtered_pre[p_idx]["box_xyxy"],
                        adapted_box=None,
                        iou_pre_gt=p_iou,
                        iou_adapt_gt=None,
                    )
                )
        elif not pre_hit and adapt_hit:
            # Adapted recovered an object missed by pretrained
            a_idx, a_iou = adapt_matched_gt[g_idx]
            best_pre_conf = None
            for p_p in pre_preds:
                if p_p.get("common_label") == c_name:
                    iou_val = iou_xyxy(p_p["box_xyxy"], g_box)
                    if iou_val >= iou_thresh:
                        best_pre_conf = max(
                            best_pre_conf or 0.0, float(p_p.get("score", 0.0))
                        )

            transitions.append(
                InstanceTransition(
                    image_id=image_id,
                    weather=weather,
                    scale_category=scale,
                    class_name=c_name,
                    transition_type="3_missed_to_recovered",
                    pretrained_conf=best_pre_conf,
                    adapted_conf=float(filtered_adapt[a_idx]["score"]),
                    gt_box=g_box,
                    pretrained_box=None,
                    adapted_box=filtered_adapt[a_idx]["box_xyxy"],
                    iou_pre_gt=None,
                    iou_adapt_gt=a_iou,
                )
            )

    # 2. False Positive transitions (States 5, 6, 7)
    pre_fp_indices = [i for i in range(len(filtered_pre)) if i not in pre_matched_preds]
    adapt_fp_indices = [
        i for i in range(len(filtered_adapt))
        if i not in adapt_matched_preds and i not in [t.adapted_box for t in transitions if t.transition_type == "4_correct_to_wrong_class"]
    ]

    # Match FPs between pretrained and adapted to see if hallucination is retained
    matched_pre_fps = set()
    matched_adapt_fps = set()

    for p_idx in pre_fp_indices:
        p_p = filtered_pre[p_idx]
        best_a_idx = None
        best_iou = 0.0
        for a_idx in adapt_fp_indices:
            if a_idx in matched_adapt_fps:
                continue
            a_p = filtered_adapt[a_idx]
            iou_val = iou_xyxy(p_p["box_xyxy"], a_p["box_xyxy"])
            if iou_val >= iou_thresh and iou_val > best_iou:
                best_iou = iou_val
                best_a_idx = a_idx

        if best_a_idx is not None:
            matched_pre_fps.add(p_idx)
            matched_adapt_fps.add(best_a_idx)
            a_p = filtered_adapt[best_a_idx]
            transitions.append(
                InstanceTransition(
                    image_id=image_id,
                    weather=weather,
                    scale_category=get_scale_category(p_p["box_xyxy"]),
                    class_name=p_p["common_label"],
                    transition_type="7_retained_false_positive",
                    pretrained_conf=float(p_p["score"]),
                    adapted_conf=float(a_p["score"]),
                    gt_box=None,
                    pretrained_box=p_p["box_xyxy"],
                    adapted_box=a_p["box_xyxy"],
                    iou_pre_gt=None,
                    iou_adapt_gt=None,
                )
            )
        else:
            # Pretrained FP was corrected / suppressed in adapted
            transitions.append(
                InstanceTransition(
                    image_id=image_id,
                    weather=weather,
                    scale_category=get_scale_category(p_p["box_xyxy"]),
                    class_name=p_p["common_label"],
                    transition_type="6_fp_to_corrected",
                    pretrained_conf=float(p_p["score"]),
                    adapted_conf=None,
                    gt_box=None,
                    pretrained_box=p_p["box_xyxy"],
                    adapted_box=None,
                    iou_pre_gt=None,
                    iou_adapt_gt=None,
                )
            )

    # Remaining adapted FPs are newly introduced / additional FPs
    for a_idx in adapt_fp_indices:
        if a_idx in matched_adapt_fps:
            continue
        a_p = filtered_adapt[a_idx]
        transitions.append(
            InstanceTransition(
                image_id=image_id,
                weather=weather,
                scale_category=get_scale_category(a_p["box_xyxy"]),
                class_name=a_p["common_label"],
                transition_type="5_additional_false_positive",
                pretrained_conf=None,
                adapted_conf=float(a_p["score"]),
                gt_box=None,
                pretrained_box=None,
                adapted_box=a_p["box_xyxy"],
                iou_pre_gt=None,
                iou_adapt_gt=None,
            )
        )

    return transitions


def summarize_transitions(transitions: List[InstanceTransition]) -> Dict[str, Any]:
    """Computes comprehensive aggregations across states, classes, scales, and weathers."""
    counts = defaultdict(int)
    by_class = defaultdict(lambda: defaultdict(int))
    by_scale = defaultdict(lambda: defaultdict(int))
    by_weather = defaultdict(lambda: defaultdict(int))

    conf_pre_by_state = defaultdict(list)
    conf_adapt_by_state = defaultdict(list)

    for t in transitions:
        st = t.transition_type
        counts[st] += 1
        by_class[t.class_name][st] += 1
        by_scale[t.scale_category][st] += 1
        by_weather[t.weather][st] += 1

        if t.pretrained_conf is not None:
            conf_pre_by_state[st].append(t.pretrained_conf)
        if t.adapted_conf is not None:
            conf_adapt_by_state[st].append(t.adapted_conf)

    total_instances = len(transitions)
    percentages = {k: (v / total_instances * 100.0) if total_instances > 0 else 0.0 for k, v in counts.items()}

    # Confidence statistics
    confidence_stats = {}
    all_states = [
        "1_correct_to_correct",
        "2_correct_to_missed",
        "3_missed_to_recovered",
        "4_correct_to_wrong_class",
        "5_additional_false_positive",
        "6_fp_to_corrected",
        "7_retained_false_positive",
    ]
    for st in all_states:
        pre_scores = conf_pre_by_state.get(st, [])
        adapt_scores = conf_adapt_by_state.get(st, [])
        confidence_stats[st] = {
            "pretrained": {
                "count": len(pre_scores),
                "mean": float(np.mean(pre_scores)) if pre_scores else None,
                "median": float(np.median(pre_scores)) if pre_scores else None,
                "std": float(np.std(pre_scores)) if pre_scores else None,
            },
            "adapted": {
                "count": len(adapt_scores),
                "mean": float(np.mean(adapt_scores)) if adapt_scores else None,
                "median": float(np.median(adapt_scores)) if adapt_scores else None,
                "std": float(np.std(adapt_scores)) if adapt_scores else None,
            },
        }

    return {
        "total_transitions": total_instances,
        "state_counts": dict(counts),
        "state_percentages": percentages,
        "by_class": {k: dict(v) for k, v in by_class.items()},
        "by_scale": {k: dict(v) for k, v in by_scale.items()},
        "by_weather": {k: dict(v) for k, v in by_weather.items()},
        "confidence_stats": confidence_stats,
    }
