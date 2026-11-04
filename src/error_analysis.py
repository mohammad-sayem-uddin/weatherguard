"""WeatherGuard Error Taxonomy and Per-Condition Analysis Framework.

Provides structured, rigorous error categorization for object detection outputs,
stratified across scale, weather condition, object class, and spatial contrast.
Supports research evaluation across Missions 1-17 without data leakage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class CategorizedPrediction:
    pred_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    category: str  # TP, Localization, Class_confusion, Duplicate, FP_background
    matched_gt_id: Optional[int] = None
    matched_iou: float = 0.0
    scale: str = "medium"
    contrast: Optional[float] = None


@dataclass
class CategorizedGroundTruth:
    gt_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    category: str  # Detected, Missed_localization, Missed_class, Missed_unproposed
    matched_pred_id: Optional[int] = None
    matched_iou: float = 0.0
    scale: str = "medium"
    contrast: Optional[float] = None


def compute_box_iou(boxA: Sequence[float], boxB: Sequence[float]) -> float:
    """Computes IoU between two (x1, y1, x2, y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0.0:
        return 0.0

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def get_scale_bucket(width: float, height: float) -> str:
    """Categorizes box into canonical scale buckets based on max dimension."""
    max_d = max(width, height)
    if max_d < 16.0:
        return "ultra_small"
    if max_d < 32.0:
        return "small"
    if max_d < 96.0:
        return "medium"
    return "large"


def extract_box_contrast(
    image: Optional[np.ndarray],
    box: Sequence[float],
) -> Optional[float]:
    """Computes RMS contrast (standard deviation of pixel intensities) inside bounding box."""
    if image is None:
        return None
    try:
        h, w = image.shape[:2]
        x1 = max(0, min(int(round(box[0])), w - 1))
        y1 = max(0, min(int(round(box[1])), h - 1))
        x2 = max(x1 + 1, min(int(round(box[2])), w))
        y2 = max(y1 + 1, min(int(round(box[3])), h))

        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        if patch.ndim == 3:
            # Luminance: 0.299 R + 0.587 G + 0.114 B
            lum = 0.299 * patch[:, :, 2] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 0]
        else:
            lum = patch.astype(np.float32)
        return float(np.std(lum))
    except Exception:
        return None


def parse_box_coords(box_obj: Any) -> Tuple[float, float, float, float]:
    """Helper to extract (x1, y1, x2, y2) from dataclass, dict, or tuple/list."""
    if hasattr(box_obj, "x1") and hasattr(box_obj, "y1"):
        return (float(box_obj.x1), float(box_obj.y1), float(box_obj.x2), float(box_obj.y2))
    if isinstance(box_obj, dict):
        if "box_xyxy" in box_obj:
            b = box_obj["box_xyxy"]
            return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        if "x1" in box_obj:
            return (float(box_obj["x1"]), float(box_obj["y1"]), float(box_obj["x2"]), float(box_obj["y2"]))
    if isinstance(box_obj, (list, tuple)) and len(box_obj) >= 4:
        return (float(box_obj[0]), float(box_obj[1]), float(box_obj[2]), float(box_obj[3]))
    raise ValueError(f"Unable to parse box coordinates from {box_obj}")


def parse_class_name(box_obj: Any) -> str:
    """Helper to extract class name."""
    if hasattr(box_obj, "class_name"):
        return str(box_obj.class_name).lower()
    if isinstance(box_obj, dict):
        if "common_label" in box_obj and box_obj["common_label"]:
            return str(box_obj["common_label"]).lower()
        if "class_name" in box_obj:
            return str(box_obj["class_name"]).lower()
        if "label" in box_obj:
            return str(box_obj["label"]).lower()
    return "unknown"


def parse_confidence(box_obj: Any) -> float:
    """Helper to extract confidence score."""
    if hasattr(box_obj, "confidence"):
        return float(box_obj.confidence)
    if isinstance(box_obj, dict):
        if "score" in box_obj:
            return float(box_obj["score"])
        if "confidence" in box_obj:
            return float(box_obj["confidence"])
    return 1.0


def categorize_image_errors(
    predictions: Sequence[Any],
    ground_truths: Sequence[Any],
    iou_thresh: float = 0.50,
    loc_thresh: float = 0.10,
    image: Optional[np.ndarray] = None,
) -> Tuple[List[CategorizedPrediction], List[CategorizedGroundTruth]]:
    """Classifies predictions and ground truth in one image using fine-grained taxonomy."""
    # Parse GTs
    parsed_gts: List[Tuple[int, str, Tuple[float, float, float, float], str, Optional[float]]] = []
    for gi, g in enumerate(ground_truths):
        b = parse_box_coords(g)
        cls = parse_class_name(g)
        scale = get_scale_bucket(max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1]))
        contrast = extract_box_contrast(image, b)
        parsed_gts.append((gi, cls, b, scale, contrast))

    # Parse and sort predictions descending by confidence
    parsed_preds: List[Tuple[int, str, float, Tuple[float, float, float, float], str, Optional[float]]] = []
    for pi, p in enumerate(predictions):
        b = parse_box_coords(p)
        cls = parse_class_name(p)
        conf = parse_confidence(p)
        scale = get_scale_bucket(max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1]))
        contrast = extract_box_contrast(image, b)
        parsed_preds.append((pi, cls, conf, b, scale, contrast))
    parsed_preds.sort(key=lambda x: x[2], reverse=True)

    matched_gt_for_tp = set()
    cat_preds: List[CategorizedPrediction] = []

    # Map GT matches
    gt_best_match: Dict[int, Tuple[str, int, float]] = {}  # gt_id -> (status, pred_id, iou)

    for orig_idx, p_cls, p_conf, p_box, p_scale, p_contrast in parsed_preds:
        # Step 1: Look for highest IoU matching GT with same class
        best_same_iou = 0.0
        best_same_gi = -1
        for gi, g_cls, g_box, _, _ in parsed_gts:
            if g_cls == p_cls:
                iou = compute_box_iou(p_box, g_box)
                if iou > best_same_iou:
                    best_same_iou = iou
                    best_same_gi = gi

        # Step 2: Look for highest IoU matching GT of any class
        best_any_iou = 0.0
        best_any_gi = -1
        for gi, _, g_box, _, _ in parsed_gts:
            iou = compute_box_iou(p_box, g_box)
            if iou > best_any_iou:
                best_any_iou = iou
                best_any_gi = gi

        # Classify prediction
        if best_same_iou >= iou_thresh and best_same_gi >= 0:
            if best_same_gi not in matched_gt_for_tp:
                matched_gt_for_tp.add(best_same_gi)
                cat_preds.append(CategorizedPrediction(
                    pred_id=orig_idx, class_name=p_cls, confidence=p_conf, bbox=p_box,
                    category="TP", matched_gt_id=best_same_gi, matched_iou=best_same_iou,
                    scale=p_scale, contrast=p_contrast,
                ))
                gt_best_match[best_same_gi] = ("Detected", orig_idx, best_same_iou)
            else:
                cat_preds.append(CategorizedPrediction(
                    pred_id=orig_idx, class_name=p_cls, confidence=p_conf, bbox=p_box,
                    category="Duplicate", matched_gt_id=best_same_gi, matched_iou=best_same_iou,
                    scale=p_scale, contrast=p_contrast,
                ))
        elif best_any_iou >= iou_thresh and best_any_gi >= 0:
            cat_preds.append(CategorizedPrediction(
                pred_id=orig_idx, class_name=p_cls, confidence=p_conf, bbox=p_box,
                category="Class_confusion", matched_gt_id=best_any_gi, matched_iou=best_any_iou,
                scale=p_scale, contrast=p_contrast,
            ))
            if best_any_gi not in gt_best_match:
                gt_best_match[best_any_gi] = ("Missed_class", orig_idx, best_any_iou)
        elif best_any_iou >= loc_thresh and best_any_gi >= 0:
            cat_preds.append(CategorizedPrediction(
                pred_id=orig_idx, class_name=p_cls, confidence=p_conf, bbox=p_box,
                category="Localization", matched_gt_id=best_any_gi, matched_iou=best_any_iou,
                scale=p_scale, contrast=p_contrast,
            ))
            if best_any_gi not in gt_best_match:
                gt_best_match[best_any_gi] = ("Missed_localization", orig_idx, best_any_iou)
        else:
            cat_preds.append(CategorizedPrediction(
                pred_id=orig_idx, class_name=p_cls, confidence=p_conf, bbox=p_box,
                category="FP_background", matched_gt_id=None, matched_iou=best_any_iou,
                scale=p_scale, contrast=p_contrast,
            ))

    # Classify Ground Truths
    cat_gts: List[CategorizedGroundTruth] = []
    for gi, g_cls, g_box, g_scale, g_contrast in parsed_gts:
        if gi in gt_best_match:
            status, pred_id, match_iou = gt_best_match[gi]
            cat_gts.append(CategorizedGroundTruth(
                gt_id=gi, class_name=g_cls, bbox=g_box, category=status,
                matched_pred_id=pred_id, matched_iou=match_iou, scale=g_scale, contrast=g_contrast,
            ))
        else:
            cat_gts.append(CategorizedGroundTruth(
                gt_id=gi, class_name=g_cls, bbox=g_box, category="Missed_unproposed",
                matched_pred_id=None, matched_iou=0.0, scale=g_scale, contrast=g_contrast,
            ))

    return cat_preds, cat_gts


class ErrorTaxonomyAnalyzer:
    """Aggregates and summarizes detection errors across datasets, scales, and weather conditions."""

    def __init__(self, iou_thresh: float = 0.50, loc_thresh: float = 0.10) -> None:
        self.iou_thresh = iou_thresh
        self.loc_thresh = loc_thresh
        self.records: List[Dict[str, Any]] = []

    def add_image_result(
        self,
        image_id: str,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
        condition: str = "all",
        image: Optional[np.ndarray] = None,
    ) -> Tuple[List[CategorizedPrediction], List[CategorizedGroundTruth]]:
        """Evaluates an image and logs its categorized predictions and ground truths."""
        preds, gts = categorize_image_errors(
            predictions=predictions,
            ground_truths=ground_truths,
            iou_thresh=self.iou_thresh,
            loc_thresh=self.loc_thresh,
            image=image,
        )
        self.records.append({
            "image_id": image_id,
            "condition": condition.lower(),
            "predictions": preds,
            "ground_truths": gts,
        })
        return preds, gts

    def compute_summary(self) -> Dict[str, Any]:
        """Generates comprehensive aggregated metrics stratified by category, scale, and condition."""
        pred_cat_counts: Dict[str, int] = {
            "TP": 0, "Localization": 0, "Class_confusion": 0, "Duplicate": 0, "FP_background": 0,
        }
        gt_cat_counts: Dict[str, int] = {
            "Detected": 0, "Missed_localization": 0, "Missed_class": 0, "Missed_unproposed": 0,
        }

        by_condition: Dict[str, Dict[str, Any]] = {}
        by_scale: Dict[str, Dict[str, Any]] = {
            "ultra_small": {"gt_total": 0, "detected": 0, "missed": 0, "unproposed": 0},
            "small": {"gt_total": 0, "detected": 0, "missed": 0, "unproposed": 0},
            "medium": {"gt_total": 0, "detected": 0, "missed": 0, "unproposed": 0},
            "large": {"gt_total": 0, "detected": 0, "missed": 0, "unproposed": 0},
        }

        total_preds = 0
        total_gts = 0

        for r in self.records:
            cond = r["condition"]
            if cond not in by_condition:
                by_condition[cond] = {
                    "images": 0, "total_preds": 0, "total_gts": 0,
                    "pred_categories": {"TP": 0, "Localization": 0, "Class_confusion": 0, "Duplicate": 0, "FP_background": 0},
                    "gt_categories": {"Detected": 0, "Missed_localization": 0, "Missed_class": 0, "Missed_unproposed": 0},
                    "scale_breakdown": {
                        "ultra_small": {"total": 0, "detected": 0},
                        "small": {"total": 0, "detected": 0},
                        "medium": {"total": 0, "detected": 0},
                        "large": {"total": 0, "detected": 0},
                    },
                }

            by_condition[cond]["images"] += 1

            for p in r["predictions"]:
                total_preds += 1
                cat = p.category
                pred_cat_counts[cat] = pred_cat_counts.get(cat, 0) + 1
                by_condition[cond]["total_preds"] += 1
                by_condition[cond]["pred_categories"][cat] = by_condition[cond]["pred_categories"].get(cat, 0) + 1

            for g in r["ground_truths"]:
                total_gts += 1
                cat = g.category
                sc = g.scale
                gt_cat_counts[cat] = gt_cat_counts.get(cat, 0) + 1
                by_condition[cond]["total_gts"] += 1
                by_condition[cond]["gt_categories"][cat] = by_condition[cond]["gt_categories"].get(cat, 0) + 1

                if sc in by_scale:
                    by_scale[sc]["gt_total"] += 1
                    if cat == "Detected":
                        by_scale[sc]["detected"] += 1
                    else:
                        by_scale[sc]["missed"] += 1
                        if cat == "Missed_unproposed":
                            by_scale[sc]["unproposed"] += 1

                if sc in by_condition[cond]["scale_breakdown"]:
                    by_condition[cond]["scale_breakdown"][sc]["total"] += 1
                    if cat == "Detected":
                        by_condition[cond]["scale_breakdown"][sc]["detected"] += 1

        # Calculate high-level precision, recall, and rates
        tp = pred_cat_counts["TP"]
        fp = total_preds - tp
        fn = total_gts - tp
        precision = round(tp / max(1, total_preds), 4) if total_preds > 0 else 0.0
        recall = round(tp / max(1, total_gts), 4) if total_gts > 0 else 0.0
        f1 = round(2 * precision * recall / max(1e-6, precision + recall), 4) if (precision + recall) > 0 else 0.0

        # Scale recall summaries
        scale_recalls = {}
        for sc, stats in by_scale.items():
            tot = stats["gt_total"]
            scale_recalls[sc] = {
                "gt_count": tot,
                "recall": round(stats["detected"] / max(1, tot), 4) if tot > 0 else 0.0,
                "unproposed_fraction": round(stats["unproposed"] / max(1, stats["missed"]), 4) if stats["missed"] > 0 else 0.0,
            }

        return {
            "total_images": len(self.records),
            "total_predictions": total_preds,
            "total_ground_truths": total_gts,
            "overall_metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            },
            "prediction_taxonomy": {
                cat: {
                    "count": cnt,
                    "fraction": round(cnt / max(1, total_preds), 4) if total_preds > 0 else 0.0,
                }
                for cat, cnt in pred_cat_counts.items()
            },
            "ground_truth_taxonomy": {
                cat: {
                    "count": cnt,
                    "fraction": round(cnt / max(1, total_gts), 4) if total_gts > 0 else 0.0,
                }
                for cat, cnt in gt_cat_counts.items()
            },
            "scale_analysis": scale_recalls,
            "condition_analysis": by_condition,
        }

    def to_markdown(self) -> str:
        """Generates markdown tables summarizing the error taxonomy."""
        summary = self.compute_summary()
        om = summary["overall_metrics"]
        pt = summary["prediction_taxonomy"]
        gt = summary["ground_truth_taxonomy"]
        sc = summary["scale_analysis"]
        cond = summary["condition_analysis"]

        lines = [
            "# WeatherGuard Error Taxonomy Analysis",
            "",
            f"**Total Images:** {summary['total_images']} | **Total Predictions:** {summary['total_predictions']} | **Total Ground Truths:** {summary['total_ground_truths']}",
            f"**Precision:** {om['precision']:.4f} | **Recall:** {om['recall']:.4f} | **F1 Score:** {om['f1']:.4f}",
            "",
            "## 1. Prediction Categorization",
            "| Error Category | Count | Fraction of Predictions | Description |",
            "| :--- | :--- | :--- | :--- |",
            f"| **TP** | {pt['TP']['count']} | {pt['TP']['fraction']:.2%} | Correct class & IoU ≥ 0.50 |",
            f"| **Localization** | {pt['Localization']['count']} | {pt['Localization']['fraction']:.2%} | 0.10 ≤ IoU < 0.50 (imprecise bounding box) |",
            f"| **Class Confusion** | {pt['Class_confusion']['count']} | {pt['Class_confusion']['fraction']:.2%} | IoU ≥ 0.50 with incorrect class label |",
            f"| **Duplicate** | {pt['Duplicate']['count']} | {pt['Duplicate']['fraction']:.2%} | Duplicate detection for already-matched GT |",
            f"| **FP Background** | {pt['FP_background']['count']} | {pt['FP_background']['fraction']:.2%} | IoU < 0.10 (pure clutter / hallucination) |",
            "",
            "## 2. Ground Truth Miss Categorization",
            "| GT Status | Count | Fraction of GTs | Interpretation |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Detected** | {gt['Detected']['count']} | {gt['Detected']['fraction']:.2%} | Successfully detected at IoU ≥ 0.50 |",
            f"| **Missed Localization** | {gt['Missed_localization']['count']} | {gt['Missed_localization']['fraction']:.2%} | Only proposed with poor spatial alignment |",
            f"| **Missed Class** | {gt['Missed_class']['count']} | {gt['Missed_class']['fraction']:.2%} | Detected spatially but misclassified |",
            f"| **Missed Unproposed** | {gt['Missed_unproposed']['count']} | {gt['Missed_unproposed']['fraction']:.2%} | Completely unproposed (candidate generator bottleneck) |",
            "",
            "## 3. Scale-Stratified Recall & Candidate Bottlenecks",
            "| Scale Bucket | GT Count | Recall | Unproposed Miss Fraction |",
            "| :--- | :--- | :--- | :--- |",
        ]

        for s_name in ["ultra_small", "small", "medium", "large"]:
            s_data = sc.get(s_name, {"gt_count": 0, "recall": 0.0, "unproposed_fraction": 0.0})
            lines.append(f"| **{s_name}** | {s_data['gt_count']} | {s_data['recall']:.2%} | {s_data['unproposed_fraction']:.2%} |")

        lines.extend([
            "",
            "## 4. Per-Condition Performance Breakdown",
            "| Weather Condition | Images | GTs | Recall | TP Preds | Background FPs |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for c_name, c_data in sorted(cond.items()):
            gts_cnt = c_data["total_gts"]
            det_cnt = c_data["gt_categories"].get("Detected", 0)
            rec = round(det_cnt / max(1, gts_cnt), 4) if gts_cnt > 0 else 0.0
            tp_cnt = c_data["pred_categories"].get("TP", 0)
            fp_bg = c_data["pred_categories"].get("FP_background", 0)
            lines.append(f"| **{c_name.capitalize()}** | {c_data['images']} | {gts_cnt} | {rec:.2%} | {tp_cnt} | {fp_bg} |")

        return "\n".join(lines)

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Saves JSON summary report."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.compute_summary(), f, indent=2)
