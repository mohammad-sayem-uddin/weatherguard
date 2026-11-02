"""WeatherGuard Mission 6 — Confidence Shift & Decision-Boundary Analysis.

Provides mathematical tools to analyze detector calibration and threshold sensitivity:
  1. Threshold sweep (0.05 to 0.90) for precision, recall, and F1.
  2. Expected Calibration Error (ECE) with 10 uniform bins on detection candidates.
  3. Brier score for matched candidate confidence.
  4. Error mode decomposition (Precision Drop, Recall Drop, Miscalibration, Confusion, Localization).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from src.external_dataset import CANONICAL_CLASSES
from src.metrics import greedy_match, iou_xyxy


def compute_ece_and_brier(
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Computes Expected Calibration Error (ECE) and Brier Score.
    
    Args:
        confidences: Array of predicted confidence scores in [0, 1].
        labels: Binary correctness array (1 for TP, 0 for FP).
        n_bins: Number of uniform bins (default 10).
    """
    if len(confidences) == 0:
        return {"ece": 0.0, "brier_score": 0.0, "bins": []}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []
    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > low) & (confidences <= high) if i > 0 else (confidences >= low) & (confidences <= high)
        count = int(np.sum(mask))

        if count > 0:
            bin_conf = float(np.mean(confidences[mask]))
            bin_acc = float(np.mean(labels[mask]))
            bin_err = abs(bin_acc - bin_conf)
            ece += (count / total_samples) * bin_err
            bin_data.append({
                "bin_idx": i,
                "range": [float(low), float(high)],
                "count": count,
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "calibration_error": bin_err,
            })
        else:
            bin_data.append({
                "bin_idx": i,
                "range": [float(low), float(high)],
                "count": 0,
                "confidence": float((low + high) / 2.0),
                "accuracy": 0.0,
                "calibration_error": 0.0,
            })

    brier = float(np.mean((confidences - labels) ** 2))

    return {
        "ece": float(ece),
        "brier_score": brier,
        "n_samples": total_samples,
        "bins": bin_data,
    }


def sweep_confidence_thresholds(
    eval_records: List[Dict[str, Any]],
    thresholds: List[float] = [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
        0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
    ],
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Sweeps confidence thresholds and computes precision, recall, and F1."""
    curve = []
    best_f1 = -1.0
    best_thresh = 0.20

    for th in thresholds:
        tp_tot = 0
        fp_tot = 0
        gt_tot = 0

        for r in eval_records:
            gt_objs = [g for g in r.get("ground_truth", []) if g.get("class_name") in CANONICAL_CLASSES]
            gt_tot += len(gt_objs)
            preds = [
                p for p in r.get("predictions", [])
                if p.get("score", 0.0) >= th and p.get("common_label") in CANONICAL_CLASSES
            ]

            matched_gt = set()
            for c in CANONICAL_CLASSES:
                c_p = [(i, p) for i, p in enumerate(preds) if p["common_label"] == c]
                c_g = [(j, g) for j, g in enumerate(gt_objs) if g["class_name"] == c]
                if not c_p or not c_g:
                    continue
                matches = greedy_match(
                    [p["box_xyxy"] for _, p in c_p],
                    [p["score"] for _, p in c_p],
                    [g["box_xyxy"] for _, g in c_g],
                    iou_threshold=iou_thresh,
                )
                for _, sub_g, _ in matches:
                    matched_gt.add(c_g[sub_g][0])

            tp = len(matched_gt)
            fp = len(preds) - tp
            tp_tot += tp
            fp_tot += fp

        prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
        rec = tp_tot / gt_tot if gt_tot > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th

        curve.append({
            "threshold": float(th),
            "tp": tp_tot,
            "fp": fp_tot,
            "fn": gt_tot - tp_tot,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        })

    return {
        "best_threshold": float(best_thresh),
        "best_f1": float(best_f1),
        "threshold_curve": curve,
    }
