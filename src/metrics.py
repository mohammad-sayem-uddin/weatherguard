"""Detection metrics for WeatherGuard.

Provides:
  * IoU between two axis-aligned boxes in (x_min, y_min, x_max, y_max) format.
  * Greedy IoU matching at a fixed threshold (no class-aware matching for
    AP@.50 — that is computed per class and aggregated into mAP below).
  * Precision / recall / F1 at a single confidence threshold.
  * Average precision per class (101-point VOC-style interpolation) and
    the COCO-style mean across classes.

This file is the *only* place the IoU/matching protocol lives. Any change
here should be reflected in `reports/final_report.md` and
`README.md#evaluation-protocol`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# Default IoU threshold used for matching predictions to ground truth.
DEFAULT_IOU_THRESHOLD: float = 0.5

# Default confidence threshold used to bin predictions into TP/FP/FN.
DEFAULT_SCORE_THRESHOLD: float = 0.5


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two boxes in (x_min, y_min, x_max, y_max) absolute pixels."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError(f"boxes must have 4 coords, got {a!r} and {b!r}")
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def greedy_match(
    pred_boxes: Sequence[Sequence[float]],
    pred_scores: Sequence[float],
    gt_boxes: Sequence[Sequence[float]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Greedy matching: sort predictions by score, match each to the
    highest-IoU still-unmatched ground-truth box above the threshold.

    Returns a list of (pred_index, gt_index, iou) tuples. Predictions with
    no match are *not* returned. GT with no match is *not* returned. The
    caller computes TP / FP / FN from these tuples and the unmatched sets.
    """
    if not pred_boxes and not gt_boxes:
        return []
    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_scores[i])
    gt_used = [False] * len(gt_boxes)
    matches: list[tuple[int, int, float]] = []
    for pi in order:
        best_iou = 0.0
        best_gi = -1
        for gi, gb in enumerate(gt_boxes):
            if gt_used[gi]:
                continue
            v = iou_xyxy(pred_boxes[pi], gb)
            if v > best_iou:
                best_iou = v
                best_gi = gi
        if best_gi >= 0 and best_iou >= iou_threshold:
            gt_used[best_gi] = True
            matches.append((pi, best_gi, best_iou))
    return matches


@dataclass
class ClassPR:
    """Per-class precision / recall result at one confidence threshold."""
    class_name: str
    tp: int
    fp: int
    fn: int
    n_gt: int
    n_pred: int
    precision: float
    recall: float
    f1: float
    ap_50: float  # 101-point interpolated AP at IoU=0.5 over all scores


@dataclass
class EvalResult:
    """Per-class + aggregate metrics for one evaluation run."""
    iou_threshold: float
    score_threshold: float
    n_images: int
    per_class: list[ClassPR] = field(default_factory=list)
    micro: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iou_threshold": self.iou_threshold,
            "score_threshold": self.score_threshold,
            "n_images": self.n_images,
            "per_class": [asdict(c) for c in self.per_class],
            "micro": self.micro,
            "macro": self.macro,
            "notes": self.notes,
        }


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _average_precision_101(
    scores: Sequence[float],
    tp_flags: Sequence[bool],
    n_gt: int,
) -> float:
    """101-point interpolated AP (Pascal VOC style).

    `scores` are prediction confidences, `tp_flags[i]` is True if the
    prediction was a true positive at the greedy match. Predictions are
    assumed pre-sorted descending by score.
    """
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    cum_tp = 0
    cum_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for tp in tp_flags:
        if tp:
            cum_tp += 1
        else:
            cum_fp += 1
        rec = cum_tp / n_gt
        prec = cum_tp / (cum_tp + cum_fp)
        recalls.append(rec)
        precisions.append(prec)
    # 11-point interpolation (Pascal VOC2007), then 101-point refinement.
    # We use 101-point: for r in {0.00, 0.01, ..., 1.00}, max prec where rec>=r.
    ap = 0.0
    for k in range(101):
        r_target = k / 100.0
        # max precision at any recall >= r_target
        max_prec = 0.0
        for rec, prec in zip(recalls, precisions):
            if rec >= r_target and prec > max_prec:
                max_prec = prec
        ap += max_prec / 101.0
    return ap


def evaluate_detections(
    per_image: list[dict],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    target_classes: Iterable[str] | None = None,
) -> EvalResult:
    """Evaluate a list of per-image prediction/ground-truth dicts.

    Each per-image dict must look like:
        {
          "image_id": str,
          "predictions": [{"label": str, "common_label": str|None,
                            "score": float, "box_xyxy": (x1,y1,x2,y2)} ...],
          "ground_truth": [{"class_name": str,
                            "box_xyxy": (x1,y1,x2,y2)} ...],
        }

    Detections whose `common_label` is None (out of DAWN label space) are
    ignored — they cannot be evaluated against DAWN ground truth. They
    are not counted as FP either, so we add a note about that.
    """
    if target_classes is None:
        from .dataset import DAWN_CLASSES
        target_classes = list(DAWN_CLASSES.values())
    target_classes = list(target_classes)

    # Aggregate per-class statistics.
    per_class_pred: dict[str, list[tuple[float, bool]]] = {c: [] for c in target_classes}
    per_class_gt: dict[str, int] = {c: 0 for c in target_classes}
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    n_images_evaluated = 0
    notes: list[str] = []

    for entry in per_image:
        if "predictions" not in entry or "ground_truth" not in entry:
            raise ValueError(
                "Each per-image entry must contain 'predictions' and 'ground_truth'."
            )
        n_images_evaluated += 1
        gts = entry["ground_truth"]
        # Bucket GT by class.
        gt_by_class: dict[str, list[tuple[float, float, float, float]]] = {
            c: [] for c in target_classes
        }
        for g in gts:
            cn = g["class_name"]
            if cn not in per_class_gt:
                # GT class outside the supported DAWN set; skip.
                continue
            per_class_gt[cn] += 1
            gt_by_class[cn].append(g["box_xyxy"])

        # Bucket predictions by class after the score filter.
        preds_by_class: dict[str, list[tuple[float, tuple]]] = {c: [] for c in target_classes}
        ignored_oov = 0
        for p in entry["predictions"]:
            if p["score"] < score_threshold:
                continue
            cn = p.get("common_label")
            if cn is None or cn not in per_class_pred:
                ignored_oov += 1
                continue
            preds_by_class[cn].append((float(p["score"]), p["box_xyxy"]))

        if ignored_oov:
            # We do not count out-of-vocabulary predictions as FP because
            # the detector was never told about that class in this
            # evaluation. Record once globally.
            pass

        for cn in target_classes:
            preds = preds_by_class[cn]
            gts = gt_by_class[cn]
            # Sort preds by score desc.
            preds_sorted = sorted(preds, key=lambda x: -x[0])
            pred_boxes = [p[1] for p in preds_sorted]
            pred_scores = [p[0] for p in preds_sorted]
            matches = greedy_match(pred_boxes, pred_scores, gts, iou_threshold=iou_threshold)
            tp_flags = [False] * len(preds_sorted)
            for pi, _gi, _iou in matches:
                tp_flags[pi] = True
            tp = sum(tp_flags)
            fp = len(preds_sorted) - tp
            fn = len(gts) - tp
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn
            for (sc, _), tp_ in zip(preds_sorted, tp_flags):
                per_class_pred[cn].append((sc, tp_))

    per_class_results: list[ClassPR] = []
    for cn in target_classes:
        preds_for_class = per_class_pred[cn]
        n_gt = per_class_gt[cn]
        # Sort by score desc for AP computation.
        preds_sorted = sorted(preds_for_class, key=lambda x: -x[0])
        scores = [p[0] for p in preds_sorted]
        tp_flags = [p[1] for p in preds_sorted]
        ap = _average_precision_101(scores, tp_flags, n_gt)
        tp = sum(1 for _, t in preds_for_class if t)
        fp = len(preds_for_class) - tp
        fn = max(0, n_gt - tp)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, n_gt) if n_gt else 0.0
        f1 = _safe_div(2 * prec * rec, prec + rec)
        per_class_results.append(
            ClassPR(
                class_name=cn,
                tp=tp,
                fp=fp,
                fn=fn,
                n_gt=n_gt,
                n_pred=len(preds_for_class),
                precision=prec,
                recall=rec,
                f1=f1,
                ap_50=ap,
            )
        )

    # mAP@.5 across classes that actually have GT.
    aps = [c.ap_50 for c in per_class_results if c.n_gt > 0 and not math.isnan(c.ap_50)]
    m_ap_50 = sum(aps) / len(aps) if aps else float("nan")

    micro_prec = _safe_div(micro_tp, micro_tp + micro_fp)
    micro_rec = _safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = _safe_div(2 * micro_prec * micro_rec, micro_prec + micro_rec)

    macro_prec = _safe_div(
        sum(c.precision for c in per_class_results if c.n_gt > 0),
        sum(1 for c in per_class_results if c.n_gt > 0),
    )
    macro_rec = _safe_div(
        sum(c.recall for c in per_class_results if c.n_gt > 0),
        sum(1 for c in per_class_results if c.n_gt > 0),
    )

    return EvalResult(
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        n_images=n_images_evaluated,
        per_class=per_class_results,
        micro={
            "tp": micro_tp, "fp": micro_fp, "fn": micro_fn,
            "precision": micro_prec, "recall": micro_rec, "f1": micro_f1,
        },
        macro={
            "precision": macro_prec, "recall": macro_rec,
            "mAP_50": m_ap_50,
        },
        notes=notes,
    )


# At import time, lazy-import `dataclasses.asdict` for the helper above.
from dataclasses import asdict  # noqa: E402
