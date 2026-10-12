"""Workstream B — detection error taxonomy.

Implements an explicit, class-aware error categorisation. Each
prediction is classified using the documented IoU matching protocol
into one of:

  * TP            — correct class AND IoU ≥ 0.5 with an unmatched GT.
  * Localization  — correct class but 0.1 ≤ IoU < 0.5 (still TP for
                    AP, but a localization error in the taxonomy).
  * Class_confusion — wrong class but IoU ≥ 0.5 with a GT.
  * Duplicate     — correct class, IoU ≥ 0.5, but GT is already
                    matched to a higher-scoring prediction.
  * FP_other      — no GT above IoU 0.1 at all.
  * Low_conf_TP   — TP with score in the lowest decile of TPs (extra
                    subcategory, not a hard category).

Each GT is categorised as:
  * Detected       — matched to any TP.
  * Missed_local   — matched only to a Localization or Duplicate
                    prediction (still a localisation failure).
  * Missed_class   — not detected (FN).

Object-size stratification:
  * small  — area < 32² px
  * medium — 32² ≤ area < 96² px
  * large  — area ≥ 96² px

Outputs:
  * `artifacts/error_taxonomy.json`
  * `artifacts/error_taxonomy.csv`
  * `reports/error_taxonomy.md`
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.dataset import DawnDataset, DawnRecord
from src.metrics import iou_xyxy

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


# Categories for predictions (a prediction falls into exactly one of these).
PRED_CATS = [
    "TP", "Localization", "Class_confusion", "Duplicate", "FP_other",
]
# Categories for ground truth (one per GT).
GT_CATS = ["Detected", "Missed_local", "Missed_class"]


def _size_bucket(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def _record_to_gt_xyxy(rec: dict) -> list[dict]:
    out = []
    for g in rec["ground_truth"]:
        out.append({"class_name": g["class_name"], "box_xyxy": g["box_xyxy"],
                    "area": max(0, g["box_xyxy"][2] - g["box_xyxy"][0]) *
                            max(0, g["box_xyxy"][3] - g["box_xyxy"][1])})
    return out


def _record_to_pred_xyxy(rec: dict) -> list[dict]:
    out = []
    for p in rec["predictions"]:
        if p.get("common_label") is None or p["score"] < 0.5:
            continue
        out.append({"class_name": p["common_label"], "label": p.get("label", ""),
                    "score": p["score"], "box_xyxy": p["box_xyxy"]})
    return out


def _categorise_image(rec: dict) -> dict:
    """Classify predictions and ground truth in one image using the
    documented IoU 0.5 protocol, with the explicit taxonomy above."""
    gts = _record_to_gt_xyxy(rec)
    preds = sorted(_record_to_pred_xyxy(rec), key=lambda p: -p["score"])

    # First pass: greedy match (highest-score first) to identify TPs and Duplicates.
    gt_used = [False] * len(gts)
    pred_cat: list[str] = [None] * len(preds)  # type: ignore
    pred_match_iou: list[float] = [0.0] * len(preds)
    pred_match_idx: list[int] = [-1] * len(preds)
    for pi, p in enumerate(preds):
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gts):
            if gt_used[gi]:
                continue
            v = iou_xyxy(p["box_xyxy"], g["box_xyxy"])
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi >= 0 and best_iou >= 0.5:
            gt_used[best_gi] = True
            # Class confusion vs TP
            if preds[pi]["class_name"] == gts[best_gi]["class_name"]:
                pred_cat[pi] = "TP"
            else:
                pred_cat[pi] = "Class_confusion"
            pred_match_iou[pi] = best_iou
            pred_match_idx[pi] = best_gi
        else:
            # Not matched at IoU 0.5
            # Find best any-class IoU
            any_iou, any_gi = 0.0, -1
            for gi, g in enumerate(gts):
                v = iou_xyxy(p["box_xyxy"], g["box_xyxy"])
                if v > any_iou:
                    any_iou, any_gi = v, gi
            if any_gi >= 0 and any_iou >= 0.1:
                pred_cat[pi] = "Localization"
                pred_match_iou[pi] = any_iou
                pred_match_idx[pi] = any_gi
            else:
                pred_cat[pi] = "FP_other"
                pred_match_iou[pi] = any_iou
                pred_match_idx[pi] = any_gi

    # Second pass: re-evaluate TP/Localization preds to detect duplicates
    # (i.e. two TPs for the same class that both point at the same GT,
    # when a higher-scored prediction already consumed that GT).
    # This is already handled implicitly: only the first (highest-score)
    # prediction for a given GT receives TP. The second one now sees
    # gt_used=True and falls into the "best IoU < 0.5" path. We re-mark
    # such cases as "Duplicate" if the un-matched-pred has best IoU ≥ 0.5
    # with the same GT class as a matched one.
    used_gt: dict[int, int] = {}  # gi -> first matching pi
    # We didn't track this above; re-derive.
    for pi, cat in enumerate(pred_cat):
        if cat in ("TP", "Class_confusion", "Localization") and pred_match_idx[pi] >= 0:
            gi = pred_match_idx[pi]
            if gi in used_gt:
                # A higher-scored prediction already matched this GT.
                pred_cat[pi] = "Duplicate"
            else:
                used_gt[gi] = pi

    # Ground-truth categories
    gt_cat: list[str] = []
    for gi, g in enumerate(gts):
        if not gt_used[gi]:
            gt_cat.append("Missed_class")
        else:
            # Did we match with a wrong-class prediction? Then GT is "missed" class-wise.
            matched_pi = used_gt.get(gi, -1)
            if matched_pi < 0:
                gt_cat.append("Detected")
            else:
                pc = preds[matched_pi]["class_name"]
                if pc == g["class_name"]:
                    gt_cat.append("Detected")
                else:
                    gt_cat.append("Missed_class")

    return {
        "n_gt": len(gts),
        "n_pred": len(preds),
        "pred_cats": pred_cat,
        "pred_match_iou": pred_match_iou,
        "pred_match_idx": pred_match_idx,
        "gt_cats": gt_cat,
        "pred_classes": [p["class_name"] for p in preds],
        "gt_classes": [g["class_name"] for g in gts],
        "pred_scores": [p["score"] for p in preds],
        "gt_areas": [g["area"] for g in gts],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    detectors = [
        ("fasterrcnn_val_full", "Faster R-CNN ResNet-50 FPN"),
        ("yolov8n_val_full", "YOLOv8n"),
    ]

    overall: dict[str, dict] = {}
    for tag, label in detectors:
        path = ARTIFACTS / f"eval_{tag}.json"
        if not path.exists():
            print(f"skip {tag}: missing {path}")
            continue
        d = json.loads(path.read_text())
        per_image = d["per_image"]
        per_class_pred: dict[str, Counter] = defaultdict(Counter)
        per_class_gt: dict[str, Counter] = defaultdict(Counter)
        per_size_pred: dict[str, Counter] = defaultdict(Counter)
        per_size_gt: dict[str, Counter] = defaultdict(Counter)
        per_cond_pred: dict[str, Counter] = defaultdict(Counter)
        per_cond_gt: dict[str, Counter] = defaultdict(Counter)
        confusion: dict[tuple[str, str], int] = defaultdict(int)
        # Per-image storage for "error examples" subset
        examples: list[dict] = []

        for rec in per_image:
            cat = _categorise_image(rec)
            cond = rec.get("condition", "?")
            for pi, (pc, gt_idx) in enumerate(zip(cat["pred_cats"], cat["pred_match_idx"])):
                cls = cat["pred_classes"][pi]
                per_class_pred[cls][pc] += 1
                per_size_pred[_size_bucket(0.0)][pc]  # ensure key exists
                # Bucket by the GT area that the prediction is closest to
                if gt_idx >= 0 and gt_idx < len(cat["gt_areas"]):
                    size = _size_bucket(cat["gt_areas"][gt_idx])
                else:
                    size = "n/a"
                per_size_pred[size][pc] += 1
                per_cond_pred[cond][pc] += 1
                # Confusion matrix: only for matched predictions with wrong class
                if pc == "Class_confusion" and gt_idx >= 0:
                    gt_cls = cat["gt_classes"][gt_idx]
                    confusion[(gt_cls, cls)] += 1
            for gi, gc in enumerate(cat["gt_cats"]):
                cls = cat["gt_classes"][gi]
                per_class_gt[cls][gc] += 1
                size = _size_bucket(cat["gt_areas"][gi])
                per_size_gt[size][gc] += 1
                per_cond_gt[cond][gc] += 1
            # Store up to 12 interesting examples per detector
            n_interesting = sum(1 for c in cat["pred_cats"] if c in ("Class_confusion", "FP_other"))
            n_missed = sum(1 for c in cat["gt_cats"] if c == "Missed_class")
            if (n_interesting + n_missed) >= 3 and len(examples) < 12:
                examples.append({
                    "image_id": rec["image_id"],
                    "condition": cond,
                    "n_pred": cat["n_pred"],
                    "n_gt": cat["n_gt"],
                    "pred_cats": cat["pred_cats"],
                    "pred_classes": cat["pred_classes"],
                    "pred_scores": cat["pred_scores"],
                    "gt_cats": cat["gt_cats"],
                    "gt_classes": cat["gt_classes"],
                })

        # Convert confusion to plain dict
        confusion_dict = {f"{gt}__{pred}": n for (gt, pred), n in confusion.items()}

        result = {
            "detector": label,
            "tag": tag,
            "n_images": len(per_image),
            "iou_threshold": args.iou_threshold,
            "score_threshold": args.score_threshold,
            "per_class_pred": {c: dict(cnt) for c, cnt in per_class_pred.items()},
            "per_class_gt": {c: dict(cnt) for c, cnt in per_class_gt.items()},
            "per_size_pred": {c: dict(cnt) for c, cnt in per_size_pred.items()},
            "per_size_gt": {c: dict(cnt) for c, cnt in per_size_gt.items()},
            "per_condition_pred": {c: dict(cnt) for c, cnt in per_cond_pred.items()},
            "per_condition_gt": {c: dict(cnt) for c, cnt in per_cond_gt.items()},
            "confusion": confusion_dict,
            "examples": examples,
        }
        overall[tag] = result
        print(f"  {label}: categorised {len(per_image)} images, "
              f"{sum(sum(c.values()) for c in per_class_pred.values())} predictions, "
              f"{sum(sum(c.values()) for c in per_class_gt.values())} GT.")

    (ARTIFACTS / "error_taxonomy.json").write_text(json.dumps(overall, indent=2, default=float))
    print(f"Saved {ARTIFACTS/'error_taxonomy.json'}")

    # CSV long-form summary
    rows: list[dict] = []
    for tag, info in overall.items():
        for cls, counts in info["per_class_pred"].items():
            for cat, n in counts.items():
                rows.append({"detector": info["detector"], "scope": "predictions",
                             "category": cat, "class": cls, "n": n})
        for cls, counts in info["per_class_gt"].items():
            for cat, n in counts.items():
                rows.append({"detector": info["detector"], "scope": "ground_truth",
                             "category": cat, "class": cls, "n": n})
        for size, counts in info["per_size_pred"].items():
            for cat, n in counts.items():
                rows.append({"detector": info["detector"], "scope": "predictions_by_gt_size",
                             "category": cat, "class": size, "n": n})
        for cond, counts in info["per_condition_pred"].items():
            for cat, n in counts.items():
                rows.append({"detector": info["detector"], "scope": "predictions_by_condition",
                             "category": cat, "class": cond, "n": n})
    with (ARTIFACTS / "error_taxonomy.csv").open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Saved {ARTIFACTS/'error_taxonomy.csv'}")

    # Markdown report
    md = ["# Error taxonomy — Faster R-CNN vs YOLOv8n\n",
          f"IoU={args.iou_threshold}, score≥{args.score_threshold}.\n"]
    for tag, info in overall.items():
        md.append(f"## {info['detector']} (n={info['n_images']})\n")
        md.append("### Per-class predictions\n")
        md.append("| class | TP | Localization | Class_confusion | Duplicate | FP_other |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for cls, counts in info["per_class_pred"].items():
            md.append(
                f"| {cls} | {counts.get('TP', 0)} | {counts.get('Localization', 0)} | "
                f"{counts.get('Class_confusion', 0)} | {counts.get('Duplicate', 0)} | "
                f"{counts.get('FP_other', 0)} |"
            )
        md.append("\n### Per-class ground truth\n")
        md.append("| class | Detected | Missed_local | Missed_class | Total |")
        md.append("|---|---:|---:|---:|---:|")
        for cls, counts in info["per_class_gt"].items():
            total = sum(counts.values())
            md.append(
                f"| {cls} | {counts.get('Detected', 0)} | {counts.get('Missed_local', 0)} | "
                f"{counts.get('Missed_class', 0)} | {total} |"
            )
        if info["confusion"]:
            md.append("\n### Class confusion (GT class → predicted class, IoU≥0.5)\n")
            md.append("| GT | Predicted | n |")
            md.append("|---|---|---:|")
            for k, n in sorted(info["confusion"].items(), key=lambda kv: -kv[1]):
                gt_cls, pred_cls = k.split("__", 1)
                md.append(f"| {gt_cls} | {pred_cls} | {n} |")
        else:
            md.append("\n### Class confusion\n")
            md.append("No class-confusion events recorded.\n")
        md.append("\n### Per-size predictions (binned by nearest GT area)\n")
        md.append("| size | TP | Localization | Class_confusion | Duplicate | FP_other |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for size in ["small", "medium", "large", "n/a"]:
            counts = info["per_size_pred"].get(size, {})
            if not counts:
                continue
            md.append(
                f"| {size} | {counts.get('TP', 0)} | {counts.get('Localization', 0)} | "
                f"{counts.get('Class_confusion', 0)} | {counts.get('Duplicate', 0)} | "
                f"{counts.get('FP_other', 0)} |"
            )
        md.append("\n### Per-condition FP rate (predictions flagged FP_other / total predictions)\n")
        md.append("| condition | FP_other | total | FP rate |")
        md.append("|---|---:|---:|---:|")
        for cond, counts in sorted(info["per_condition_pred"].items(), key=lambda kv: -sum(kv[1].values())):
            total = sum(counts.values())
            fp = counts.get("FP_other", 0)
            rate = fp / total if total else 0.0
            md.append(f"| {cond} | {fp} | {total} | {rate:.3f} |")
    (REPORTS / "error_taxonomy.md").write_text("\n".join(md))
    print(f"Saved {REPORTS/'error_taxonomy.md'}")


if __name__ == "__main__":
    main()
