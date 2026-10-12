"""Stage 5: error analysis — confidence distributions, per-class
calibration sketch (Brier + reliability bins), and failure breakdown.

Reads artifacts/eval_fasterrcnn_val_full.json. Writes a Markdown report
under reports/error_analysis.md and a CSV summary of reliability bins.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def _brier(predictions_outcome: list[tuple[float, int]]) -> float:
    """Brier score: mean over (p - y)^2."""
    if not predictions_outcome:
        return float("nan")
    return sum((p - y) ** 2 for p, y in predictions_outcome) / len(predictions_outcome)


def _reliability_bins(predictions_outcome: list[tuple[float, int]], n_bins: int = 10):
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in predictions_outcome:
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, y))
    summary = []
    for i, items in enumerate(bins):
        if not items:
            summary.append({"bin_lo": i / n_bins, "bin_hi": (i + 1) / n_bins,
                            "count": 0, "mean_conf": None, "frac_correct": None})
            continue
        mean_conf = sum(p for p, _ in items) / len(items)
        frac_correct = sum(y for _, y in items) / len(items)
        summary.append({
            "bin_lo": i / n_bins,
            "bin_hi": (i + 1) / n_bins,
            "count": len(items),
            "mean_conf": mean_conf,
            "frac_correct": frac_correct,
        })
    return summary


def main() -> None:
    for tag, label in [("fasterrcnn_val_full", "Faster R-CNN"),
                       ("yolov8n_val_full", "YOLOv8n")]:
        path = ARTIFACTS / f"eval_{tag}.json"
        if not path.exists():
            print(f"skip {tag}: {path} not found")
            continue
        data = json.loads(path.read_text())
        # Build (score, outcome=1 if TP) per detection at IoU>=0.5.
        from src.metrics import iou_xyxy
        per_class_p_o: dict[str, list[tuple[float, int]]] = defaultdict(list)
        per_cond_counts: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for er in data["per_image"]:
            cond = er["condition"]
            gt_by_class: dict[str, list] = defaultdict(list)
            for g in er["ground_truth"]:
                gt_by_class[g["class_name"]].append(g["box_xyxy"])
            preds_by_class: dict[str, list] = defaultdict(list)
            for p in er["predictions"]:
                if p.get("common_label") is None or p["score"] < 0.5:
                    continue
                preds_by_class[p["common_label"]].append(p)
            for cn, gts in gt_by_class.items():
                preds = sorted(preds_by_class.get(cn, []), key=lambda x: -x["score"])
                gt_used = [False] * len(gts)
                for p in preds:
                    best_iou, best_gi = 0.0, -1
                    for gi, gb in enumerate(gts):
                        if gt_used[gi]:
                            continue
                        v = iou_xyxy(p["box_xyxy"], gb)
                        if v > best_iou:
                            best_iou, best_gi = v, gi
                    if best_gi >= 0 and best_iou >= 0.5:
                        gt_used[best_gi] = True
                        per_class_p_o[cn].append((p["score"], 1))
                        per_cond_counts[cond]["tp"] += 1
                    else:
                        per_class_p_o[cn].append((p["score"], 0))
                        per_cond_counts[cond]["fp"] += 1
                # FN = unmatched GT
                per_cond_counts[cond]["fn"] += sum(1 for u in gt_used if not u)

        # Aggregate scores by class
        all_po = [(p, y) for items in per_class_p_o.values() for p, y in items]
        brier = _brier(all_po)
        bins = _reliability_bins(all_po, n_bins=10)

        # Write CSV of reliability bins
        rel_csv = ARTIFACTS / f"reliability_{tag}.csv"
        with rel_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["bin_lo", "bin_hi", "count", "mean_conf", "frac_correct"])
            w.writeheader()
            for row in bins:
                w.writerow(row)
        print(f"wrote {rel_csv}")

        # Per-class Brier
        per_class_brier = {cn: _brier(items) for cn, items in per_class_p_o.items()}

        # Per-condition confusion summary
        per_cond_rows = []
        for cond, c in sorted(per_cond_counts.items(), key=lambda kv: -kv[1]["tp"]):
            per_cond_rows.append({
                "condition": cond,
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                "precision": c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0,
                "recall": c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0,
            })

        # Write report
        out = []
        out.append(f"# Error analysis — {label} on DAWN val (n={data['n_images']})\n")
        out.append("## Brier score (lower = better calibration)")
        out.append("")
        out.append(f"- Overall (score>=0.5, IoU>=0.5): **{brier:.4f}**  ")
        out.append("- Per class:")
        for cn, b in sorted(per_class_brier.items(), key=lambda kv: kv[1]):
            n = len(per_class_p_o[cn])
            out.append(f"  - {cn}: {b:.4f}  (n={n})")
        out.append("")
        out.append("> Detector confidences are **not** calibrated probabilities. "
                   "Treat them as relative scores only.\n")
        out.append("## Reliability bins (10 equal-width confidence bins)\n")
        out.append("| bin | count | mean conf | frac correct |")
        out.append("|---|---:|---:|---:|")
        for row in bins:
            mc = f"{row['mean_conf']:.3f}" if row["mean_conf"] is not None else "-"
            fc = f"{row['frac_correct']:.3f}" if row["frac_correct"] is not None else "-"
            out.append(f"| {row['bin_lo']:.1f}–{row['bin_hi']:.1f} | {row['count']} | {mc} | {fc} |")
        out.append("")
        out.append("## Per-condition TP / FP / FN\n")
        out.append("| condition | TP | FP | FN | P | R |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for r in per_cond_rows:
            out.append(f"| {r['condition']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                       f"{r['precision']:.3f} | {r['recall']:.3f} |")
        out.append("")
        (REPORTS / f"error_analysis_{tag}.md").write_text("\n".join(out))
        print(f"wrote {REPORTS}/error_analysis_{tag}.md")


if __name__ == "__main__":
    main()
