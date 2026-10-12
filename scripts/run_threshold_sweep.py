"""Workstream C — confidence-threshold sweep.

Sweeps IoU=0.5 fixed, varying score thresholds in {0.05, 0.10, …, 0.90}.
Reports precision, recall, F1, false positives, false negatives, and
the validation-selected F1-maximising threshold.

Important: the best-threshold selection is done on the validation set.
The result is therefore a *validation-selected operating point*, not an
unbiased estimate of generalization. This is documented in the output
and in `reports/threshold_sweep.md`.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from src.metrics import evaluate_detections, DEFAULT_IOU_THRESHOLD

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
PLOTS = ARTIFACTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def _sweep(per_image: list[dict], iou: float, thresholds: list[float]) -> list[dict]:
    rows = []
    for th in thresholds:
        res = evaluate_detections(per_image, iou_threshold=iou, score_threshold=th)
        rows.append({
            "score_threshold": th,
            "iou_threshold": iou,
            "n_images": res.n_images,
            "micro_tp": res.micro["tp"],
            "micro_fp": res.micro["fp"],
            "micro_fn": res.micro["fn"],
            "micro_precision": res.micro["precision"],
            "micro_recall": res.micro["recall"],
            "micro_f1": res.micro["f1"],
            "mAP_50": res.macro["mAP_50"],
        })
    return rows


def _select_best(rows: list[dict], metric: str = "micro_f1") -> dict:
    valid = [r for r in rows if r[metric] is not None]
    if not valid:
        return {}
    return max(valid, key=lambda r: r[metric])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=str,
                        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    detectors = [
        ("fasterrcnn_val_full", "Faster R-CNN ResNet-50 FPN"),
        ("yolov8n_val_full", "YOLOv8n"),
    ]
    summary: dict = {"config": {"thresholds": thresholds, "iou": args.iou},
                     "detectors": {}}

    for tag, label in detectors:
        path = ARTIFACTS / f"eval_{tag}.json"
        if not path.exists():
            print(f"skip {tag}: missing {path}")
            continue
        d = json.loads(path.read_text())
        per_image = d["per_image"]
        t0 = time.perf_counter()
        rows = _sweep(per_image, args.iou, thresholds)
        elapsed = time.perf_counter() - t0
        best_f1 = _select_best(rows, "micro_f1")
        best_ap = _select_best(rows, "mAP_50")
        summary["detectors"][tag] = {
            "label": label,
            "n_images": len(per_image),
            "sweep_seconds": elapsed,
            "rows": rows,
            "best_micro_f1": best_f1,
            "best_mAP_50": best_ap,
            "n_images": len(per_image),
        }
        print(f"  {label}: {len(rows)} thresholds in {elapsed:.2f}s, "
              f"best F1 = {best_f1['micro_f1']:.4f} at score={best_f1['score_threshold']}")

    (ARTIFACTS / "threshold_sweep.json").write_text(json.dumps(summary, indent=2, default=float))
    rows_csv: list[dict] = []
    for tag, info in summary["detectors"].items():
        for r in info["rows"]:
            rows_csv.append({"detector": info["label"], "tag": tag, **r})
    with (ARTIFACTS / "threshold_sweep.csv").open("w", newline="") as f:
        if rows_csv:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)
    print(f"Saved {ARTIFACTS/'threshold_sweep.{json,csv}'}")

    # Plot precision-recall curves and F1 vs threshold.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for tag, info in summary["detectors"].items():
            rows = info["rows"]
            ts = [r["score_threshold"] for r in rows]
            ps = [r["micro_precision"] for r in rows]
            rs = [r["micro_recall"] for r in rows]
            f1s = [r["micro_f1"] for r in rows]
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(rs, ps, marker="o")
            ax[0].set_xlabel("Recall")
            ax[0].set_ylabel("Precision")
            ax[0].set_title(f"P–R curve — {info['label']}")
            ax[0].set_xlim(0, 1)
            ax[0].set_ylim(0, 1)
            ax[0].grid(True, alpha=0.3)
            ax[1].plot(ts, f1s, marker="o", label="Micro F1")
            ax[1].set_xlabel("Score threshold")
            ax[1].set_ylabel("Micro F1")
            ax[1].set_title(f"F1 vs threshold — {info['label']}")
            ax[1].set_ylim(0, 1)
            ax[1].grid(True, alpha=0.3)
            fig.tight_layout()
            out = PLOTS / f"threshold_sweep_{tag}.png"
            fig.savefig(out, dpi=110)
            plt.close(fig)
            print(f"  plot {out}")
    except Exception as exc:
        print(f"  (matplotlib not available or failed: {exc})")

    # Markdown
    md = ["# Confidence-threshold sweep\n",
          f"IoU fixed at {args.iou}. Score thresholds: {thresholds}.\n",
          "> The best-F1 and best-mAP points are **validation-selected operating "
          "points**. They are not unbiased estimates of generalisation; they "
          "are useful for picking an operating point for downstream use on "
          "this same validation set.\n"]
    for tag, info in summary["detectors"].items():
        md.append(f"## {info['label']} (n={info['n_images']})\n")
        md.append("| score | P | R | F1 | mAP@.5 | TP | FP | FN |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in info["rows"]:
            md.append(
                f"| {r['score_threshold']:.2f} | {r['micro_precision']:.3f} | "
                f"{r['micro_recall']:.3f} | {r['micro_f1']:.3f} | {r['mAP_50']:.3f} | "
                f"{r['micro_tp']} | {r['micro_fp']} | {r['micro_fn']} |"
            )
        bf = info["best_micro_f1"]
        ba = info["best_mAP_50"]
        md.append(f"\n**Validation-selected F1 max:** score={bf['score_threshold']:.2f} → "
                  f"P={bf['micro_precision']:.3f} R={bf['micro_recall']:.3f} F1={bf['micro_f1']:.3f}\n")
        md.append(f"**Validation-selected mAP@.5 max:** score={ba['score_threshold']:.2f} → "
                  f"mAP@.5={ba['mAP_50']:.4f}\n")
    (REPORTS / "threshold_sweep.md").write_text("\n".join(md))
    print(f"Saved {REPORTS/'threshold_sweep.md'}")


if __name__ == "__main__":
    main()
