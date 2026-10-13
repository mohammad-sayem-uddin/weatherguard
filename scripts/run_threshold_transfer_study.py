"""Phase 3/4 of the focused study: threshold selection on dev, then
held-out evaluation of the locked threshold.

Inputs (already produced by run_transfer_eval.py):
  artifacts/eval_dev_fasterrcnn.json
  artifacts/eval_dev_yolov8n.json
  artifacts/eval_heldout_fasterrcnn.json
  artifacts/eval_heldout_yolov8n.json

Outputs:
  artifacts/threshold_transfer.json  (full)
  artifacts/threshold_transfer.csv   (long form)
  artifacts/plots/threshold_transfer.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import evaluate_detections, DEFAULT_IOU_THRESHOLD, DEFAULT_SCORE_THRESHOLD

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
PLOTS = ARTIFACTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [round(x * 0.05, 2) for x in range(1, 19)]  # 0.05..0.90
HIGH_RECALL_TARGET = 0.85
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 20261012


def sweep(per_image, thresholds, iou=DEFAULT_IOU_THRESHOLD):
    """Sweep a list of score thresholds; return list of metric dicts."""
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


def best_f1(rows):
    """Threshold maximising micro F1 (ties broken by higher recall)."""
    return max(rows, key=lambda r: (r["micro_f1"], r["micro_recall"]))


def high_recall(rows, target=HIGH_RECALL_TARGET):
    """Among thresholds achieving recall >= target on dev, pick the
    one with the highest precision. If none reach target, return None."""
    candidates = [r for r in rows if r["micro_recall"] >= target]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["micro_precision"], r["micro_recall"]))


def pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (q / 100) * (len(s) - 1)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def ci95(vals):
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    return (pct(vals, 2.5), pct(vals, 50), pct(vals, 97.5))


def bootstrap_paired(recs_a, recs_b, metric_fn, n=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Paired image-level bootstrap. Resample indices jointly.
    `metric_fn(per_image_subset) -> float`."""
    rng = random.Random(seed)
    L = len(recs_a)
    assert L == len(recs_b)
    diffs = []
    # Use common seed sequence for reproducibility
    for _ in range(n):
        idxs = [rng.randrange(L) for _ in range(L)]
        sub_a = [recs_a[i] for i in idxs]
        sub_b = [recs_b[i] for i in idxs]
        diffs.append(metric_fn(sub_a) - metric_fn(sub_b))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boots", type=int, default=BOOTSTRAP_REPS)
    args = ap.parse_args()

    dev_paths = {
        "fasterrcnn_resnet50_fpn": ARTIFACTS / "eval_dev_fasterrcnn.json",
        "yolov8n":                 ARTIFACTS / "eval_dev_yolov8n.json",
    }
    held_paths = {
        "fasterrcnn_resnet50_fpn": ARTIFACTS / "eval_heldout_fasterrcnn.json",
        "yolov8n":                 ARTIFACTS / "eval_heldout_yolov8n.json",
    }

    # Load all
    bundles = {}
    for det, p in dev_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}; run run_transfer_eval.py first.")
        bundles[det] = {
            "dev": json.loads(p.read_text()),
            "heldout": json.loads(held_paths[det].read_text()),
        }

    # Sweep on both dev and heldout
    for det, b in bundles.items():
        b["dev_sweep"] = sweep(b["dev"]["per_image"], THRESHOLDS)
        b["held_sweep"] = sweep(b["heldout"]["per_image"], THRESHOLDS)
        b["dev_best_f1"] = best_f1(b["dev_sweep"])
        b["dev_high_recall"] = high_recall(b["dev_sweep"])
        b["held_best_f1"] = best_f1(b["held_sweep"])  # diagnostic only
        print(f"\n=== {det} ===")
        print(f"  dev   best F1 = {b['dev_best_f1']['micro_f1']:.4f} at score={b['dev_best_f1']['score_threshold']}")
        if b["dev_high_recall"]:
            print(f"  dev   high-recall = score={b['dev_high_recall']['score_threshold']} "
                  f"recall={b['dev_high_recall']['micro_recall']:.4f} "
                  f"precision={b['dev_high_recall']['micro_precision']:.4f}")
        else:
            print(f"  dev   high-recall: none reached recall >= {HIGH_RECALL_TARGET}")
        print(f"  held  best F1 (diagnostic) = {b['held_best_f1']['micro_f1']:.4f} "
              f"at score={b['held_best_f1']['score_threshold']}")

    # Locked threshold evaluation: dev-selected threshold applied to heldout
    locked_eval: dict[str, dict] = {}
    for det, b in bundles.items():
        # Default
        b["held_default"] = evaluate_detections(
            b["heldout"]["per_image"], iou_threshold=0.5, score_threshold=0.5
        )
        # Dev-F1
        thr_f1 = b["dev_best_f1"]["score_threshold"]
        b["held_dev_f1"] = evaluate_detections(
            b["heldout"]["per_image"], iou_threshold=0.5, score_threshold=thr_f1
        )
        # Dev-high-recall
        if b["dev_high_recall"]:
            thr_hr = b["dev_high_recall"]["score_threshold"]
            b["held_dev_hr"] = evaluate_detections(
                b["heldout"]["per_image"], iou_threshold=0.5, score_threshold=thr_hr
            )
        # Diagnostic val-oracle
        b["held_val_oracle"] = evaluate_detections(
            b["heldout"]["per_image"], iou_threshold=0.5,
            score_threshold=b["held_best_f1"]["score_threshold"]
        )
        locked_eval[det] = {
            "default": b["held_default"],
            "dev_f1_threshold": b["dev_best_f1"]["score_threshold"],
            "dev_f1": b["held_dev_f1"],
            "dev_hr_threshold": (b["dev_high_recall"] or {}).get("score_threshold"),
            "dev_hr": b.get("held_dev_hr"),
            "val_oracle_threshold": b["held_best_f1"]["score_threshold"],
            "val_oracle": b["held_val_oracle"],
        }

    # Per-condition transfer (for RQ3)
    per_cond_dev = {}
    per_cond_held = {}
    for det, b in bundles.items():
        per_cond_dev[det] = {}
        per_cond_held[det] = {}
        # Per-condition dev sweep
        by_cond = defaultdict(list)
        for rec in b["dev"]["per_image"]:
            by_cond[rec.get("condition", "?")].append(rec)
        for cond, items in by_cond.items():
            per_cond_dev[det][cond] = sweep(items, THRESHOLDS)
        by_cond_h = defaultdict(list)
        for rec in b["heldout"]["per_image"]:
            by_cond_h[rec.get("condition", "?")].append(rec)
        for cond, items in by_cond_h.items():
            per_cond_held[det][cond] = sweep(items, THRESHOLDS)

    # Per-class transfer (for RQ3)
    per_class_dev = {}
    per_class_held = {}
    for det, b in bundles.items():
        per_class_dev[det] = {}
        per_class_held[det] = {}
        # Per-class dev evaluation at each operating point
        # Build per-class records
        for op_name, op_eval, op_thr in [
            ("default", b["held_default"], 0.5),
            ("dev_f1",  b["held_dev_f1"],  b["dev_best_f1"]["score_threshold"]),
            ("dev_hr",  b.get("held_dev_hr"), (b["dev_high_recall"] or {}).get("score_threshold")),
        ]:
            if op_eval is None:
                continue
            for c in op_eval.per_class:
                per_class_held[det].setdefault(c.class_name, {})[op_name] = {
                    "score_threshold": op_thr,
                    "tp": c.tp, "fp": c.fp, "fn": c.fn,
                    "n_gt": c.n_gt, "n_pred": c.n_pred,
                    "precision": c.precision, "recall": c.recall, "f1": c.f1,
                    "ap_50": c.ap_50,
                }
        # Per-class dev
        for c_name in {g["class_name"] for rec in b["dev"]["per_image"] for g in rec["ground_truth"]}:
            sub = [r for r in b["dev"]["per_image"]
                   if any(g["class_name"] == c_name for g in r["ground_truth"])]
            for thr in THRESHOLDS:
                res = evaluate_detections(sub, iou_threshold=0.5, score_threshold=thr)
                cls = next((c for c in res.per_class if c.class_name == c_name), None)
                if cls is not None:
                    per_class_dev[det].setdefault(c_name, {})[f"score_{thr:.2f}"] = {
                        "precision": cls.precision, "recall": cls.recall, "f1": cls.f1,
                        "n_gt": cls.n_gt, "tp": cls.tp, "fp": cls.fp, "fn": cls.fn,
                    }

    # Bootstrap CIs: paired differences in micro F1 on the held-out set
    boot_results = {}
    rng = random.Random(BOOTSTRAP_SEED)
    dets = list(bundles.keys())
    for det in dets:
        b = bundles[det]
        per_image = b["heldout"]["per_image"]
        # Default vs dev-F1
        thr_def = 0.5
        thr_df1 = b["dev_best_f1"]["score_threshold"]
        diffs_default_minus_df1 = []
        for _ in range(args.boots):
            idxs = [rng.randrange(len(per_image)) for _ in range(len(per_image))]
            sub = [per_image[i] for i in idxs]
            r_def = evaluate_detections(sub, iou_threshold=0.5, score_threshold=thr_def)
            r_df1 = evaluate_detections(sub, iou_threshold=0.5, score_threshold=thr_df1)
            diffs_default_minus_df1.append(r_def.micro["f1"] - r_df1.micro["f1"])
        boot_results[det] = {
            "default_vs_dev_f1_paired_diff_f1": {
                "mean": sum(diffs_default_minus_df1)/len(diffs_default_minus_df1),
                "median": pct(diffs_default_minus_df1, 50),
                "ci95": list(ci95(diffs_default_minus_df1)),
            }
        }
    # Detector vs detector at the dev-F1 threshold (paired on heldout)
    if len(dets) == 2:
        a, b = dets
        per_image_a = bundles[a]["heldout"]["per_image"]
        per_image_b = bundles[b]["heldout"]["per_image"]
        # Align by image_id
        a_by_id = {r["image_id"]: r for r in per_image_a}
        b_by_id = {r["image_id"]: r for r in per_image_b}
        common = sorted(set(a_by_id) & set(b_by_id))
        a_aligned = [a_by_id[i] for i in common]
        b_aligned = [b_by_id[i] for i in common]
        diffs_detector_f1 = []
        for _ in range(args.boots):
            idxs = [rng.randrange(len(common)) for _ in range(len(common))]
            sub_a = [a_aligned[i] for i in idxs]
            sub_b = [b_aligned[i] for i in idxs]
            thr = bundles[a]["dev_best_f1"]["score_threshold"]
            # Both detectors evaluated at THEIR dev-F1 thresholds (different)
            r_a = evaluate_detections(sub_a, iou_threshold=0.5, score_threshold=thr)
            r_b = evaluate_detections(sub_b, iou_threshold=0.5,
                                     score_threshold=bundles[b]["dev_best_f1"]["score_threshold"])
            diffs_detector_f1.append(r_a.micro["f1"] - r_b.micro["f1"])
        boot_results["detector_comparison_dev_f1_threshold"] = {
            "fasterrcnn_minus_yolov8n_micro_f1": {
                "mean": sum(diffs_detector_f1)/len(diffs_detector_f1),
                "median": pct(diffs_detector_f1, 50),
                "ci95": list(ci95(diffs_detector_f1)),
            }
        }

    # Latency summary (from the run artifacts)
    latency = {}
    for det, b in bundles.items():
        latency[det] = {
            "load_seconds_dev": b["dev"].get("detector_load_seconds"),
            "load_seconds_heldout": b["heldout"].get("detector_load_seconds"),
            "mean_seconds_per_image_dev": b["dev"].get("inference_mean_seconds"),
            "median_seconds_per_image_dev": b["dev"].get("inference_median_seconds"),
            "mean_seconds_per_image_heldout": b["heldout"].get("inference_mean_seconds"),
            "median_seconds_per_image_heldout": b["heldout"].get("inference_median_seconds"),
            "total_seconds_dev": b["dev"].get("inference_total_seconds"),
            "total_seconds_heldout": b["heldout"].get("inference_total_seconds"),
            "device": b["dev"].get("device"),
            "warmup": True,
        }

    # Assemble output
    output = {
        "config": {
            "thresholds": THRESHOLDS,
            "iou_threshold": DEFAULT_IOU_THRESHOLD,
            "default_score_threshold": DEFAULT_SCORE_THRESHOLD,
            "high_recall_target": HIGH_RECALL_TARGET,
            "bootstrap_reps": args.boots,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "resampling_unit": "image",
            "matching_protocol": "greedy_iou_descending_score",
            "iou_threshold": 0.5,
        },
        "splits": {
            det: {
                "dev": {
                    "parquet": b["dev"]["parquet"],
                    "n_images": b["dev"]["n_images"],
                    "excluded_image_ids": b["dev"].get("excluded_image_ids", []),
                },
                "heldout": {
                    "parquet": b["heldout"]["parquet"],
                    "n_images": b["heldout"]["n_images"],
                    "excluded_image_ids": b["heldout"].get("excluded_image_ids", []),
                },
            } for det, b in bundles.items()
        },
        "per_detector": {
            det: {
                "label": det,
                "dev_sweep": b["dev_sweep"],
                "heldout_sweep": b["held_sweep"],
                "dev_best_f1_threshold": b["dev_best_f1"]["score_threshold"],
                "dev_best_f1_on_dev": {
                    "micro_precision": b["dev_best_f1"]["micro_precision"],
                    "micro_recall": b["dev_best_f1"]["micro_recall"],
                    "micro_f1": b["dev_best_f1"]["micro_f1"],
                    "micro_tp": b["dev_best_f1"]["micro_tp"],
                    "micro_fp": b["dev_best_f1"]["micro_fp"],
                    "micro_fn": b["dev_best_f1"]["micro_fn"],
                },
                "dev_high_recall_threshold": (b["dev_high_recall"] or {}).get("score_threshold"),
                "dev_high_recall_on_dev": (
                    {k: b["dev_high_recall"][k] for k in
                     ("score_threshold","micro_precision","micro_recall","micro_f1",
                      "micro_tp","micro_fp","micro_fn")}
                    if b["dev_high_recall"] else None
                ),
                "held_default_eval": b["held_default"].to_dict(),
                "held_dev_f1_eval": b["held_dev_f1"].to_dict(),
                "held_dev_hr_eval": b.get("held_dev_hr").to_dict() if b.get("held_dev_hr") else None,
                "held_val_oracle_eval": b["held_val_oracle"].to_dict(),
                "held_val_oracle_threshold": b["held_best_f1"]["score_threshold"],
                "per_condition_dev": per_cond_dev[det],
                "per_condition_heldout": per_cond_held[det],
                "per_class_dev": per_class_dev[det],
                "per_class_heldout_by_operating_point": per_class_held[det],
            } for det, b in bundles.items()
        },
        "bootstrap": boot_results,
        "latency": latency,
    }

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "threshold_transfer.json").write_text(json.dumps(output, indent=2, default=float))
    print(f"Saved {ARTIFACTS/'threshold_transfer.json'}")

    # CSV (long form)
    rows_csv = []
    for det, b in bundles.items():
        for op_name, eval_obj, thr in [
            ("default_0.5",  b["held_default"], 0.5),
            ("dev_F1",       b["held_dev_f1"],  b["dev_best_f1"]["score_threshold"]),
            ("val_oracle_diagnostic", b["held_val_oracle"], b["held_best_f1"]["score_threshold"]),
        ]:
            r = eval_obj
            rows_csv.append({
                "scope": "heldout_overall",
                "detector": det,
                "operating_point": op_name,
                "score_threshold": thr,
                "n_images": r.n_images,
                "micro_tp": r.micro["tp"], "micro_fp": r.micro["fp"], "micro_fn": r.micro["fn"],
                "micro_precision": r.micro["precision"],
                "micro_recall": r.micro["recall"],
                "micro_f1": r.micro["f1"],
                "mAP_50": r.macro["mAP_50"],
            })
        if b.get("held_dev_hr"):
            r = b["held_dev_hr"]
            rows_csv.append({
                "scope": "heldout_overall",
                "detector": det,
                "operating_point": "dev_high_recall",
                "score_threshold": b["dev_high_recall"]["score_threshold"],
                "n_images": r.n_images,
                "micro_tp": r.micro["tp"], "micro_fp": r.micro["fp"], "micro_fn": r.micro["fn"],
                "micro_precision": r.micro["precision"],
                "micro_recall": r.micro["recall"],
                "micro_f1": r.micro["f1"],
                "mAP_50": r.macro["mAP_50"],
            })
        for cond, rows in per_cond_held[det].items():
            for r in rows:
                rows_csv.append({
                    "scope": f"heldout_condition:{cond}",
                    "detector": det,
                    "operating_point": f"sweep@{r['score_threshold']:.2f}",
                    "score_threshold": r["score_threshold"],
                    "n_images": r["n_images"],
                    "micro_tp": r["micro_tp"], "micro_fp": r["micro_fp"], "micro_fn": r["micro_fn"],
                    "micro_precision": r["micro_precision"],
                    "micro_recall": r["micro_recall"],
                    "micro_f1": r["micro_f1"],
                    "mAP_50": r["mAP_50"],
                })
    out_csv = ARTIFACTS / "threshold_transfer.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"Saved {out_csv}  ({len(rows_csv)} rows)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for det, b in bundles.items():
        ds = b["dev_sweep"]
        hs = b["held_sweep"]
        ts_d = [r["score_threshold"] for r in ds]
        ts_h = [r["score_threshold"] for r in hs]
        axes[0].plot(ts_d, [r["micro_f1"] for r in ds], "--", label=f"dev {det}")
        axes[0].plot(ts_h, [r["micro_f1"] for r in hs], "-", label=f"heldout {det}")
    axes[0].axvline(0.5, color="grey", lw=0.5, ls=":")
    axes[0].set_xlabel("Score threshold")
    axes[0].set_ylabel("Micro F1")
    axes[0].set_title("F1 vs threshold — development (dashed) vs held-out (solid)")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    for det, b in bundles.items():
        hs = b["held_sweep"]
        axes[1].plot([r["micro_recall"] for r in hs],
                     [r["micro_precision"] for r in hs],
                     "-o", markersize=3, label=f"heldout {det}")
    axes[1].set_xlabel("Recall (held-out)")
    axes[1].set_ylabel("Precision (held-out)")
    axes[1].set_title("P–R curve on held-out (n=202)")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out_png = PLOTS / "threshold_transfer.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
