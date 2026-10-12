"""Workstream A — paired image-level bootstrap resampling.

Re-uses the existing evaluation pipeline (`src/metrics.evaluate_detections`)
on resampled subsets of the per-image prediction records. The resampling
unit is the **image** — boxes within an image are not independent
observations, so box-level bootstraps would overstate confidence.

Outputs:
  * `artifacts/statistical_analysis.json`
  * `artifacts/statistical_analysis.csv`
  * `reports/statistical_analysis.md`

The script is deterministic for a fixed seed.
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

from src.metrics import evaluate_detections, DEFAULT_IOU_THRESHOLD, DEFAULT_SCORE_THRESHOLD

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile. q in [0, 100]."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (q / 100.0) * (len(sorted_values) - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo))


def _ci95(values: list[float]) -> tuple[float, float, float]:
    s = sorted(values)
    return _percentile(s, 2.5), _percentile(s, 50.0), _percentile(s, 97.5)


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "std": None, "ci95": [None, None]}
    lo, med, hi = _ci95(values)
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    return {"mean": mean, "median": med, "std": math.sqrt(var),
            "ci95": [lo, hi]}


def _eval_subset(records: list[dict], iou: float, score: float) -> dict:
    """Re-evaluate a subset using the same protocol as the original eval."""
    res = evaluate_detections(records, iou_threshold=iou, score_threshold=score)
    return {
        "mAP_50": res.macro["mAP_50"],
        "micro_precision": res.micro["precision"],
        "micro_recall": res.micro["recall"],
        "micro_f1": res.micro["f1"],
        "n_images": res.n_images,
    }


def _per_condition_metric(records: list[dict], condition: str,
                          iou: float, score: float) -> dict | None:
    subset = [r for r in records if r.get("condition") == condition]
    if not subset:
        return None
    return _eval_subset(subset, iou, score)


def _load_per_image(tag: str) -> tuple[list[dict], dict]:
    path = ARTIFACTS / f"eval_{tag}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval artifact: {path}")
    d = json.loads(path.read_text())
    return d["per_image"], d


def _condition_records(per_image: list[dict], corruption_name: str, baseline_tag: str) -> list[dict] | None:
    """For corruption experiments: load per_image predictions for a specific corruption
    from the corruption JSON, matched by image_id to the baseline records (which
    carry the ground truth and the original image_id)."""
    path = ARTIFACTS / f"corruption_{baseline_tag}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if corruption_name not in d:
        return None
    return None  # corruption JSON stores aggregate only; we need per-image preds. See below.


def _make_pair_records(baseline_eval: dict, corruption_tag: str,
                       corruption_name: str) -> list[dict] | None:
    """Build a per_image list for one corruption level by combining:
       * ground truth + image_id from the baseline full-eval records
       * predictions from the corruption experiment run

    The corruption experiment re-runs inference on the *same* stratified
    78-image subset used by the existing `corruption_*.json`. To keep
    the per-image predictions, we need them in the JSON; the existing
    `run_corruption_experiment.py` only writes the aggregate. We
    therefore fall back to loading predictions from the most recent
    per-image artifact if present, or skip the per-corruption bootstrap.
    """
    return None  # not enough per-image data in existing corruption JSONs


def _bootstrap_records(records: list[dict], seed: int) -> list[dict]:
    """Sample with replacement at the IMAGE level."""
    rng = random.Random(seed)
    n = len(records)
    return [records[rng.randrange(n)] for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    args = parser.parse_args()

    print(f"Statistical analysis — {args.replicates} replicates, seed={args.seed}, "
          f"IoU={args.iou_threshold}, score={args.score_threshold}")

    detectors = [
        ("fasterrcnn_val_full", "Faster R-CNN ResNet-50 FPN"),
        ("yolov8n_val_full", "YOLOv8n"),
    ]
    per_detector: dict[str, list[dict]] = {}
    baseline_metrics: dict[str, dict] = {}
    for tag, label in detectors:
        per_image, full = _load_per_image(tag)
        per_detector[tag] = per_image
        baseline_metrics[tag] = _eval_subset(per_image, args.iou_threshold, args.score_threshold)
        baseline_metrics[tag]["label"] = label
        baseline_metrics[tag]["n_images"] = len(per_image)
        print(f"  {label}: n={len(per_image)} baseline mAP@.5={baseline_metrics[tag]['mAP_50']:.4f}")

    t0 = time.perf_counter()
    # Bootstrap mAP, P, R, F1 for each detector; and paired differences.
    boot: dict[str, dict[str, list[float]]] = {}
    for tag, _ in detectors:
        boot[tag] = {"mAP_50": [], "micro_precision": [], "micro_recall": [], "micro_f1": []}
    # We'll also collect per-condition mAPs for conditions with >=15 images.
    min_n_for_ci = 15
    condition_boots: dict[str, dict[str, dict[str, list[float]]]] = {}
    for tag, _ in detectors:
        condition_boots[tag] = {}
        for rec in per_detector[tag]:
            cond = rec.get("condition")
            if cond:
                condition_boots[tag].setdefault(cond, {"mAP_50": []})

    paired_diff: dict[str, list[float]] = {"mAP_50": [], "micro_f1": []}

    rng = random.Random(args.seed)
    seeds = [rng.randrange(2**31) for _ in range(args.replicates)]
    n0 = len(per_detector[detectors[0][0]])
    n1 = len(per_detector[detectors[1][0]])
    # Restrict to common image_ids for paired comparison (they should match).
    common_ids = {r["image_id"] for r in per_detector[detectors[0][0]]} & \
                 {r["image_id"] for r in per_detector[detectors[1][0]]}
    print(f"  common image_ids for paired bootstrap: {len(common_ids)}")
    recs0 = [r for r in per_detector[detectors[0][0]] if r["image_id"] in common_ids]
    recs1 = [r for r in per_detector[detectors[1][0]] if r["image_id"] in common_ids]
    # Sort so the two lists are aligned (we resample by index).
    recs0.sort(key=lambda r: r["image_id"])
    recs1.sort(key=lambda r: r["image_id"])
    n_pair = len(recs0)
    assert n_pair == len(recs1)

    for rep, s in enumerate(seeds):
        rng_b = random.Random(s)
        # Per-detector bootstrap
        for tag, recs in [(detectors[0][0], per_detector[detectors[0][0]]),
                          (detectors[1][0], per_detector[detectors[1][0]])]:
            boot_recs = [recs[rng_b.randrange(len(recs))] for _ in range(len(recs))]
            res = _eval_subset(boot_recs, args.iou_threshold, args.score_threshold)
            boot[tag]["mAP_50"].append(res["mAP_50"])
            boot[tag]["micro_precision"].append(res["micro_precision"])
            boot[tag]["micro_recall"].append(res["micro_recall"])
            boot[tag]["micro_f1"].append(res["micro_f1"])
            # Per-condition mAP (cheap: only over resampled images that match condition)
            for cond in condition_boots[tag]:
                cond_recs = [r for r in boot_recs if r.get("condition") == cond]
                if cond_recs:
                    cres = _eval_subset(cond_recs, args.iou_threshold, args.score_threshold)
                    if not math.isnan(cres["mAP_50"]):
                        condition_boots[tag][cond]["mAP_50"].append(cres["mAP_50"])
        # Paired bootstrap (FRCNN - YOLO), same resample indices
        idxs = [rng_b.randrange(n_pair) for _ in range(n_pair)]
        recs0_b = [recs0[i] for i in idxs]
        recs1_b = [recs1[i] for i in idxs]
        e0 = _eval_subset(recs0_b, args.iou_threshold, args.score_threshold)
        e1 = _eval_subset(recs1_b, args.iou_threshold, args.score_threshold)
        paired_diff["mAP_50"].append(e0["mAP_50"] - e1["mAP_50"])
        paired_diff["micro_f1"].append(e0["micro_f1"] - e1["micro_f1"])
        if (rep + 1) % 100 == 0:
            print(f"    replicate {rep+1}/{args.replicates} done in {time.perf_counter()-t0:.1f}s")

    bootstrap_runtime = time.perf_counter() - t0
    print(f"Bootstrap complete in {bootstrap_runtime:.1f}s")

    # Assemble output
    summary = {
        "config": {
            "replicates": args.replicates,
            "seed": args.seed,
            "score_threshold": args.score_threshold,
            "iou_threshold": args.iou_threshold,
            "runtime_seconds": bootstrap_runtime,
        },
        "baseline": {tag: baseline_metrics[tag] for tag, _ in detectors},
        "bootstrap": {
            tag: {
                k: _summarize(v) for k, v in boot[tag].items()
            } for tag, _ in detectors
        },
        "paired_difference": {
            "metric": "Faster R-CNN minus YOLOv8n (on common image_ids, resampled jointly)",
            "n_common_images": n_pair,
            **{k: _summarize(v) for k, v in paired_diff.items() if k not in ("n_common_images", "metric")},
        },
        "per_condition_ci": {},
        "warnings": [],
    }

    # Per-condition CI for conditions with n>=15 in the val set
    for cond in sorted(condition_boots[detectors[0][0]].keys()):
        n_cond = sum(1 for r in per_detector[detectors[0][0]] if r.get("condition") == cond)
        per_det = {}
        for tag, _ in detectors:
            vals = condition_boots[tag][cond]["mAP_50"]
            per_det[tag] = {"n_val": n_cond, "n_resamples": len(vals), "summary": _summarize(vals)}
        summary["per_condition_ci"][cond] = {"n_val": n_cond, "by_detector": per_det}
        if n_cond < min_n_for_ci:
            summary["warnings"].append(
                f"Condition '{cond}' has only {n_cond} val images; "
                "bootstrap CIs are exploratory and should not be used for inference."
            )

    # Corruption degradation bootstrap (only if per-image corruption artifacts exist)
    corr_results: dict[str, dict] = {}
    for corr_tag in ["fasterrcnn_corruption_v2", "yolov8n_corruption_v2",
                     "fasterrcnn_corruption", "yolov8n_corruption"]:
        per_image_path = ARTIFACTS / f"corruption_{corr_tag}_per_image.json"
        if not per_image_path.exists():
            continue
        d = json.loads(per_image_path.read_text())
        baseline_recs = d.get("identity")
        if not baseline_recs:
            continue
        # For each non-identity corruption, bootstrap the mAP paired with identity.
        corr_results[corr_tag] = {"per_corruption": {}}
        # Bootstrap baseline mAP distribution
        boot_baseline: list[float] = []
        for s in seeds:
            rng_b = random.Random(s)
            sub = [baseline_recs[rng_b.randrange(len(baseline_recs))] for _ in range(len(baseline_recs))]
            res = _eval_subset(sub, args.iou_threshold, args.score_threshold)
            boot_baseline.append(res["mAP_50"])
        corr_results[corr_tag]["baseline_bootstrap_mAP"] = _summarize(boot_baseline)
        # Paired mAP drop per corruption
        for cname, recs in d.items():
            if cname == "identity":
                continue
            # Align by image_id to baseline
            base_by_id = {r["image_id"]: r for r in baseline_recs}
            corr_by_id = {r["image_id"]: r for r in recs}
            common = sorted(set(base_by_id.keys()) & set(corr_by_id.keys()))
            base_aligned = [base_by_id[i] for i in common]
            corr_aligned = [corr_by_id[i] for i in common]
            n = len(common)
            if n == 0:
                continue
            boot_drops: list[float] = []
            for s in seeds:
                rng_b = random.Random(s)
                idxs = [rng_b.randrange(n) for _ in range(n)]
                base_b = [base_aligned[i] for i in idxs]
                corr_b = [corr_aligned[i] for i in idxs]
                base_res = _eval_subset(base_b, args.iou_threshold, args.score_threshold)
                corr_res = _eval_subset(corr_b, args.iou_threshold, args.score_threshold)
                if not (math.isnan(base_res["mAP_50"]) or math.isnan(corr_res["mAP_50"])):
                    boot_drops.append(corr_res["mAP_50"] - base_res["mAP_50"])
            corr_results[corr_tag]["per_corruption"][cname] = {
                "n_common_images": n,
                "mAP_drop_summary": _summarize(boot_drops),
            }
        print(f"  corruption bootstrap: {corr_tag} done")

    summary["corruption_bootstrap"] = corr_results

    # Save JSON
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "statistical_analysis.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"Saved {ARTIFACTS/'statistical_analysis.json'}")

    # Save CSV (long form: detector, metric, mean, median, ci95_lo, ci95_hi)
    rows = []
    for tag, label in detectors:
        for metric, summ in summary["bootstrap"][tag].items():
            rows.append({
                "scope": "overall",
                "detector": label,
                "metric": metric,
                "n": args.replicates,
                "mean": summ["mean"],
                "median": summ["median"],
                "ci95_lo": summ["ci95"][0],
                "ci95_hi": summ["ci95"][1],
            })
    for metric, summ in summary["paired_difference"].items():
        if metric in ("n_common_images", "metric"):
            continue
        rows.append({
            "scope": "paired_diff",
            "detector": "Faster R-CNN - YOLOv8n",
            "metric": metric,
            "n": args.replicates,
            "mean": summ["mean"],
            "median": summ["median"],
            "ci95_lo": summ["ci95"][0],
            "ci95_hi": summ["ci95"][1],
        })
    for cond, info in summary["per_condition_ci"].items():
        for tag, label in detectors:
            summ = info["by_detector"][tag]["summary"]
            rows.append({
                "scope": f"condition:{cond}",
                "detector": label,
                "metric": "mAP_50",
                "n": info["by_detector"][tag]["n_resamples"],
                "mean": summ["mean"],
                "median": summ["median"],
                "ci95_lo": summ["ci95"][0],
                "ci95_hi": summ["ci95"][1],
            })
    with (ARTIFACTS / "statistical_analysis.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {ARTIFACTS/'statistical_analysis.csv'}")

    # Markdown report
    md = [
        "# Statistical analysis — paired image-level bootstrap\n",
        f"- Replicates: **{args.replicates}**, seed: {args.seed}, "
        f"IoU={args.iou_threshold}, score={args.score_threshold}.",
        f"- Runtime: {bootstrap_runtime:.1f} s.\n",
        "## Baseline (point estimates on full val set)\n",
        "| Detector | n | mAP@.5 | Micro P | Micro R | Micro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tag, label in detectors:
        b = baseline_metrics[tag]
        md.append(
            f"| {label} | {b['n_images']} | {b['mAP_50']:.4f} | "
            f"{b['micro_precision']:.4f} | {b['micro_recall']:.4f} | {b['micro_f1']:.4f} |"
        )
    md.append("\n## Bootstrap 95% CIs (image-level resampling with replacement)\n")
    md.append("| Detector | Metric | Mean | Median | 95% CI |")
    md.append("|---|---|---:|---:|---|")
    for tag, label in detectors:
        for metric, summ in summary["bootstrap"][tag].items():
            md.append(
                f"| {label} | {metric} | {summ['mean']:.4f} | {summ['median']:.4f} | "
                f"[{summ['ci95'][0]:.4f}, {summ['ci95'][1]:.4f}] |"
            )
    md.append("\n## Paired difference (Faster R-CNN − YOLOv8n)\n")
    md.append(f"- Resampling is *joint*: the same image_ids are drawn for both detectors. "
               f"Common n = {n_pair}.\n")
    md.append("| Metric | Mean | Median | 95% CI |")
    md.append("|---|---:|---:|---|")
    for metric, summ in summary["paired_difference"].items():
        if metric in ("n_common_images", "metric"):
            continue
        md.append(
            f"| {metric} | {summ['mean']:.4f} | {summ['median']:.4f} | "
            f"[{summ['ci95'][0]:.4f}, {summ['ci95'][1]:.4f}] |"
        )
    md.append(
        "\n> Overlapping CIs are **not** a formal significance test. "
        "The paired bootstrap gives a defensible non-parametric interval for the "
        "detector-difference distribution.\n"
    )

    md.append("\n## Per-condition mAP@.5 bootstrap CIs\n")
    md.append("| Condition | n_val | Faster R-CNN mAP (95% CI) | YOLOv8n mAP (95% CI) | Note |")
    md.append("|---|---:|---|---|---|")
    for cond, info in summary["per_condition_ci"].items():
        n = info["n_val"]
        s_fr = info["by_detector"]["fasterrcnn_val_full"]["summary"]
        s_yo = info["by_detector"]["yolov8n_val_full"]["summary"]
        note = "exploratory (n<15)" if n < 15 else ""
        md.append(
            f"| {cond} | {n} | "
            f"{s_fr['median']:.3f} [{s_fr['ci95'][0]:.3f}, {s_fr['ci95'][1]:.3f}] | "
            f"{s_yo['median']:.3f} [{s_yo['ci95'][0]:.3f}, {s_yo['ci95'][1]:.3f}] | {note} |"
        )
    if summary["warnings"]:
        md.append("\n## Warnings\n")
        for w in summary["warnings"]:
            md.append(f"- {w}")
    (REPORTS / "statistical_analysis.md").write_text("\n".join(md))
    print(f"Saved {REPORTS/'statistical_analysis.md'}")


if __name__ == "__main__":
    main()
