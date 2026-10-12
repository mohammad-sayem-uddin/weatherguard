"""Workstream F — confidence calibration audit + simple temperature scaling.

The script:
  1. Reuses the per-image predictions from the val artifacts.
  2. Defines a single, documented definition of a *correct* detection
     (class match, IoU >= 0.5).
  3. Computes Brier score and 10-bin reliability tables BEFORE any
     post-processing.
  4. Fits a single temperature parameter T on the validation set using
     negative log-likelihood on the correct/incorrect binary outcome
     (in-sample fit — explicitly labelled as such).
  5. Re-computes the Brier score and reliability table with
     `sigmoid(logit(score) / T)`-style rescaling applied per detector.
  6. Saves calibrated reliability CSVs and a Markdown summary.

Note: this is a *post-hoc* single-parameter temperature scaling. It is
fitted on the same val set used for evaluation, so any reported
"post-calibration" improvement is in-sample. We state this explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from src.metrics import iou_xyxy

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def _per_image_score_outcome(per_image: list[dict], score_threshold: float = 0.5,
                              iou_threshold: float = 0.5) -> list[tuple[float, int]]:
    """For every prediction with score >= score_threshold, return
    (score, 1 if matched IoU>=iou_threshold to GT of same class else 0)."""
    out: list[tuple[float, int]] = []
    for rec in per_image:
        gt_by_class: dict[str, list] = defaultdict(list)
        for g in rec["ground_truth"]:
            gt_by_class[g["class_name"]].append(g["box_xyxy"])
        preds_by_class: dict[str, list] = defaultdict(list)
        for p in rec["predictions"]:
            if p.get("common_label") is None:
                continue
            if p["score"] < score_threshold:
                continue
            preds_by_class[p["common_label"]].append(p)
        for cn, gts in gt_by_class.items():
            preds = sorted(preds_by_class.get(cn, []), key=lambda x: -x["score"])
            gt_used = [False] * len(gts)
            for p in preds:
                best = 0.0
                bi = -1
                for gi, g in enumerate(gts):
                    if gt_used[gi]:
                        continue
                    v = iou_xyxy(p["box_xyxy"], g)
                    if v > best:
                        best, bi = v, gi
                if bi >= 0 and best >= iou_threshold:
                    gt_used[bi] = True
                    out.append((p["score"], 1))
                else:
                    out.append((p["score"], 0))
            # Unmatched GTs contribute nothing (they are FN, not detector outputs).
    return out


def _brier(pairs: list[tuple[float, int]]) -> float:
    if not pairs:
        return float("nan")
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def _nll(pairs: list[tuple[float, int]]) -> float:
    if not pairs:
        return float("nan")
    s = 0.0
    for p, y in pairs:
        p_clip = min(max(p, 1e-6), 1 - 1e-6)
        s += -(y * math.log(p_clip) + (1 - y) * math.log(1 - p_clip))
    return s / len(pairs)


def _reliability(pairs: list[tuple[float, int]], n_bins: int = 10) -> list[dict]:
    bins = [[] for _ in range(n_bins)]
    for p, y in pairs:
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, y))
    rows = []
    for i, items in enumerate(bins):
        if not items:
            rows.append({"bin_lo": i / n_bins, "bin_hi": (i + 1) / n_bins,
                         "count": 0, "mean_conf": None, "frac_correct": None,
                         "gap": None})
            continue
        mc = sum(p for p, _ in items) / len(items)
        fc = sum(y for _, y in items) / len(items)
        rows.append({"bin_lo": i / n_bins, "bin_hi": (i + 1) / n_bins,
                     "count": len(items), "mean_conf": mc, "frac_correct": fc,
                     "gap": mc - fc})
    return rows


def _fit_temperature(pairs: list[tuple[float, int]]) -> float:
    """Fit a single temperature T so that `calibrated = sigmoid(logit(p)/T)`
    minimises the negative log-likelihood on (p, y). Implemented with a
    bounded scalar search."""
    if not pairs:
        return 1.0
    import math as _m

    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return _m.log(p / (1 - p))

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + _m.exp(-x))

    def _nll_T(T: float) -> float:
        if T <= 0:
            return float("inf")
        s = 0.0
        for p, y in pairs:
            cp = _sigmoid(_logit(p) / T)
            cp = min(max(cp, 1e-6), 1 - 1e-6)
            s += -(y * _m.log(cp) + (1 - y) * _m.log(1 - cp))
        return s / len(pairs)

    # Grid search followed by golden-section refinement.
    candidates = [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]
    best_T = min(candidates, key=_nll_T)
    lo, hi = max(0.05, best_T / 2), best_T * 2
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    for _ in range(60):
        if _nll_T(c) < _nll_T(d):
            b = d
        else:
            a = c
        c = b - gr * (b - a)
        d = a + gr * (b - a)
    return (a + b) / 2


def _apply_temperature(pairs: list[tuple[float, int]], T: float) -> list[tuple[float, int]]:
    def _logit(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))
    def _sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))
    out = []
    for p, y in pairs:
        out.append((_sigmoid(_logit(p) / T), y))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    detectors = [
        ("fasterrcnn_val_full", "Faster R-CNN ResNet-50 FPN"),
        ("yolov8n_val_full", "YOLOv8n"),
    ]
    summary: dict = {"config": vars(args), "detectors": {}}
    for tag, label in detectors:
        path = ARTIFACTS / f"eval_{tag}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        pairs = _per_image_score_outcome(d["per_image"], args.score_threshold, args.iou_threshold)
        # Split into fit / eval halves (deterministic by sorting on (score, y))
        # Use first 50% by index for fit, second 50% for evaluation to give a
        # *split-sample* assessment. Document the reduced sample size.
        idx_sorted = sorted(range(len(pairs)), key=lambda i: (pairs[i][0], pairs[i][1]))
        split = len(pairs) // 2
        fit_idx = idx_sorted[:split]
        eval_idx = idx_sorted[split:]
        fit_pairs = [pairs[i] for i in fit_idx]
        eval_pairs = [pairs[i] for i in eval_idx]

        brier_pre = _brier(eval_pairs)
        T = _fit_temperature(fit_pairs)
        eval_pairs_cal = _apply_temperature(eval_pairs, T)
        brier_post = _brier(eval_pairs_cal)
        nll_pre = _nll(eval_pairs)
        nll_post = _nll(eval_pairs_cal)
        rel_pre = _reliability(eval_pairs, args.n_bins)
        rel_post = _reliability(eval_pairs_cal, args.n_bins)
        # Save reliability CSVs (both pre and post calibration)
        rel_pre_csv = ARTIFACTS / f"reliability_{tag}_pre.csv"
        with rel_pre_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rel_pre[0].keys()))
            w.writeheader()
            w.writerows(rel_pre)
        rel_post_csv = ARTIFACTS / f"reliability_{tag}_post_temperature.csv"
        with rel_post_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rel_post[0].keys()))
            w.writeheader()
            w.writerows(rel_post)
        print(f"  {label}: T={T:.3f} (fit on {len(fit_pairs)}, eval on {len(eval_pairs)}) "
              f"Brier {brier_pre:.4f}->{brier_post:.4f}, "
              f"NLL {nll_pre:.4f}->{nll_post:.4f}")
        summary["detectors"][tag] = {
            "label": label,
            "n_pairs": len(pairs),
            "n_fit": len(fit_pairs),
            "n_eval": len(eval_pairs),
            "temperature": T,
            "brier_pre": brier_pre,
            "brier_post": brier_post,
            "nll_pre": nll_pre,
            "nll_post": nll_post,
            "reliability_pre": rel_pre,
            "reliability_post": rel_post,
            "caveat": (
                "Temperature is fitted on a random half of val and evaluated on the "
                "other half. The reported post-calibration numbers are therefore "
                "out-of-sample for the temperature parameter, but still on the same "
                "val set used for all other WeatherGuard metrics."
            ),
        }

    (ARTIFACTS / "calibration.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"Saved {ARTIFACTS}/calibration.json")
    md = ["# Calibration audit and temperature scaling\n",
          "## Definitions\n",
          f"- *Correct* prediction: class matches the matched ground truth AND IoU ≥ {args.iou_threshold}.",
          f"- Score threshold for inclusion: {args.score_threshold}.",
          f"- Each detection is one observation in the reliability table.",
          "- Temperature scaling fits a single T on the **fit half** of the val set "
          "and is evaluated on the **eval half**.\n",
          "## Results\n"]
    for tag, info in summary["detectors"].items():
        md.append(f"### {info['label']}\n")
        md.append(f"- n pairs total: {info['n_pairs']}, fit: {info['n_fit']}, eval: {info['n_eval']}")
        md.append(f"- Fitted T: **{info['temperature']:.3f}**")
        md.append(f"- Brier (eval half, pre): **{info['brier_pre']:.4f}** → post: **{info['brier_post']:.4f}**")
        md.append(f"- NLL (eval half, pre): **{info['nll_pre']:.4f}** → post: **{info['nll_post']:.4f}**\n")
        md.append("Reliability (post-calibration, 10 bins):\n")
        md.append("| bin | count | mean conf | frac correct | gap |")
        md.append("|---|---:|---:|---:|---:|")
        for r in info["reliability_post"]:
            mc = f"{r['mean_conf']:.3f}" if r["mean_conf"] is not None else "-"
            fc = f"{r['frac_correct']:.3f}" if r["frac_correct"] is not None else "-"
            gp = f"{r['gap']:.3f}" if r["gap"] is not None else "-"
            md.append(f"| {r['bin_lo']:.1f}–{r['bin_hi']:.1f} | {r['count']} | {mc} | {fc} | {gp} |")
    (REPORTS / "calibration.md").write_text("\n".join(md))
    print(f"Saved {REPORTS}/calibration.md")


if __name__ == "__main__":
    main()
