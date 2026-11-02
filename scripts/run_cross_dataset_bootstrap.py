"""Statistical bootstrap analysis for cross-dataset generalization.

Implements:
1. Paired image-level bootstrap comparing Faster R-CNN vs YOLOv8n on external images.
2. Paired image-level bootstrap comparing DAWN-selected thresholds vs default 0.50.
3. Independent two-sample bootstrap comparing DAWN held-out vs BDD100K external.

Generates artifacts/cross_dataset_bootstrap.json with 95% confidence intervals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import evaluate_detections

ARTIFACTS = ROOT / "artifacts"
BOOTSTRAP_SEED = 20261101
N_BOOTSTRAP = 1000


def bootstrap_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    sorted_vals = np.sort(values)
    lower = float(np.percentile(sorted_vals, 100 * (alpha / 2)))
    upper = float(np.percentile(sorted_vals, 100 * (1 - alpha / 2)))
    return lower, upper


def evaluate_records_at_threshold(records: list[dict], threshold: float) -> tuple[int, int, int]:
    """Fast tally of TP, FP, FN for a collection of per-image eval records."""
    ev = evaluate_detections(records, iou_threshold=0.5, score_threshold=threshold)
    return ev.micro["tp"], ev.micro["fp"], ev.micro["fn"]


def compute_f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0


def compute_precision_from_counts(tp: int, fp: int) -> float:
    denom = tp + fp
    return (float(tp) / denom) if denom > 0 else 0.0


def compute_recall_from_counts(tp: int, fn: int) -> float:
    denom = tp + fn
    return (float(tp) / denom) if denom > 0 else 0.0


def main() -> None:
    print("=" * 70)
    print("WEATHERGUARD MISSION 2 — STATISTICAL BOOTSTRAP ANALYSIS")
    print("=" * 70)

    # 1. Load prediction files
    frcnn_ext_file = ARTIFACTS / "eval_external_fasterrcnn_resnet50_fpn.json"
    yolo_ext_file = ARTIFACTS / "eval_external_yolov8n.json"
    frcnn_dawn_file = ARTIFACTS / "eval_heldout_fasterrcnn.json"
    yolo_dawn_file = ARTIFACTS / "eval_heldout_yolov8n.json"

    for p in [frcnn_ext_file, yolo_ext_file, frcnn_dawn_file, yolo_dawn_file]:
        if not p.exists():
            raise FileNotFoundError(f"Required eval file missing: {p}")

    with open(frcnn_ext_file) as f:
        frcnn_ext = json.load(f)["per_image"]
    with open(yolo_ext_file) as f:
        yolo_ext = json.load(f)["per_image"]
    with open(frcnn_dawn_file) as f:
        frcnn_dawn = json.load(f)["per_image"]
    with open(yolo_dawn_file) as f:
        yolo_dawn = json.load(f)["per_image"]

    n_ext = len(frcnn_ext)
    n_dawn = len(frcnn_dawn)
    print(f"Loaded: External n={n_ext}, DAWN held-out n={n_dawn}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # Precompute per-image TP/FP/FN for fast vectorised resampling
    # For external:
    ext_frcnn_dawn_th = [evaluate_records_at_threshold([rec], 0.60) for rec in frcnn_ext]
    ext_frcnn_def_th = [evaluate_records_at_threshold([rec], 0.50) for rec in frcnn_ext]
    ext_yolo_dawn_th = [evaluate_records_at_threshold([rec], 0.20) for rec in yolo_ext]
    ext_yolo_def_th = [evaluate_records_at_threshold([rec], 0.50) for rec in yolo_ext]

    # For DAWN heldout:
    dawn_frcnn_dawn_th = [evaluate_records_at_threshold([rec], 0.60) for rec in frcnn_dawn]
    dawn_yolo_dawn_th = [evaluate_records_at_threshold([rec], 0.20) for rec in yolo_dawn]

    # -------------------------------------------------------------
    # 1. PAIRED BOOTSTRAP: Faster R-CNN vs YOLOv8n on External Data
    # -------------------------------------------------------------
    print("\nRunning Analysis 1: Paired bootstrap (FRCNN @ 0.60 vs YOLO @ 0.20 on external)...")
    delta_f1_ext_model = []
    delta_p_ext_model = []
    delta_r_ext_model = []

    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n_ext, size=n_ext, replace=True)
        # FRCNN
        tp_f = sum(ext_frcnn_dawn_th[i][0] for i in idx)
        fp_f = sum(ext_frcnn_dawn_th[i][1] for i in idx)
        fn_f = sum(ext_frcnn_dawn_th[i][2] for i in idx)
        f1_f = compute_f1_from_counts(tp_f, fp_f, fn_f)
        p_f = compute_precision_from_counts(tp_f, fp_f)
        r_f = compute_recall_from_counts(tp_f, fn_f)

        # YOLO
        tp_y = sum(ext_yolo_dawn_th[i][0] for i in idx)
        fp_y = sum(ext_yolo_dawn_th[i][1] for i in idx)
        fn_y = sum(ext_yolo_dawn_th[i][2] for i in idx)
        f1_y = compute_f1_from_counts(tp_y, fp_y, fn_y)
        p_y = compute_precision_from_counts(tp_y, fp_y)
        r_y = compute_recall_from_counts(tp_y, fn_y)

        delta_f1_ext_model.append(f1_f - f1_y)
        delta_p_ext_model.append(p_f - p_y)
        delta_r_ext_model.append(r_f - r_y)

    # -------------------------------------------------------------
    # 2. PAIRED BOOTSTRAP: DAWN Threshold vs Default 0.50 Threshold
    # -------------------------------------------------------------
    print("Running Analysis 2: Paired bootstrap (DAWN-selected threshold vs default 0.50)...")
    delta_f1_frcnn_th = []
    delta_f1_yolo_th = []

    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n_ext, size=n_ext, replace=True)
        # FRCNN @ 0.60 vs 0.50
        tp_f6 = sum(ext_frcnn_dawn_th[i][0] for i in idx)
        fp_f6 = sum(ext_frcnn_dawn_th[i][1] for i in idx)
        fn_f6 = sum(ext_frcnn_dawn_th[i][2] for i in idx)
        f1_f6 = compute_f1_from_counts(tp_f6, fp_f6, fn_f6)

        tp_f5 = sum(ext_frcnn_def_th[i][0] for i in idx)
        fp_f5 = sum(ext_frcnn_def_th[i][1] for i in idx)
        fn_f5 = sum(ext_frcnn_def_th[i][2] for i in idx)
        f1_f5 = compute_f1_from_counts(tp_f5, fp_f5, fn_f5)
        delta_f1_frcnn_th.append(f1_f6 - f1_f5)

        # YOLO @ 0.20 vs 0.50
        tp_y2 = sum(ext_yolo_dawn_th[i][0] for i in idx)
        fp_y2 = sum(ext_yolo_dawn_th[i][1] for i in idx)
        fn_y2 = sum(ext_yolo_dawn_th[i][2] for i in idx)
        f1_y2 = compute_f1_from_counts(tp_y2, fp_y2, fn_y2)

        tp_y5 = sum(ext_yolo_def_th[i][0] for i in idx)
        fp_y5 = sum(ext_yolo_def_th[i][1] for i in idx)
        fn_y5 = sum(ext_yolo_def_th[i][2] for i in idx)
        f1_y5 = compute_f1_from_counts(tp_y5, fp_y5, fn_y5)
        delta_f1_yolo_th.append(f1_y2 - f1_y5)

    # -------------------------------------------------------------
    # 3. INDEPENDENT TWO-SAMPLE BOOTSTRAP: External vs DAWN Heldout
    # -------------------------------------------------------------
    print("Running Analysis 3: Independent bootstrap (External vs DAWN heldout)...")
    delta_f1_cross_frcnn = []
    delta_f1_cross_yolo = []

    for _ in range(N_BOOTSTRAP):
        idx_ext = rng.choice(n_ext, size=n_ext, replace=True)
        idx_dawn = rng.choice(n_dawn, size=n_dawn, replace=True)

        # FRCNN cross
        tp_ext_f = sum(ext_frcnn_dawn_th[i][0] for i in idx_ext)
        fp_ext_f = sum(ext_frcnn_dawn_th[i][1] for i in idx_ext)
        fn_ext_f = sum(ext_frcnn_dawn_th[i][2] for i in idx_ext)
        f1_ext_f = compute_f1_from_counts(tp_ext_f, fp_ext_f, fn_ext_f)

        tp_dwn_f = sum(dawn_frcnn_dawn_th[i][0] for i in idx_dawn)
        fp_dwn_f = sum(dawn_frcnn_dawn_th[i][1] for i in idx_dawn)
        fn_dwn_f = sum(dawn_frcnn_dawn_th[i][2] for i in idx_dawn)
        f1_dwn_f = compute_f1_from_counts(tp_dwn_f, fp_dwn_f, fn_dwn_f)
        delta_f1_cross_frcnn.append(f1_ext_f - f1_dwn_f)

        # YOLO cross
        tp_ext_y = sum(ext_yolo_dawn_th[i][0] for i in idx_ext)
        fp_ext_y = sum(ext_yolo_dawn_th[i][1] for i in idx_ext)
        fn_ext_y = sum(ext_yolo_dawn_th[i][2] for i in idx_ext)
        f1_ext_y = compute_f1_from_counts(tp_ext_y, fp_ext_y, fn_ext_y)

        tp_dwn_y = sum(dawn_yolo_dawn_th[i][0] for i in idx_dawn)
        fp_dwn_y = sum(dawn_yolo_dawn_th[i][1] for i in idx_dawn)
        fn_dwn_y = sum(dawn_yolo_dawn_th[i][2] for i in idx_dawn)
        f1_dwn_y = compute_f1_from_counts(tp_dwn_y, fp_dwn_y, fn_dwn_y)
        delta_f1_cross_yolo.append(f1_ext_y - f1_dwn_y)

    # Assemble summary
    bootstrap_results = {
        "metadata": {
            "n_bootstrap": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
            "n_external_images": n_ext,
            "n_dawn_images": n_dawn,
            "confidence_level": 0.95,
        },
        "paired_model_comparison_external": {
            "comparison": "Faster R-CNN (0.60) minus YOLOv8n (0.20) on External Cohort",
            "delta_f1": {
                "mean": float(np.mean(delta_f1_ext_model)),
                "std_err": float(np.std(delta_f1_ext_model)),
                "ci_95": list(bootstrap_ci(delta_f1_ext_model)),
            },
            "delta_precision": {
                "mean": float(np.mean(delta_p_ext_model)),
                "std_err": float(np.std(delta_p_ext_model)),
                "ci_95": list(bootstrap_ci(delta_p_ext_model)),
            },
            "delta_recall": {
                "mean": float(np.mean(delta_r_ext_model)),
                "std_err": float(np.std(delta_r_ext_model)),
                "ci_95": list(bootstrap_ci(delta_r_ext_model)),
            },
        },
        "paired_threshold_comparison_external": {
            "fasterrcnn_0.60_minus_0.50": {
                "mean": float(np.mean(delta_f1_frcnn_th)),
                "std_err": float(np.std(delta_f1_frcnn_th)),
                "ci_95": list(bootstrap_ci(delta_f1_frcnn_th)),
            },
            "yolov8n_0.20_minus_0.50": {
                "mean": float(np.mean(delta_f1_yolo_th)),
                "std_err": float(np.std(delta_f1_yolo_th)),
                "ci_95": list(bootstrap_ci(delta_f1_yolo_th)),
            },
        },
        "independent_cross_dataset_comparison": {
            "fasterrcnn_external_minus_dawn_heldout": {
                "mean": float(np.mean(delta_f1_cross_frcnn)),
                "std_err": float(np.std(delta_f1_cross_frcnn)),
                "ci_95": list(bootstrap_ci(delta_f1_cross_frcnn)),
            },
            "yolov8n_external_minus_dawn_heldout": {
                "mean": float(np.mean(delta_f1_cross_yolo)),
                "std_err": float(np.std(delta_f1_cross_yolo)),
                "ci_95": list(bootstrap_ci(delta_f1_cross_yolo)),
            },
        },
    }

    out_json = ARTIFACTS / "cross_dataset_bootstrap.json"
    with open(out_json, "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"\nSaved cross_dataset_bootstrap.json to {out_json}")
    print("Paired Model F1 Diff (FRCNN - YOLO): "
          f"{bootstrap_results['paired_model_comparison_external']['delta_f1']['mean']:.4f} "
          f"95% CI {bootstrap_results['paired_model_comparison_external']['delta_f1']['ci_95']}")
    print("YOLO Threshold Transfer Benefit (0.20 - 0.50): "
          f"{bootstrap_results['paired_threshold_comparison_external']['yolov8n_0.20_minus_0.50']['mean']:.4f} "
          f"95% CI {bootstrap_results['paired_threshold_comparison_external']['yolov8n_0.20_minus_0.50']['ci_95']}")


if __name__ == "__main__":
    main()
