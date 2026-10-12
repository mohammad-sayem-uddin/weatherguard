"""Stage 2/3: run the chosen detector on the entire DAWN val split.

Writes per-image predictions and aggregate metrics to `artifacts/`.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_full_eval.py
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from src.dataset import DawnDataset
from src.model import build_detector
from src.inference import run_inference, predictions_to_eval_records
from src.metrics import evaluate_detections

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=ROOT / "val-00000-of-00001.parquet")
    parser.add_argument("--detector", type=str, default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--tag", type=str, default="fasterrcnn_val")
    args = parser.parse_args()

    ds = DawnDataset.from_parquet(args.parquet)
    records = list(ds)
    if args.limit:
        records = ds.deterministic_sample(args.limit, seed=42)
    print(f"Loaded {len(records)} records from {args.parquet}")

    detector = build_detector(args.detector, score_threshold=args.score_threshold)
    print(f"Loading detector: {detector.name}")
    detector.load()
    print(f"  loaded in {detector.load_seconds:.2f}s on {detector.device}")

    run = run_inference(detector, records, score_threshold=args.score_threshold, warmup=True)
    print(
        f"Inference: total {run.total_inference_seconds:.2f}s, "
        f"mean {run.mean_inference_seconds*1000:.1f}ms, "
        f"median {run.median_inference_seconds*1000:.1f}ms"
    )

    eval_records = predictions_to_eval_records(records, run.predictions)
    eval50 = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=0.5)
    eval30 = evaluate_detections(eval_records, iou_threshold=0.3, score_threshold=0.5)
    print(f"\nmAP@.5 (score>=0.5): {eval50.macro['mAP_50']:.3f}")
    print(f"mAP@.3 (score>=0.5): {eval30.macro['mAP_50']:.3f}")
    print(f"Micro P/R/F1 @ IoU=.5 score>=.5: {eval50.micro['precision']:.3f}/{eval50.micro['recall']:.3f}/{eval50.micro['f1']:.3f}")

    # Per-condition metrics
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for er in eval_records:
        by_cond[er["condition"]].append(er)
    per_condition = {}
    for cond, items in by_cond.items():
        e = evaluate_detections(items, iou_threshold=0.5, score_threshold=0.5)
        per_condition[cond] = {
            "n_images": e.n_images,
            "micro": e.micro,
            "macro": e.macro,
            "per_class": [c.__dict__ for c in e.per_class],
        }

    out = {
        "tag": args.tag,
        "detector_name": detector.name,
        "detector_load_seconds": detector.load_seconds,
        "device": detector.device,
        "n_images": len(records),
        "score_threshold_at_filter": args.score_threshold,
        "weights_provenance": detector.weights_provenance,
        "license_note": detector.license_note,
        "inference_total_seconds": run.total_inference_seconds,
        "inference_mean_seconds": run.mean_inference_seconds,
        "inference_median_seconds": run.median_inference_seconds,
        "eval_iou0.5_score0.5": eval50.to_dict(),
        "eval_iou0.3_score0.5": eval30.to_dict(),
        "per_condition_iou0.5_score0.5": per_condition,
        "per_image": [
            {
                "image_id": er["image_id"],
                "condition": er["condition"],
                "predictions": er["predictions"],
                "ground_truth": er["ground_truth"],
            }
            for er in eval_records
        ],
    }
    out_path = ARTIFACTS / f"eval_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"Saved {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
