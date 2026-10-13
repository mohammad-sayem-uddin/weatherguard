"""Run a detector on a DAWN split and write a per-image eval record.

Used by the focused threshold-transfer study to:
  * Run each detector on the train (DEVELOPMENT) split.
  * Run each detector on the clean 202-image val (HELD-OUT) split.

Output JSON has the same schema as the existing eval_*.json files so
that downstream scripts (threshold sweep, bootstrap, calibration) can
reuse it directly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.dataset import DawnDataset
from src.inference import run_inference, predictions_to_eval_records
from src.metrics import evaluate_detections
from src.model import build_detector

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--detector", type=str, default="fasterrcnn_resnet50_fpn")
    ap.add_argument("--score-threshold", type=float, default=0.0)
    ap.add_argument("--heldout-image-ids", type=Path, default=None,
                    help="Optional JSON file with key 'heldout_image_ids'; "
                         "if provided, only those images are used.")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--split-role", type=str, choices=["development", "heldout"],
                    required=True)
    args = ap.parse_args()

    ds = DawnDataset.from_parquet(args.parquet)
    if args.heldout_image_ids is not None:
        info = json.loads(args.heldout_image_ids.read_text())
        keep = set(info["heldout_image_ids"])
        records = [r for r in ds if r.image_id in keep]
        excluded = info.get("excluded_image_ids", [])
    else:
        records = list(ds)
        excluded = []
    records.sort(key=lambda r: r.image_id)
    print(f"Loaded {len(ds)} records; using {len(records)} after filter.")

    detector = build_detector(args.detector, score_threshold=args.score_threshold)
    print(f"Loading detector: {detector.name}")
    detector.load()
    print(f"  loaded in {detector.load_seconds:.2f}s on {detector.device}")

    run = run_inference(detector, records, score_threshold=args.score_threshold, warmup=True)
    print(f"Inference: total {run.total_inference_seconds:.2f}s, "
          f"mean {run.mean_inference_seconds*1000:.1f}ms, "
          f"median {run.median_inference_seconds*1000:.1f}ms")

    eval_records = predictions_to_eval_records(records, run.predictions)
    eval50 = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=0.5)
    print(f"mAP@.5: {eval50.macro['mAP_50']:.4f} | "
          f"Micro P/R/F1: {eval50.micro['precision']:.3f}/"
          f"{eval50.micro['recall']:.3f}/{eval50.micro['f1']:.3f}")

    out = {
        "tag": args.tag,
        "split_role": args.split_role,
        "parquet": str(args.parquet),
        "excluded_image_ids": excluded,
        "n_images": len(records),
        "image_ids": [r.image_id for r in records],
        "detector_name": detector.name,
        "detector_load_seconds": detector.load_seconds,
        "device": detector.device,
        "weights_provenance": detector.weights_provenance,
        "license_note": detector.license_note,
        "inference_total_seconds": run.total_inference_seconds,
        "inference_mean_seconds": run.mean_inference_seconds,
        "inference_median_seconds": run.median_inference_seconds,
        "inference_warmup": True,
        "per_image": eval_records,
        "eval_iou0.5_score0.5": eval50.to_dict(),
    }
    out_path = ARTIFACTS / f"eval_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"Saved {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
