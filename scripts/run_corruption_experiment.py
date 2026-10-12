"""Stage 4 — controlled corruption robustness experiment.

For each corruption in `default_corruption_grid()`:

  1. Apply the corruption to the *same* source images.
  2. Run the *same* detector and the *same* evaluation protocol.
  3. Save per-corruption aggregate metrics as JSON + a per-image CSV.

Bbox annotations are reused unchanged because every supported corruption
is photometric and preserves object geometry.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

from src.dataset import DawnDataset, DawnRecord
from src.experiments import CorruptionConfig, corrupt_image, default_corruption_grid
from src.model import build_detector
from src.inference import predictions_to_eval_records
from src.metrics import evaluate_detections

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"


def _record_with_image(rec: DawnRecord, img: Image.Image) -> DawnRecord:
    """Return a shallow copy of `rec` with overridden PIL image bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    new = DawnRecord(
        image_id=rec.image_id,
        image_bytes=buf.getvalue(),
        width=img.width,
        height=img.height,
        objects=rec.objects,
    )
    return new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=ROOT / "val-00000-of-00001.parquet")
    parser.add_argument("--detector", type=str, default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--limit", type=int, default=30, help="images per condition")
    parser.add_argument("--tag", type=str, default="fasterrcnn_corruption")
    args = parser.parse_args()

    ds = DawnDataset.from_parquet(args.parquet)
    sample = ds.stratified_condition_sample(per_condition=args.limit, seed=42)
    print(f"Stratified sample: {len(sample)} images across {len({s.condition for s in sample})} conditions")

    detector = build_detector(args.detector, score_threshold=0.0)
    print(f"Loading detector: {detector.name}")
    detector.load()
    print(f"  loaded in {detector.load_seconds:.2f}s on {detector.device}")

    grid = default_corruption_grid()
    rows_csv: list[dict] = []
    per_corruption_summary: dict[str, dict] = {}
    per_image_records: dict[str, list[dict]] = {}

    for cfg in grid:
        print(f"\n=== Corruption: {cfg.name} ({cfg.description}) ===")
        # Apply corruption to each image, then run inference.
        corrupted_records = []
        for r in sample:
            img = r.to_pil()
            cimg = corrupt_image(img, cfg, seed=0)
            corrupted_records.append(_record_with_image(r, cimg))

        t0 = time.perf_counter()
        from src.inference import run_inference
        run = run_inference(detector, corrupted_records, score_threshold=0.0, warmup=False)
        elapsed = time.perf_counter() - t0

        eval_records = predictions_to_eval_records(corrupted_records, run.predictions)
        eval50 = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=0.5)

        per_image_records[cfg.name] = eval_records
        per_corruption_summary[cfg.name] = {
            "description": cfg.description,
            "params": cfg.params,
            "inference_total_seconds": run.total_inference_seconds,
            "inference_mean_seconds": run.mean_inference_seconds,
            "inference_median_seconds": run.median_inference_seconds,
            "n_images": eval50.n_images,
            "micro": eval50.micro,
            "macro": eval50.macro,
            "per_class": [c.__dict__ for c in eval50.per_class],
        }
        print(f"  mAP@.5 = {eval50.macro['mAP_50']:.3f}, "
              f"P={eval50.micro['precision']:.3f} R={eval50.micro['recall']:.3f} F1={eval50.micro['f1']:.3f} "
              f"(elapsed {elapsed:.1f}s)")

        for er in eval_records:
            n_pred = sum(1 for p in er["predictions"] if p["common_label"] is not None and p["score"] >= 0.5)
            n_tp_proxy = None  # not computed per-image here; use aggregate
            rows_csv.append(
                {
                    "corruption": cfg.name,
                    "image_id": er["image_id"],
                    "condition": er["condition"],
                    "n_predictions_score>=0.5": n_pred,
                    "n_ground_truth": len(er["ground_truth"]),
                }
            )

    out_json = ARTIFACTS / f"corruption_{args.tag}.json"
    out_json.write_text(json.dumps(per_corruption_summary, indent=2, default=float))
    print(f"\nSaved {out_json}")

    out_csv = ARTIFACTS / f"corruption_{args.tag}.csv"
    with out_csv.open("w", newline="") as f:
        if rows_csv:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)
    print(f"Saved {out_csv}")

    # Also save per-corruption per-image predictions (for downstream bootstrap).
    out_per_image = ARTIFACTS / f"corruption_{args.tag}_per_image.json"
    out_per_image.write_text(json.dumps(per_image_records, indent=2, default=float))
    print(f"Saved {out_per_image}")


if __name__ == "__main__":
    main()
