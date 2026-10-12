"""Stage 1 smoke test: run Faster R-CNN on 8 deterministic DAWN val images.

Saves a JSON report under `artifacts/smoke_test_report.json` and a
Markdown summary under `reports/smoke_test_report.md`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.dataset import DawnDataset
from src.model import build_detector
from src.inference import run_inference, predictions_to_eval_records
from src.metrics import evaluate_detections

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
VAL_PARQUET = ROOT / "val-00000-of-00001.parquet"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
ARTIFACTS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def main() -> None:
    print("Loading DAWN validation parquet...")
    t0 = time.perf_counter()
    ds = DawnDataset.from_parquet(VAL_PARQUET)
    load_seconds = time.perf_counter() - t0
    print(f"  loaded {len(ds)} records in {load_seconds:.2f}s")

    sample = ds.deterministic_sample(k=8, seed=42)
    print(f"  selected deterministic sample of {len(sample)} records")

    detector = build_detector("fasterrcnn_resnet50_fpn", score_threshold=0.0)
    print(f"Loading detector: {detector.name} ...")
    t0 = time.perf_counter()
    detector.load()
    print(f"  loaded in {time.perf_counter() - t0:.2f}s (device={detector.device})")

    run = run_inference(detector, sample, score_threshold=0.0, warmup=True)
    n_preds = sum(len(p.detections) for p in run.predictions)
    print(f"  inference: total {run.total_inference_seconds:.2f}s, "
          f"mean {run.mean_inference_seconds*1000:.1f}ms, "
          f"median {run.median_inference_seconds*1000:.1f}ms, "
          f"total detections: {n_preds}")

    eval_records = predictions_to_eval_records(sample, run.predictions)
    eval50 = evaluate_detections(eval_records, iou_threshold=0.5, score_threshold=0.5)
    print("\nPer-class (IoU=0.5, score>=0.5):")
    for c in eval50.per_class:
        print(f"  {c.class_name:<10s} GT={c.n_gt:>4d} PRED={c.n_pred:>4d} "
              f"TP={c.tp:>3d} FP={c.fp:>3d} FN={c.fn:>3d} "
              f"P={c.precision:.3f} R={c.recall:.3f} F1={c.f1:.3f} AP50={c.ap_50:.3f}")
    print(f"  mAP@.5 = {eval50.macro['mAP_50']:.3f}")
    print(f"  micro P/R/F1 = {eval50.micro['precision']:.3f}/"
          f"{eval50.micro['recall']:.3f}/{eval50.micro['f1']:.3f}")

    # Save report
    out = {
        "stage": "smoke_test",
        "n_images": len(sample),
        "dataset_source": str(VAL_PARQUET),
        "detector_name": detector.name,
        "detector_load_seconds": detector.load_seconds,
        "detector_device": detector.device,
        "weights_provenance": detector.weights_provenance,
        "license_note": detector.license_note,
        "inference_total_seconds": run.total_inference_seconds,
        "inference_mean_seconds": run.mean_inference_seconds,
        "inference_median_seconds": run.median_inference_seconds,
        "total_detections": n_preds,
        "per_image_prediction_counts": [len(p.detections) for p in run.predictions],
        "image_ids": [r.image_id for r in sample],
        "eval_at_iou0.5_score0.5": eval50.to_dict(),
    }
    (ARTIFACTS / "smoke_test_report.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved {ARTIFACTS/'smoke_test_report.json'}")

    md_lines = [
        "# Smoke Test Report",
        "",
        f"- Detector: `{detector.name}`",
        f"- Device: `{detector.device}`",
        f"- Weights provenance: {detector.weights_provenance}",
        f"- License note: {detector.license_note}",
        f"- n_images: {len(sample)} (deterministic, seed=42)",
        f"- Detector load seconds: {detector.load_seconds:.2f}",
        f"- Total inference seconds: {run.total_inference_seconds:.2f}",
        f"- Mean per-image inference: {run.mean_inference_seconds*1000:.1f} ms",
        f"- Median per-image inference: {run.median_inference_seconds*1000:.1f} ms",
        f"- Total detections: {n_preds}",
        "",
        "## Per-class (IoU=0.5, score>=0.5)",
        "",
        "| class | GT | PRED | TP | FP | FN | P | R | F1 | AP@.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in eval50.per_class:
        md_lines.append(
            f"| {c.class_name} | {c.n_gt} | {c.n_pred} | {c.tp} | {c.fp} | {c.fn} | "
            f"{c.precision:.3f} | {c.recall:.3f} | {c.f1:.3f} | {c.ap_50:.3f} |"
        )
    md_lines += [
        f"| **mAP@.5** |  |  |  |  |  |  |  |  | **{eval50.macro['mAP_50']:.3f}** |",
        "",
        f"**Micro** P={eval50.micro['precision']:.3f} R={eval50.micro['recall']:.3f} F1={eval50.micro['f1']:.3f}",
        "",
    ]
    (REPORTS / "smoke_test_report.md").write_text("\n".join(md_lines))
    print(f"Saved {REPORTS/'smoke_test_report.md'}")


if __name__ == "__main__":
    main()
