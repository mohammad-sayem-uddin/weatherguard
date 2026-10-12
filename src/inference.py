"""Run inference and assemble per-image evaluation records."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from PIL import Image

from .dataset import DawnRecord
from .model import Detector, Prediction


@dataclass
class InferenceRun:
    """All outputs from a single end-to-end inference run."""
    detector_name: str
    n_images: int
    predictions: list[Prediction] = field(default_factory=list)
    load_seconds: float = 0.0
    total_inference_seconds: float = 0.0
    mean_inference_seconds: float = 0.0
    median_inference_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dicts(self) -> list[dict]:
        return {
            "detector_name": self.detector_name,
            "n_images": self.n_images,
            "load_seconds": self.load_seconds,
            "total_inference_seconds": self.total_inference_seconds,
            "mean_inference_seconds": self.mean_inference_seconds,
            "median_inference_seconds": self.median_inference_seconds,
            "n_predictions": [len(p.detections) for p in self.predictions],
            "notes": self.notes,
        }


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return float(s[n // 2])
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def run_inference(
    detector: Detector,
    records: Sequence[DawnRecord],
    *,
    score_threshold: float = 0.0,
    warmup: bool = True,
) -> InferenceRun:
    """Run `detector` on each DawnRecord and return a structured run."""
    if not records:
        raise ValueError("run_inference requires at least one record.")
    if not detector.__class__.__name__.startswith("Base") and not getattr(detector, "model", None):
        detector.load()
    if warmup:
        try:
            detector.warmup()
        except Exception as exc:  # warmup failures must not stop inference
            detector.__dict__.setdefault("notes", []).append(f"warmup failed: {exc}")

    t0 = time.perf_counter()
    predictions: list[Prediction] = []
    latencies: list[float] = []
    for rec in records:
        img = rec.to_pil()
        pred = detector.predict(img, image_id=rec.image_id)
        # Apply score threshold (filtering can also happen inside detector.predict).
        if score_threshold > 0:
            pred.detections = [d for d in pred.detections if d.score >= score_threshold]
        predictions.append(pred)
        latencies.append(pred.inference_seconds)
    elapsed = time.perf_counter() - t0

    load_seconds = getattr(detector, "_load_seconds", 0.0) or getattr(detector, "load_seconds", 0.0)
    return InferenceRun(
        detector_name=detector.name,
        n_images=len(records),
        predictions=predictions,
        load_seconds=float(load_seconds),
        total_inference_seconds=elapsed,
        mean_inference_seconds=sum(latencies) / len(latencies) if latencies else 0.0,
        median_inference_seconds=_median(latencies),
    )


def predictions_to_eval_records(
    records: Sequence[DawnRecord],
    predictions: Sequence[Prediction],
) -> list[dict]:
    """Convert (records, predictions) into the per-image format expected
    by `metrics.evaluate_detections`."""
    if len(records) != len(predictions):
        raise ValueError(
            f"records and predictions length mismatch: {len(records)} vs {len(predictions)}"
        )
    out: list[dict] = []
    for rec, pred in zip(records, predictions):
        out.append(
            {
                "image_id": rec.image_id,
                "condition": rec.condition,
                "predictions": [
                    {
                        "label": d.label,
                        "common_label": d.common_label,
                        "score": d.score,
                        "box_xyxy": d.box_xyxy,
                    }
                    for d in pred.detections
                ],
                "ground_truth": [
                    {
                        "class_name": b.class_name,
                        "box_xyxy": b.to_xyxy(),
                    }
                    for b in rec.objects
                ],
            }
        )
    return out
