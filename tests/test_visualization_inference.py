"""Tests for visualization + inference postprocessing helpers.

These tests do not load a real detector; they verify the
postprocessing path and rendering utilities that run after inference.
"""
from __future__ import annotations

import io
import unittest
from pathlib import Path

from PIL import Image

from src.dataset import BBox, DawnRecord
from src.inference import (
    _median,
    predictions_to_eval_records,
    run_inference,
)
from src.model import Detection, Prediction
from src.visualization import draw_boxes, render_prediction, render_record


def _make_record(image_id: str = "foggy-001", w: int = 100, h: int = 80) -> DawnRecord:
    img = Image.new("RGB", (w, h), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return DawnRecord(
        image_id=image_id,
        image_bytes=buf.getvalue(),
        width=w,
        height=h,
        objects=[BBox(10, 5, 30, 20, "Car"), BBox(60, 30, 35, 25, "Truck")],
    )


class _DummyDetector:
    name = "dummy"
    weights_provenance = "test"
    license_note = "test"
    device = "cpu"
    load_seconds = 0.0

    def __init__(self):
        self._loaded = True

    def load(self):
        self._loaded = True

    def predict(self, image, image_id=""):
        return Prediction(
            image_id=image_id,
            width=image.width,
            height=image.height,
            detections=[
                Detection(
                    box_xyxy=(10.0, 5.0, 40.0, 25.0),
                    score=0.9,
                    label="car",
                    common_label="Car",
                ),
                Detection(
                    box_xyxy=(60.0, 30.0, 95.0, 55.0),
                    score=0.4,
                    label="truck",
                    common_label="Truck",
                ),
            ],
            inference_seconds=0.01,
        )

    def warmup(self):
        return None


class TestInferencePostprocess(unittest.TestCase):
    def test_predictions_to_eval_records(self):
        rec = _make_record()
        pred = _DummyDetector().predict(rec.to_pil(), rec.image_id)
        records = predictions_to_eval_records([rec], [pred])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["image_id"], "foggy-001")
        self.assertEqual(records[0]["condition"], "foggy")
        self.assertEqual(len(records[0]["predictions"]), 2)
        self.assertEqual(len(records[0]["ground_truth"]), 2)
        for g in records[0]["ground_truth"]:
            self.assertEqual(len(g["box_xyxy"]), 4)

    def test_predictions_length_mismatch_raises(self):
        rec = _make_record()
        pred = _DummyDetector().predict(rec.to_pil(), rec.image_id)
        with self.assertRaises(ValueError):
            predictions_to_eval_records([rec, rec], [pred])

    def test_score_threshold_filtering(self):
        rec = _make_record()
        det = _DummyDetector()
        det.load()
        run = run_inference(det, [rec], score_threshold=0.5, warmup=False)
        # The 0.4-score truck should be filtered, leaving only the car.
        self.assertEqual(len(run.predictions[0].detections), 1)
        self.assertEqual(run.predictions[0].detections[0].label, "car")

    def test_median(self):
        self.assertEqual(_median([5, 1, 3]), 3)
        self.assertEqual(_median([1, 2, 3, 4]), 2.5)
        self.assertEqual(_median([]), 0.0)

    def test_run_inference_empty_raises(self):
        det = _DummyDetector()
        with self.assertRaises(ValueError):
            run_inference(det, [], score_threshold=0.0, warmup=False)


class TestVisualization(unittest.TestCase):
    def test_draw_boxes_returns_image(self):
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        out = draw_boxes(img, [(0, 0, 50, 50)], ["hello"], color=(0, 0, 255))
        self.assertEqual(out.size, (100, 100))
        # Output is a new image, not the same object.
        self.assertIsNot(out, img)

    def test_render_record(self):
        rec = _make_record()
        out = render_record(rec)
        self.assertEqual(out.size, (100, 80))

    def test_render_prediction(self):
        rec = _make_record()
        pred = _DummyDetector().predict(rec.to_pil(), rec.image_id)
        out = render_prediction(rec, pred, score_threshold=0.0)
        self.assertEqual(out.size, (100, 80))
        out2 = render_prediction(rec, pred, score_threshold=0.5)
        # Same size, possibly different pixels.
        self.assertEqual(out2.size, (100, 80))


if __name__ == "__main__":
    unittest.main()
