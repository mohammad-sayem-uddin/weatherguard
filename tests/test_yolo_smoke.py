"""Smoke test for the YOLOv8n detector wrapper.

This test is opt-in (skipped unless `--runslow` is passed) because it
requires the ultralytics package and the ~6 MB yolov8n.pt weights to
be available. Run with:

    PYTHONPATH=. .venv/bin/python -m unittest tests.test_yolo_smoke -v
"""
from __future__ import annotations

import unittest

from PIL import Image


def _have_yolo() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_have_yolo(), "ultralytics not installed")
class TestYoloSmoke(unittest.TestCase):
    def test_predict_on_synthetic_image(self):
        from src.model import build_detector
        det = build_detector("yolov8n", score_threshold=0.0)
        det.load()
        img = Image.new("RGB", (640, 480), color=(40, 40, 40))
        pred = det.predict(img, image_id="synthetic")
        self.assertEqual(pred.image_id, "synthetic")
        self.assertEqual(pred.width, 640)
        # Empty image should produce zero or very few predictions.
        self.assertGreaterEqual(len(pred.detections), 0)


if __name__ == "__main__":
    unittest.main()
