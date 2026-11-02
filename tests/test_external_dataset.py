"""Tests for src/external_dataset — BDD100K adapter and quality control.

Uses mock JSON structures to keep tests deterministic, fast, and hermetic.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.dataset import BBox
from src.external_dataset import (
    BDD100K_TO_CANONICAL,
    CANONICAL_CLASSES,
    Bdd100kDataset,
    ExternalRecord,
)


class TestExternalDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_class_mapping_taxonomy(self) -> None:
        """Verify that target 6 classes map correctly and rider is strictly excluded."""
        self.assertEqual(BDD100K_TO_CANONICAL["pedestrian"], "Person")
        self.assertEqual(BDD100K_TO_CANONICAL["car"], "Car")
        self.assertEqual(BDD100K_TO_CANONICAL["truck"], "Truck")
        self.assertEqual(BDD100K_TO_CANONICAL["bus"], "Bus")
        self.assertEqual(BDD100K_TO_CANONICAL["motorcycle"], "Motorcycle")
        self.assertEqual(BDD100K_TO_CANONICAL["bicycle"], "Bicycle")
        # Strict exclusion of rider and non-target categories
        self.assertIsNone(BDD100K_TO_CANONICAL["rider"])
        self.assertIsNone(BDD100K_TO_CANONICAL["traffic light"])
        self.assertIsNone(BDD100K_TO_CANONICAL["traffic sign"])
        self.assertIsNone(BDD100K_TO_CANONICAL["train"])

    def test_mock_json_parsing_and_coordinate_conversion(self) -> None:
        """Verify relative-to-absolute box conversion and clipping."""
        mock_data = {
            "samples": [
                {
                    "filepath": "data/sample_001.jpg",
                    "metadata": {"width": 1280, "height": 720},
                    "weather": {"label": "rainy"},
                    "timeofday": {"label": "daytime"},
                    "scene": {"label": "city street"},
                    "detections": {
                        "detections": [
                            {
                                "label": "car",
                                "bounding_box": [0.1, 0.2, 0.3, 0.4],  # x, y, w, h
                            },
                            {
                                "label": "pedestrian",
                                "bounding_box": [0.5, 0.6, 0.05, 0.1],
                            },
                            {
                                "label": "rider",  # Should be excluded from objects
                                "bounding_box": [0.6, 0.6, 0.05, 0.1],
                            },
                            {
                                "label": "car",
                                "bounding_box": [0.95, 0.95, 0.1, 0.1],  # Overflows boundary
                            },
                            {
                                "label": "bus",
                                "bounding_box": [0.2, 0.2, -0.1, 0.1],  # Invalid negative width
                            },
                        ]
                    },
                }
            ]
        }
        json_path = self.root / "mock_samples.json"
        json_path.write_text(json.dumps(mock_data))

        dataset = Bdd100kDataset.from_fiftyone_json(json_path)
        self.assertEqual(len(dataset), 1)

        rec = dataset[0]
        self.assertEqual(rec.image_id, "sample_001.jpg")
        self.assertEqual(rec.condition, "rainy")
        self.assertEqual(rec.timeofday, "daytime")
        self.assertEqual(rec.width, 1280)
        self.assertEqual(rec.height, 720)

        # Target objects should contain car (valid), pedestrian (valid), car (clipped).
        # Rider should be excluded, negative width bus should be rejected.
        self.assertEqual(len(rec.objects), 3)

        # Check Car 1 coordinates: x=128, y=144, w=384, h=288
        car1 = rec.objects[0]
        self.assertEqual(car1.class_name, "Car")
        self.assertAlmostEqual(car1.x_min, 128.0)
        self.assertAlmostEqual(car1.y_min, 144.0)
        self.assertAlmostEqual(car1.width, 384.0)
        self.assertAlmostEqual(car1.height, 288.0)

        # Check Pedestrian
        ped = rec.objects[1]
        self.assertEqual(ped.class_name, "Person")
        self.assertAlmostEqual(ped.x_min, 640.0)
        self.assertAlmostEqual(ped.y_min, 432.0)

        # Check Clipped Car
        car_clipped = rec.objects[2]
        self.assertEqual(car_clipped.class_name, "Car")
        self.assertTrue(car_clipped.x_min + car_clipped.width <= 1280.0)
        self.assertTrue(car_clipped.y_min + car_clipped.height <= 720.0)

        # Check unmapped raw objects preserved for audit
        self.assertEqual(len(rec.all_objects), 5)

    def test_deterministic_stratified_sampling(self) -> None:
        """Verify sampling counts and deterministic reproducibility with seed."""
        samples = []
        for i in range(100):
            cond = "rainy" if i < 30 else ("snowy" if i < 60 else ("clear" if i < 80 else "overcast"))
            samples.append(
                {
                    "filepath": f"data/img_{i:03d}.jpg",
                    "metadata": {"width": 1280, "height": 720},
                    "weather": {"label": cond},
                    "timeofday": {"label": "daytime"},
                    "scene": {"label": "highway"},
                    "detections": {"detections": []},
                }
            )
        json_path = self.root / "mock_cohort.json"
        json_path.write_text(json.dumps({"samples": samples}))

        ds = Bdd100kDataset.from_fiftyone_json(json_path)

        targets = {"rainy": 10, "snowy": 10, "clear": 5, "overcast": 5}
        cohort1 = ds.select_weather_stratified_cohort(targets=targets, seed=42)
        cohort2 = ds.select_weather_stratified_cohort(targets=targets, seed=42)
        cohort3 = ds.select_weather_stratified_cohort(targets=targets, seed=99)

        self.assertEqual(len(cohort1), 30)
        self.assertEqual([r.image_id for r in cohort1], [r.image_id for r in cohort2])
        self.assertNotEqual([r.image_id for r in cohort1], [r.image_id for r in cohort3])

    def test_record_image_loading_and_hashing(self) -> None:
        """Verify ExternalRecord loads PIL images and calculates SHA256 checksum."""
        img_path = self.root / "test_img.jpg"
        test_img = Image.new("RGB", (1280, 720), color=(100, 150, 200))
        test_img.save(img_path)

        rec = ExternalRecord(
            image_id="test_img.jpg",
            condition="snowy",
            timeofday="daytime",
            scene="city street",
            width=1280,
            height=720,
            image_path=img_path,
        )

        pil_img = rec.to_pil()
        self.assertEqual(pil_img.size, (1280, 720))

        sha = rec.sha256()
        self.assertEqual(len(sha), 64)
        self.assertIsInstance(sha, str)


if __name__ == "__main__":
    unittest.main()
