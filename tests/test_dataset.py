"""Tests for src/dataset — schema parsing and sampling helpers.

The tests construct small in-memory parquet files so they don't depend
on the real DAWN files. This keeps the tests deterministic and fast.
"""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from src.dataset import (
    DAWN_CLASSES,
    DawnDataset,
    DawnRecord,
    BBox,
    COCO_TO_DAWN,
)


def _make_parquet(path: Path, rows: list[dict]) -> None:
    """Build a minimal DAWN-shaped parquet at `path`."""
    image_ids = []
    images = []
    widths = []
    heights = []
    objects = []
    for r in rows:
        image_ids.append(r["image_id"])
        img = r["image"]
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        images.append({"bytes": buf.getvalue(), "path": r["image_id"] + ".jpg"})
        widths.append(img.width)
        heights.append(img.height)
        objects.append(r["objects"])
    table = pa.table(
        {
            "image_id": pa.array(image_ids, type=pa.string()),
            "image": pa.array(images, type=pa.struct(
                [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
            )),
            "width": pa.array(widths, type=pa.int64()),
            "height": pa.array(heights, type=pa.int64()),
            "objects": pa.array(
                objects,
                type=pa.list_(pa.struct(
                    [
                        pa.field("class", pa.int64()),
                        pa.field("class_name", pa.string()),
                        pa.field("height", pa.float64()),
                        pa.field("width", pa.float64()),
                        pa.field("x_min", pa.float64()),
                        pa.field("y_min", pa.float64()),
                    ]
                )),
            ),
        }
    )
    pq.write_table(table, str(path))


class TestDawnDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "fake.parquet"
        rows = [
            {
                "image_id": "foggy-001",
                "image": Image.new("RGB", (100, 80), color=(120, 120, 120)),
                "objects": [
                    {
                        "class": 3,
                        "class_name": "Car",
                        "x_min": 10.0,
                        "y_min": 5.0,
                        "width": 30.0,
                        "height": 20.0,
                    },
                    {
                        "class": 0,
                        "class_name": "Truck",
                        "x_min": 60.0,
                        "y_min": 30.0,
                        "width": 35.0,
                        "height": 25.0,
                    },
                ],
            },
            {
                "image_id": "foggy-002",
                "image": Image.new("RGB", (200, 150), color=(80, 80, 80)),
                "objects": [
                    {
                        "class": 1,
                        "class_name": "Person",
                        "x_min": 50.0,
                        "y_min": 50.0,
                        "width": 20.0,
                        "height": 60.0,
                    }
                ],
            },
            {
                "image_id": "rain_storm-001",
                "image": Image.new("RGB", (640, 480), color=(50, 50, 50)),
                "objects": [],
            },
        ]
        _make_parquet(self.path, rows)
        self.ds = DawnDataset.from_parquet(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_count(self):
        self.assertEqual(len(self.ds), 3)

    def test_image_id_and_condition(self):
        self.assertEqual(self.ds[0].image_id, "foggy-001")
        self.assertEqual(self.ds[0].condition, "foggy")
        self.assertEqual(self.ds[2].condition, "rain_storm")

    def test_bbox_xyxy(self):
        rec = self.ds[0]
        car = rec.objects[0]
        self.assertEqual(car.to_xyxy(), (10.0, 5.0, 40.0, 25.0))
        self.assertAlmostEqual(car.area(), 30 * 20)

    def test_by_condition_groups(self):
        groups = self.ds.by_condition()
        self.assertEqual(set(groups.keys()), {"foggy", "rain_storm"})
        self.assertEqual(len(groups["foggy"]), 2)
        self.assertEqual(len(groups["rain_storm"]), 1)

    def test_deterministic_sample_is_stable(self):
        a = self.ds.deterministic_sample(k=2, seed=7)
        b = self.ds.deterministic_sample(k=2, seed=7)
        self.assertEqual([r.image_id for r in a], [r.image_id for r in b])

    def test_stratified_condition_sample_respects_per_condition(self):
        sample = self.ds.stratified_condition_sample(per_condition=1, seed=1)
        self.assertEqual(len(sample), 2)
        conds = {s.condition for s in sample}
        self.assertEqual(conds, {"foggy", "rain_storm"})

    def test_summary_counts(self):
        s = self.ds.summary()
        self.assertEqual(s["n_images"], 3)
        self.assertEqual(s["n_objects"], 3)
        self.assertEqual(s["per_class_n_objects"]["Car"], 1)
        self.assertEqual(s["per_class_n_objects"]["Truck"], 1)
        self.assertEqual(s["per_class_n_objects"]["Person"], 1)

    def test_to_pil_returns_rgb(self):
        img = self.ds[0].to_pil()
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size, (100, 80))

    def test_coco_to_dawn_mapping_complete(self):
        # All six DAWN classes must be reachable from COCO labels.
        for cls in DAWN_CLASSES.values():
            self.assertIn(cls, set(COCO_TO_DAWN.values()))


class TestParquetErrors(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            DawnDataset.from_parquet("/nonexistent/dawn.parquet")


if __name__ == "__main__":
    unittest.main()
