"""DAWN dataset loading and parsing.

The DAWN parquet files use a simple schema:

    image_id: string              (weather condition is the prefix before '-')
    image:    struct<bytes, path> (JPEG-encoded bytes)
    width:    int64
    height:   int64
    objects:  list<struct<class:int64, class_name:str,
                            x_min:double, y_min:double,
                            width:double, height:double>>

Bounding boxes are stored in COCO absolute-pixel (x_min, y_min, w, h)
format. The `class` integer is a DAWN-specific ID that does **not** match
the COCO 80-class ID space; we therefore expose only the class name
through the public API and use a string-based label space.
"""
from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Sequence

import pyarrow.parquet as pq
from PIL import Image

# Canonical DAWN class mapping as observed in the verified dataset.
DAWN_CLASSES: dict[int, str] = {
    0: "Truck",
    1: "Person",
    2: "Bicycle",
    3: "Car",
    4: "Motorcycle",
    5: "Bus",
}

# Detector (COCO-pretrained) → DAWN class mapping. We keep the detector
# label space (string) explicit and let metrics.py handle the join.
COCO_TO_DAWN: dict[str, str | None] = {
    "person": "Person",
    "bicycle": "Bicycle",
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
}


@dataclass
class BBox:
    x_min: float
    y_min: float
    width: float
    height: float
    class_name: str

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_min + self.width, self.y_min + self.height)

    def area(self) -> float:
        return float(self.width) * float(self.height)


@dataclass
class DawnRecord:
    image_id: str
    image_bytes: bytes
    width: int
    height: int
    objects: list[BBox] = field(default_factory=list)

    @property
    def condition(self) -> str:
        if "-" in self.image_id:
            return self.image_id.split("-", 1)[0]
        return "<unknown>"

    def to_pil(self) -> Image.Image:
        img = Image.open(io.BytesIO(self.image_bytes))
        img.load()
        return img.convert("RGB")


class DawnDataset:
    """In-memory DAWN dataset loaded from a single parquet file."""

    def __init__(self, records: list[DawnRecord], source: str):
        self.records = records
        self.source = source

    @classmethod
    def from_parquet(cls, path: str | Path) -> "DawnDataset":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DAWN parquet not found: {path}")
        table = pq.read_table(path)
        records: list[DawnRecord] = []
        ids = table.column("image_id").to_pylist()
        imgs = table.column("image").to_pylist()
        widths = table.column("width").to_pylist()
        heights = table.column("height").to_pylist()
        objs = table.column("objects").to_pylist()
        for i in range(len(ids)):
            boxes: list[BBox] = []
            for o in objs[i] or []:
                boxes.append(
                    BBox(
                        x_min=float(o["x_min"]),
                        y_min=float(o["y_min"]),
                        width=float(o["width"]),
                        height=float(o["height"]),
                        class_name=str(o["class_name"]),
                    )
                )
            records.append(
                DawnRecord(
                    image_id=str(ids[i]),
                    image_bytes=imgs[i]["bytes"],
                    width=int(widths[i]),
                    height=int(heights[i]),
                    objects=boxes,
                )
            )
        return cls(records, source=str(path))

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[DawnRecord]:
        return iter(self.records)

    def __getitem__(self, idx: int) -> DawnRecord:
        return self.records[idx]

    def by_condition(self) -> dict[str, list[DawnRecord]]:
        out: dict[str, list[DawnRecord]] = {}
        for r in self.records:
            out.setdefault(r.condition, []).append(r)
        return out

    def deterministic_sample(self, k: int, seed: int = 0) -> list[DawnRecord]:
        if k >= len(self.records):
            return list(self.records)
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(self.records)), k))
        return [self.records[i] for i in indices]

    def stratified_condition_sample(
        self, per_condition: int, seed: int = 0
    ) -> list[DawnRecord]:
        rng = random.Random(seed)
        out: list[DawnRecord] = []
        for cond, recs in sorted(self.by_condition().items()):
            if not recs:
                continue
            k = min(per_condition, len(recs))
            chosen = sorted(rng.sample(range(len(recs)), k))
            out.extend(recs[i] for i in chosen)
        return out

    def summary(self) -> dict:
        conds = self.by_condition()
        per_cond = {c: len(v) for c, v in conds.items()}
        n_objs = sum(len(r.objects) for r in self.records)
        per_class: dict[str, int] = {c: 0 for c in DAWN_CLASSES.values()}
        for r in self.records:
            for b in r.objects:
                per_class[b.class_name] = per_class.get(b.class_name, 0) + 1
        return {
            "source": self.source,
            "n_images": len(self.records),
            "n_objects": n_objs,
            "mean_objects_per_image": n_objs / max(1, len(self.records)),
            "per_condition_n_images": per_cond,
            "per_condition_n_objects": {
                c: sum(len(b.objects) for b in v) for c, v in conds.items()
            },
            "per_class_n_objects": per_class,
            "image_size_min": (min(r.width for r in self.records), min(r.height for r in self.records)),
            "image_size_max": (max(r.width for r in self.records), max(r.height for r in self.records)),
        }


def serialize_record_summary(record: DawnRecord) -> dict:
    return {
        "image_id": record.image_id,
        "condition": record.condition,
        "width": record.width,
        "height": record.height,
        "n_objects": len(record.objects),
        "objects": [
            {
                "x_min": b.x_min,
                "y_min": b.y_min,
                "width": b.width,
                "height": b.height,
                "class_name": b.class_name,
            }
            for b in record.objects
        ],
    }
