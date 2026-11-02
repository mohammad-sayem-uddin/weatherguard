"""BDD100K external dataset adapter and quality validation.

Normalises BDD100K annotations into WeatherGuard's internal BBox/Record
representation for fair cross-dataset evaluation.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image

from src.dataset import BBox

# Canonical mapping from BDD100K categories to WeatherGuard 6-class space.
# Crucially, 'rider' is deliberately excluded per protocol to avoid
# semantic conflation with 'Person'.
BDD100K_TO_CANONICAL: dict[str, str | None] = {
    "pedestrian": "Person",
    "car": "Car",
    "truck": "Truck",
    "bus": "Bus",
    "motorcycle": "Motorcycle",
    "bicycle": "Bicycle",
    # Explicitly unmapped / excluded categories:
    "rider": None,
    "traffic light": None,
    "traffic sign": None,
    "train": None,
    "other vehicle": None,
    "trailer": None,
    "other person": None,
}

CANONICAL_CLASSES: list[str] = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]


@dataclass
class ExternalRecord:
    image_id: str
    condition: str  # weather: 'rainy', 'snowy', 'foggy', 'overcast', 'clear'
    timeofday: str  # 'daytime', 'night', 'dawn/dusk'
    scene: str
    width: int
    height: int
    objects: list[BBox] = field(default_factory=list)
    image_path: Path | None = None
    image_url: str | None = None
    all_objects: list[dict] = field(default_factory=list)  # raw unmapped objects for audit

    def to_pil(self) -> Image.Image:
        if self.image_path and self.image_path.exists():
            img = Image.open(self.image_path)
            img.load()
            return img.convert("RGB")
        raise FileNotFoundError(f"Image file not found for {self.image_id} at {self.image_path}")

    def sha256(self) -> str:
        if self.image_path and self.image_path.exists():
            h = hashlib.sha256()
            with open(self.image_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()
        raise FileNotFoundError(f"Image file not found for {self.image_id}")


class Bdd100kDataset:
    """External BDD100K dataset adapter."""

    def __init__(self, records: list[ExternalRecord], source: str):
        self.records = records
        self.source = source

    @classmethod
    def from_fiftyone_json(
        cls,
        json_path: str | Path,
        images_dir: str | Path | None = None,
    ) -> "Bdd100kDataset":
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"BDD100K samples JSON not found: {json_path}")
        images_dir = Path(images_dir) if images_dir else None

        with open(json_path, "r") as f:
            data = json.load(f)

        records: list[ExternalRecord] = []
        samples = data.get("samples", [])

        for s in samples:
            meta = s.get("metadata", {})
            width = int(meta.get("width", 1280))
            height = int(meta.get("height", 720))

            weather_label = s.get("weather", {}).get("label", "undefined")
            time_label = s.get("timeofday", {}).get("label", "undefined")
            scene_label = s.get("scene", {}).get("label", "undefined")

            filepath_raw = s.get("filepath", "")
            image_id = Path(filepath_raw).name  # e.g., 'b1c66a42-6f7d68ca.jpg'
            image_path = (images_dir / image_id) if images_dir else None
            image_url = f"https://huggingface.co/datasets/dgural/bdd100k/resolve/main/{filepath_raw}"

            dets_raw = s.get("detections", {}).get("detections", [])
            valid_boxes: list[BBox] = []
            all_objs: list[dict] = []

            for d in dets_raw:
                label = d.get("label", "")
                bbox = d.get("bounding_box", [])  # [x_rel, y_rel, w_rel, h_rel]
                if len(bbox) != 4:
                    continue

                x_rel, y_rel, w_rel, h_rel = bbox
                xmin = float(x_rel * width)
                ymin = float(y_rel * height)
                bw = float(w_rel * width)
                bh = float(h_rel * height)

                all_objs.append({
                    "raw_label": label,
                    "x_min": xmin,
                    "y_min": ymin,
                    "width": bw,
                    "height": bh,
                })

                canonical = BDD100K_TO_CANONICAL.get(label)
                if canonical is not None:
                    # Enforce coordinate validation bounds: non-negative width/height
                    if bw > 0 and bh > 0:
                        # Clip coordinates to image boundary if slightly overflowing
                        xmin = max(0.0, min(float(width - 1), xmin))
                        ymin = max(0.0, min(float(height - 1), ymin))
                        bw = max(1.0, min(float(width - xmin), bw))
                        bh = max(1.0, min(float(height - ymin), bh))

                        valid_boxes.append(
                            BBox(
                                x_min=xmin,
                                y_min=ymin,
                                width=bw,
                                height=bh,
                                class_name=canonical,
                            )
                        )

            records.append(
                ExternalRecord(
                    image_id=image_id,
                    condition=weather_label,
                    timeofday=time_label,
                    scene=scene_label,
                    width=width,
                    height=height,
                    objects=valid_boxes,
                    image_path=image_path,
                    image_url=image_url,
                    all_objects=all_objs,
                )
            )

        return cls(records, source=str(json_path))

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ExternalRecord]:
        return iter(self.records)

    def __getitem__(self, idx: int) -> ExternalRecord:
        return self.records[idx]

    def select_weather_stratified_cohort(
        self,
        targets: dict[str, int] | None = None,
        seed: int = 42,
    ) -> list[ExternalRecord]:
        """Selects a weather-stratified subset deterministically according to protocol."""
        if targets is None:
            targets = {
                "rainy": 60,
                "snowy": 60,
                "foggy": 13,
                "overcast": 35,
                "clear": 35,
            }

        rng = random.Random(seed)
        selected: list[ExternalRecord] = []

        # Stratify
        for condition, target_count in targets.items():
            if condition == "foggy":
                # Foggy is rare in BDD100K val (13 total), include all available regardless of timeofday
                candidates = [r for r in self.records if r.condition == "foggy"]
            else:
                # Target daytime adverse / control scenes to avoid confounding with night darkness
                candidates = [
                    r for r in self.records
                    if r.condition == condition and r.timeofday == "daytime"
                ]

            # Sort deterministically by image_id
            candidates = sorted(candidates, key=lambda r: r.image_id)

            if len(candidates) <= target_count:
                chosen = candidates
            else:
                sampled_indices = sorted(rng.sample(range(len(candidates)), target_count))
                chosen = [candidates[i] for i in sampled_indices]

            selected.extend(chosen)

        return selected
