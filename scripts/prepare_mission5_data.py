"""Prepare YOLO-formatted DAWN dataset for Mission 5 fine-tuning experiments.

Extracts training and validation images and bounding boxes from DAWN parquet files:
  - train-00000-of-00001.parquet (N=821)
  - val-00000-of-00001.parquet   (N=206)

Generates:
  - data/dawn_yolo/images/train/*.jpg
  - data/dawn_yolo/labels/train/*.txt
  - data/dawn_yolo/images/val/*.jpg
  - data/dawn_yolo/labels/val/*.txt
  - data/dawn_yolo/train_full.txt
  - data/dawn_yolo/train_200.txt (stratified 200-image subset across adverse weather conditions)
  - data/dawn_yolo/val.txt
  - data/dawn_yolo/dawn_200.yaml
  - data/dawn_yolo/dawn_full.yaml

Labels are remapped to canonical 6 classes:
  0: person, 1: bicycle, 2: car, 3: motorcycle, 4: bus, 5: truck
"""

from __future__ import annotations

import collections
from pathlib import Path
import random
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
from src.dataset import DawnDataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "dawn_yolo"
IMAGES_TRAIN = DATA_DIR / "images" / "train"
LABELS_TRAIN = DATA_DIR / "labels" / "train"
IMAGES_VAL = DATA_DIR / "images" / "val"
LABELS_VAL = DATA_DIR / "labels" / "val"

DAWN_TO_YOLO: Dict[str, int] = {
    "Person": 0,
    "Bicycle": 1,
    "Car": 2,
    "Motorcycle": 3,
    "Bus": 4,
    "Truck": 5,
}

CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def extract_split(parquet_name: str, images_dir: Path, labels_dir: Path) -> List[Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = ROOT / parquet_name
    print(f"Loading {parquet_path}...")
    ds = DawnDataset.from_parquet(parquet_path)

    saved_images: List[Path] = []
    class_counts = collections.Counter()
    total_boxes = 0

    for rec in ds.records:
        img_filename = f"{rec.image_id}.jpg"
        img_path = images_dir / img_filename
        lbl_filename = f"{rec.image_id}.txt"
        lbl_path = labels_dir / lbl_filename

        # Save image
        with open(img_path, "wb") as f:
            f.write(rec.image_bytes)
        saved_images.append(img_path.resolve())

        # Save labels
        lines = []
        w_img = float(rec.width)
        h_img = float(rec.height)

        for obj in rec.objects:
            if obj.class_name not in DAWN_TO_YOLO:
                continue
            cls_id = DAWN_TO_YOLO[obj.class_name]
            # Convert (x_min, y_min, w, h) to normalized (x_center, y_center, w, h)
            xc = (obj.x_min + obj.width / 2.0) / w_img
            yc = (obj.y_min + obj.height / 2.0) / h_img
            wn = obj.width / w_img
            hn = obj.height / h_img

            # Clamp
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            wn = max(0.0, min(1.0, wn))
            hn = max(0.0, min(1.0, hn))

            if wn > 0.0 and hn > 0.0:
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
                class_counts[obj.class_name] += 1
                total_boxes += 1

        with open(lbl_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    print(f"Extracted {len(saved_images)} images and {total_boxes} boxes to {images_dir.parent}")
    print(f"Class breakdown: {dict(class_counts)}")
    return saved_images


def create_stratified_subset(
    train_images: List[Path],
    target_n: int = 200,
    seed: int = 20261101,
) -> List[Path]:
    """Select a stratified subset of DAWN training images proportional to weather condition."""
    rng = random.Random(seed)
    by_condition: Dict[str, List[Path]] = collections.defaultdict(list)

    for p in train_images:
        stem = p.stem
        cond = stem.split("-", 1)[0] if "-" in stem else "unknown"
        by_condition[cond].append(p)

    total_images = len(train_images)
    subset: List[Path] = []

    # Proportional allocation
    allocations: Dict[str, int] = {}
    remaining = target_n
    for cond, paths in sorted(by_condition.items(), key=lambda x: -len(x[1])):
        k = max(1, round(target_n * (len(paths) / total_images)))
        k = min(k, len(paths))
        allocations[cond] = k
        remaining -= k

    # Adjust rounding discrepancy if any
    cond_keys = sorted(by_condition.keys(), key=lambda c: len(by_condition[c]), reverse=True)
    idx = 0
    while remaining > 0:
        c = cond_keys[idx % len(cond_keys)]
        if allocations[c] < len(by_condition[c]):
            allocations[c] += 1
            remaining -= 1
        idx += 1
    while remaining < 0:
        c = cond_keys[idx % len(cond_keys)]
        if allocations[c] > 1:
            allocations[c] -= 1
            remaining += 1
        idx += 1

    for cond, paths in sorted(by_condition.items()):
        sampled = rng.sample(paths, allocations[cond])
        subset.extend(sampled)

    print(f"Stratified subset of {len(subset)} images created with allocations: {allocations}")
    return subset


def main():
    print("=== Preparing DAWN YOLO Dataset for Mission 5 ===")
    train_images = extract_split("train-00000-of-00001.parquet", IMAGES_TRAIN, LABELS_TRAIN)
    val_images = extract_split("val-00000-of-00001.parquet", IMAGES_VAL, LABELS_VAL)

    train_subset_200 = create_stratified_subset(train_images, target_n=200, seed=20261101)

    # Write txt lists
    train_full_txt = DATA_DIR / "train_full.txt"
    with open(train_full_txt, "w") as f:
        f.write("\n".join(str(p) for p in sorted(train_images)) + "\n")

    train_200_txt = DATA_DIR / "train_200.txt"
    with open(train_200_txt, "w") as f:
        f.write("\n".join(str(p) for p in sorted(train_subset_200)) + "\n")

    val_txt = DATA_DIR / "val.txt"
    with open(val_txt, "w") as f:
        f.write("\n".join(str(p) for p in sorted(val_images)) + "\n")

    # Write dataset YAMLs
    yaml_200_content = f"""# WeatherGuard Mission 5: DAWN 200 Stratified Training Dataset
path: {DATA_DIR}
train: {train_200_txt}
val: {val_txt}
nc: 6
names: {CLASS_NAMES}
"""
    yaml_200_path = DATA_DIR / "dawn_200.yaml"
    with open(yaml_200_path, "w") as f:
        f.write(yaml_200_content)

    yaml_full_content = f"""# WeatherGuard Mission 5: DAWN Full (821) Training Dataset
path: {DATA_DIR}
train: {train_full_txt}
val: {val_txt}
nc: 6
names: {CLASS_NAMES}
"""
    yaml_full_path = DATA_DIR / "dawn_full.yaml"
    with open(yaml_full_path, "w") as f:
        f.write(yaml_full_content)

    print(f"Wrote {yaml_200_path}")
    print(f"Wrote {yaml_full_path}")
    print("Data preparation completed successfully.")


if __name__ == "__main__":
    main()
