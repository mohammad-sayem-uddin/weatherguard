#!/usr/bin/env python3
"""Audits all existing WeatherGuard image datasets, manifests, and class/size distributions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from PIL import Image
import pandas as pd

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
DATA_DIR = ROOT / "data"

CANONICAL_CLASSES = ["truck", "person", "bicycle", "car", "motorcycle", "bus"]


def get_scale_bucket(max_dim: float) -> str:
    if max_dim < 16.0:
        return "ultra_small"
    elif max_dim < 32.0:
        return "small"
    elif max_dim < 96.0:
        return "medium"
    else:
        return "large"


def audit_dawn_yolo() -> Dict[str, Any]:
    val_txt = DATA_DIR / "dawn_yolo" / "val.txt"
    train_txt = DATA_DIR / "dawn_yolo" / "train.txt"
    labels_dir = DATA_DIR / "dawn_yolo" / "labels"

    # Read val and train paths
    val_imgs = [l.strip() for l in val_txt.read_text().splitlines() if l.strip()] if val_txt.exists() else []
    train_imgs = [l.strip() for l in train_txt.read_text().splitlines() if l.strip()] if train_txt.exists() else []

    total_images = len(val_imgs) + len(train_imgs)
    class_counts = {c: 0 for c in CANONICAL_CLASSES}
    scale_counts = {"ultra_small": 0, "small": 0, "medium": 0, "large": 0}
    weather_counts = {"fog": 0, "rain": 0, "snow": 0, "sand": 0, "other": 0}
    resolutions = set()

    all_img_paths = [Path(p) for p in val_imgs + train_imgs]
    for p in all_img_paths:
        name_lower = p.name.lower()
        if "fog" in name_lower:
            weather_counts["fog"] += 1
        elif "rain" in name_lower:
            weather_counts["rain"] += 1
        elif "snow" in name_lower:
            weather_counts["snow"] += 1
        elif "sand" in name_lower:
            weather_counts["sand"] += 1
        else:
            weather_counts["other"] += 1

        if p.exists():
            try:
                with Image.open(p) as im:
                    resolutions.add(f"{im.width}x{im.height}")
                    w_im, h_im = im.width, im.height
            except Exception:
                w_im, h_im = 1280.0, 720.0
        else:
            w_im, h_im = 1280.0, 720.0

        lbl_file = labels_dir / f"{p.stem}.txt"
        if not lbl_file.exists():
            lbl_file = labels_dir / "val" / f"{p.stem}.txt"
        if not lbl_file.exists():
            lbl_file = labels_dir / "train" / f"{p.stem}.txt"

        if lbl_file.exists():
            DAWN_MAP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "bus", 5: "truck"}
            for line in lbl_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    bw = float(parts[3]) * w_im
                    bh = float(parts[4]) * h_im
                    max_d = max(bw, bh)
                    cname = DAWN_MAP.get(cid, "car")
                    if cname in class_counts:
                        class_counts[cname] += 1
                    scale_counts[get_scale_bucket(max_d)] += 1

    return {
        "dataset_name": "DAWN (Vehicle Detection in Adverse Weather Nature)",
        "source_category": "Public Real-World Dataset",
        "reference": "Kenan et al. (2020), CC BY-NC 3.0",
        "total_images": total_images,
        "splits": {"train": len(train_imgs), "validation": len(val_imgs)},
        "resolutions": sorted(list(resolutions))[:5],
        "weather_distribution": weather_counts,
        "class_distribution": class_counts,
        "scale_distribution": scale_counts,
    }


def audit_flir_thermal() -> Dict[str, Any]:
    flir_pq = DATA_DIR / "flir_thermal" / "validation.parquet"
    if not flir_pq.exists():
        return {"status": "NOT_FOUND"}

    df = pd.read_parquet(flir_pq)
    n_images = len(df)
    class_counts = {"car": 0, "person": 0, "bicycle": 0}
    scale_counts = {"ultra_small": 0, "small": 0, "medium": 0, "large": 0}

    for _, row in df.iterrows():
        objs = row.get("objects", {})
        if isinstance(objs, dict):
            boxes = objs.get("bbox", [])
            categories = objs.get("category", [])
        else:
            boxes = []
            categories = []

        FLIR_MAP = {1: "bicycle", 2: "car", 4: "person"}
        for b, l in zip(boxes, categories):
            cname = FLIR_MAP.get(int(l))
            if cname in class_counts:
                class_counts[cname] += 1
            if len(b) >= 4:
                bw = float(b[2])
                bh = float(b[3])
                max_d = max(bw, bh)
                scale_counts[get_scale_bucket(max_d)] += 1

    return {
        "dataset_name": "FLIR ADAS Thermal Dataset v1",
        "source_category": "Public Real-World Dataset",
        "sensor_modality": "Thermal Long-Wave Infrared (LWIR 8-14 um)",
        "reference": "Teledyne FLIR Academic Research Terms",
        "total_images": n_images,
        "native_resolution": "640x512",
        "class_distribution": class_counts,
        "scale_distribution": scale_counts,
    }


def audit_bdd100k_video() -> Dict[str, Any]:
    bdd_dir = DATA_DIR / "bdd100k"
    seq_dirs = [d for d in bdd_dir.glob("*") if d.is_dir()] if bdd_dir.exists() else []

    total_frames = 0
    seq_summary = {}
    for sd in seq_dirs:
        imgs = list(sd.glob("*.jpg")) + list(sd.glob("*.png"))
        total_frames += len(imgs)
        seq_summary[sd.name] = len(imgs)

    return {
        "dataset_name": "BDD100K Adverse Weather MOT Video Cohort",
        "source_category": "Public Real-World Driving Video",
        "reference": "Yu et al. (2020), Academic Research License",
        "total_video_frames": total_frames,
        "sequences_count": len(seq_dirs),
        "sequence_frames": seq_summary,
        "native_resolution": "1280x720 (16:9)",
        "cohort_roles": {
            "development_sequences": ["b24380ab", "b255cd6c"],
            "locked_reference_sequences": ["b1e1a7b8", "b1d0a191", "b24d283f", "b23c9e00"]
        }
    }


if __name__ == "__main__":
    audit = {
        "timestamp": "2026-11-05T05:41:00Z",
        "mission": "Master Autonomous Mission — Phase 2: Existing Dataset Audit",
        "dawn": audit_dawn_yolo(),
        "flir": audit_flir_thermal(),
        "bdd100k": audit_bdd100k_video(),
    }
    out_path = ROOT / "artifacts" / "existing_dataset_audit.json"
    out_path.write_text(json.dumps(audit, indent=2))
    print(f"Audit saved to {out_path}")
    print(json.dumps(audit, indent=2))
