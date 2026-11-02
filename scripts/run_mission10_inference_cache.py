"""Inference caching script for Mission 10 independent sequences.

Runs YOLOv8n @ 640px and YOLOv8n @ 832px (and Head-Only @ 640px) across the 6
independent sequences (1,213 continuous frames), saving raw detections with confidence,
coordinates, scale, and classes into data/mission10_independent/predictions/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
import sys
sys.path.insert(0, str(ROOT))

from src.video_dataset import (
    CANONICAL_CLASSES,
    CANONICAL_MAPPING,
    DetectionBox,
    VideoDataset,
    VideoSequence,
)

DATA_DIR = ROOT / "data" / "mission10_independent"
PREDS_CACHE_DIR = DATA_DIR / "predictions"
ARTIFACTS_DIR = ROOT / "artifacts"

CONF_THRESHOLD = 0.10  # Cache down to 0.10 to allow low-confidence ByteTrack association
IOU_THRESHOLD = 0.50

MODELS = {
    "yolov8n_640": {"weights": ROOT / "yolov8n.pt", "imgsz": 640, "desc": "Pretrained YOLOv8n 640px baseline"},
    "yolov8n_832": {"weights": ROOT / "yolov8n.pt", "imgsz": 832, "desc": "Pretrained YOLOv8n 832px high-res"},
}


def run_inference_and_cache():
    PREDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load sequence manifest
    manifest_path = ARTIFACTS_DIR / "mission10_sequence_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_seq_ids = [s["sequence_id"] for s in manifest["partitions"]["independent_development_cohort"]["sequences"]] + \
                  [s["sequence_id"] for s in manifest["partitions"]["independent_locked_test_cohort"]["sequences"]]

    print(f"Loading {len(all_seq_ids)} sequences from {DATA_DIR}...")
    dataset = VideoDataset.load(DATA_DIR, sequence_ids=all_seq_ids)

    for model_key, m_info in MODELS.items():
        w_path = Path(m_info["weights"])
        if not w_path.exists():
            print(f"Model weights not found at {w_path}, skipping {model_key}...")
            continue

        print(f"\n========================================================")
        print(f"Running inference for {model_key} ({m_info['desc']})")
        print(f"========================================================")
        model = YOLO(str(w_path))

        for sid in all_seq_ids:
            cache_file = PREDS_CACHE_DIR / f"{sid}_{model_key}.json"
            if cache_file.exists():
                print(f"Cache exists for {sid} [{model_key}], skipping...")
                continue

            seq = dataset.sequences[sid]
            print(f"Inferencing {sid} ({len(seq.frames)} frames @ {m_info['imgsz']}px)...")
            t0 = time.time()
            all_frame_detections = []

            for frame in seq.frames:
                res = model(str(frame.image_path), imgsz=m_info["imgsz"], conf=CONF_THRESHOLD, verbose=False)[0]
                frame_boxes = []

                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy()

                    for i in range(len(xyxy)):
                        cls_id = int(classes[i])
                        raw_name = model.names[cls_id]
                        canon = CANONICAL_MAPPING.get(raw_name.lower(), raw_name)
                        if canon in CANONICAL_CLASSES:
                            frame_boxes.append({
                                "class_name": canon,
                                "raw_class": raw_name,
                                "confidence": round(float(confs[i]), 4),
                                "x1": round(float(xyxy[i, 0]), 2),
                                "y1": round(float(xyxy[i, 1]), 2),
                                "x2": round(float(xyxy[i, 2]), 2),
                                "y2": round(float(xyxy[i, 3]), 2),
                            })
                all_frame_detections.append(frame_boxes)

            elapsed = time.time() - t0
            fps = len(seq.frames) / elapsed
            print(f"Done {sid} in {elapsed:.2f}s ({fps:.1f} FPS, {sum(len(b) for b in all_frame_detections)} detections)")

            with open(cache_file, "w") as f:
                json.dump({
                    "sequence_id": sid,
                    "model_key": model_key,
                    "imgsz": m_info["imgsz"],
                    "frames_count": len(seq.frames),
                    "total_detections": sum(len(b) for b in all_frame_detections),
                    "inference_seconds": round(elapsed, 3),
                    "fps": round(fps, 2),
                    "predictions": all_frame_detections,
                }, f)

    print("\nAll inferences complete and cached successfully!")


if __name__ == "__main__":
    run_inference_and_cache()
