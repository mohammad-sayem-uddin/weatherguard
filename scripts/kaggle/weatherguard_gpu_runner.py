#!/usr/bin/env python3
"""WeatherGuard — Reusable Kaggle GPU Training & Evaluation Runner.

Target Environment: Kaggle Cloud, NVIDIA Tesla T4 GPU (16 GB VRAM).
Capabilities:
  - CUDA GPU device validation and VRAM profiling.
  - Reproducibility seeding (Python, NumPy, PyTorch, CUDA deterministic).
  - Fine-tunes YOLOv8n on WeatherGuard V2 train split (667 frames) with AdamW, AMP.
  - Checkpoint selection strictly on validation split (173 frames).
  - Frozen independent evaluation on locked test split (160 frames, 0 augmented).
  - Metrics: mAP@0.5, mAP@0.5:0.95, per-class AP, 4 scale categories, and 4 weather domains.
  - Zero-shot baseline vs Fine-tuned comparison.
  - Exports artifacts to /kaggle/working/artifacts/.
"""
from __future__ import annotations

import csv
import glob
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WeatherGuard-KaggleGPU")

SEED = 42

CANONICAL_CLASSES = {
    0: "truck",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "bus",
}

# COCO to WeatherGuard canonical class ID mapping
# COCO: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
COCO_TO_WG_CLASS_ID = {
    0: 1,  # person -> 1
    1: 2,  # bicycle -> 2
    2: 3,  # car -> 3
    3: 4,  # motorcycle -> 4
    5: 5,  # bus -> 5
    7: 0,  # truck -> 0
}


def set_reproducibility(seed: int = SEED) -> None:
    """Sets random seeds across all libraries for deterministic execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Deterministic seed %d initialized.", seed)


def audit_gpu_environment() -> Dict[str, Any]:
    """Inspects and returns the hardware runtime profile."""
    is_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    total_mem_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if is_cuda else 0.0
    )

    profile = {
        "platform": "Kaggle Cloud",
        "cuda_available": is_cuda,
        "device_name": device_name,
        "gpu_memory_gb": round(total_mem_gb, 2),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if is_cuda else None,
        "is_nvidia_t4": "T4" in device_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    logger.info("GPU Audit: %s (%.2f GB VRAM, CUDA %s)",
                device_name, total_mem_gb, profile["cuda_version"])
    return profile


def get_scale_bucket(width: float, height: float) -> str:
    """Categorizes box into canonical scale buckets based on max dimension in pixels."""
    max_d = max(width, height)
    if max_d < 16.0:
        return "ultra_small"
    if max_d < 32.0:
        return "small"
    if max_d < 96.0:
        return "medium"
    return "large"


def get_weather_domain(filename: str) -> str:
    """Extracts adverse weather domain from filename."""
    name = Path(filename).name.lower()
    if "b24d283f" in name or "rain" in name:
        return "rain"
    if "snow" in name:
        return "snow"
    if "b255cd6c" in name or "night" in name:
        return "night"
    if "dense_fog" in name or "fog" in name:
        return "dense_fog"
    return "other"


def compute_box_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union for [xmin, ymin, xmax, ymax]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def evaluate_fine_grained(
    model: Any,
    test_img_dir: Path,
    test_lbl_dir: Path,
    is_zero_shot: bool = False,
    conf_threshold: float = 0.25,
    iou_match_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluates detections against ground truth split by scale and weather condition."""
    scale_stats = {
        "ultra_small": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0},
        "small": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0},
        "medium": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0},
        "large": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0},
    }
    weather_stats = {
        "rain": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0, "images": 0},
        "snow": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0, "images": 0},
        "night": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0, "images": 0},
        "dense_fog": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0, "images": 0},
    }

    img_files = sorted([p for p in test_img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    logger.info("Running fine-grained evaluation on %d test images (zero_shot=%s)...", len(img_files), is_zero_shot)

    for img_p in img_files:
        lbl_p = test_lbl_dir / f"{img_p.stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        weather = get_weather_domain(img_p.name)
        if weather in weather_stats:
            weather_stats[weather]["images"] += 1

        # Read GT
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    xmin = (cx - bw / 2.0) * w
                    ymin = (cy - bh / 2.0) * h
                    xmax = (cx + bw / 2.0) * w
                    ymax = (cy + bh / 2.0) * h
                    pw = bw * w
                    ph = bh * h
                    scale = get_scale_bucket(pw, ph)
                    gt_boxes.append({
                        "cls_id": cls_id,
                        "box": [xmin, ymin, xmax, ymax],
                        "scale": scale,
                        "matched": False,
                    })

        # Count GT totals
        for gt in gt_boxes:
            scale_stats[gt["scale"]]["total_gt"] += 1
            if weather in weather_stats:
                weather_stats[weather]["total_gt"] += 1

        # Run inference
        results = model.predict(source=str(img_p), conf=conf_threshold, iou=0.45, verbose=False)[0]
        pred_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                c = int(b.cls[0].item())
                if is_zero_shot:
                    if c not in COCO_TO_WG_CLASS_ID:
                        continue
                    c = COCO_TO_WG_CLASS_ID[c]
                xyxy = b.xyxy[0].tolist()
                pw = xyxy[2] - xyxy[0]
                ph = xyxy[3] - xyxy[1]
                scale = get_scale_bucket(pw, ph)
                pred_boxes.append({
                    "cls_id": c,
                    "box": xyxy,
                    "scale": scale,
                    "matched": False,
                })

        # Match preds to GT (Greedy IoU)
        for pred in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if not gt["matched"] and gt["cls_id"] == pred["cls_id"]:
                    iou = compute_box_iou(pred["box"], gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            if best_iou >= iou_match_threshold and best_gt_idx >= 0:
                gt_boxes[best_gt_idx]["matched"] = True
                pred["matched"] = True
                gt_scale = gt_boxes[best_gt_idx]["scale"]
                scale_stats[gt_scale]["tp"] += 1
                if weather in weather_stats:
                    weather_stats[weather]["tp"] += 1
            else:
                scale_stats[pred["scale"]]["fp"] += 1
                if weather in weather_stats:
                    weather_stats[weather]["fp"] += 1

        # Unmatched GT are FN
        for gt in gt_boxes:
            if not gt["matched"]:
                scale_stats[gt["scale"]]["fn"] += 1
                if weather in weather_stats:
                    weather_stats[weather]["fn"] += 1

    # Compute Precision, Recall, F1
    scale_summary = {}
    for s, data in scale_stats.items():
        tp = data["tp"]
        fp = data["fp"]
        fn = data["fn"]
        p = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        r = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
        scale_summary[s] = {
            "total_gt": data["total_gt"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        }

    weather_summary = {}
    for w_name, data in weather_stats.items():
        tp = data["tp"]
        fp = data["fp"]
        fn = data["fn"]
        p = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        r = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
        weather_summary[w_name] = {
            "images": data["images"],
            "total_gt": data["total_gt"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        }

    return {
        "scale_metrics": scale_summary,
        "weather_metrics": weather_summary,
    }


def run_full_vision_benchmark(
    data_yaml: str = "/kaggle/input/weatherguard-v2-corpus/data.yaml",
    weights_path: str = "yolov8n.pt",
    epochs: int = 15,
    imgsz: int = 640,
    batch_size: int = 16,
    output_dir: str = "/kaggle/working/runs/train",
    experiment_id: str = "weatherguard_t4_real",
) -> Dict[str, Any]:
    """Orchestrates end-to-end training and evaluation on Kaggle GPU."""
    from ultralytics import YOLO

    set_reproducibility(SEED)
    gpu_profile = audit_gpu_environment()

    data_dir = Path(data_yaml).parent
    test_img_dir = data_dir / "test" / "images"
    test_lbl_dir = data_dir / "test" / "labels"

    # Step 1: Zero-shot baseline evaluation on test set
    logger.info("--- Step 1: Evaluating Pretrained YOLOv8n (Zero-Shot Baseline) ---")
    base_model = YOLO("yolov8n.pt")
    zero_shot_fine_grained = evaluate_fine_grained(
        base_model, test_img_dir, test_lbl_dir, is_zero_shot=True, conf_threshold=0.25
    )

    # Step 2: Fine-Tuning YOLOv8n on Train split (667 frames)
    logger.info("--- Step 2: Fine-Tuning YOLOv8n on WeatherGuard V2 Train Split ---")
    device = 0 if torch.cuda.is_available() else "cpu"
    train_model = YOLO("yolov8n.pt")

    start_train_time = time.perf_counter()
    train_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=output_dir,
        name=experiment_id,
        seed=SEED,
        deterministic=True,
        amp=torch.cuda.is_available(),
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        save=True,
        val=True,
        plots=True,
        workers=2,
    )
    training_time_s = time.perf_counter() - start_train_time
    logger.info("Training finished in %.2f seconds.", training_time_s)

    best_ckpt = Path(output_dir) / experiment_id / "weights" / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found at {best_ckpt}")

    # Step 3: Evaluate fine-tuned best model on Validation split (model selection check)
    logger.info("--- Step 3: Validating Best Checkpoint on Validation Split ---")
    val_eval_model = YOLO(str(best_ckpt))
    val_results = val_eval_model.val(data=data_yaml, split="val", device=device, plots=False)
    val_metrics = {
        "precision": round(float(val_results.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(val_results.results_dict.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(val_results.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
    }

    # Step 4: Frozen Independent Evaluation on Locked Test split (160 frames, 0 augmented)
    logger.info("--- Step 4: Independent Evaluation on Locked Test Split ---")
    test_results = val_eval_model.val(data=data_yaml, split="test", device=device, plots=True)
    test_metrics = {
        "precision": round(float(test_results.results_dict.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(test_results.results_dict.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(test_results.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(test_results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
    }

    # Per-class AP on test split
    per_class_test = {}
    if hasattr(test_results.box, "ap50") and test_results.box.ap50 is not None:
        ap50_list = test_results.box.ap50
        ap_list = test_results.box.maps if hasattr(test_results.box, "maps") else [0.0] * len(ap50_list)
        for idx, name in CANONICAL_CLASSES.items():
            if idx < len(ap50_list):
                per_class_test[name] = {
                    "mAP50": round(float(ap50_list[idx]), 4),
                    "mAP50_95": round(float(ap_list[idx]), 4),
                }

    # Fine-grained scale and weather evaluation on locked test split
    ft_fine_grained = evaluate_fine_grained(
        val_eval_model, test_img_dir, test_lbl_dir, is_zero_shot=False, conf_threshold=0.25
    )

    # Export artifacts
    artifacts_dir = Path("/kaggle/working/artifacts/vision")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Copy key files
    for item in ["best.pt", "last.pt"]:
        src_f = Path(output_dir) / experiment_id / "weights" / item
        if src_f.exists():
            shutil.copy2(src_f, artifacts_dir / item)

    for item in ["results.csv", "results.png", "confusion_matrix.png"]:
        src_f = Path(output_dir) / experiment_id / item
        if src_f.exists():
            shutil.copy2(src_f, artifacts_dir / item)

    benchmark_summary = {
        "experiment_id": experiment_id,
        "dataset_yaml": data_yaml,
        "gpu_profile": gpu_profile,
        "training": {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "amp": True,
            "training_time_seconds": round(training_time_s, 2),
            "best_checkpoint": str(best_ckpt),
        },
        "validation_split_metrics": val_metrics,
        "locked_test_split_metrics": {
            "overall": test_metrics,
            "per_class": per_class_test,
            "scale_breakdown": ft_fine_grained["scale_metrics"],
            "weather_breakdown": ft_fine_grained["weather_metrics"],
        },
        "zero_shot_baseline_comparison": {
            "scale_breakdown": zero_shot_fine_grained["scale_metrics"],
            "weather_breakdown": zero_shot_fine_grained["weather_metrics"],
        },
    }

    out_json = artifacts_dir / "mission20_vision_gpu_results.json"
    out_json.write_text(json.dumps(benchmark_summary, indent=2))
    logger.info("Saved complete vision benchmark results to %s", out_json)

    return benchmark_summary


if __name__ == "__main__":
    profile = audit_gpu_environment()
    print(json.dumps(profile, indent=2))
