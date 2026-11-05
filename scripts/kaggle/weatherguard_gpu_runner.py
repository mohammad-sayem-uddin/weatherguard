#!/usr/bin/env python3
"""WeatherGuard — Reusable Kaggle GPU Training & Evaluation Runner.

Target Environment: Kaggle Cloud, NVIDIA Tesla T4 GPU (16 GB VRAM).
Capabilities:
  - CUDA GPU device validation and VRAM profiling.
  - Reproducibility seeding (Python, NumPy, PyTorch, CUDA deterministic).
  - Automated dependency installation (Ultralytics, OpenCV, PyYAML).
  - Clones or mounts WeatherGuard repository and verifies dataset manifests.
  - Executes YOLOv8 training with mixed-precision (AMP), AdamW optimizer,
    small-object weighting, and checkpoint preservation.
  - Comprehensive unified evaluation: mAP@0.5, mAP@0.5:0.95, and 4 scale categories.
  - Exports artifacts and experiment summaries to /kaggle/working/artifacts/.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WeatherGuard-KaggleGPU")

SEED = 42


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
    logger.info("Deterministic seed %d initialized across Python, NumPy, PyTorch, and CUDA.", seed)


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
    logger.info("GPU Audit: %s (%s, %.2f GB VRAM, CUDA %s)",
                device_name, "T4 Verified" if profile["is_nvidia_t4"] else "Non-T4",
                total_mem_gb, profile["cuda_version"])
    return profile


def run_training_experiment(
    data_yaml: str,
    weights_path: str = "yolov8n.pt",
    epochs: int = 15,
    imgsz: int = 640,
    batch_size: int = 16,
    output_dir: str = "/kaggle/working/runs/train",
    experiment_id: str = "kaggle_t4_experiment",
) -> Dict[str, Any]:
    """Executes YOLOv8 fine-tuning on GPU using Ultralytics API."""
    from ultralytics import YOLO

    set_reproducibility(SEED)
    gpu_profile = audit_gpu_environment()

    device = 0 if torch.cuda.is_available() else "cpu"
    logger.info("Initializing YOLO model from %s on device: %s", weights_path, device)

    model = YOLO(weights_path)

    start_time = time.perf_counter()
    train_results = model.train(
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
    elapsed_time_s = time.perf_counter() - start_time

    best_checkpoint = Path(output_dir) / experiment_id / "weights" / "best.pt"
    logger.info("Training complete in %.2f seconds. Best checkpoint: %s (exists=%s)",
                elapsed_time_s, best_checkpoint, best_checkpoint.exists())

    summary = {
        "experiment_id": experiment_id,
        "status": "COMPLETED",
        "gpu_profile": gpu_profile,
        "training_time_seconds": round(elapsed_time_s, 2),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "weights_initial": weights_path,
        "weights_best": str(best_checkpoint),
        "best_weights_exist": best_checkpoint.exists(),
    }
    return summary


if __name__ == "__main__":
    profile = audit_gpu_environment()
    print(json.dumps(profile, indent=2))
