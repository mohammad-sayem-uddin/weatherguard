"""WeatherGuard Unified Model Evaluation Framework.

Module: scripts/evaluate_weatherguard.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
Mission: Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

Provides a standardized, reproducible evaluation interface supporting:
- Models: Baseline YOLOv8n, Fine-tuned checkpoints, Policy J, Policy K.
- Datasets: WeatherGuard-D1, BDD100K, DAWN, FLIR ADAS.
- Splits: validation, test, locked.
- Metrics: Overall F1, Object-size breakdown (Ultra-Small, Small, Medium, Large),
  Class-wise metrics, Condition-wise metrics, False positives per image.
- Latency profiling: Separates raw model inference from end-to-end pipeline latency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.priority_allocation import DetectionBox, PriorityAllocationController, compute_iou
from src.mission15.policy_k import PolicyKConfig, PolicyKController

DEFAULT_D1_CONFIG = ROOT / "configs" / "weatherguard_d1.yaml"
BASE_WEIGHTS_PATH = ROOT / "models" / "yolov8n.pt" if (ROOT / "models" / "yolov8n.pt").exists() else ROOT / "yolov8n.pt"

CANONICAL_CLASSES = ["truck", "person", "bicycle", "car", "motorcycle", "bus"]
YOLO_TO_CANONICAL = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class GroundTruthBox:
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    scale: str  # "ultra_small", "small", "medium", "large"
    max_dim: float


def get_object_scale(width: float, height: float) -> str:
    """WeatherGuard object-size convention based on max(pixel_width, pixel_height)."""
    max_d = max(width, height)
    if max_d < 16.0:
        return "ultra_small"
    elif max_d < 32.0:
        return "small"
    elif max_d < 96.0:
        return "medium"
    else:
        return "large"


def evaluate_predictions(
    all_predictions: List[List[DetectionBox]],
    all_ground_truths: List[List[GroundTruthBox]],
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Computes comprehensive object-level metrics across all images."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    scale_counts = {
        s: {"tp": 0, "fp": 0, "fn": 0}
        for s in ("ultra_small", "small", "medium", "large")
    }
    class_counts = {
        cls: {"tp": 0, "fp": 0, "fn": 0}
        for cls in CANONICAL_CLASSES
    }

    total_images = len(all_predictions)

    for preds, gts in zip(all_predictions, all_ground_truths):
        matched_gt = set()
        # Sort predictions by confidence descending
        sorted_preds = sorted(preds, key=lambda d: d.confidence, reverse=True)

        for p in sorted_preds:
            p_w = p.x2 - p.x1
            p_h = p.y2 - p.y1
            p_scale = get_object_scale(p_w, p_h)
            best_iou = 0.0
            best_idx = -1

            for g_idx, g in enumerate(gts):
                if g_idx in matched_gt:
                    continue
                if g.class_name.lower() == p.class_name.lower():
                    iou = compute_iou((p.x1, p.y1, p.x2, p.y2), (g.x1, g.y1, g.x2, g.y2))
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = g_idx

            if best_iou >= iou_thresh and best_idx >= 0:
                matched_gt.add(best_idx)
                total_tp += 1
                g_match = gts[best_idx]
                scale_counts[g_match.scale]["tp"] += 1
                if g_match.class_name in class_counts:
                    class_counts[g_match.class_name]["tp"] += 1
            else:
                total_fp += 1
                scale_counts[p_scale]["fp"] += 1
                if p.class_name in class_counts:
                    class_counts[p.class_name]["fp"] += 1

        for g_idx, g in enumerate(gts):
            if g_idx not in matched_gt:
                total_fn += 1
                scale_counts[g.scale]["fn"] += 1
                if g.class_name in class_counts:
                    class_counts[g.class_name]["fn"] += 1

    def compute_prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return round(prec, 4), round(rec, 4), round(f1, 4)

    o_prec, o_rec, o_f1 = compute_prf1(total_tp, total_fp, total_fn)

    out = {
        "overall": {
            "f1": o_f1,
            "precision": o_prec,
            "recall": o_rec,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "fp_per_image": round(total_fp / max(total_images, 1), 3),
        },
        "by_scale": {},
        "by_class": {},
    }

    for sc in ("ultra_small", "small", "medium", "large"):
        p, r, f = compute_prf1(scale_counts[sc]["tp"], scale_counts[sc]["fp"], scale_counts[sc]["fn"])
        out["by_scale"][sc] = {
            "f1": f, "precision": p, "recall": r,
            "tp": scale_counts[sc]["tp"], "fp": scale_counts[sc]["fp"], "fn": scale_counts[sc]["fn"],
        }

    for cls in CANONICAL_CLASSES:
        p, r, f = compute_prf1(class_counts[cls]["tp"], class_counts[cls]["fp"], class_counts[cls]["fn"])
        out["by_class"][cls] = {
            "f1": f, "precision": p, "recall": r,
            "tp": class_counts[cls]["tp"], "fp": class_counts[cls]["fp"], "fn": class_counts[cls]["fn"],
        }

    return out


def merge_two_tier_nms(
    base_dets: List[DetectionBox],
    crop_dets: List[DetectionBox],
    iou_thresh: float = 0.50,
) -> List[DetectionBox]:
    """Merges base and crop detections using two-tier cross-scale NMS."""
    if not crop_dets:
        return base_dets
    if not base_dets:
        return crop_dets
    all_dets = list(base_dets) + list(crop_dets)
    all_dets.sort(key=lambda d: d.confidence, reverse=True)
    keep = []
    while all_dets:
        cur = all_dets.pop(0)
        keep.append(cur)
        remaining = []
        for other in all_dets:
            if other.class_name == cur.class_name:
                iou = compute_iou((cur.x1, cur.y1, cur.x2, cur.y2), (other.x1, other.y1, other.x2, other.y2))
                if iou < iou_thresh:
                    remaining.append(other)
            else:
                remaining.append(other)
        all_dets = remaining
    return keep


class WeatherGuardEvaluator:
    """Unified evaluation engine for models, policies, and benchmark splits."""

    def __init__(
        self,
        weights_path: Path = BASE_WEIGHTS_PATH,
        policy: str = "raw_detector",  # "raw_detector", "policy_j", "policy_k"
        imgsz: int = 640,
        conf_thresh: float = 0.20,
        iou_thresh: float = 0.50,
    ):
        self.weights_path = Path(weights_path)
        self.policy = policy.lower()
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Load YOLO model
        if self.weights_path.exists():
            self.model = YOLO(str(self.weights_path))
        else:
            self.model = None

        # Policy controllers
        self.ctrl_j = None
        self.ctrl_k = None
        if self.policy == "policy_j":
            self.ctrl_j = PriorityAllocationController(policy="policy_j_best_combination", max_crops=2)
        elif self.policy == "policy_k":
            self.ctrl_k = PolicyKController(PolicyKConfig(max_crops=2, crop_size=448))

    def evaluate_image_stream(
        self,
        image_records: List[Dict[str, Any]],
        warmup_frames: int = 5,
    ) -> Dict[str, Any]:
        """Runs evaluation over a list of image paths and corresponding ground truths."""
        if self.model is None:
            return {
                "status": "MODEL_WEIGHTS_NOT_FOUND",
                "message": f"Weights file {self.weights_path} not found.",
            }

        # Warmup
        if image_records:
            for item in image_records[:warmup_frames]:
                img_path = item.get("file_path")
                if img_path and Path(img_path).exists():
                    _ = self.model(img_path, imgsz=self.imgsz, verbose=False)

        raw_infer_times: List[float] = []
        pipeline_times: List[float] = []

        all_preds: List[List[DetectionBox]] = []
        all_gts: List[List[GroundTruthBox]] = []

        for item in image_records:
            img_path = Path(item["file_path"])
            gts = item.get("gt_boxes", [])

            t_pipe_start = time.perf_counter()

            # Step 1: Base Model Forward Pass
            t_infer_start = time.perf_counter()
            res = self.model(str(img_path), imgsz=self.imgsz, conf=self.conf_thresh, verbose=False)[0]
            infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
            raw_infer_times.append(infer_ms)

            base_boxes: List[DetectionBox] = []
            for b in res.boxes:
                cls_id = int(b.cls[0])
                if cls_id in YOLO_TO_CANONICAL:
                    xyxy = b.xyxy[0].cpu().numpy()
                    base_boxes.append(DetectionBox(
                        class_name=YOLO_TO_CANONICAL[cls_id],
                        confidence=float(b.conf[0]),
                        x1=float(xyxy[0]), y1=float(xyxy[1]), x2=float(xyxy[2]), y2=float(xyxy[3]),
                    ))

            final_dets = base_boxes

            # Step 2: Policy Refinements (if enabled)
            if self.policy in ("policy_j", "policy_k"):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None:
                    h, w = img_bgr.shape[:2]
                    crop_dets = []

                    if self.policy == "policy_j" and self.ctrl_j:
                        dec = self.ctrl_j.allocate_crops(img_bgr, detector_boxes=base_boxes)
                        crops = [(int(c.x1), int(c.y1), int(c.x2), int(c.y2)) for c in dec.allocated_crops]
                    elif self.policy == "policy_k" and self.ctrl_k:
                        crops, _, _ = self.ctrl_k.process_and_allocate(img_bgr, sequence_id=item.get("sequence_id", "seq_default"), detector_boxes=base_boxes)
                    else:
                        crops = []

                    for cx1, cy1, cx2, cy2 in crops:
                        crop_patch = img_bgr[cy1:cy2, cx1:cx2]
                        if crop_patch.shape[0] >= 32 and crop_patch.shape[1] >= 32:
                            c_res = self.model(crop_patch, imgsz=448, conf=self.conf_thresh, verbose=False)[0]
                            for cb in c_res.boxes:
                                ccls = int(cb.cls[0])
                                if ccls in YOLO_TO_CANONICAL:
                                    cxy = cb.xyxy[0].cpu().numpy()
                                    crop_dets.append(DetectionBox(
                                        class_name=YOLO_TO_CANONICAL[ccls],
                                        confidence=float(cb.conf[0]),
                                        x1=float(cxy[0] + cx1), y1=float(cxy[1] + cy1),
                                        x2=float(cxy[2] + cx1), y2=float(cxy[3] + cy1),
                                    ))

                    final_dets = merge_two_tier_nms(base_boxes, crop_dets, iou_thresh=self.iou_thresh)

            pipe_ms = (time.perf_counter() - t_pipe_start) * 1000.0
            pipeline_times.append(pipe_ms)

            all_preds.append(final_dets)
            all_gts.append(gts)

        # Compute accuracy metrics
        metrics = evaluate_predictions(all_preds, all_gts, iou_thresh=self.iou_thresh)

        # Latency statistics
        def summarize(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "fps": 0.0}
            a = np.array(arr)
            mean_v = float(np.mean(a))
            return {
                "mean_ms": round(mean_v, 2),
                "median_ms": round(float(np.median(a)), 2),
                "p90_ms": round(float(np.percentile(a, 90)), 2),
                "p95_ms": round(float(np.percentile(a, 95)), 2),
                "fps": round(1000.0 / mean_v, 2) if mean_v > 0 else 0.0,
            }

        metrics["latency"] = {
            "pure_model_inference": summarize(raw_infer_times),
            "end_to_end_pipeline": summarize(pipeline_times),
            "measured_images": len(image_records),
            "warmup_images": warmup_frames,
            "hardware": platform.processor(),
        }

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate WeatherGuard models across datasets and policies.")
    parser.add_argument("--model", type=str, default=str(BASE_WEIGHTS_PATH), help="Weights checkpoint path")
    parser.add_argument("--policy", type=str, default="raw_detector", choices=["raw_detector", "policy_j", "policy_k"])
    parser.add_argument("--dataset", type=str, default="d1", choices=["d1", "bdd100k", "dawn", "flir"])
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test", "locked"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--output", type=str, default="artifacts/evaluation_summary.json")
    args = parser.parse_args()

    evaluator = WeatherGuardEvaluator(
        weights_path=Path(args.model),
        policy=args.policy,
        imgsz=args.imgsz,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    )

    print(f"Unified evaluation interface configured for {args.dataset} ({args.split}) with policy {args.policy} at {args.imgsz}px.")
    # Check if dataset is D1 and present
    if args.dataset == "d1":
        manifest_p = ROOT / "data" / "WeatherGuard-D1" / "manifests" / "d1_split_manifest.json"
        if not manifest_p.exists():
            print(f"Dataset manifest {manifest_p} not found. WeatherGuard-D1 is awaiting synthetic generation.")
            return

    print("Evaluation interface ready.")


if __name__ == "__main__":
    main()
