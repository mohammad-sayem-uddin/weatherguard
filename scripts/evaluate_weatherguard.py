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

COCO_TO_CANONICAL = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

D1_TO_CANONICAL = {
    0: "truck",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "bus",
}

# Backward compatibility alias
YOLO_TO_CANONICAL = dict(COCO_TO_CANONICAL)


class ModelClassMappingError(ValueError):
    """Raised when model class metadata cannot be mapped to the canonical WeatherGuard taxonomy."""
    pass


def resolve_model_class_mapping(
    model_names: Optional[Dict[Any, str] | List[str]] = None,
    model_type_hint: Optional[str] = None,
) -> Dict[int, str]:
    """Resolves an unambiguous mapping from model output class indices to canonical class names.

    Supports:
    1. Direct inspection of model.names dictionary or list.
    2. Explicit model_type_hint ('coco', 'd1', 'weatherguard_d1').
    3. Rejection of incompatible, unknown, or ambiguous class taxonomies.
    """
    if model_type_hint:
        hint = model_type_hint.lower().strip()
        if hint in ("coco", "coco80"):
            return dict(COCO_TO_CANONICAL)
        elif hint in ("d1", "weatherguard_d1", "custom6"):
            return dict(D1_TO_CANONICAL)
        else:
            raise ModelClassMappingError(f"Unknown model_type_hint: '{model_type_hint}'. Allowed: 'coco', 'd1'.")

    if model_names is None:
        raise ModelClassMappingError(
            "Cannot resolve class mapping: neither model.names nor a valid model_type_hint was provided."
        )

    # Convert list to dict if needed
    if isinstance(model_names, list):
        names_dict = {i: str(name) for i, name in enumerate(model_names)}
    elif isinstance(model_names, dict):
        names_dict = {int(k): str(v) for k, v in model_names.items()}
    else:
        raise ModelClassMappingError(f"Unsupported model_names type: {type(model_names)}")

    if not names_dict:
        raise ModelClassMappingError("model.names is empty.")

    # Match names against canonical taxonomy (case-insensitive)
    canonical_lookup = {c.lower(): c for c in CANONICAL_CLASSES}
    mapping: Dict[int, str] = {}

    for idx, name in names_dict.items():
        norm_name = str(name).strip().lower()
        if norm_name in canonical_lookup:
            mapping[idx] = canonical_lookup[norm_name]

    # Verification: Check if we mapped at least one class
    if not mapping:
        sample_names = list(names_dict.values())[:10]
        raise ModelClassMappingError(
            f"No canonical driving classes found in model.names: {sample_names}... "
            f"Expected classes matching {CANONICAL_CLASSES}."
        )

    # Check for duplicate target mappings
    targets = list(mapping.values())
    if len(targets) != len(set(targets)):
        raise ModelClassMappingError(
            f"Ambiguous model.names: multiple indices map to the same canonical class ({mapping})."
        )

    # If the model has 80 classes (standard COCO), verify expected indices
    if len(names_dict) == 80:
        for k, v in COCO_TO_CANONICAL.items():
            if k not in mapping or mapping[k] != v:
                raise ModelClassMappingError(
                    f"COCO-80 model has unexpected class names at index {k}: expected '{v}', found '{names_dict.get(k)}'."
                )
        return mapping

    # If the model has 6 classes (WeatherGuard-D1), verify all 6 are covered
    if len(names_dict) == 6:
        missing = set(CANONICAL_CLASSES) - set(mapping.values())
        if missing:
            raise ModelClassMappingError(
                f"6-class model is missing canonical classes: {missing}. Found: {names_dict}"
            )
        return mapping

    return mapping


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
        model_type_hint: Optional[str] = None,
    ):
        self.weights_path = Path(weights_path)
        self.policy = policy.lower()
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model_type_hint = model_type_hint

        # Load YOLO model
        if self.weights_path.exists():
            self.model = YOLO(str(self.weights_path))
            self.class_mapping = resolve_model_class_mapping(
                model_names=getattr(self.model, "names", None),
                model_type_hint=self.model_type_hint,
            )
        else:
            self.model = None
            self.class_mapping = resolve_model_class_mapping(
                model_names=None,
                model_type_hint=self.model_type_hint or "coco",
            )

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
                if cls_id in self.class_mapping:
                    xyxy = b.xyxy[0].cpu().numpy()
                    base_boxes.append(DetectionBox(
                        class_name=self.class_mapping[cls_id],
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
                                if ccls in self.class_mapping:
                                    cxy = cb.xyxy[0].cpu().numpy()
                                    crop_dets.append(DetectionBox(
                                        class_name=self.class_mapping[ccls],
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


def load_dataset_records(
    dataset_name: str,
    split: str = "validation",
    max_samples: Optional[int] = None,
    allow_locked_test: bool = False,
) -> List[Dict[str, Any]]:
    """Loads image paths and ground-truth bounding boxes for a requested dataset and split.

    Strictly enforces locked-test protections during development.
    """
    clean_split = split.lower().strip()
    is_locked = clean_split in ("test", "locked", "frozen_internal_test", "group_c")

    if is_locked and not allow_locked_test:
        raise PermissionError(
            f"Access to locked test split '{split}' is forbidden during development. "
            f"Pass --allow-locked-test explicitly only for formal locked evaluation."
        )

    records: List[Dict[str, Any]] = []

    if dataset_name == "d1":
        # Check canonical manifest locations
        manifest_candidates = [
            ROOT / "data" / "WeatherGuard-D1" / "manifests" / "d1_split_manifest.json",
            ROOT / "artifacts" / "d1_splits_manifest.json",
            ROOT / "artifacts" / "d1_split_manifest.json",
        ]
        manifest_p = None
        for cand in manifest_candidates:
            if cand.exists():
                manifest_p = cand
                break

        if manifest_p is None:
            raise FileNotFoundError(
                f"WeatherGuard-D1 split manifest not found in {[str(p) for p in manifest_candidates]}. "
                f"Awaiting synthetic dataset generation and split generation."
            )

        with open(manifest_p, "r") as f:
            manifest_data = json.load(f)

        split_key = "validation" if clean_split in ("val", "validation") else clean_split
        img_ids = manifest_data.get(split_key, {}).get("image_ids", [])
        if not img_ids:
            return []

        img_dir = ROOT / "data" / "WeatherGuard-D1" / "images"
        lbl_dir = ROOT / "data" / "WeatherGuard-D1" / "labels"

        # Check if actual images are present on disk
        target_ids = img_ids[:max_samples] if max_samples else img_ids
        for img_id in target_ids:
            img_path = None
            for ext in (".jpg", ".png", ".jpeg"):
                p = img_dir / f"{img_id}{ext}"
                if p.exists():
                    img_path = p
                    break
                # Check condition subfolders
                for cond in ("day", "night", "rain", "fog"):
                    sub_p = img_dir / cond / f"{img_id}{ext}"
                    if sub_p.exists():
                        img_path = sub_p
                        break
                if img_path:
                    break

            if not img_path:
                continue

            # Load YOLO label txt
            gt_boxes: List[GroundTruthBox] = []
            lbl_file = lbl_dir / f"{img_id}.txt"
            if lbl_file.exists():
                with open(lbl_file) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            xc = float(parts[1]) * 1280.0
                            yc = float(parts[2]) * 720.0
                            w = float(parts[3]) * 1280.0
                            h = float(parts[4]) * 720.0
                            x1 = xc - w / 2.0
                            y1 = yc - h / 2.0
                            x2 = xc + w / 2.0
                            y2 = yc + h / 2.0
                            cname = D1_TO_CANONICAL.get(cid, "car")
                            scale = get_object_scale(w, h)
                            gt_boxes.append(GroundTruthBox(
                                class_name=cname,
                                x1=round(x1, 2), y1=round(y1, 2),
                                x2=round(x2, 2), y2=round(y2, 2),
                                scale=scale, max_dim=round(max(w, h), 2),
                            ))

            records.append({
                "image_id": img_id,
                "file_path": str(img_path),
                "gt_boxes": gt_boxes,
            })

    elif dataset_name == "bdd100k":
        samples_path = ROOT / "data" / "bdd100k" / "samples.json"
        img_dir = ROOT / "data" / "bdd100k" / "images"

        if not samples_path.exists() or not img_dir.exists():
            raise FileNotFoundError(f"BDD100K data not found at {samples_path} or {img_dir}.")

        with open(samples_path) as f:
            samples_data = json.load(f)

        bdd_cat_map = {
            "pedestrian": "person",
            "car": "car",
            "truck": "truck",
            "bus": "bus",
            "motorcycle": "motorcycle",
            "bicycle": "bicycle",
        }

        # Filter split
        for item in samples_data:
            img_name = item.get("name")
            img_file = img_dir / img_name
            if not img_file.exists():
                continue

            gt_boxes: List[GroundTruthBox] = []
            for obj in item.get("labels", []):
                cat = obj.get("category", "").lower()
                if cat in bdd_cat_map:
                    box2d = obj.get("box2d", {})
                    x1 = float(box2d.get("x1", 0))
                    y1 = float(box2d.get("y1", 0))
                    x2 = float(box2d.get("x2", 0))
                    y2 = float(box2d.get("y2", 0))
                    w = x2 - x1
                    h = y2 - y1
                    scale = get_object_scale(w, h)
                    gt_boxes.append(GroundTruthBox(
                        class_name=bdd_cat_map[cat],
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        scale=scale, max_dim=max(w, h),
                    ))

            records.append({
                "image_id": img_name,
                "file_path": str(img_file),
                "gt_boxes": gt_boxes,
            })
            if max_samples and len(records) >= max_samples:
                break

    elif dataset_name == "dawn":
        val_txt = ROOT / "data" / "dawn_yolo" / "val.txt"
        lbl_dir = ROOT / "data" / "dawn_yolo" / "labels"

        if not val_txt.exists():
            raise FileNotFoundError(f"DAWN validation list {val_txt} not found.")

        with open(val_txt) as f:
            lines = [l.strip() for l in f if l.strip()]

        target_lines = lines[:max_samples] if max_samples else lines
        for img_p_str in target_lines:
            img_p = Path(img_p_str)
            if not img_p.exists():
                continue

            # Infer width and height from image if needed, or assume default
            try:
                with Image.open(img_p) as pil_im:
                    im_w, im_h = pil_im.size
            except Exception:
                im_w, im_h = 1280.0, 720.0

            gt_boxes: List[GroundTruthBox] = []
            lbl_file = lbl_dir / f"{img_p.stem}.txt"
            if lbl_file.exists():
                with open(lbl_file) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            xc = float(parts[1]) * im_w
                            yc = float(parts[2]) * im_h
                            w = float(parts[3]) * im_w
                            h = float(parts[4]) * im_h
                            x1 = xc - w / 2.0
                            y1 = yc - h / 2.0
                            x2 = xc + w / 2.0
                            y2 = yc + h / 2.0
                            cname = D1_TO_CANONICAL.get(cid, "car")
                            scale = get_object_scale(w, h)
                            gt_boxes.append(GroundTruthBox(
                                class_name=cname,
                                x1=round(x1, 2), y1=round(y1, 2),
                                x2=round(x2, 2), y2=round(y2, 2),
                                scale=scale, max_dim=round(max(w, h), 2),
                            ))

            records.append({
                "image_id": img_p.stem,
                "file_path": str(img_p),
                "gt_boxes": gt_boxes,
            })

    elif dataset_name == "flir":
        flir_p = ROOT / "data" / "flir_thermal" / "validation.parquet"
        if not flir_p.exists():
            raise FileNotFoundError(f"FLIR dataset not found at {flir_p}.")

        import pyarrow.parquet as pq
        table = pq.read_table(flir_p)
        df = table.to_pandas()

        flir_img_dir = ROOT / "data" / "flir_thermal" / "temp_eval_images"
        flir_img_dir.mkdir(parents=True, exist_ok=True)

        cat_map = {1: "bicycle", 2: "car", 4: "person"}
        valid_rows = [row for _, row in df.iterrows() if len(row["objects"]["category"]) > 0]
        target_rows = valid_rows[:max_samples] if max_samples else valid_rows

        for row in target_rows:
            image_id = str(row["image_id"])
            img_file = flir_img_dir / f"flir_{image_id}.jpg"
            if not img_file.exists():
                import io
                img_bytes = row["image"]["bytes"]
                with open(img_file, "wb") as f:
                    f.write(img_bytes)

            objs = row["objects"]
            gt_boxes: List[GroundTruthBox] = []
            for i in range(len(objs["category"])):
                cid = int(objs["category"][i])
                cname = cat_map.get(cid)
                if cname:
                    b = objs["bbox"][i]
                    x1 = float(b[0])
                    y1 = float(b[1])
                    w = float(b[2])
                    h = float(b[3])
                    x2 = x1 + w
                    y2 = y1 + h
                    scale = get_object_scale(w, h)
                    gt_boxes.append(GroundTruthBox(
                        class_name=cname,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        scale=scale, max_dim=max(w, h),
                    ))

            records.append({
                "image_id": image_id,
                "file_path": str(img_file),
                "gt_boxes": gt_boxes,
            })

    else:
        raise ValueError(f"Unsupported dataset: '{dataset_name}'. Allowed: 'd1', 'bdd100k', 'dawn', 'flir'.")

    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate WeatherGuard models across datasets and policies.")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment YAML config (e.g. configs/mission16/e1_baseline_yolov8n.yaml)")
    parser.add_argument("--model", type=str, default=str(BASE_WEIGHTS_PATH), help="Weights checkpoint path")
    parser.add_argument("--model-type-hint", type=str, default=None, choices=["coco", "d1"], help="Explicit class mapping hint")
    parser.add_argument("--policy", type=str, default="raw_detector", choices=["raw_detector", "policy_j", "policy_k"])
    parser.add_argument("--dataset", type=str, default="d1", choices=["d1", "bdd100k", "dawn", "flir"])
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "val", "test", "locked"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of images to evaluate (for fast profiling/dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Perform preflight check without executing full inference")
    parser.add_argument("--allow-locked-test", action="store_true", help="Permit evaluating on locked test split (formal evaluations only)")
    parser.add_argument("--output", type=str, default="artifacts/evaluation_summary.json")
    parser.add_argument("--update-registry", type=str, default=None, help="Experiment ID to update in experiment_registry.csv/json")
    args = parser.parse_args()

    # Load from config if provided
    exp_id = args.update_registry
    weights_path = Path(args.model)
    policy = args.policy
    imgsz = args.imgsz
    conf_thresh = args.conf
    iou_thresh = args.iou
    dataset_name = args.dataset
    split = args.split

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {cfg_path}")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        exp_id = cfg.get("experiment_id", exp_id)
        model_sec = cfg.get("model", {})
        eval_sec = cfg.get("evaluation", {})

        if "weights" in model_sec:
            weights_path = ROOT / model_sec["weights"] if not Path(model_sec["weights"]).is_absolute() else Path(model_sec["weights"])
        elif "output_checkpoint" in model_sec:
            weights_path = ROOT / model_sec["output_checkpoint"] if not Path(model_sec["output_checkpoint"]).is_absolute() else Path(model_sec["output_checkpoint"])

        policy = eval_sec.get("policy", policy)
        imgsz = eval_sec.get("imgsz", imgsz)
        conf_thresh = eval_sec.get("conf_thresh", conf_thresh)
        iou_thresh = eval_sec.get("iou_thresh", iou_thresh)

        # Detect dataset from config if specified
        if "datasets" in eval_sec and isinstance(eval_sec["datasets"], list) and len(eval_sec["datasets"]) > 0:
            first_d = eval_sec["datasets"][0]
            if isinstance(first_d, dict):
                dataset_name = first_d.get("name", dataset_name)
                d_splits = first_d.get("splits", [split])
                # Default to validation during development
                split = "validation" if "validation" in d_splits else d_splits[0]

    # Preflight / Dry-run handling
    if args.dry_run:
        preflight_info = {
            "status": "PREFLIGHT_VERIFIED",
            "experiment_id": exp_id or "ad_hoc_eval",
            "model_weights": str(weights_path),
            "weights_exist": weights_path.exists(),
            "policy": policy,
            "dataset": dataset_name,
            "split": split,
            "imgsz": imgsz,
            "conf_thresh": conf_thresh,
            "iou_thresh": iou_thresh,
            "hardware": platform.processor(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        print(f"Preflight evaluation verified for {exp_id or 'eval'} ({dataset_name} / {split}).")
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(preflight_info, f, indent=2)
        return

    # Initialize evaluator
    evaluator = WeatherGuardEvaluator(
        weights_path=weights_path,
        policy=policy,
        imgsz=imgsz,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        model_type_hint=args.model_type_hint,
    )

    print(f"Evaluating {weights_path.name} on {dataset_name} ({split}) using {policy}...")

    try:
        records = load_dataset_records(
            dataset_name=dataset_name,
            split=split,
            max_samples=args.max_samples,
            allow_locked_test=args.allow_locked_test,
        )
    except FileNotFoundError as e:
        print(f"Dataset notice: {e}")
        status_info = {
            "status": "DATASET_NOT_INGESTED",
            "dataset": dataset_name,
            "split": split,
            "message": str(e),
        }
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(status_info, f, indent=2)
        return
    except PermissionError as e:
        print(f"Locked-test protection triggered: {e}")
        raise

    if not records:
        print(f"No records available to evaluate for {dataset_name} ({split}).")
        status_info = {
            "status": "NO_RECORDS_AVAILABLE",
            "dataset": dataset_name,
            "split": split,
            "message": "Zero records loaded for evaluation.",
        }
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(status_info, f, indent=2)
        return

    print(f"Running inference over {len(records)} images...")
    results = evaluator.evaluate_image_stream(records)

    # Attach metadata
    results["metadata"] = {
        "experiment_id": exp_id or "untracked_eval",
        "dataset": dataset_name,
        "split": split,
        "weights": str(weights_path),
        "policy": policy,
        "imgsz": imgsz,
        "conf_thresh": conf_thresh,
        "iou_thresh": iou_thresh,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Save output JSON
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results successfully saved to {out_p}")

    # Display summary
    overall = results.get("overall", {})
    lat = results.get("latency", {}).get("end_to_end_pipeline", {})
    print(f"Results: Overall F1={overall.get('f1', 0.0):.4f} (Prec={overall.get('precision', 0.0):.4f}, Rec={overall.get('recall', 0.0):.4f}) | Mean Latency={lat.get('mean_ms', 0.0)}ms ({lat.get('fps', 0.0)} FPS)")

    # Update experiment registry if requested
    if exp_id:
        from scripts.manage_experiment_registry import ExperimentRegistry
        registry = ExperimentRegistry()
        rec_data = {
            "experiment_id": exp_id,
            "overall_f1": overall.get("f1", ""),
            "small_f1": results.get("by_scale", {}).get("small", {}).get("f1", ""),
            "ultra_small_f1": results.get("by_scale", {}).get("ultra_small", {}).get("f1", ""),
            "mean_latency_ms": lat.get("mean_ms", ""),
            "fps": lat.get("fps", ""),
            "status": "COMPLETED",
            "artifact_locations": str(out_p),
        }
        registry.upsert_experiment(rec_data)
        print(f"Registry updated for {exp_id}.")


if __name__ == "__main__":
    main()
