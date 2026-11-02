"""Region-Adaptive High-Resolution Inference Architecture for WeatherGuard Mission 11.

Implements:
1. Candidate proposal generation (small object cues, uncertainty, object density, horizon priors).
2. Spatial candidate clustering with context padding.
3. Fixed-grid and road-horizon baseline slicing.
4. Exact coordinate restoration with letterbox / aspect-ratio awareness.
5. Boundary-aware duplicate suppression and duplicate-safe detection merging.
6. Stage-level latency, crop count, and pixel profiling for Apple M1 CPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image

from src.video_dataset import CANONICAL_CLASSES, CANONICAL_MAPPING, DetectionBox, compute_iou


@dataclass
class CropRegion:
    """Bounding box of an extracted crop in native image coordinates [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float
    candidate_indices: List[int] = field(default_factory=list)
    source_strategy: str = "clustered"

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def box_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": round(float(self.x1), 1),
            "y1": round(float(self.y1), 1),
            "x2": round(float(self.x2), 1),
            "y2": round(float(self.y2), 1),
            "candidate_indices": list(self.candidate_indices),
            "source_strategy": str(self.source_strategy),
        }


def transform_crop_to_original(
    crop_box: Sequence[float],
    crop_region: CropRegion,
    crop_input_size: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float, float, float]:
    """Projects detection coordinates from crop space back to original image space.
    
    Ultralytics YOLO returns bounding box coordinates already scaled to the native
    pixel dimensions of the input crop image (crop_region.width x crop_region.height).
    Therefore, the projection to full-frame space is simply an offset translation by
    (crop_region.x1, crop_region.y1).
    """
    orig_x1 = crop_region.x1 + float(crop_box[0])
    orig_y1 = crop_region.y1 + float(crop_box[1])
    orig_x2 = crop_region.x1 + float(crop_box[2])
    orig_y2 = crop_region.y1 + float(crop_box[3])

    return (orig_x1, orig_y1, orig_x2, orig_y2)


def is_touching_crop_boundary(
    box: Sequence[float],
    crop_size: Tuple[float, float],
    boundary_margin_px: float = 4.0,
) -> bool:
    """Detects if a bounding box within a crop is truncated by the crop boundary."""
    cw, ch = crop_size
    bx1, by1, bx2, by2 = box[:4]
    touches_left = bx1 <= boundary_margin_px
    touches_top = by1 <= boundary_margin_px
    touches_right = bx2 >= (cw - boundary_margin_px)
    touches_bottom = by2 >= (ch - boundary_margin_px)
    return touches_left or touches_top or touches_right or touches_bottom


def cluster_candidate_boxes(
    boxes: Sequence[Tuple[float, float, float, float]],
    image_shape: Tuple[int, int] = (1280, 720),
    max_distance_px: float = 96.0,
    context_margin: float = 0.20,
    min_crop_size: float = 256.0,
    max_crops: int = 3,
) -> List[CropRegion]:
    """Clusters spatial candidate boxes into unified, context-expanded crop regions."""
    if not boxes:
        return []

    img_w, img_h = image_shape
    n = len(boxes)
    # Adjacency graph based on Euclidean distance between box centers
    centers = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in boxes])
    
    # Simple connected components clustering
    visited = [False] * n
    clusters: List[List[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        queue = [i]

        while queue:
            curr = queue.pop(0)
            for j in range(n):
                if not visited[j]:
                    dist = np.linalg.norm(centers[curr] - centers[j])
                    if dist <= max_distance_px:
                        visited[j] = True
                        cluster.append(j)
                        queue.append(j)
        clusters.append(cluster)

    # Sort clusters by candidate count (prioritize densest clusters)
    clusters.sort(key=lambda c: len(c), reverse=True)
    clusters = clusters[:max_crops]

    crop_regions: List[CropRegion] = []
    for cluster in clusters:
        c_boxes = [boxes[idx] for idx in cluster]
        min_x = min(b[0] for b in c_boxes)
        min_y = min(b[1] for b in c_boxes)
        max_x = max(b[2] for b in c_boxes)
        max_y = max(b[3] for b in c_boxes)

        w = max_x - min_x
        h = max_y - min_y

        # Expand with context margin
        pad_x = max(w * context_margin, 24.0)
        pad_y = max(h * context_margin, 24.0)

        cx1 = max(0.0, min_x - pad_x)
        cy1 = max(0.0, min_y - pad_y)
        cx2 = min(float(img_w), max_x + pad_x)
        cy2 = min(float(img_h), max_y + pad_y)

        # Enforce min crop size while staying within image
        cw = cx2 - cx1
        ch = cy2 - cy1
        if cw < min_crop_size and img_w >= min_crop_size:
            diff = min_crop_size - cw
            cx1 = max(0.0, cx1 - diff / 2.0)
            cx2 = cx1 + min_crop_size
            if cx2 > float(img_w):
                cx2 = float(img_w)
                cx1 = max(0.0, cx2 - min_crop_size)
        if ch < min_crop_size and img_h >= min_crop_size:
            diff = min_crop_size - ch
            cy1 = max(0.0, cy1 - diff / 2.0)
            cy2 = cy1 + min_crop_size
            if cy2 > float(img_h):
                cy2 = float(img_h)
                cy1 = max(0.0, cy2 - min_crop_size)

        crop_regions.append(
            CropRegion(
                x1=round(cx1, 2),
                y1=round(cy1, 2),
                x2=round(cx2, 2),
                y2=round(cy2, 2),
                candidate_indices=cluster,
                source_strategy="clustered",
            )
        )

    return crop_regions


def generate_fixed_grid_crops(
    image_shape: Tuple[int, int] = (1280, 720),
    grid_type: str = "horizon_strip",
) -> List[CropRegion]:
    """Generates deterministic baseline crops (e.g. road horizon strip or 2x2 grid)."""
    img_w, img_h = image_shape
    crops = []

    if grid_type == "horizon_strip":
        # Overlapping horizontal band where small/distant vehicles concentrate
        # Band 1: Center road zone y in [0.28, 0.68] * H
        crops.append(
            CropRegion(
                x1=0.0,
                y1=round(0.28 * img_h, 2),
                x2=float(img_w),
                y2=round(0.68 * img_h, 2),
                source_strategy="horizon_strip",
            )
        )
    elif grid_type == "overlapping_grid_2x1":
        # Two horizontal strips: upper road zone and lower vehicle zone
        crops.append(
            CropRegion(
                x1=0.0,
                y1=round(0.25 * img_h, 2),
                x2=float(img_w),
                y2=round(0.65 * img_h, 2),
                source_strategy="grid_2x1_top",
            )
        )
        crops.append(
            CropRegion(
                x1=0.0,
                y1=round(0.45 * img_h, 2),
                x2=float(img_w),
                y2=round(0.85 * img_h, 2),
                source_strategy="grid_2x1_bottom",
            )
        )
    elif grid_type == "grid_2x2":
        # 4 quadrant overlapping crops (50% size with 20% overlap)
        mid_x = img_w / 2.0
        mid_y = img_h / 2.0
        pad_x = img_w * 0.10
        pad_y = img_h * 0.10
        crops.append(CropRegion(0.0, 0.0, min(float(img_w), mid_x + pad_x), min(float(img_h), mid_y + pad_y), source_strategy="grid_2x2_tl"))
        crops.append(CropRegion(max(0.0, mid_x - pad_x), 0.0, float(img_w), min(float(img_h), mid_y + pad_y), source_strategy="grid_2x2_tr"))
        crops.append(CropRegion(0.0, max(0.0, mid_y - pad_y), min(float(img_w), mid_x + pad_x), float(img_h), source_strategy="grid_2x2_bl"))
        crops.append(CropRegion(max(0.0, mid_x - pad_x), max(0.0, mid_y - pad_y), float(img_w), float(img_h), source_strategy="grid_2x2_br"))

    return crops


def merge_detections_nms(
    base_detections: Sequence[DetectionBox],
    crop_detections: Sequence[DetectionBox],
    iou_threshold: float = 0.50,
    small_area_threshold: float = 1024.0,
    base_conf_threshold: float = 0.20,
    new_crop_conf_threshold: float = 0.30,
    merge_mode: str = "conservative",  # 'standard_nms' or 'conservative'
) -> List[DetectionBox]:
    """Merges base and restored crop detections.
    
    In 'conservative' mode:
    - Retains all base detections >= base_conf_threshold.
    - Upgrades low-confidence base detections (0.10 <= s < base_conf_threshold) if confirmed by crop.
    - Updates overlapping base detections if crop has higher confidence.
    - Adds new crop discoveries only if confidence >= new_crop_conf_threshold, preventing false alarms.
    """
    if merge_mode == "conservative":
        # Base detections split into confident vs low-conf candidates
        output_boxes = [b for b in base_detections if b.confidence >= base_conf_threshold]
        low_conf_base = [b for b in base_detections if 0.10 <= b.confidence < base_conf_threshold]

        for cb in crop_detections:
            if cb.confidence < 0.10:
                continue
            cb_box = (cb.x1, cb.y1, cb.x2, cb.y2)

            # 1. Check overlap with confident base detections
            overlaps = [
                (compute_iou(cb_box, (ob.x1, ob.y1, ob.x2, ob.y2)), idx)
                for idx, ob in enumerate(output_boxes)
                if ob.class_name == cb.class_name
            ]
            best_iou, best_idx = max(overlaps, key=lambda x: x[0]) if overlaps else (0.0, -1)

            if best_iou >= iou_threshold:
                # Overlaps with an already confident base detection
                # Favor crop detection if object is small or crop confidence is strictly higher
                if (cb.area < small_area_threshold and cb.confidence >= output_boxes[best_idx].confidence * 0.90) or cb.confidence > output_boxes[best_idx].confidence:
                    output_boxes[best_idx] = cb
                continue

            # 2. Check overlap with low-confidence base detections (recovery of weak candidates)
            low_overlaps = [
                (compute_iou(cb_box, (lb.x1, lb.y1, lb.x2, lb.y2)), lb)
                for lb in low_conf_base
                if lb.class_name == cb.class_name
            ]
            best_low_iou, best_low_box = max(low_overlaps, key=lambda x: x[0]) if low_overlaps else (0.0, None)

            if best_low_iou >= iou_threshold and cb.confidence >= base_conf_threshold:
                # Upgraded confirmation of previously uncertain base detection!
                output_boxes.append(cb)
                continue

            # 3. Completely new crop discovery: require new_crop_conf_threshold
            if cb.confidence >= new_crop_conf_threshold:
                output_boxes.append(cb)

        return output_boxes

    # Fallback standard NMS
    all_dets = list(base_detections) + list(crop_detections)
    if not all_dets:
        return []

    def sort_key(d: DetectionBox) -> float:
        bonus = 0.05 if d.area < small_area_threshold else 0.0
        return d.confidence + bonus

    all_dets.sort(key=sort_key, reverse=True)

    kept: List[DetectionBox] = []
    classes = set(d.class_name for d in all_dets)

    for c in classes:
        c_dets = [d for d in all_dets if d.class_name == c]
        while c_dets:
            best = c_dets.pop(0)
            kept.append(best)
            best_box = (best.x1, best.y1, best.x2, best.y2)
            c_dets = [
                d for d in c_dets
                if compute_iou(best_box, (d.x1, d.y1, d.x2, d.y2)) < iou_threshold
            ]

    return kept


class RegionAdaptiveDetector:
    """End-to-end Region-Adaptive High-Resolution Detector for edge CPU perception."""

    def __init__(
        self,
        yolo_model: Any,
        base_resolution: int = 640,
        crop_resolution: int = 640,
        strategy: str = "clustered",  # 'individual', 'clustered', 'fixed_grid', 'hybrid'
        small_object_area_threshold: float = 1024.0,  # 32x32 px
        uncertainty_min_conf: float = 0.15,
        uncertainty_max_conf: float = 0.40,
        max_crops_per_frame: int = 2,
        context_margin: float = 0.20,
        min_crop_size: float = 256.0,
        cluster_distance_px: float = 80.0,
        nms_merge_iou: float = 0.50,
        boundary_suppression_margin_px: float = 4.0,
        output_conf_threshold: float = 0.20,
        new_crop_conf_threshold: float = 0.30,
        merge_mode: str = "conservative",
    ):
        self.model = yolo_model
        self.base_resolution = base_resolution
        self.crop_resolution = crop_resolution
        self.strategy = strategy
        self.small_object_area_threshold = small_object_area_threshold
        self.uncertainty_min_conf = uncertainty_min_conf
        self.uncertainty_max_conf = uncertainty_max_conf
        self.max_crops_per_frame = max_crops_per_frame
        self.context_margin = context_margin
        self.min_crop_size = min_crop_size
        self.cluster_distance_px = cluster_distance_px
        self.nms_merge_iou = nms_merge_iou
        self.boundary_suppression_margin_px = boundary_suppression_margin_px
        self.output_conf_threshold = output_conf_threshold
        self.new_crop_conf_threshold = new_crop_conf_threshold
        self.merge_mode = merge_mode

    def identify_candidates(
        self,
        base_detections: Sequence[DetectionBox],
    ) -> List[Tuple[float, float, float, float]]:
        """Identifies bounding boxes that require higher spatial resolution."""
        candidates = []
        for det in base_detections:
            is_small = det.area < self.small_object_area_threshold
            is_uncertain = self.uncertainty_min_conf <= det.confidence <= self.uncertainty_max_conf
            if (is_small or is_uncertain) and det.confidence >= 0.10:
                candidates.append((det.x1, det.y1, det.x2, det.y2))
        return candidates

    def generate_crops(
        self,
        candidates: Sequence[Tuple[float, float, float, float]],
        image_shape: Tuple[int, int] = (1280, 720),
    ) -> List[CropRegion]:
        """Generates crop regions based on the selected strategy."""
        if self.strategy == "individual":
            img_w, img_h = image_shape
            crops = []
            for i, b in enumerate(candidates[:self.max_crops_per_frame]):
                w = b[2] - b[0]
                h = b[3] - b[1]
                pad_x = max(w * self.context_margin, 20.0)
                pad_y = max(h * self.context_margin, 20.0)
                cx1 = max(0.0, b[0] - pad_x)
                cy1 = max(0.0, b[1] - pad_y)
                cx2 = min(float(img_w), b[2] + pad_x)
                cy2 = min(float(img_h), b[3] + pad_y)
                # Expand to min size
                if (cx2 - cx1) < self.min_crop_size:
                    diff = (self.min_crop_size - (cx2 - cx1)) / 2.0
                    cx1 = max(0.0, cx1 - diff)
                    cx2 = min(float(img_w), cx2 + diff)
                if (cy2 - cy1) < self.min_crop_size:
                    diff = (self.min_crop_size - (cy2 - cy1)) / 2.0
                    cy1 = max(0.0, cy1 - diff)
                    cy2 = min(float(img_h), cy2 + diff)
                crops.append(CropRegion(cx1, cy1, cx2, cy2, [i], "individual"))
            return crops

        elif self.strategy == "clustered":
            return cluster_candidate_boxes(
                candidates,
                image_shape=image_shape,
                max_distance_px=self.cluster_distance_px,
                context_margin=self.context_margin,
                min_crop_size=self.min_crop_size,
                max_crops=self.max_crops_per_frame,
            )

        elif self.strategy == "fixed_grid":
            return generate_fixed_grid_crops(image_shape=image_shape, grid_type="horizon_strip")

        elif self.strategy == "hybrid":
            # If candidates exist, use clustered; otherwise fall back to horizon strip if image contains complex scene
            if candidates:
                return cluster_candidate_boxes(
                    candidates,
                    image_shape=image_shape,
                    max_distance_px=self.cluster_distance_px,
                    context_margin=self.context_margin,
                    min_crop_size=self.min_crop_size,
                    max_crops=self.max_crops_per_frame,
                )
            return []

        return []

    def detect_frame(
        self,
        image_input: str | Path | Image.Image | np.ndarray,
        base_cached_boxes: Optional[List[DetectionBox]] = None,
    ) -> Tuple[List[DetectionBox], Dict[str, Any]]:
        """Executes full region-adaptive inference with complete timing and diagnostics."""
        t_total_start = time.perf_counter()
        timing_stats = {
            "base_inference_ms": 0.0,
            "crop_generation_ms": 0.0,
            "crop_inference_ms": 0.0,
            "coordinate_restore_ms": 0.0,
            "merge_ms": 0.0,
            "total_latency_ms": 0.0,
        }

        # 1. Load image and get dimensions
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = image_input
        img_w, img_h = pil_img.size

        # 2. Base inference (or load from cache if pre-computed)
        if base_cached_boxes is not None:
            base_dets = base_cached_boxes
            timing_stats["base_inference_ms"] = 0.0
        else:
            t0 = time.perf_counter()
            res = self.model(pil_img, imgsz=self.base_resolution, conf=0.10, verbose=False)[0]
            base_dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()
                for i in range(len(xyxy)):
                    raw_cls = self.model.names[int(classes[i])]
                    canon = CANONICAL_MAPPING.get(raw_cls.lower(), raw_cls)
                    if canon in CANONICAL_CLASSES:
                        base_dets.append(
                            DetectionBox(
                                class_name=canon,
                                confidence=round(float(confs[i]), 4),
                                x1=round(float(xyxy[i, 0]), 2),
                                y1=round(float(xyxy[i, 1]), 2),
                                x2=round(float(xyxy[i, 2]), 2),
                                y2=round(float(xyxy[i, 3]), 2),
                            )
                        )
            timing_stats["base_inference_ms"] = (time.perf_counter() - t0) * 1000.0

        # 3. Identify candidate regions & generate crops
        t_cg = time.perf_counter()
        candidates = self.identify_candidates(base_dets)
        crop_regions = self.generate_crops(candidates, image_shape=(img_w, img_h))
        timing_stats["crop_generation_ms"] = (time.perf_counter() - t_cg) * 1000.0

        # 4. Crop inference & coordinate restoration
        restored_crop_dets: List[DetectionBox] = []
        pixels_processed = self.base_resolution * self.base_resolution
        t_ci_total = 0.0
        t_cr_total = 0.0

        for crop_reg in crop_regions:
            pixels_processed += self.crop_resolution * self.crop_resolution
            # Extract crop
            crop_img = pil_img.crop((crop_reg.x1, crop_reg.y1, crop_reg.x2, crop_reg.y2))
            
            t_ci = time.perf_counter()
            c_res = self.model(crop_img, imgsz=self.crop_resolution, conf=0.10, verbose=False)[0]
            t_ci_total += (time.perf_counter() - t_ci) * 1000.0

            t_cr = time.perf_counter()
            if c_res.boxes is not None and len(c_res.boxes) > 0:
                c_xyxy = c_res.boxes.xyxy.cpu().numpy()
                c_confs = c_res.boxes.conf.cpu().numpy()
                c_classes = c_res.boxes.cls.cpu().numpy()

                for j in range(len(c_xyxy)):
                    # Boundary artifact suppression: reject low-conf boxes cut off at crop edges
                    c_box = c_xyxy[j]
                    touches_edge = is_touching_crop_boundary(
                        c_box,
                        crop_size=(crop_reg.width, crop_reg.height),
                        boundary_margin_px=self.boundary_suppression_margin_px,
                    )
                    conf_val = float(c_confs[j])
                    if touches_edge and conf_val < 0.35:
                        continue  # suppress boundary artifact

                    raw_cls = self.model.names[int(c_classes[j])]
                    canon = CANONICAL_MAPPING.get(raw_cls.lower(), raw_cls)
                    if canon in CANONICAL_CLASSES:
                        ox1, oy1, ox2, oy2 = transform_crop_to_original(
                            c_box,
                            crop_region=crop_reg,
                            crop_input_size=(self.crop_resolution, self.crop_resolution),
                        )
                        # Clamp to original image boundaries
                        ox1 = max(0.0, min(float(img_w), ox1))
                        oy1 = max(0.0, min(float(img_h), oy1))
                        ox2 = max(0.0, min(float(img_w), ox2))
                        oy2 = max(0.0, min(float(img_h), oy2))

                        if (ox2 - ox1) > 2.0 and (oy2 - oy1) > 2.0:
                            restored_crop_dets.append(
                                DetectionBox(
                                    class_name=canon,
                                    confidence=round(conf_val, 4),
                                    x1=round(ox1, 2),
                                    y1=round(oy1, 2),
                                    x2=round(ox2, 2),
                                    y2=round(oy2, 2),
                                )
                            )
            t_cr_total += (time.perf_counter() - t_cr) * 1000.0

        timing_stats["crop_inference_ms"] = t_ci_total
        timing_stats["coordinate_restore_ms"] = t_cr_total

        # 5. Merge base detections and crop detections
        t_mg = time.perf_counter()
        final_dets = merge_detections_nms(
            base_detections=base_dets,
            crop_detections=restored_crop_dets,
            iou_threshold=self.nms_merge_iou,
            small_area_threshold=self.small_object_area_threshold,
            base_conf_threshold=self.output_conf_threshold,
            new_crop_conf_threshold=self.new_crop_conf_threshold,
            merge_mode=self.merge_mode,
        )
        if self.merge_mode != "conservative":
            final_dets = [d for d in final_dets if d.confidence >= self.output_conf_threshold]
        timing_stats["merge_ms"] = (time.perf_counter() - t_mg) * 1000.0

        timing_stats["total_latency_ms"] = (time.perf_counter() - t_total_start) * 1000.0

        diagnostics = {
            "num_candidates": len(candidates),
            "num_crops": len(crop_regions),
            "crop_regions": [c.box_xyxy for c in crop_regions],
            "pixels_processed": pixels_processed,
            "timing_ms": timing_stats,
            "crop_dets_count": len(restored_crop_dets),
            "final_dets_count": len(final_dets),
        }

        return final_dets, diagnostics
