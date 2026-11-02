"""
WeatherGuard Feature Reuse Module (Mission 8)
Implements test-time multi-scale intermediate feature reuse pathways:
1. Neck-feature upsampling (bypassing backbone + neck)
2. Backbone-feature upsampling (bypassing backbone)
3. Coordinate transformation and detection decoding
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple, Sequence, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils import ops

from src.dataset import COCO_TO_DAWN

class FeatureReuseDetector:
    """Manages secondary detection inference by reusing intermediate feature maps."""

    def __init__(
        self,
        model: YOLO,
        target_resolution: int = 832,
        base_resolution: int = 640,
        score_threshold: float = 0.20,
        iou_threshold: float = 0.50,
    ):
        self.model = model
        self.nn_model = model.model.eval()
        self.target_res = target_resolution
        self.base_res = base_resolution
        self.score_thresh = score_threshold
        self.iou_thresh = iou_threshold

        # Compute target spatial dimensions for P3 (stride 8), P4 (stride 16), P5 (stride 32)
        self.target_shapes = {
            "p3": (target_resolution // 8, target_resolution // 8),     # e.g. 104x104
            "p4": (target_resolution // 16, target_resolution // 16),   # e.g. 52x52
            "p5": (target_resolution // 32, target_resolution // 32),   # e.g. 26x26
        }

        self.head = self.nn_model.model[22]

    def decode_and_scale_predictions(
        self,
        raw_preds: torch.Tensor,
        orig_shape: Tuple[int, int],
        eval_res: int,
    ) -> List[Dict[str, Any]]:
        """
        Runs non-max suppression and scales coordinates back to original image dimensions.
        raw_preds: output of Detect head, shape (B, 4 + nc, num_anchors)
        orig_shape: (height, width) of original PIL image
        eval_res: resolution associated with anchors (e.g. 832 or 640)
        """
        # NMS produces list of tensors of shape (N, 6): [x1, y1, x2, y2, conf, cls]
        nms_list = non_max_suppression(
            raw_preds,
            conf_thres=self.score_thresh,
            iou_thres=self.iou_thresh,
        )
        if not nms_list or nms_list[0].shape[0] == 0:
            return []

        boxes_tensor = nms_list[0]
        # Scale boxes from letterbox eval_res space to original image space
        scaled_boxes = ops.scale_boxes(
            (eval_res, eval_res),
            boxes_tensor[:, :4],
            orig_shape,
        ).cpu().numpy()

        scores = boxes_tensor[:, 4].cpu().numpy()
        classes = boxes_tensor[:, 5].cpu().numpy().astype(int)

        class_names = self.model.names
        dets = []
        for i in range(len(scaled_boxes)):
            cls_id = classes[i]
            coco_name = class_names.get(cls_id, str(cls_id))
            common_lbl = COCO_TO_DAWN.get(coco_name)
            if not common_lbl:
                continue  # ignore non-traffic COCO classes

            b = scaled_boxes[i]
            dets.append({
                "common_label": common_lbl,
                "score": float(scores[i]),
                "box_xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "coco_class": coco_name,
            })
        return dets

    def predict_from_neck_features(
        self,
        neck_features: Dict[int, torch.Tensor],
        orig_shape: Tuple[int, int],
        mode: str = "bilinear",
        refinement_module: Optional[nn.Module] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Takes captured neck features [15, 18, 21], upsamples them, optionally applies refinement,
        and executes the Detect head.
        Returns (detections, latency_ms).
        """
        t0 = time.perf_counter()

        # Extract features
        f15 = neck_features[15]  # P3: (1, 64, 80, 80)
        f18 = neck_features[18]  # P4: (1, 128, 40, 40)
        f21 = neck_features[21]  # P5: (1, 256, 20, 20)

        # Upsample to target resolution
        interp_kwargs = {"align_corners": False} if mode == "bilinear" else {}
        f15_up = nn.functional.interpolate(f15, size=self.target_shapes["p3"], mode=mode, **interp_kwargs)
        f18_up = nn.functional.interpolate(f18, size=self.target_shapes["p4"], mode=mode, **interp_kwargs)
        f21_up = nn.functional.interpolate(f21, size=self.target_shapes["p5"], mode=mode, **interp_kwargs)

        if refinement_module is not None:
            f15_up, f18_up, f21_up = refinement_module(f15_up, f18_up, f21_up)

        # Forward pass through Detect head
        with torch.no_grad():
            raw_out = self.head([f15_up, f18_up, f21_up])

        raw_preds = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
        dets = self.decode_and_scale_predictions(raw_preds, orig_shape, eval_res=self.target_res)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return dets, latency_ms

    def predict_from_backbone_features(
        self,
        backbone_features: Dict[int, torch.Tensor],
        orig_shape: Tuple[int, int],
        mode: str = "bilinear",
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Takes captured backbone features [4, 6, 9], upsamples them, runs through the neck (10-21)
        and executes the Detect head (22).
        Returns (detections, latency_ms).
        """
        t0 = time.perf_counter()

        interp_kwargs = {"align_corners": False} if mode == "bilinear" else {}
        f4_up = nn.functional.interpolate(backbone_features[4], size=self.target_shapes["p3"], mode=mode, **interp_kwargs)
        f6_up = nn.functional.interpolate(backbone_features[6], size=self.target_shapes["p4"], mode=mode, **interp_kwargs)
        f9_up = nn.functional.interpolate(backbone_features[9], size=self.target_shapes["p5"], mode=mode, **interp_kwargs)

        # Execute Neck (layers 10 to 21)
        # Note from YOLOv8 graph:
        # Layer 10: Upsample from Layer 9
        # Layer 11: Concat (Layer 10, Layer 6)
        # Layer 12: C2f
        # Layer 13: Upsample from Layer 12
        # Layer 14: Concat (Layer 13, Layer 4)
        # Layer 15: C2f (Neck P3)
        # Layer 16: Conv stride 2
        # Layer 17: Concat (Layer 16, Layer 12)
        # Layer 18: C2f (Neck P4)
        # Layer 19: Conv stride 2
        # Layer 20: Concat (Layer 19, Layer 9)
        # Layer 21: C2f (Neck P5)
        m = self.nn_model.model

        with torch.no_grad():
            x10 = m[10](f9_up)
            x11 = m[11]([x10, f6_up])
            x12 = m[12](x11)
            x13 = m[13](x12)
            x14 = m[14]([x13, f4_up])
            x15 = m[15](x14)
            x16 = m[16](x15)
            x17 = m[17]([x16, x12])
            x18 = m[18](x17)
            x19 = m[19](x18)
            x20 = m[20]([x19, f9_up])
            x21 = m[21](x20)
            raw_out = self.head([x15, x18, x21])

        raw_preds = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
        dets = self.decode_and_scale_predictions(raw_preds, orig_shape, eval_res=self.target_res)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return dets, latency_ms
