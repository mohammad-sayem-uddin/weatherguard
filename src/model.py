"""Pretrained object detector wrapper.

WeatherGuard is built around the principle of *zero training*. We only
load openly licensed pretrained weights and run them on DAWN. The model
wrapper is responsible for:

  * choosing the pretrained backbone and weight URL,
  * exposing a uniform `predict(image: PIL.Image) -> Prediction` API,
  * normalising detector labels into a documented common label space,
  * exposing model metadata for the dashboard and reports.

Supported detectors:
  * `fasterrcnn_resnet50_fpn` — Torchvision, BSD-3-Clause code.
  * `yolov8n`                 — Ultralytics YOLOv8 nano, AGPL-3.0.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

# Map torchvision COCO category id -> category name (80 classes).
COCO_INSTANCE_CATEGORY_NAMES: list[str] = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    label: str  # COCO label name (e.g. "car")
    common_label: str | None  # mapped into DAWN label space when possible

    def to_dict(self) -> dict:
        return {
            "x_min": self.box_xyxy[0],
            "y_min": self.box_xyxy[1],
            "x_max": self.box_xyxy[2],
            "y_max": self.box_xyxy[3],
            "score": self.score,
            "label": self.label,
            "common_label": self.common_label,
        }


@dataclass
class Prediction:
    image_id: str
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    inference_seconds: float = 0.0

    def to_dicts(self) -> list[dict]:
        return [d.to_dict() for d in self.detections]


class Detector:
    """Base detector API."""

    name: str = "base"
    weights_provenance: str = ""
    license_note: str = ""

    def load(self) -> None:
        ...

    def predict(self, image: Image.Image, image_id: str = "") -> Prediction:
        ...

    def warmup(self) -> None:
        """Optional warm-up to amortise first-call costs out of latency."""
        ...


class FasterRCNNResNet50FPN(Detector):
    name = "fasterrcnn_resnet50_fpn"
    weights_provenance = (
        "Torchvision pretrained weights (COCO 2017 train). "
        "Model architecture: Faster R-CNN with ResNet-50 FPN backbone, "
        "shipped via torchvision.models.detection."
    )
    license_note = (
        "Torchvision code: BSD-3-Clause. Pretrained COCO weights: "
        "research-use terms per torchvision model registry. "
        "See https://github.com/pytorch/vision/blob/main/LICENSE for code and "
        "https://pytorch.org/vision/stable/models.html for weight notes."
    )

    def __init__(self, score_threshold: float = 0.0, device: str | None = None):
        self.score_threshold = float(score_threshold)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_seconds: float = 0.0

    def load(self) -> None:
        t0 = time.perf_counter()
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.eval()
        self.model.to(self.device)
        self._load_seconds = time.perf_counter() - t0

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    def _common_label(self, coco_label: str) -> str | None:
        from .dataset import COCO_TO_DAWN
        return COCO_TO_DAWN.get(coco_label)

    @torch.no_grad()
    def predict(self, image: Image.Image, image_id: str = "") -> Prediction:
        if self.model is None:
            self.load()
        # torchvision Faster R-CNN expects a tensor in [0,1] (it normalises).
        from torchvision.transforms.functional import to_tensor

        x = to_tensor(image)
        t0 = time.perf_counter()
        outputs = self.model([x])[0]
        elapsed = time.perf_counter() - t0

        dets: list[Detection] = []
        boxes = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        for box, lab, sc in zip(boxes, labels, scores):
            sc_f = float(sc)
            if sc_f < self.score_threshold:
                continue
            label = (
                COCO_INSTANCE_CATEGORY_NAMES[int(lab)]
                if 0 <= int(lab) < len(COCO_INSTANCE_CATEGORY_NAMES)
                else f"coco_{int(lab)}"
            )
            dets.append(
                Detection(
                    box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    score=sc_f,
                    label=label,
                    common_label=self._common_label(label),
                )
            )
        return Prediction(
            image_id=image_id,
            width=image.width,
            height=image.height,
            detections=dets,
            inference_seconds=elapsed,
        )

    @torch.no_grad()
    def warmup(self) -> None:
        if self.model is None:
            self.load()
        dummy = Image.new("RGB", (320, 240), color=(128, 128, 128))
        self.predict(dummy, image_id="<warmup>")


def build_detector(name: str, **kwargs) -> Detector:
    name = name.lower()
    if name in ("fasterrcnn_resnet50_fpn", "faster_rcnn", "frcnn"):
        return FasterRCNNResNet50FPN(**kwargs)
    if name in ("yolov8n", "yolov8", "yolo_nano"):
        return YOLOv8N(**kwargs)
    raise ValueError(
        f"Unknown detector: {name!r}. Supported: fasterrcnn_resnet50_fpn, yolov8n."
    )


class YOLOv8N(Detector):
    name = "yolov8n"
    weights_provenance = (
        "Ultralytics YOLOv8 nano pretrained weights (COCO 80 classes). "
        "Source: https://github.com/ultralytics/assets — yolov8n.pt."
    )
    license_note = (
        "Ultralytics package: AGPL-3.0. Pretrained weights: AGPL-3.0 per the "
        "Ultralytics model registry. Commercial use of Ultralytics code or "
        "weights requires a separate commercial licence from Ultralytics."
    )

    # YOLOv8 native COCO label order. From the Ultralytics source.
    _YOLO_COCO_NAMES: list[str] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    def __init__(self, score_threshold: float = 0.0, weights_path: str = "yolov8n.pt"):
        self.score_threshold = float(score_threshold)
        self.weights_path = weights_path
        self.device = "cpu"  # YOLOv8n runs on CPU via Ultralytics in this project
        self._model = None
        self._load_seconds: float = 0.0

    def load(self) -> None:
        t0 = time.perf_counter()
        from ultralytics import YOLO
        self._model = YOLO(self.weights_path)
        self._load_seconds = time.perf_counter() - t0

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    def _common_label(self, label: str) -> str | None:
        from .dataset import COCO_TO_DAWN
        return COCO_TO_DAWN.get(label)

    def predict(self, image: Image.Image, image_id: str = "") -> Prediction:
        if self._model is None:
            self.load()
        import numpy as np
        arr = np.asarray(image)
        t0 = time.perf_counter()
        # verbose=False to silence per-image logs. conf=0.0 keeps all scores;
        # thresholding is applied below.
        results = self._model.predict(arr, conf=0.0, verbose=False)
        elapsed = time.perf_counter() - t0
        dets: list[Detection] = []
        if not results:
            return Prediction(image_id=image_id, width=image.width, height=image.height,
                              detections=dets, inference_seconds=elapsed)
        r = results[0]
        names = r.names  # mapping id -> label
        if r.boxes is None:
            return Prediction(image_id=image_id, width=image.width, height=image.height,
                              detections=dets, inference_seconds=elapsed)
        for box in r.boxes:
            try:
                score = float(box.conf.item())
            except Exception:
                score = float(box.conf)
            if score < self.score_threshold:
                continue
            try:
                cls_id = int(box.cls.item())
            except Exception:
                cls_id = int(box.cls)
            label = names.get(cls_id, str(cls_id))
            xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
            dets.append(
                Detection(
                    box_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    score=score,
                    label=label,
                    common_label=self._common_label(label),
                )
            )
        return Prediction(
            image_id=image_id,
            width=image.width,
            height=image.height,
            detections=dets,
            inference_seconds=elapsed,
        )

    def warmup(self) -> None:
        if self._model is None:
            self.load()
        dummy = Image.new("RGB", (320, 240), color=(128, 128, 128))
        self.predict(dummy, image_id="<warmup>")
