"""Video Dataset abstractions and temporal evaluation metrics for WeatherGuard Mission 9.

Supports loading frame sequences, reading ground truth tracking annotations,
running frame-level and sequence-level object detection evaluation, and computing
temporal stability metrics (flicker rate, track coverage, fragmentation, dropout gaps).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from PIL import Image

CANONICAL_CLASSES = ("Car", "Person", "Truck", "Bus", "Bicycle", "Motorcycle")

CANONICAL_MAPPING = {
    "car": "Car",
    "pedestrian": "Person",
    "rider": "Person",
    "other person": "Person",
    "truck": "Truck",
    "bus": "Bus",
    "bicycle": "Bicycle",
    "motorcycle": "Motorcycle",
}


@dataclass
class GroundTruthBox:
    track_id: str
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    occluded: bool = False
    truncated: bool = False

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def scale(self) -> str:
        a = self.area
        if a < 1024:
            return "small"
        elif a <= 9216:
            return "medium"
        return "large"


@dataclass
class DetectionBox:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[str] = None

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def scale(self) -> str:
        a = self.area
        if a < 1024:
            return "small"
        elif a <= 9216:
            return "medium"
        return "large"


@dataclass
class VideoFrame:
    frame_index: int
    image_path: Path
    timestamp_seconds: float
    gt_boxes: List[GroundTruthBox] = field(default_factory=list)

    def load_image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")


@dataclass
class VideoSequence:
    sequence_id: str
    condition: str
    timeofday: str
    scene: str
    fps: float
    frames: List[VideoFrame] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_seconds(self) -> float:
        return len(self.frames) / self.fps if self.fps > 0 else 0.0

    def get_all_track_ids(self) -> Set[Tuple[str, str]]:
        tracks = set()
        for f in self.frames:
            for b in f.gt_boxes:
                tracks.add((b.class_name, b.track_id))
        return tracks


class VideoDataset:
    def __init__(self, sequences: Dict[str, VideoSequence]):
        self.sequences = sequences

    @classmethod
    def load(
        cls,
        base_dir: str | Path,
        sequence_ids: Optional[List[str]] = None,
        fps: float = 5.0,
    ) -> "VideoDataset":
        base_path = Path(base_dir)
        seq_dir = base_path / "sequences"
        labels_dir = base_path / "labels"

        available_labels = sorted(list(labels_dir.glob("*.json")))
        sequences = {}

        for lpath in available_labels:
            sid = lpath.stem
            if sequence_ids is not None and sid not in sequence_ids:
                continue

            with open(lpath, "r") as f:
                raw_data = json.load(f)

            img_folder = seq_dir / sid
            frame_files = sorted(list(img_folder.glob("*.jpg")))
            assert len(frame_files) == len(raw_data), f"Mismatch in frames for {sid}"

            # Simple heuristic for weather/lighting if not supplied externally
            frames: List[VideoFrame] = []
            for idx, (img_path, f_data) in enumerate(zip(frame_files, raw_data)):
                gt_boxes: List[GroundTruthBox] = []
                for obj in f_data.get("labels", []):
                    cat = obj.get("category", "").lower()
                    canon = CANONICAL_MAPPING.get(cat)
                    if not canon:
                        continue
                    b = obj.get("box2d", {})
                    x1, y1, x2, y2 = b.get("x1", 0.0), b.get("y1", 0.0), b.get("x2", 0.0), b.get("y2", 0.0)
                    if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                        continue
                    attr = obj.get("attributes", {})
                    gt_boxes.append(
                        GroundTruthBox(
                            track_id=str(obj.get("id", f"tid_{idx}")),
                            class_name=canon,
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            occluded=bool(attr.get("occluded", False)),
                            truncated=bool(attr.get("truncated", False)),
                        )
                    )

                frames.append(
                    VideoFrame(
                        frame_index=idx,
                        image_path=img_path,
                        timestamp_seconds=idx / fps,
                        gt_boxes=gt_boxes,
                    )
                )

            sequences[sid] = VideoSequence(
                sequence_id=sid,
                condition="adverse",
                timeofday="mixed",
                scene="driving",
                fps=fps,
                frames=frames,
            )

        return cls(sequences)


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_video_sequence(
    sequence: VideoSequence,
    predictions_per_frame: List[List[DetectionBox]],
    iou_threshold: float = 0.5,
) -> dict:
    """Evaluates frame-level detection accuracy and temporal stability metrics across a video sequence."""
    assert len(predictions_per_frame) == len(sequence.frames), "Predictions length must match sequence frames"

    total_tp = 0
    total_fp = 0
    total_fn = 0

    class_tp = Counter()
    class_fp = Counter()
    class_fn = Counter()

    scale_tp = Counter()
    scale_fp = Counter()
    scale_fn = Counter()

    # Track-level temporal dynamics
    # Map: (class_name, track_id) -> list of bool (whether detected in each frame the track exists)
    track_detection_timeline: Dict[Tuple[str, str], List[bool]] = {}
    track_scale: Dict[Tuple[str, str], str] = {}

    for f_idx, (frame, preds) in enumerate(zip(sequence.frames, predictions_per_frame)):
        gts = frame.gt_boxes
        matched_gt_indices: Set[int] = set()

        # Sort predictions by confidence descending
        sorted_preds = sorted(preds, key=lambda p: p.confidence, reverse=True)

        for pred in sorted_preds:
            best_iou = 0.0
            best_gt_idx = -1

            p_box = (pred.x1, pred.y1, pred.x2, pred.y2)

            for g_idx, gt in enumerate(gts):
                if g_idx in matched_gt_indices:
                    continue
                if gt.class_name != pred.class_name:
                    continue

                g_box = (gt.x1, gt.y1, gt.x2, gt.y2)
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)
                total_tp += 1
                class_tp[pred.class_name] += 1
                scale_tp[gts[best_gt_idx].scale] += 1
            else:
                total_fp += 1
                class_fp[pred.class_name] += 1
                scale_fp[pred.scale] += 1

        # False negatives are unmatched ground truth
        for g_idx, gt in enumerate(gts):
            is_detected = g_idx in matched_gt_indices
            if not is_detected:
                total_fn += 1
                class_fn[gt.class_name] += 1
                scale_fn[gt.scale] += 1

            track_key = (gt.class_name, gt.track_id)
            if track_key not in track_detection_timeline:
                track_detection_timeline[track_key] = []
                track_scale[track_key] = gt.scale
            track_detection_timeline[track_key].append(is_detected)

    # Compute Standard Metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute Temporal Stability Metrics
    # 1. Track Coverage: fraction of active frames where track was detected
    coverages = [sum(timeline) / len(timeline) for timeline in track_detection_timeline.values() if len(timeline) > 0]
    mean_coverage = float(np.mean(coverages)) if coverages else 0.0

    # 2. Flicker / Fragmentation Rate: state transitions (detected <-> missed) divided by lifetime
    flicker_events = 0
    total_lifetime_steps = 0
    dropout_gaps = []

    for timeline in track_detection_timeline.values():
        if len(timeline) < 2:
            continue
        total_lifetime_steps += len(timeline) - 1
        current_gap = 0
        for i in range(1, len(timeline)):
            if timeline[i] != timeline[i - 1]:
                flicker_events += 1

            # Count consecutive missed frames
            if not timeline[i]:
                current_gap += 1
            else:
                if current_gap > 0:
                    dropout_gaps.append(current_gap)
                    current_gap = 0
        if current_gap > 0:
            dropout_gaps.append(current_gap)

    flicker_rate = flicker_events / total_lifetime_steps if total_lifetime_steps > 0 else 0.0
    mean_dropout_gap = float(np.mean(dropout_gaps)) if dropout_gaps else 0.0

    # Scale-specific metrics
    scale_metrics = {}
    for sc in ("small", "medium", "large"):
        tp = scale_tp[sc]
        fp = scale_fp[sc]
        fn = scale_fn[sc]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sc_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        scale_metrics[sc] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(sc_f1, 4), "tp": tp, "fp": fp, "fn": fn}

    # Class-specific metrics
    class_metrics = {}
    for cl in CANONICAL_CLASSES:
        tp = class_tp[cl]
        fp = class_fp[cl]
        fn = class_fn[cl]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cl_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        class_metrics[cl] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(cl_f1, 4), "tp": tp, "fp": fp, "fn": fn}

    return {
        "sequence_id": sequence.sequence_id,
        "frames_evaluated": len(sequence.frames),
        "total_gt": total_tp + total_fn,
        "total_preds": total_tp + total_fp,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "temporal_stability": {
            "mean_track_coverage": round(mean_coverage, 4),
            "flicker_rate": round(flicker_rate, 4),
            "total_flicker_events": flicker_events,
            "mean_dropout_gap_frames": round(mean_dropout_gap, 2),
            "total_dropout_events": len(dropout_gaps),
            "dropout_gap_histogram": dict(Counter(dropout_gaps)),
        },
        "scale_breakdown": scale_metrics,
        "class_breakdown": class_metrics,
    }
