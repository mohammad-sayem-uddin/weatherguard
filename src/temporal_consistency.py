"""
WeatherGuard Temporal Detection Consistency Module (Mission 8)
Implements online lightweight IoU association, track-level confidence smoothing,
and dropout recovery to measure detection consistency across video frames.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Sequence, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from src.metrics import iou_xyxy

@dataclass
class Track:
    track_id: int
    class_name: str
    box_xyxy: List[float]
    confidence: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    history_scores: List[float] = field(default_factory=list)

class TemporalConsistencyTracker:
    """
    Lightweight online IoU-association tracker for assessing temporal consistency
    and filtering transient single-frame weather noise.
    """
    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age: int = 2,
        min_hits: int = 2,
        momentum: float = 0.60,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.momentum = momentum
        self.tracks: List[Track] = []
        self.next_id: int = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def update(self, detections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates tracks with detections from current frame.
        detections: list of dicts with 'box_xyxy', 'score', 'common_label'
        Returns list of temporally consistent confirmed detections.
        """
        # Increment time_since_update for existing tracks
        for t in self.tracks:
            t.time_since_update += 1
            t.age += 1

        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        if self.tracks and detections:
            # Greedy IoU matching
            for t_idx, track in enumerate(self.tracks):
                best_iou = 0.0
                best_d_idx = -1
                for d_idx in unmatched_dets:
                    d = detections[d_idx]
                    lbl = d.get("common_label") or d.get("class_name") or d.get("label", "")
                    if lbl != track.class_name:
                        continue
                    iou = iou_xyxy(track.box_xyxy, d["box_xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_d_idx = d_idx

                if best_d_idx >= 0 and best_iou >= self.iou_threshold:
                    d = detections[best_d_idx]
                    # Update track state
                    track.box_xyxy = [float(x) for x in d["box_xyxy"]]
                    track.confidence = self.momentum * track.confidence + (1.0 - self.momentum) * float(d.get("score", 0.5))
                    track.hits += 1
                    track.time_since_update = 0
                    track.history_scores.append(float(d.get("score", 0.5)))
                    matched_tracks.add(t_idx)
                    unmatched_dets.remove(best_d_idx)

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            d = detections[d_idx]
            lbl = d.get("common_label") or d.get("class_name") or d.get("label", "")
            s = float(d.get("score", 0.5))
            new_t = Track(
                track_id=self.next_id,
                class_name=lbl,
                box_xyxy=[float(x) for x in d["box_xyxy"]],
                confidence=s,
                age=1,
                hits=1,
                time_since_update=0,
                history_scores=[s],
            )
            self.next_id += 1
            self.tracks.append(new_t)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Output confirmed detections (tracks with hits >= min_hits or high instantaneous confidence)
        confirmed_dets = []
        for t in self.tracks:
            if t.time_since_update == 0:
                is_confirmed = (t.hits >= self.min_hits) or (t.confidence >= 0.50)
                confirmed_dets.append({
                    "track_id": t.track_id,
                    "common_label": t.class_name,
                    "box_xyxy": t.box_xyxy,
                    "score": t.confidence,
                    "is_confirmed": is_confirmed,
                    "track_hits": t.hits,
                    "track_age": t.age,
                })
        return confirmed_dets
