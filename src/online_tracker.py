"""Lightweight Online Multi-Object Tracker (Adverse-Weather ByteTrack & Kalman Filter).

Implements two-stage association to recover low-confidence detections degraded by
rain streaks, snow spray, and nighttime glare without propagating false alarms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.video_dataset import DetectionBox, compute_iou


class KalmanFilterBBox:
    """A simple constant-velocity Kalman Filter for 2D bounding boxes.

    State: [xc, yc, s, r, v_xc, v_yc, v_s] where:
      xc, yc = box center
      s = area (w * h)
      r = aspect ratio (w / h)
      v_* = velocities
    """

    def __init__(self, bbox: Tuple[float, float, float, float]):
        # bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        xc = x1 + w / 2.0
        yc = y1 + h / 2.0
        s = w * h
        r = w / h

        # State vector (7x1)
        self.x = np.array([xc, yc, s, r, 0.0, 0.0, 0.0], dtype=np.float64)

        # State transition matrix F (7x7)
        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        # Measurement matrix H (4x7)
        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # State covariance P (7x7)
        self.P = np.diag([10.0, 10.0, 100.0, 1.0, 100.0, 100.0, 100.0])

        # Process noise Q (7x7)
        self.Q = np.diag([1.0, 1.0, 10.0, 0.01, 10.0, 10.0, 10.0])

        # Measurement noise R (4x4)
        self.R = np.diag([5.0, 5.0, 50.0, 0.1])

    def predict(self) -> Tuple[float, float, float, float]:
        """Projects state and error covariance forward by one time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_bbox()

    def update(self, bbox: Tuple[float, float, float, float]) -> None:
        """Corrects state with observed measurement."""
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        xc = x1 + w / 2.0
        yc = y1 + h / 2.0
        s = w * h
        r = w / h
        z = np.array([xc, yc, s, r], dtype=np.float64)

        # Innovation
        y = z - (self.H @ self.x)
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + (K @ y)
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(7) - (K @ self.H)
        self.P = (I_KH @ self.P @ I_KH.T) + (K @ self.R @ K.T)

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Converts internal state [xc, yc, s, r] back to [x1, y1, x2, y2]."""
        xc, yc, s, r = self.x[0], self.x[1], max(1.0, self.x[2]), max(0.01, self.x[3])
        w = np.sqrt(s * r)
        h = s / max(1.0, w)
        x1 = xc - w / 2.0
        y1 = yc - h / 2.0
        x2 = xc + w / 2.0
        y2 = yc + h / 2.0
        return (float(x1), float(y1), float(x2), float(y2))


@dataclass
class Track:
    track_id: int
    class_name: str
    kf: KalmanFilterBBox
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    state: str = "tentative"  # 'tentative', 'confirmed', 'deleted'
    confidence: float = 0.50
    confidence_history: List[float] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def predict(self) -> Tuple[float, float, float, float]:
        self.bbox = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, detection: DetectionBox) -> None:
        det_bbox = (detection.x1, detection.y1, detection.x2, detection.y2)
        self.kf.update(det_bbox)
        self.bbox = self.kf.get_bbox()
        self.hits += 1
        self.time_since_update = 0
        self.confidence = detection.confidence
        self.confidence_history.append(detection.confidence)
        if self.state == "tentative" and self.hits >= 2:
            self.state = "confirmed"


class AdverseWeatherByteTracker:
    """ByteTrack online tracker adapted for adverse weather multi-object tracking.

    Implements:
    - High-threshold association (primary matching)
    - Low-threshold association (recovering weather-degraded detections)
    - Kalman motion smoothing
    - Birth confirmation to suppress transient false alarms (water spray, reflections)
    """

    def __init__(
        self,
        high_thresh: float = 0.40,
        low_thresh: float = 0.15,
        match_thresh: float = 0.50,
        low_match_thresh: float = 0.40,
        confirm_frames: int = 2,
        max_age: int = 6,  # 1.2s at 5 fps
        decay_factor: float = 0.85,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.low_match_thresh = low_match_thresh
        self.confirm_frames = confirm_frames
        self.max_age = max_age
        self.decay_factor = decay_factor

        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, detections: List[DetectionBox]) -> List[DetectionBox]:
        """Processes detections from the current frame and returns tracked detection boxes."""
        # 1. Predict forward all existing tracks
        for t in self.tracks:
            t.predict()

        # Separate detections into high and low confidence pools
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        confirmed_tracks = [t for t in self.tracks if t.state == "confirmed"]
        tentative_tracks = [t for t in self.tracks if t.state == "tentative"]

        # 2. First association: match confirmed tracks with high-confidence detections
        matched_tracks_1, unmatched_tracks_1, unmatched_dets_high = self._match(
            confirmed_tracks, high_dets, self.match_thresh
        )

        for t, d in matched_tracks_1:
            t.update(d)

        # 3. Second association: match remaining confirmed tracks with low-confidence detections
        matched_tracks_2, unmatched_tracks_2, _ = self._match(
            unmatched_tracks_1, low_dets, self.low_match_thresh
        )

        for t, d in matched_tracks_2:
            t.update(d)

        # 4. Third association: match tentative tracks with remaining unmatched high-confidence detections
        matched_tentative, unmatched_tentative, remaining_high_dets = self._match(
            tentative_tracks, unmatched_dets_high, self.match_thresh
        )

        for t, d in matched_tentative:
            t.update(d)

        # Initialize new tentative tracks from remaining unmatched high detections
        for d in remaining_high_dets:
            bbox = (d.x1, d.y1, d.x2, d.y2)
            kf = KalmanFilterBBox(bbox)
            new_track = Track(
                track_id=self.next_id,
                class_name=d.class_name,
                kf=kf,
                confidence=d.confidence,
                confidence_history=[d.confidence],
                bbox=bbox,
                state="tentative" if self.confirm_frames > 1 else "confirmed",
            )
            self.next_id += 1
            self.tracks.append(new_track)

        # Apply confidence decay to unupdated confirmed tracks
        for t in unmatched_tracks_2:
            t.confidence *= self.decay_factor

        # Mark dead tracks
        active_tracks: List[Track] = []
        for t in self.tracks:
            if t.state == "tentative" and t.time_since_update > 1:
                t.state = "deleted"
            elif t.time_since_update > self.max_age:
                t.state = "deleted"
            elif t.confidence < 0.05:
                t.state = "deleted"

            if t.state != "deleted":
                active_tracks.append(t)

        self.tracks = active_tracks

        # Return confirmed tracks as DetectionBoxes
        output_boxes: List[DetectionBox] = []
        for t in self.tracks:
            if t.state == "confirmed" and t.time_since_update <= 1:
                x1, y1, x2, y2 = t.bbox
                output_boxes.append(
                    DetectionBox(
                        class_name=t.class_name,
                        confidence=float(t.confidence),
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        track_id=str(t.track_id),
                    )
                )

        return output_boxes

    def _match(
        self, tracks: List[Track], detections: List[DetectionBox], iou_thresh: float
    ) -> Tuple[List[Tuple[Track, DetectionBox]], List[Track], List[DetectionBox]]:
        """Greedy IoU association partitioned by class."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], tracks, detections

        matched: List[Tuple[Track, DetectionBox]] = []
        matched_track_indices: Set[int] = set()
        matched_det_indices: Set[int] = set()

        # Build IoU cost matrix
        # Compute IoU between all tracks and detections of matching class
        pairs = []
        for t_idx, t in enumerate(tracks):
            t_box = t.bbox
            for d_idx, d in enumerate(detections):
                if t.class_name != d.class_name:
                    continue
                d_box = (d.x1, d.y1, d.x2, d.y2)
                iou = compute_iou(t_box, d_box)
                if iou >= iou_thresh:
                    pairs.append((iou, t_idx, d_idx))

        # Sort descending by IoU
        pairs.sort(key=lambda x: x[0], reverse=True)

        for iou, t_idx, d_idx in pairs:
            if t_idx in matched_track_indices or d_idx in matched_det_indices:
                continue
            matched_track_indices.add(t_idx)
            matched_det_indices.add(d_idx)
            matched.append((tracks[t_idx], detections[d_idx]))

        unmatched_tracks = [tracks[i] for i in range(len(tracks)) if i not in matched_track_indices]
        unmatched_dets = [detections[i] for i in range(len(detections)) if i not in matched_det_indices]

        return matched, unmatched_tracks, unmatched_dets
