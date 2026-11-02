"""Temporal Confidence Smoothing, Track-Level Fusion, and Detection Recovery for Mission 9.

Smooths frame-by-frame confidence oscillations along tracked trajectories using an
Exponential Moving Average (EMA) and recovers true positives during momentary dropout gaps.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.video_dataset import DetectionBox


class TemporalConfidenceFilter:
    """Applies temporal confidence smoothing and optional detection recovery along tracks."""

    def __init__(
        self,
        alpha: float = 0.60,  # EMA weight for current observation (higher = more reactive, lower = smoother)
        recovery_max_gap: int = 2,  # Maximum consecutive missed frames to recover via trajectory extrapolation
        decay_factor: float = 0.80,  # Confidence decay multiplier for unobserved recovered boxes
        output_threshold: float = 0.20,
    ):
        self.alpha = alpha
        self.recovery_max_gap = recovery_max_gap
        self.decay_factor = decay_factor
        self.output_threshold = output_threshold

        # Track state dictionary: track_id -> dict
        # { 'smoothed_conf': float, 'last_box': DetectionBox, 'missed_count': int }
        self.track_states: Dict[str, dict] = {}

    def reset(self) -> None:
        self.track_states.clear()

    def process_frame(
        self,
        detections: List[DetectionBox],
        enable_recovery: bool = True,
    ) -> Tuple[List[DetectionBox], Dict[str, any]]:
        """Processes detections for a single frame.

        detections: list of DetectionBox with track_id set by the tracker.
        Returns:
            smoothed_boxes: list of DetectionBox with smoothed / recovered confidences
            diagnostics: count of recovered boxes, smoothed count, dropped boxes
        """
        observed_track_ids = set()
        output_boxes: List[DetectionBox] = []
        recovered_count = 0
        smoothed_count = 0

        # 1. Process observed detections
        for det in detections:
            tid = det.track_id
            if tid is None:
                # Untracked detection, pass through directly
                if det.confidence >= self.output_threshold:
                    output_boxes.append(det)
                continue

            observed_track_ids.add(tid)
            if tid in self.track_states:
                # Update existing track state with EMA
                prior_conf = self.track_states[tid]["smoothed_conf"]
                smoothed_conf = self.alpha * det.confidence + (1.0 - self.alpha) * prior_conf
                self.track_states[tid]["smoothed_conf"] = smoothed_conf
                self.track_states[tid]["last_box"] = det
                self.track_states[tid]["missed_count"] = 0
                smoothed_count += 1
            else:
                # Initialize new track state
                smoothed_conf = det.confidence
                self.track_states[tid] = {
                    "smoothed_conf": smoothed_conf,
                    "last_box": det,
                    "missed_count": 0,
                    "class_name": det.class_name,
                }

            if smoothed_conf >= self.output_threshold:
                output_boxes.append(
                    DetectionBox(
                        class_name=det.class_name,
                        confidence=float(smoothed_conf),
                        x1=det.x1,
                        y1=det.y1,
                        x2=det.x2,
                        y2=det.y2,
                        track_id=det.track_id,
                    )
                )

        # 2. Recovery: check tracks that were not observed in this frame
        if enable_recovery:
            active_tids = list(self.track_states.keys())
            for tid in active_tids:
                if tid not in observed_track_ids:
                    state = self.track_states[tid]
                    state["missed_count"] += 1
                    if state["missed_count"] <= self.recovery_max_gap:
                        # Extrapolate prior box with decayed confidence
                        decayed_conf = state["smoothed_conf"] * self.decay_factor
                        state["smoothed_conf"] = decayed_conf
                        last_b = state["last_box"]

                        if decayed_conf >= self.output_threshold:
                            output_boxes.append(
                                DetectionBox(
                                    class_name=state["class_name"],
                                    confidence=float(decayed_conf),
                                    x1=last_b.x1,
                                    y1=last_b.y1,
                                    x2=last_b.x2,
                                    y2=last_b.y2,
                                    track_id=tid,
                                )
                            )
                            recovered_count += 1
                    else:
                        # Expired track, remove from state
                        del self.track_states[tid]

        diagnostics = {
            "observed_boxes": len(detections),
            "output_boxes": len(output_boxes),
            "smoothed_count": smoothed_count,
            "recovered_count": recovered_count,
            "active_tracks_tracked": len(self.track_states),
        }

        return output_boxes, diagnostics

    def process_frame_separate(
        self,
        detections: List[DetectionBox],
        enable_recovery: bool = True,
    ) -> Tuple[List[DetectionBox], List[DetectionBox], Dict[str, any]]:
        """Processes detections returning (observed_smoothed_boxes, extrapolated_only_boxes, diagnostics)."""
        observed_track_ids = set()
        observed_smoothed_boxes: List[DetectionBox] = []
        extrapolated_boxes: List[DetectionBox] = []
        recovered_count = 0
        smoothed_count = 0

        # 1. Process observed detections
        for det in detections:
            tid = det.track_id
            if tid is None:
                if det.confidence >= self.output_threshold:
                    observed_smoothed_boxes.append(det)
                continue

            observed_track_ids.add(tid)
            if tid in self.track_states:
                prior_conf = self.track_states[tid]["smoothed_conf"]
                smoothed_conf = self.alpha * det.confidence + (1.0 - self.alpha) * prior_conf
                self.track_states[tid]["smoothed_conf"] = smoothed_conf
                self.track_states[tid]["last_box"] = det
                self.track_states[tid]["missed_count"] = 0
                smoothed_count += 1
            else:
                smoothed_conf = det.confidence
                self.track_states[tid] = {
                    "smoothed_conf": smoothed_conf,
                    "last_box": det,
                    "missed_count": 0,
                    "class_name": det.class_name,
                }

            if smoothed_conf >= self.output_threshold:
                observed_smoothed_boxes.append(
                    DetectionBox(
                        class_name=det.class_name,
                        confidence=float(smoothed_conf),
                        x1=det.x1,
                        y1=det.y1,
                        x2=det.x2,
                        y2=det.y2,
                        track_id=det.track_id,
                    )
                )

        # 2. Recovery: unobserved tracks
        if enable_recovery:
            active_tids = list(self.track_states.keys())
            for tid in active_tids:
                if tid not in observed_track_ids:
                    state = self.track_states[tid]
                    state["missed_count"] += 1
                    if state["missed_count"] <= self.recovery_max_gap:
                        decayed_conf = state["smoothed_conf"] * self.decay_factor
                        state["smoothed_conf"] = decayed_conf
                        last_b = state["last_box"]

                        if decayed_conf >= self.output_threshold:
                            extrapolated_boxes.append(
                                DetectionBox(
                                    class_name=state["class_name"],
                                    confidence=float(decayed_conf),
                                    x1=last_b.x1,
                                    y1=last_b.y1,
                                    x2=last_b.x2,
                                    y2=last_b.y2,
                                    track_id=tid,
                                )
                            )
                            recovered_count += 1
                    else:
                        del self.track_states[tid]

        diagnostics = {
            "observed_boxes": len(detections),
            "output_boxes": len(observed_smoothed_boxes) + len(extrapolated_boxes),
            "smoothed_count": smoothed_count,
            "recovered_count": recovered_count,
            "active_tracks_tracked": len(self.track_states),
        }

        return observed_smoothed_boxes, extrapolated_boxes, diagnostics
