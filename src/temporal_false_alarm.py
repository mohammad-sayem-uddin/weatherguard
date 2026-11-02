"""Temporal False-Alarm Suppressor for Adverse Weather Video Driving.

Filters ephemeral adverse-weather visual artifacts (rain streaks, wet pavement glare,
water spray plumes, headlight flare) by enforcing temporal track persistence
while rigorously preserving stationary vehicles and stopped pedestrians.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
from src.video_dataset import DetectionBox, compute_iou


class TemporalFalseAlarmSuppressor:
    """Filters spurious, non-persistent detections across time.

    Key principles:
    - Fleeting artifacts (1 frame duration) are suppressed.
    - True objects that persist for >= min_persistence frames are confirmed.
    - Stationary objects (stopped at red lights or parked) are explicitly preserved.
    """

    def __init__(
        self,
        min_persistence_frames: int = 2,
        stationary_iou_thresh: float = 0.85,
        max_speed_pixels_per_frame: float = 80.0,  # ~120 km/h projected bounds
    ):
        self.min_persistence_frames = min_persistence_frames
        self.stationary_iou_thresh = stationary_iou_thresh
        self.max_speed_pixels_per_frame = max_speed_pixels_per_frame

        # Track history: track_id -> dict
        # { 'consecutive_frames': int, 'last_center': (xc, yc), 'is_stationary': bool, 'boxes': list }
        self.track_registry: Dict[str, dict] = {}

    def reset(self) -> None:
        self.track_registry.clear()

    def filter_frame(
        self,
        tracked_boxes: List[DetectionBox],
    ) -> Tuple[List[DetectionBox], Dict[str, any]]:
        """Filters detections in the current frame based on temporal persistence."""
        output_boxes: List[DetectionBox] = []
        suppressed_count = 0
        preserved_stationary_count = 0
        active_track_ids = set()

        for det in tracked_boxes:
            tid = det.track_id
            if tid is None:
                # Detections without tracking IDs cannot be verified temporally; suppress
                suppressed_count += 1
                continue

            active_track_ids.add(tid)
            xc = (det.x1 + det.x2) / 2.0
            yc = (det.y1 + det.y2) / 2.0

            if tid in self.track_registry:
                reg = self.track_registry[tid]
                reg["consecutive_frames"] += 1

                # Check velocity
                prev_xc, prev_yc = reg["last_center"]
                dist = np.hypot(xc - prev_xc, yc - prev_yc) if "np" in globals() else ((xc - prev_xc) ** 2 + (yc - prev_yc) ** 2) ** 0.5
                reg["last_center"] = (xc, yc)

                # Check if stationary (IoU with previous box > threshold)
                prev_box = reg["last_box"]
                iou = compute_iou((det.x1, det.y1, det.x2, det.y2), (prev_box.x1, prev_box.y1, prev_box.x2, prev_box.y2))
                if iou >= self.stationary_iou_thresh:
                    reg["is_stationary"] = True
                    preserved_stationary_count += 1

                reg["last_box"] = det
            else:
                self.track_registry[tid] = {
                    "consecutive_frames": 1,
                    "last_center": (xc, yc),
                    "is_stationary": False,
                    "last_box": det,
                }

            # Enforce persistence criterion
            track_age = self.track_registry[tid]["consecutive_frames"]
            if track_age >= self.min_persistence_frames or self.track_registry[tid]["is_stationary"]:
                output_boxes.append(det)
            else:
                suppressed_count += 1

        # Clean up stale tracks not seen in this frame
        stale_tids = [tid for tid in self.track_registry if tid not in active_track_ids]
        for tid in stale_tids:
            del self.track_registry[tid]

        diagnostics = {
            "input_boxes": len(tracked_boxes),
            "output_boxes": len(output_boxes),
            "suppressed_ephemeral_boxes": suppressed_count,
            "preserved_stationary_boxes": preserved_stationary_count,
        }

        return output_boxes, diagnostics
