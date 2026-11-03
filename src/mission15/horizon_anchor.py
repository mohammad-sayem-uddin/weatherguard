"""Adaptive Vanishing-Point and Horizon-Anchored Candidate Generator.

Module: src/mission15/horizon_anchor.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-03
Mission: Mission 15 — Temporal Evidence Accumulation & Horizon-Anchored Candidate Generation

Design Principles:
1. Dynamic Horizon & Vanishing Point Estimation: Instead of hardcoding a static horizon row,
   estimates road line convergence dynamically via Hough line intersections and robust median filtering.
2. Temporal Horizon Smoothing: Uses an online causal EMA filter to smooth vanishing point
   trajectories across video frames, handling road dips, hills, and camera pitch changes.
3. Constrained Horizon Cone: Concentrates candidate generation strictly in the distant road convergence
   trapezoid, avoiding wasteful proposal generation across sky, trees, or peripheral buildings.
4. Structural Background Rejection: Distinguishes compact vehicle silhouettes from overhead signs,
   bridges, and horizon treelines using vertical placement and aspect ratio constraints.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.priority_allocation import CandidateProposal


@dataclass
class HorizonAnchorConfig:
    """Hyperparameters for adaptive vanishing-point and horizon anchoring."""
    horizon_band_half_height: int = 45  # Half-height (dy) of adaptive horizon search band
    min_line_length: int = 35
    max_line_gap: int = 15
    canny_thresh1: int = 50
    canny_thresh2: int = 150
    ema_beta: float = 0.85  # Smoothing factor for vanishing point tracking
    min_candidate_size: int = 6
    max_candidate_size: int = 32
    max_proposals: int = 10
    default_vp: Tuple[float, float] = (640.0, 360.0)  # Default center-frame prior


class AdaptiveHorizonAnchor:
    """Estimates dynamic vanishing-point geometry and generates horizon-anchored candidate proposals."""

    def __init__(self, config: Optional[HorizonAnchorConfig] = None):
        self.config = config or HorizonAnchorConfig()
        self.current_sequence_id: Optional[str] = None
        self.smoothed_vp: Tuple[float, float] = self.config.default_vp
        self.frame_index: int = 0
        self.last_estimated_vp: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        """Resets vanishing point tracking state across sequence boundaries."""
        self.current_sequence_id = None
        self.smoothed_vp = self.config.default_vp
        self.frame_index = 0
        self.last_estimated_vp = None

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        sequence_id: str,
        frame_id: Optional[str] = None,
    ) -> List[CandidateProposal]:
        """Estimates vanishing point adaptively and extracts horizon-anchored candidate proposals.
        
        Args:
            frame_rgb: Current video frame (H x W x 3 RGB).
            sequence_id: Video sequence identifier.
            frame_id: Optional frame name / index.
            
        Returns:
            List of CandidateProposal dataclass instances.
        """
        # Strict sequence isolation
        if self.current_sequence_id != sequence_id:
            self.reset()
            self.current_sequence_id = sequence_id

        h, w = frame_rgb.shape[:2]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Estimate Instantaneous Vanishing Point
        instant_vp = self._estimate_vanishing_point(gray, w, h)
        if instant_vp is not None:
            self.last_estimated_vp = instant_vp
            if self.frame_index == 0:
                self.smoothed_vp = instant_vp
            else:
                beta = self.config.ema_beta
                sx = beta * self.smoothed_vp[0] + (1.0 - beta) * instant_vp[0]
                sy = beta * self.smoothed_vp[1] + (1.0 - beta) * instant_vp[1]
                self.smoothed_vp = (sx, sy)
        else:
            # Fall back to smoothed VP
            self.last_estimated_vp = self.smoothed_vp

        self.frame_index += 1
        vp_x, vp_y = self.smoothed_vp

        # 2. Extract Horizon Road Convergence Region
        proposals = self._extract_horizon_candidates(gray, w, h, vp_x, vp_y)
        return proposals

    def _estimate_vanishing_point(
        self,
        gray: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> Optional[Tuple[float, float]]:
        """Detects road lines via Hough Transform and computes the dominant intersection."""
        # Focus edge detection on the lower-middle road region [0.35*H to 0.85*H]
        y1_roi = int(0.35 * img_h)
        y2_roi = int(0.85 * img_h)
        roi_gray = gray[y1_roi:y2_roi, :]

        # Edge detection
        edges = cv2.Canny(roi_gray, self.config.canny_thresh1, self.config.canny_thresh2)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap,
        )

        if lines is None or len(lines) < 4:
            return None

        # Filter diagonal road lines into left and right clusters
        left_lines = []
        right_lines = []

        for line in lines:
            l = line.reshape(-1)
            x1, y1, x2, y2 = int(l[0]), int(l[1]), int(l[2]), int(l[3])
            # Convert back to full image coordinates
            y1 += y1_roi
            y2 += y1_roi

            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1e-4:
                continue

            slope = float(dy) / float(dx)
            # Retain typical lane angle: 18 deg to 72 deg (|slope| between 0.32 and 3.0)
            if 0.32 <= abs(slope) <= 3.2:
                # Intercept b = y - slope * x
                b = y1 - slope * x1
                if slope < -0.32:
                    left_lines.append((slope, b))
                elif slope > 0.32:
                    right_lines.append((slope, b))

        if not left_lines or not right_lines:
            return None

        # Compute pairwise intersections between left and right lines
        intersections = []
        # Sample up to 15 pairs to bound CPU compute to < 0.5 ms
        for s_l, b_l in left_lines[:15]:
            for s_r, b_r in right_lines[:15]:
                denom = s_l - s_r
                if abs(denom) < 1e-4:
                    continue
                ix = (b_r - b_l) / denom
                iy = s_l * ix + b_l

                # Check if intersection falls in plausible central vanishing region
                if 0.30 * img_w <= ix <= 0.70 * img_w and 0.25 * img_h <= iy <= 0.65 * img_h:
                    intersections.append((ix, iy))

        if len(intersections) < 3:
            return None

        # Robust spatial median of valid intersections
        xs = [p[0] for p in intersections]
        ys = [p[1] for p in intersections]
        median_x = float(np.median(xs))
        median_y = float(np.median(ys))
        return (median_x, median_y)

    def _extract_horizon_candidates(
        self,
        gray: np.ndarray,
        img_w: int,
        img_h: int,
        vp_x: float,
        vp_y: float,
    ) -> List[CandidateProposal]:
        """Extracts candidate proposals within the adaptive road horizon convergence cone."""
        # Define adaptive horizon band: [vp_y - dy, vp_y + dy]
        dy = self.config.horizon_band_half_height
        y_top = max(0, int(vp_y - dy * 0.4))  # Limit above horizon to avoid sky
        y_bottom = min(img_h, int(vp_y + dy * 1.6))  # Extend downward into road

        # Road convergence trapezoid: width expands downward from vanishing point
        # At horizon: width is ~200px; at y_bottom: width is ~450px
        cone_w_top = 220.0
        cone_w_bottom = 480.0

        x_left_top = max(0.0, vp_x - cone_w_top / 2.0)
        x_right_top = min(float(img_w), vp_x + cone_w_top / 2.0)
        x_left_bottom = max(0.0, vp_x - cone_w_bottom / 2.0)
        x_right_bottom = min(float(img_w), vp_x + cone_w_bottom / 2.0)

        # Create binary mask of the road convergence cone
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        pts = np.array([
            [x_left_top, y_top],
            [x_right_top, y_top],
            [x_right_bottom, y_bottom],
            [x_left_bottom, y_bottom],
        ], dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)

        # High-pass contrast detection inside masked region
        blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
        diff = cv2.absdiff(gray, blur)
        diff_masked = cv2.bitwise_and(diff, diff, mask=mask)

        # Threshold for salient anomalies in the distant horizon
        # Adaptive threshold based on 98th percentile of road region
        masked_pixels = diff_masked[mask > 0]
        if len(masked_pixels) == 0:
            return []

        p98 = np.percentile(masked_pixels, 98.0)
        tau = max(18.0, float(p98 * 0.70))

        thresh = np.uint8((diff_masked > tau) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_candidates = []
        min_sz = self.config.min_candidate_size
        max_sz = self.config.max_candidate_size

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Ultra-small / small size constraints
            if bw < min_sz or bh < min_sz:
                continue
            if bw > max_sz or bh > max_sz:
                continue

            aspect = float(bw) / float(bh + 1e-6)
            if aspect < 0.25 or aspect > 4.0:
                continue

            # Candidate score based on local contrast and proximity to vanishing point
            patch = diff_masked[y : y + bh, x : x + bw]
            patch_contrast = float(np.mean(patch)) / 255.0

            # Proximity factor: distant vehicles cluster close to the vanishing point
            dist_to_vp = np.hypot((x + bw / 2.0) - vp_x, (y + bh / 2.0) - vp_y)
            proximity_weight = max(0.20, 1.0 - (dist_to_vp / 350.0))

            final_score = float(np.clip(patch_contrast * 2.5 * proximity_weight, 0.15, 0.95))

            # Context padding
            pad = 4.0
            cx1 = max(0.0, float(x) - pad)
            cy1 = max(0.0, float(y) - pad)
            cx2 = min(float(img_w), float(x + bw) + pad)
            cy2 = min(float(img_h), float(y + bh) + pad)

            raw_candidates.append({
                "x1": cx1,
                "y1": cy1,
                "x2": cx2,
                "y2": cy2,
                "score": final_score,
                "bw": bw,
                "bh": bh,
                "aspect": aspect,
                "dist_vp": dist_to_vp,
            })

        raw_candidates.sort(key=lambda c: c["score"], reverse=True)

        # Deduplication / NMS on horizon candidates
        proposals: List[CandidateProposal] = []
        for cand in raw_candidates[: self.config.max_proposals * 2]:
            overlap = False
            for p in proposals:
                ix1 = max(cand["x1"], p.x1)
                iy1 = max(cand["y1"], p.y1)
                ix2 = min(cand["x2"], p.x2)
                iy2 = min(cand["y2"], p.y2)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                cand_area = (cand["x2"] - cand["x1"]) * (cand["y2"] - cand["y1"])
                if inter / (cand_area + 1e-6) > 0.40:
                    overlap = True
                    break

            if not overlap:
                cand_w = cand["x2"] - cand["x1"]
                cand_h = cand["y2"] - cand["y1"]
                prop = CandidateProposal(
                    x1=cand["x1"],
                    y1=cand["y1"],
                    x2=cand["x2"],
                    y2=cand["y2"],
                    score=cand["score"],
                    source="horizon_anchor",
                    area=cand_w * cand_h,
                    aspect_ratio=cand["aspect"],
                    metadata={
                        "vp_x": vp_x,
                        "vp_y": vp_y,
                        "dist_to_vp": cand["dist_vp"],
                        "frame_index": self.frame_index,
                    },
                )
                proposals.append(prop)
                if len(proposals) >= self.config.max_proposals:
                    break

        return proposals
