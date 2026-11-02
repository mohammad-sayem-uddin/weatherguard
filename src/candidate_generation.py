"""Detector-Independent Candidate Generation & Proposal Fusion Architecture for WeatherGuard Mission 12.

Implements:
1. CandidateProposal dataclass with spatial bounds, confidence, source modality, and scale prior.
2. FrameDifferencingProposal: Temporal difference between consecutive frames with wiper and illumination filtering.
3. SparseOpticalFlowProposal: Lucas-Kanade feature motion tracking with global ego-motion compensation.
4. StaticSalienceProposal: Multi-scale gradient magnitude and contrast salience within road geometry.
5. SpatialPriorProposal: Vanishing-point and horizon-band scale prior.
6. ProposalFusionController: Strictly causal online fusion, deduplication, budget enforcement (K <= 2, 3), and ROI clustering.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.roi_inference import CropRegion, cluster_candidate_boxes


@dataclass
class CandidateProposal:
    """A spatial proposal for potential high-resolution ROI magnification."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: str  # 'detector', 'frame_diff', 'sparse_flow', 'static_salience', 'spatial_prior'
    area: float = 0.0
    aspect_ratio: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        w = max(0.0, self.x2 - self.x1)
        h = max(0.0, self.y2 - self.y1)
        self.area = w * h
        self.aspect_ratio = w / max(h, 1e-4)

    @property
    def box_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class FrameDifferenceProposal:
    """Detects candidate motion regions via causal consecutive frame differencing."""

    def __init__(
        self,
        diff_threshold: float = 25.0,
        min_area: float = 40.0,
        max_area: float = 40000.0,
        max_dim: float = 500.0,
        horizon_y_min: float = 0.20,
        horizon_y_max: float = 0.82,
    ):
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.max_dim = max_dim
        self.horizon_y_min = horizon_y_min
        self.horizon_y_max = horizon_y_max
        self.prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_gray = None

    def generate_proposals(self, current_frame_bgr: np.ndarray) -> List[CandidateProposal]:
        """Causal proposal generation using only current and previous frame."""
        h, w = current_frame_bgr.shape[:2]
        # Downscale for edge CPU speed (320x180)
        proc_w, proc_h = 320, 180
        curr_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(curr_gray, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        if self.prev_gray is None:
            self.prev_gray = small_gray
            return []

        # Absolute frame difference
        diff = cv2.absdiff(small_gray, self.prev_gray)
        self.prev_gray = small_gray

        # Apply road horizon mask in downscaled space
        y_min_idx = int(self.horizon_y_min * proc_h)
        y_max_idx = int(self.horizon_y_max * proc_h)
        diff_masked = np.zeros_like(diff)
        diff_masked[y_min_idx:y_max_idx, :] = diff[y_min_idx:y_max_idx, :]

        # Thresholding
        _, thresh = cv2.threshold(diff_masked, int(self.diff_threshold), 255, cv2.THRESH_BINARY)
        
        # Morphological opening and closing to remove speckles and connect vehicle parts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        
        scale_x = w / float(proc_w)
        scale_y = h / float(proc_h)

        proposals = []
        for i in range(1, n_labels):
            sx, sy, sw, sh, sarea = stats[i]
            
            # Map back to full-frame dimensions
            bx1 = sx * scale_x
            by1 = sy * scale_y
            bw = sw * scale_x
            bh = sh * scale_y
            bx2 = bx1 + bw
            by2 = by1 + bh
            barea = bw * bh

            # Filter: ignore noise, whole-frame wiper sweeps, or gigantic background shifts
            if self.min_area <= barea <= self.max_area and bw <= self.max_dim and bh <= self.max_dim:
                # Score based on motion intensity inside component
                comp_mask = (labels == i)
                mean_intensity = float(np.mean(diff_masked[comp_mask])) if np.any(comp_mask) else 0.0
                score = min(1.0, mean_intensity / 80.0)

                proposals.append(
                    CandidateProposal(
                        x1=round(bx1, 1),
                        y1=round(by1, 1),
                        x2=round(bx2, 1),
                        y2=round(by2, 1),
                        score=round(score, 3),
                        source="frame_diff",
                        metadata={"downscaled_area": sarea, "mean_diff": mean_intensity},
                    )
                )

        # Sort by score and truncate to top 15
        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals[:15]


class SparseOpticalFlowProposal:
    """Detects candidate motion regions via Lucas-Kanade sparse optical flow with ego-motion compensation."""

    def __init__(
        self,
        max_corners: int = 150,
        quality_level: float = 0.02,
        min_distance: float = 12.0,
        residual_threshold_px: float = 1.8,
        cluster_radius_px: float = 48.0,
        horizon_y_min: float = 0.25,
        horizon_y_max: float = 0.80,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.residual_threshold_px = residual_threshold_px
        self.cluster_radius_px = cluster_radius_px
        self.horizon_y_min = horizon_y_min
        self.horizon_y_max = horizon_y_max
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pts = None

    def generate_proposals(self, current_frame_bgr: np.ndarray) -> List[CandidateProposal]:
        """Causal sparse optical flow proposal generator."""
        h, w = current_frame_bgr.shape[:2]
        curr_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            # Find initial corners
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(self.horizon_y_min * h):int(self.horizon_y_max * h), :] = 255
            self.prev_pts = cv2.goodFeaturesToTrack(
                curr_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                mask=mask,
            )
            return []

        if self.prev_pts is None or len(self.prev_pts) < 8:
            # Re-seed points
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(self.horizon_y_min * h):int(self.horizon_y_max * h), :] = 255
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                mask=mask,
            )
            if self.prev_pts is None or len(self.prev_pts) < 4:
                self.prev_gray = curr_gray
                return []

        # Pyramidal Lucas-Kanade optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            self.prev_pts,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        good_new = curr_pts[status == 1]
        good_old = self.prev_pts[status == 1]

        # Update previous frame and points for next timestep
        self.prev_gray = curr_gray
        # Re-sample next points on curr_gray
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(self.horizon_y_min * h):int(self.horizon_y_max * h), :] = 255
        self.prev_pts = cv2.goodFeaturesToTrack(
            curr_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
        )

        if len(good_new) < 5:
            return []

        # Motion displacement vectors
        displacements = good_new - good_old
        
        # Estimate ego-motion via median flow vector
        median_flow = np.median(displacements, axis=0)
        
        # Compute residual independent motion after ego-motion subtraction
        residuals = displacements - median_flow
        residual_magnitudes = np.linalg.norm(residuals, axis=1)

        # Filter independent moving points
        moving_indices = np.where(residual_magnitudes >= self.residual_threshold_px)[0]
        if len(moving_indices) == 0:
            return []

        moving_pts = good_new[moving_indices]
        moving_mags = residual_magnitudes[moving_indices]

        # Simple clustering of moving points
        n_pts = len(moving_pts)
        visited = [False] * n_pts
        clusters = []

        for i in range(n_pts):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            queue = [i]
            while queue:
                curr = queue.pop(0)
                for j in range(n_pts):
                    if not visited[j]:
                        dist = np.linalg.norm(moving_pts[curr] - moving_pts[j])
                        if dist <= self.cluster_radius_px:
                            visited[j] = True
                            cluster.append(j)
                            queue.append(j)
            clusters.append(cluster)

        proposals = []
        for cl in clusters:
            pts = moving_pts[cl]
            mags = moving_mags[cl]
            x_min = float(np.min(pts[:, 0]))
            y_min = float(np.min(pts[:, 1]))
            x_max = float(np.max(pts[:, 0]))
            y_max = float(np.max(pts[:, 1]))

            # Pad bounding box around points
            pad_x = max(24.0, (x_max - x_min) * 0.35)
            pad_y = max(18.0, (y_max - y_min) * 0.35)

            bx1 = max(0.0, x_min - pad_x)
            by1 = max(0.0, y_min - pad_y)
            bx2 = min(float(w), x_max + pad_x)
            by2 = min(float(h), y_max + pad_y)

            mean_res = float(np.mean(mags))
            score = min(1.0, mean_res / 10.0 + len(cl) * 0.05)

            proposals.append(
                CandidateProposal(
                    x1=round(bx1, 1),
                    y1=round(by1, 1),
                    x2=round(bx2, 1),
                    y2=round(by2, 1),
                    score=round(score, 3),
                    source="sparse_flow",
                    metadata={"points_count": len(cl), "mean_residual_flow": round(mean_res, 2)},
                )
            )

        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals[:10]


class StaticSalienceProposal:
    """Detects candidate objects via multi-scale edge gradient and local contrast salience."""

    def __init__(
        self,
        percentile_thresh: float = 94.0,
        min_area: float = 45.0,
        max_area: float = 35000.0,
        horizon_y_min: float = 0.28,
        horizon_y_max: float = 0.72,
    ):
        self.percentile_thresh = percentile_thresh
        self.min_area = min_area
        self.max_area = max_area
        self.horizon_y_min = horizon_y_min
        self.horizon_y_max = horizon_y_max

    def generate_proposals(self, current_frame_bgr: np.ndarray) -> List[CandidateProposal]:
        """Single-frame static salience generator."""
        h, w = current_frame_bgr.shape[:2]
        proc_w, proc_h = 320, 180
        curr_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(curr_gray, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        # Sobel gradients
        sobelx = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobelx, sobely)

        # Road horizon weighting
        y_min_idx = int(self.horizon_y_min * proc_h)
        y_max_idx = int(self.horizon_y_max * proc_h)
        masked_grad = np.zeros_like(grad_mag)
        masked_grad[y_min_idx:y_max_idx, :] = grad_mag[y_min_idx:y_max_idx, :]

        # Distance weighting: prioritize center vanishing band y ~ 0.45 * H
        y_coords = np.arange(proc_h)[:, None]
        center_y = 0.45 * proc_h
        sigma_y = 0.18 * proc_h
        gauss_weight = np.exp(-0.5 * ((y_coords - center_y) / sigma_y) ** 2)
        weighted_grad = masked_grad * gauss_weight

        # Threshold top percentile
        val_sub = weighted_grad[y_min_idx:y_max_idx, :]
        if len(val_sub) == 0:
            return []
        th = np.percentile(val_sub, self.percentile_thresh)
        if th <= 1.0:
            max_val = float(np.max(val_sub))
            if max_val >= 10.0:
                th = max_val * 0.25
            else:
                return []

        sal_mask = (weighted_grad > th).astype(np.uint8) * 255
        
        # Morphological grouping
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        closed = cv2.morphologyEx(sal_mask, cv2.MORPH_CLOSE, kernel)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

        scale_x = w / float(proc_w)
        scale_y = h / float(proc_h)

        proposals = []
        for i in range(1, n_labels):
            sx, sy, sw, sh, sarea = stats[i]
            bx1 = sx * scale_x
            by1 = sy * scale_y
            bw = sw * scale_x
            bh = sh * scale_y
            bx2 = bx1 + bw
            by2 = by1 + bh
            barea = bw * bh

            if self.min_area <= barea <= self.max_area and bw < 500 and bh < 300:
                comp_mask = (labels == i)
                mean_grad = float(np.mean(weighted_grad[comp_mask])) if np.any(comp_mask) else 0.0
                score = min(1.0, mean_grad / 120.0)

                proposals.append(
                    CandidateProposal(
                        x1=round(bx1, 1),
                        y1=round(by1, 1),
                        x2=round(bx2, 1),
                        y2=round(by2, 1),
                        score=round(score, 3),
                        source="static_salience",
                        metadata={"gradient_salience": round(mean_grad, 2)},
                    )
                )

        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals[:15]


class SpatialPriorProposal:
    """Road geometry and perspective-based spatial prior for vanishing-horizon regions."""

    def __init__(
        self,
        horizon_band: Tuple[float, float] = (0.35, 0.58),
        num_horizontal_cells: int = 3,
    ):
        self.horizon_band = horizon_band
        self.num_horizontal_cells = num_horizontal_cells

    def generate_proposals(self, image_shape: Tuple[int, int]) -> List[CandidateProposal]:
        """Proposes structured spatial sub-windows along the perspective vanishing band."""
        h, w = image_shape
        y1 = self.horizon_band[0] * h
        y2 = self.horizon_band[1] * h
        cell_w = w / float(self.num_horizontal_cells)

        proposals = []
        for c in range(self.num_horizontal_cells):
            x1 = c * cell_w
            x2 = (c + 1) * cell_w
            # Center cells get higher perspective probability
            center_dist = abs(c - (self.num_horizontal_cells - 1) / 2.0)
            score = max(0.40, 0.70 - center_dist * 0.15)

            proposals.append(
                CandidateProposal(
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    x2=round(x2, 1),
                    y2=round(y2, 1),
                    score=round(score, 3),
                    source="spatial_prior",
                    metadata={"cell_index": c},
                )
            )
        return proposals


class ProposalFusionController:
    """Unified online controller for multi-modal candidate generation, deduplication, and budget-aware ROI clustering."""

    def __init__(
        self,
        use_detector: bool = True,
        use_frame_diff: bool = False,
        use_sparse_flow: bool = False,
        use_static_salience: bool = False,
        use_spatial_prior: bool = False,
        max_crops: int = 2,
        context_margin: float = 0.20,
        min_crop_size: float = 256.0,
        cluster_distance_px: float = 120.0,
        iou_dedup_threshold: float = 0.40,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.use_detector = use_detector
        self.use_frame_diff = use_frame_diff
        self.use_sparse_flow = use_sparse_flow
        self.use_static_salience = use_static_salience
        self.use_spatial_prior = use_spatial_prior
        self.max_crops = max_crops
        self.context_margin = context_margin
        self.min_crop_size = min_crop_size
        self.cluster_distance_px = cluster_distance_px
        self.iou_dedup_threshold = iou_dedup_threshold

        self.weights = weights or {
            "detector": 1.0,
            "sparse_flow": 0.85,
            "frame_diff": 0.65,
            "static_salience": 0.70,
            "spatial_prior": 0.50,
        }

        # Initialize sub-generators
        self.diff_gen = FrameDifferenceProposal() if use_frame_diff else None
        self.flow_gen = SparseOpticalFlowProposal() if use_sparse_flow else None
        self.sal_gen = StaticSalienceProposal() if use_static_salience else None
        self.prior_gen = SpatialPriorProposal() if use_spatial_prior else None

    def reset(self) -> None:
        """Resets temporal generators at video sequence boundaries."""
        if self.diff_gen:
            self.diff_gen.reset()
        if self.flow_gen:
            self.flow_gen.reset()

    def fuse_proposals(
        self,
        frame_bgr: np.ndarray,
        detector_boxes: Optional[Sequence[Any]] = None,
    ) -> Tuple[List[CandidateProposal], List[CropRegion], Dict[str, float]]:
        """Executes candidate generation, deduplication, budget ranking, and returns final CropRegions."""
        h, w = frame_bgr.shape[:2]
        timings: Dict[str, float] = {}
        all_proposals: List[CandidateProposal] = []

        # 1. Detector Candidates (Small or Uncertain Boxes)
        t0 = time.time()
        if self.use_detector and detector_boxes:
            for d in detector_boxes:
                # Accept small boxes (< 48px) or low-confidence boxes (0.15 <= conf < 0.45)
                dw = d.x2 - d.x1
                dh = d.y2 - d.y1
                area = dw * dh
                is_small = (area < 2304.0) or (dw < 48.0 or dh < 48.0)
                is_uncertain = (0.15 <= d.confidence < 0.45)
                
                if is_small or is_uncertain:
                    all_proposals.append(
                        CandidateProposal(
                            x1=float(d.x1),
                            y1=float(d.y1),
                            x2=float(d.x2),
                            y2=float(d.y2),
                            score=float(d.confidence) * self.weights["detector"],
                            source="detector",
                            metadata={"class_name": d.class_name, "raw_conf": d.confidence},
                        )
                    )
        timings["detector_candidates_ms"] = (time.time() - t0) * 1000.0

        # 2. Frame Differencing
        if self.use_frame_diff and self.diff_gen:
            t0 = time.time()
            diff_props = self.diff_gen.generate_proposals(frame_bgr)
            for p in diff_props:
                p.score *= self.weights["frame_diff"]
                all_proposals.append(p)
            timings["frame_diff_ms"] = (time.time() - t0) * 1000.0

        # 3. Sparse Optical Flow
        if self.use_sparse_flow and self.flow_gen:
            t0 = time.time()
            flow_props = self.flow_gen.generate_proposals(frame_bgr)
            for p in flow_props:
                p.score *= self.weights["sparse_flow"]
                all_proposals.append(p)
            timings["sparse_flow_ms"] = (time.time() - t0) * 1000.0

        # 4. Static Salience
        if self.use_static_salience and self.sal_gen:
            t0 = time.time()
            sal_props = self.sal_gen.generate_proposals(frame_bgr)
            for p in sal_props:
                p.score *= self.weights["static_salience"]
                all_proposals.append(p)
            timings["static_salience_ms"] = (time.time() - t0) * 1000.0

        # 5. Spatial Prior
        if self.use_spatial_prior and self.prior_gen:
            t0 = time.time()
            prior_props = self.prior_gen.generate_proposals((h, w))
            for p in prior_props:
                p.score *= self.weights["spatial_prior"]
                all_proposals.append(p)
            timings["spatial_prior_ms"] = (time.time() - t0) * 1000.0

        # 6. Deduplication & Cross-Modal NMS
        t0 = time.time()
        deduped = self._deduplicate_proposals(all_proposals, self.iou_dedup_threshold)
        
        # Sort by composite score
        deduped.sort(key=lambda p: p.score, reverse=True)
        timings["dedup_and_sort_ms"] = (time.time() - t0) * 1000.0

        # 7. Spatial Clustering into Bounded Crop Regions (K <= max_crops)
        t0 = time.time()
        candidate_boxes = [(p.x1, p.y1, p.x2, p.y2) for p in deduped]
        
        crop_regions = cluster_candidate_boxes(
            boxes=candidate_boxes,
            image_shape=(w, h),
            max_distance_px=self.cluster_distance_px,
            context_margin=self.context_margin,
            min_crop_size=self.min_crop_size,
            max_crops=self.max_crops,
        )
        timings["crop_clustering_ms"] = (time.time() - t0) * 1000.0

        return deduped, crop_regions, timings

    @staticmethod
    def _deduplicate_proposals(
        proposals: List[CandidateProposal],
        iou_thresh: float = 0.40,
    ) -> List[CandidateProposal]:
        """Greedy non-maximum suppression across heterogeneous proposal sources."""
        if not proposals:
            return []

        proposals_sorted = sorted(proposals, key=lambda p: p.score, reverse=True)
        kept: List[CandidateProposal] = []

        for cand in proposals_sorted:
            box_c = cand.box_xyxy
            overlap = False
            for k in kept:
                # Compute IoU
                box_k = k.box_xyxy
                ix1 = max(box_c[0], box_k[0])
                iy1 = max(box_c[1], box_k[1])
                ix2 = min(box_c[2], box_k[2])
                iy2 = min(box_c[3], box_k[3])
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                union = cand.area + k.area - inter
                iou = inter / union if union > 0 else 0.0
                if iou >= iou_thresh:
                    overlap = True
                    break
            if not overlap:
                kept.append(cand)

        return kept
