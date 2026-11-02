"""WeatherGuard Mission 13 — Priority-Gated Candidate Allocation Controller.

Implements unified candidate representation, coverage-aware priority ranking,
independent proposal admission gating, uncertainty-aware routing, and adaptive crop budgeting.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.video_dataset import compute_iou, DetectionBox
from src.candidate_generation import (
    CandidateProposal,
    FrameDifferenceProposal,
    StaticSalienceProposal,
    SpatialPriorProposal,
)
from src.roi_inference import (
    CropRegion,
    cluster_candidate_boxes,
)


@dataclass
class AllocatedCandidate:
    """Unified representation of a region-of-interest candidate across all sources."""
    candidate_id: str
    source: str  # 'detector_high', 'detector_low', 'frame_diff', 'static_salience', 'spatial_horizon'
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    scale_bucket: str  # 'ultra_small', 'small', 'medium', 'large'
    frame_idx: int = 0
    timestamp: float = 0.0
    generation_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    cluster_id: Optional[int] = None
    max_detector_overlap_iou: float = 0.0
    is_new_spatial_coverage: bool = True
    priority_score: float = 0.0
    allocation_status: str = "pending"  # 'allocated', 'rejected', 'suppressed'
    rejection_reason: Optional[str] = None

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

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "score": round(self.score, 4),
            "scale_bucket": self.scale_bucket,
            "priority_score": round(self.priority_score, 4),
            "allocation_status": self.allocation_status,
            "rejection_reason": self.rejection_reason,
            "max_detector_overlap_iou": round(self.max_detector_overlap_iou, 4),
            "is_new_spatial_coverage": self.is_new_spatial_coverage,
            "metadata": self.metadata,
        }


@dataclass
class CropAllocationDecision:
    """Audit record capturing all allocation choices for a single frame."""
    frame_idx: int
    initial_budget: int
    used_budget: int
    allocated_crops: List[CropRegion]
    all_candidates: List[AllocatedCandidate]
    retained_detector_count: int
    admitted_independent_count: int
    displaced_detector_count: int
    policy_name: str
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "initial_budget": self.initial_budget,
            "used_budget": self.used_budget,
            "allocated_crops": [
                c.to_dict() if hasattr(c, "to_dict") else {
                    "x1": round(float(c.x1), 1),
                    "y1": round(float(c.y1), 1),
                    "x2": round(float(c.x2), 1),
                    "y2": round(float(c.y2), 1),
                    "candidate_indices": list(getattr(c, "candidate_indices", [])),
                    "source_strategy": str(getattr(c, "source_strategy", "clustered")),
                }
                for c in self.allocated_crops
            ],
            "total_candidates": len(self.all_candidates),
            "retained_detector_count": self.retained_detector_count,
            "admitted_independent_count": self.admitted_independent_count,
            "displaced_detector_count": self.displaced_detector_count,
            "policy_name": self.policy_name,
            "timings_ms": self.timings_ms,
        }


class PriorityAllocationController:
    """Priority-Gated Candidate Allocation Controller for Edge Object Detection.

    Supports 6 configurable allocation policies:
    - 'policy_a_detector_first': Detector proposals strictly prioritized; independent proposals fill remaining budget.
    - 'policy_b_confidence_coverage': Joint scoring based on confidence, spatial novelty, and scale.
    - 'policy_c_independent_gate': Explicit multi-criteria admission gates for independent proposals.
    - 'policy_d_uncertainty_aware': Prioritizes low-confidence detector candidates and high-entropy zones.
    - 'policy_e_adaptive_budget': Dynamic crop budget K in {0, 1, 2, 3}; executes 0 crops when no candidate is worthy.
    - 'policy_j_best_combination': Hierarchically gated detector-first + admission gating + adaptive budgeting.
    """

    def __init__(
        self,
        policy: str = "policy_a_detector_first",
        max_crops: int = 2,
        context_margin: float = 0.20,
        min_crop_size: float = 256.0,
        cluster_distance_px: float = 120.0,
        use_frame_diff: bool = True,
        use_static_salience: bool = True,
        use_spatial_horizon: bool = True,
        min_independent_score: float = 0.45,
        horizon_band: Tuple[float, float] = (0.35, 0.58),
        overlap_redundancy_thresh: float = 0.20,
    ):
        self.policy = policy
        self.max_crops = max_crops
        self.context_margin = context_margin
        self.min_crop_size = min_crop_size
        self.cluster_distance_px = cluster_distance_px
        self.use_frame_diff = use_frame_diff
        self.use_static_salience = use_static_salience
        self.use_spatial_horizon = use_spatial_horizon
        self.min_independent_score = min_independent_score
        self.horizon_band = horizon_band
        self.overlap_redundancy_thresh = overlap_redundancy_thresh

        # Initialize underlying generators
        self.diff_gen = FrameDifferenceProposal() if use_frame_diff else None
        self.sal_gen = StaticSalienceProposal() if use_static_salience else None
        self.horizon_gen = SpatialPriorProposal(horizon_band=horizon_band) if use_spatial_horizon else None

        self.frame_counter = 0

    def reset(self) -> None:
        """Resets causal state across video sequence boundaries."""
        if self.diff_gen:
            self.diff_gen.reset()
        self.frame_counter = 0

    def generate_unified_candidates(
        self,
        frame_bgr: np.ndarray,
        detector_boxes: Optional[Sequence[DetectionBox]] = None,
    ) -> Tuple[List[AllocatedCandidate], Dict[str, float]]:
        """Extracts and standardizes candidates from all active sources into a unified list."""
        h, w = frame_bgr.shape[:2]
        timings: Dict[str, float] = {}
        unified: List[AllocatedCandidate] = []
        cand_idx = 0

        # Helper to classify scale
        def get_scale(area: float) -> str:
            if area < 256.0:
                return "ultra_small"
            elif area < 1024.0:
                return "small"
            elif area < 9216.0:
                return "medium"
            else:
                return "large"

        detector_cand_boxes: List[Tuple[float, float, float, float]] = []

        # 1. Detector Proposals (High and Low Confidence)
        t0 = time.perf_counter()
        if detector_boxes:
            for d in detector_boxes:
                dw = d.x2 - d.x1
                dh = d.y2 - d.y1
                area = dw * dh
                is_small = (area < 2304.0) or (dw < 48.0 or dh < 48.0)
                is_low_conf = (0.10 <= d.confidence < 0.35)
                is_high_conf = (d.confidence >= 0.35)

                if is_small or is_low_conf:
                    source_tag = "detector_high" if is_high_conf else "detector_low"
                    cand = AllocatedCandidate(
                        candidate_id=f"det_{cand_idx}",
                        source=source_tag,
                        x1=float(d.x1),
                        y1=float(d.y1),
                        x2=float(d.x2),
                        y2=float(d.y2),
                        score=float(d.confidence),
                        scale_bucket=get_scale(area),
                        frame_idx=self.frame_counter,
                        timestamp=time.time(),
                        metadata={"class_name": d.class_name, "raw_confidence": d.confidence},
                        max_detector_overlap_iou=1.0,
                        is_new_spatial_coverage=True,
                    )
                    unified.append(cand)
                    detector_cand_boxes.append((d.x1, d.y1, d.x2, d.y2))
                    cand_idx += 1
        timings["detector_candidates_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Frame Difference Proposals
        if self.diff_gen:
            t0 = time.perf_counter()
            diff_props = self.diff_gen.generate_proposals(frame_bgr)
            gen_ms = (time.perf_counter() - t0) * 1000.0
            timings["frame_diff_ms"] = gen_ms

            for p in diff_props:
                # Check overlap with existing detector proposals
                max_iou = 0.0
                p_box = (p.x1, p.y1, p.x2, p.y2)
                for db in detector_cand_boxes:
                    iou = compute_iou(p_box, db)
                    if iou > max_iou:
                        max_iou = iou

                cand = AllocatedCandidate(
                    candidate_id=f"diff_{cand_idx}",
                    source="frame_diff",
                    x1=float(p.x1),
                    y1=float(p.y1),
                    x2=float(p.x2),
                    y2=float(p.y2),
                    score=float(p.score),
                    scale_bucket=get_scale(p.area),
                    frame_idx=self.frame_counter,
                    timestamp=time.time(),
                    generation_latency_ms=gen_ms,
                    metadata=p.metadata,
                    max_detector_overlap_iou=max_iou,
                    is_new_spatial_coverage=(max_iou < self.overlap_redundancy_thresh),
                )
                unified.append(cand)
                cand_idx += 1

        # 3. Static Salience Proposals
        if self.sal_gen:
            t0 = time.perf_counter()
            sal_props = self.sal_gen.generate_proposals(frame_bgr)
            gen_ms = (time.perf_counter() - t0) * 1000.0
            timings["static_salience_ms"] = gen_ms

            for p in sal_props:
                max_iou = 0.0
                p_box = (p.x1, p.y1, p.x2, p.y2)
                for db in detector_cand_boxes:
                    iou = compute_iou(p_box, db)
                    if iou > max_iou:
                        max_iou = iou

                cand = AllocatedCandidate(
                    candidate_id=f"sal_{cand_idx}",
                    source="static_salience",
                    x1=float(p.x1),
                    y1=float(p.y1),
                    x2=float(p.x2),
                    y2=float(p.y2),
                    score=float(p.score),
                    scale_bucket=get_scale(p.area),
                    frame_idx=self.frame_counter,
                    timestamp=time.time(),
                    generation_latency_ms=gen_ms,
                    metadata=p.metadata,
                    max_detector_overlap_iou=max_iou,
                    is_new_spatial_coverage=(max_iou < self.overlap_redundancy_thresh),
                )
                unified.append(cand)
                cand_idx += 1

        # 4. Spatial Horizon Proposals
        if self.horizon_gen:
            t0 = time.perf_counter()
            horiz_props = self.horizon_gen.generate_proposals((h, w))
            gen_ms = (time.perf_counter() - t0) * 1000.0
            timings["spatial_horizon_ms"] = gen_ms

            for p in horiz_props:
                max_iou = 0.0
                p_box = (p.x1, p.y1, p.x2, p.y2)
                for db in detector_cand_boxes:
                    iou = compute_iou(p_box, db)
                    if iou > max_iou:
                        max_iou = iou

                cand = AllocatedCandidate(
                    candidate_id=f"horiz_{cand_idx}",
                    source="spatial_horizon",
                    x1=float(p.x1),
                    y1=float(p.y1),
                    x2=float(p.x2),
                    y2=float(p.y2),
                    score=float(p.score),
                    scale_bucket=get_scale(p.area),
                    frame_idx=self.frame_counter,
                    timestamp=time.time(),
                    generation_latency_ms=gen_ms,
                    metadata=p.metadata,
                    max_detector_overlap_iou=max_iou,
                    is_new_spatial_coverage=(max_iou < self.overlap_redundancy_thresh),
                )
                unified.append(cand)
                cand_idx += 1

        self.frame_counter += 1
        return unified, timings

    def allocate_crops(
        self,
        frame_bgr: np.ndarray,
        detector_boxes: Optional[Sequence[DetectionBox]] = None,
    ) -> CropAllocationDecision:
        """Executes candidate allocation under the active policy and returns the allocation decision."""
        t_start = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        # 1. Generate unified candidates
        candidates, timings = self.generate_unified_candidates(frame_bgr, detector_boxes)

        # 2. Dispatch policy allocation
        t_alloc = time.perf_counter()
        if self.policy == "policy_a_detector_first":
            decision = self._allocate_policy_a(candidates, (w, h))
        elif self.policy == "policy_b_confidence_coverage":
            decision = self._allocate_policy_b(candidates, (w, h))
        elif self.policy == "policy_c_independent_gate":
            decision = self._allocate_policy_c(candidates, (w, h))
        elif self.policy == "policy_d_uncertainty_aware":
            decision = self._allocate_policy_d(candidates, (w, h))
        elif self.policy == "policy_e_adaptive_budget":
            decision = self._allocate_policy_e(candidates, (w, h))
        elif self.policy == "policy_j_best_combination":
            decision = self._allocate_policy_j(candidates, (w, h))
        else:
            raise ValueError(f"Unknown allocation policy: {self.policy}")

        timings["allocation_policy_ms"] = (time.perf_counter() - t_alloc) * 1000.0
        timings["total_candidate_pipeline_ms"] = (time.perf_counter() - t_start) * 1000.0
        decision.timings_ms = timings

        return decision

    # -------------------------------------------------------------------------
    # Policy A: Detector-First Allocation
    # -------------------------------------------------------------------------
    def _allocate_policy_a(
        self,
        candidates: List[AllocatedCandidate],
        image_shape: Tuple[int, int],
    ) -> CropAllocationDecision:
        """Ranks all detector candidates first. Independent candidates only fill remaining budget if non-redundant."""
        img_w, img_h = image_shape
        max_k = self.max_crops

        detector_cands = [c for c in candidates if c.source.startswith("detector")]
        independent_cands = [c for c in candidates if not c.source.startswith("detector")]

        # Sort detector candidates by confidence descending
        detector_cands.sort(key=lambda c: c.score, reverse=True)
        # Sort independent candidates by proposal score descending
        independent_cands.sort(key=lambda c: c.score, reverse=True)

        for c in detector_cands:
            c.priority_score = 1.0 + c.score

        for c in independent_cands:
            # Independent priority strictly below 1.0
            c.priority_score = c.score * (0.8 if c.is_new_spatial_coverage else 0.2)

        # Cluster detector candidates first
        det_boxes = [(c.x1, c.y1, c.x2, c.y2) for c in detector_cands]
        allocated_crops: List[CropRegion] = []
        retained_det = 0
        admitted_indep = 0

        if det_boxes:
            det_crops = cluster_candidate_boxes(
                boxes=det_boxes,
                image_shape=image_shape,
                max_distance_px=self.cluster_distance_px,
                context_margin=self.context_margin,
                min_crop_size=self.min_crop_size,
                max_crops=max_k,
            )
            allocated_crops.extend(det_crops)
            retained_det = len(det_boxes)
            for c in detector_cands:
                c.allocation_status = "allocated"

        # Check remaining budget
        remaining_budget = max_k - len(allocated_crops)

        if remaining_budget > 0 and independent_cands:
            # Admit only independent candidates that do NOT overlap already allocated crops
            eligible_indep = []
            for ic in independent_cands:
                # Check overlap with existing crops
                overlaps_existing = False
                for cr in allocated_crops:
                    iou = compute_iou(ic.box_xyxy, cr.box_xyxy)
                    if iou >= self.overlap_redundancy_thresh:
                        overlaps_existing = True
                        break

                if not overlaps_existing and ic.score >= self.min_independent_score:
                    eligible_indep.append(ic)
                else:
                    ic.allocation_status = "rejected"
                    ic.rejection_reason = "redundant_coverage" if overlaps_existing else "low_score"

            if eligible_indep:
                indep_boxes = [(c.x1, c.y1, c.x2, c.y2) for c in eligible_indep]
                indep_crops = cluster_candidate_boxes(
                    boxes=indep_boxes,
                    image_shape=image_shape,
                    max_distance_px=self.cluster_distance_px,
                    context_margin=self.context_margin,
                    min_crop_size=self.min_crop_size,
                    max_crops=remaining_budget,
                )
                allocated_crops.extend(indep_crops)
                admitted_indep = len(indep_crops)
                for c in eligible_indep[:admitted_indep]:
                    c.allocation_status = "allocated"

        return CropAllocationDecision(
            frame_idx=self.frame_counter,
            initial_budget=max_k,
            used_budget=len(allocated_crops),
            allocated_crops=allocated_crops,
            all_candidates=candidates,
            retained_detector_count=retained_det,
            admitted_independent_count=admitted_indep,
            displaced_detector_count=0,
            policy_name="policy_a_detector_first",
        )

    # -------------------------------------------------------------------------
    # Policy B: Confidence & Coverage Joint Allocation
    # -------------------------------------------------------------------------
    def _allocate_policy_b(
        self,
        candidates: List[AllocatedCandidate],
        image_shape: Tuple[int, int],
    ) -> CropAllocationDecision:
        """Joint scoring function incorporating detector confidence, proposal strength, scale, and spatial novelty."""
        max_k = self.max_crops

        for c in candidates:
            # Base score
            base_s = c.score
            # Modality weight
            w_mod = 1.0 if c.source.startswith("detector") else (0.75 if c.source == "static_salience" else 0.65)
            # Spatial novelty multiplier
            w_cov = 1.2 if c.is_new_spatial_coverage else 0.4
            # Scale priority (ultra-small and small get highest priority)
            w_scale = 1.3 if c.scale_bucket in ("ultra_small", "small") else 0.9

            c.priority_score = base_s * w_mod * w_cov * w_scale

        # Rank all candidates jointly
        candidates_sorted = sorted(candidates, key=lambda c: c.priority_score, reverse=True)
        # Select top non-redundant candidates
        kept_candidates: List[AllocatedCandidate] = []
        for cand in candidates_sorted:
            if cand.priority_score < 0.20:
                cand.allocation_status = "rejected"
                cand.rejection_reason = "below_priority_threshold"
                continue

            # Greedy non-redundancy check
            redundant = False
            for k in kept_candidates:
                if compute_iou(cand.box_xyxy, k.box_xyxy) >= 0.40:
                    redundant = True
                    break

            if not redundant:
                kept_candidates.append(cand)
                cand.allocation_status = "allocated"
            else:
                cand.allocation_status = "rejected"
                cand.rejection_reason = "intra_list_redundant"

        boxes = [(c.x1, c.y1, c.x2, c.y2) for c in kept_candidates]
        allocated_crops = cluster_candidate_boxes(
            boxes=boxes,
            image_shape=image_shape,
            max_distance_px=self.cluster_distance_px,
            context_margin=self.context_margin,
            min_crop_size=self.min_crop_size,
            max_crops=max_k,
        )

        retained_det = sum(1 for c in kept_candidates if c.source.startswith("detector"))
        admitted_indep = sum(1 for c in kept_candidates if not c.source.startswith("detector"))
        det_total = sum(1 for c in candidates if c.source.startswith("detector"))
        displaced_det = max(0, det_total - retained_det)

        return CropAllocationDecision(
            frame_idx=self.frame_counter,
            initial_budget=max_k,
            used_budget=len(allocated_crops),
            allocated_crops=allocated_crops,
            all_candidates=candidates,
            retained_detector_count=retained_det,
            admitted_independent_count=admitted_indep,
            displaced_detector_count=displaced_det,
            policy_name="policy_b_confidence_coverage",
        )

    # -------------------------------------------------------------------------
    # Policy C: Independent-Proposal Admission Gate
    # -------------------------------------------------------------------------
    def _allocate_policy_c(
        self,
        candidates: List[AllocatedCandidate],
        image_shape: Tuple[int, int],
    ) -> CropAllocationDecision:
        """Independent candidates admitted only if satisfying strict spatial and evidence admission gates."""
        img_w, img_h = image_shape
        max_k = self.max_crops

        horizon_y_min = self.horizon_band[0] * img_h
        horizon_y_max = self.horizon_band[1] * img_h

        gated_candidates: List[AllocatedCandidate] = []

        for c in candidates:
            if c.source.startswith("detector"):
                c.priority_score = 1.0 + c.score
                c.allocation_status = "allocated"
                gated_candidates.append(c)
            else:
                # Gate 1: Spatial plausibility (must intersect horizon band)
                cy = c.center[1]
                in_horizon = (horizon_y_min - 40.0) <= cy <= (horizon_y_max + 60.0)
                if not in_horizon:
                    c.allocation_status = "rejected"
                    c.rejection_reason = "outside_horizon_gate"
                    continue

                # Gate 2: Non-redundant spatial coverage
                if not c.is_new_spatial_coverage:
                    c.allocation_status = "rejected"
                    c.rejection_reason = "redundant_coverage_gate"
                    continue

                # Gate 3: Geometric aspect ratio (filter wiper blades and puddle lines)
                aspect = c.width / max(1.0, c.height)
                if aspect > 4.5 or aspect < 0.20 or c.height > 300.0:
                    c.allocation_status = "rejected"
                    c.rejection_reason = "aspect_ratio_nuisance_gate"
                    continue

                # Gate 4: Minimum evidence threshold
                if c.score < self.min_independent_score:
                    c.allocation_status = "rejected"
                    c.rejection_reason = "below_evidence_gate"
                    continue

                c.priority_score = 0.5 + c.score * 0.5
                c.allocation_status = "allocated"
                gated_candidates.append(c)

        # Cluster gated candidates
        boxes = [(c.x1, c.y1, c.x2, c.y2) for c in gated_candidates]
        allocated_crops = cluster_candidate_boxes(
            boxes=boxes,
            image_shape=image_shape,
            max_distance_px=self.cluster_distance_px,
            context_margin=self.context_margin,
            min_crop_size=self.min_crop_size,
            max_crops=max_k,
        )

        retained_det = sum(1 for c in gated_candidates if c.source.startswith("detector"))
        admitted_indep = sum(1 for c in gated_candidates if not c.source.startswith("detector"))

        return CropAllocationDecision(
            frame_idx=self.frame_counter,
            initial_budget=max_k,
            used_budget=len(allocated_crops),
            allocated_crops=allocated_crops,
            all_candidates=candidates,
            retained_detector_count=retained_det,
            admitted_independent_count=admitted_indep,
            displaced_detector_count=0,
            policy_name="policy_c_independent_gate",
        )

    # -------------------------------------------------------------------------
    # Policy D: Uncertainty-Aware Allocation
    # -------------------------------------------------------------------------
    def _allocate_policy_d(
        self,
        candidates: List[AllocatedCandidate],
        image_shape: Tuple[int, int],
    ) -> CropAllocationDecision:
        """Prioritizes low-confidence detector candidates and high-entropy regions over redundant high-confidence cues."""
        max_k = self.max_crops

        for c in candidates:
            if c.source == "detector_low":
                # Low-confidence detector detections (0.10 <= conf < 0.35) have the highest uncertainty value
                # A crop here resolves classification doubt
                uncertainty_weight = 1.50
                c.priority_score = 0.50 + (0.35 - c.score) * 2.0 + uncertainty_weight
            elif c.source == "detector_high":
                # High-confidence detector boxes already have good certainty; lower urgency unless small
                if c.scale_bucket in ("ultra_small", "small"):
                    c.priority_score = 1.40 + c.score * 0.5
                else:
                    c.priority_score = 0.60 + c.score * 0.4
            elif c.source == "static_salience":
                # Salience with no detector box is uncertain
                c.priority_score = 1.0 + c.score * (1.0 if c.is_new_spatial_coverage else 0.3)
            elif c.source == "frame_diff":
                c.priority_score = 0.9 + c.score * (1.0 if c.is_new_spatial_coverage else 0.3)
            else:
                c.priority_score = 0.40

        candidates_sorted = sorted(candidates, key=lambda c: c.priority_score, reverse=True)
        selected_candidates: List[AllocatedCandidate] = []

        for cand in candidates_sorted:
            if cand.priority_score < 0.50:
                cand.allocation_status = "rejected"
                cand.rejection_reason = "low_uncertainty_value"
                continue

            # Overlap suppression with already selected
            redundant = False
            for sc in selected_candidates:
                if compute_iou(cand.box_xyxy, sc.box_xyxy) >= 0.35:
                    redundant = True
                    break

            if not redundant:
                selected_candidates.append(cand)
                cand.allocation_status = "allocated"
            else:
                cand.allocation_status = "rejected"
                cand.rejection_reason = "redundant_uncertainty_zone"

        boxes = [(c.x1, c.y1, c.x2, c.y2) for c in selected_candidates]
        allocated_crops = cluster_candidate_boxes(
            boxes=boxes,
            image_shape=image_shape,
            max_distance_px=self.cluster_distance_px,
            context_margin=self.context_margin,
            min_crop_size=self.min_crop_size,
            max_crops=max_k,
        )

        retained_det = sum(1 for c in selected_candidates if c.source.startswith("detector"))
        admitted_indep = sum(1 for c in selected_candidates if not c.source.startswith("detector"))
        det_total = sum(1 for c in candidates if c.source.startswith("detector"))
        displaced_det = max(0, det_total - retained_det)

        return CropAllocationDecision(
            frame_idx=self.frame_counter,
            initial_budget=max_k,
            used_budget=len(allocated_crops),
            allocated_crops=allocated_crops,
            all_candidates=candidates,
            retained_detector_count=retained_det,
            admitted_independent_count=admitted_indep,
            displaced_detector_count=displaced_det,
            policy_name="policy_d_uncertainty_aware",
        )

    # -------------------------------------------------------------------------
    # Policy E: Adaptive Crop Budget
    # -------------------------------------------------------------------------
    def _allocate_policy_e(
        self,
        candidates: List[AllocatedCandidate],
        image_shape: Tuple[int, int],
    ) -> CropAllocationDecision:
        """Dynamic crop budget K in {0, 1, 2, 3}. Never allocates a crop unless a candidate exceeds the value hurdle."""
        max_k = self.max_crops

        # Filter candidates meeting strict value threshold
        worthy_candidates: List[AllocatedCandidate] = []
        for c in candidates:
            # Detector small/uncertain: threshold 0.15
            if c.source.startswith("detector") and c.score >= 0.15:
                c.priority_score = 1.0 + c.score
                c.allocation_status = "allocated"
                worthy_candidates.append(c)
            # Independent non-redundant: threshold min_independent_score
            elif not c.source.startswith("detector") and c.is_new_spatial_coverage and c.score >= self.min_independent_score:
                c.priority_score = 0.5 + c.score * 0.5
                c.allocation_status = "allocated"
                worthy_candidates.append(c)
            else:
                c.allocation_status = "rejected"
                c.rejection_reason = "below_adaptive_value_hurdle"

        # Determine dynamic budget: if 0 worthy candidates, K=0! If 1 isolated cluster, K=1!
        if not worthy_candidates:
            dynamic_k = 0
            allocated_crops = []
        else:
            boxes = [(c.x1, c.y1, c.x2, c.y2) for c in worthy_candidates]
            # First cluster with max_crops = max_k
            pre_crops = cluster_candidate_boxes(
                boxes=boxes,
                image_shape=image_shape,
                max_distance_px=self.cluster_distance_px,
                context_margin=self.context_margin,
                min_crop_size=self.min_crop_size,
                max_crops=max_k,
            )
            # Dynamic budget is exactly the number of distinct clusters formed (could be 1 or 2)
            dynamic_k = min(len(pre_crops), max_k)
            allocated_crops = pre_crops[:dynamic_k]

        retained_det = sum(1 for c in worthy_candidates if c.source.startswith("detector"))
        admitted_indep = sum(1 for c in worthy_candidates if not c.source.startswith("detector"))

        return CropAllocationDecision(
            frame_idx=self.frame_counter,
            initial_budget=max_k,
            used_budget=len(allocated_crops),
            allocated_crops=allocated_crops,
            all_candidates=candidates,
            retained_detector_count=retained_det,
            admitted_independent_count=admitted_indep,
            displaced_detector_count=0,
            policy_name="policy_e_adaptive_budget",
        )

    # -------------------------------------------------------------------------
    # Policy J: Best Priority-Gated Combination
    # -------------------------------------------------------------------------
    def _allocate_policy_j(
        self,
        candidates: List[AllocatedCandidate],
        image_shape: Tuple[int, int],
    ) -> CropAllocationDecision:
        """Best combination: Detector-first reservation + multi-criteria admission gating + adaptive budgeting."""
        img_w, img_h = image_shape
        max_k = self.max_crops

        horizon_y_min = self.horizon_band[0] * img_h
        horizon_y_max = self.horizon_band[1] * img_h

        detector_cands = [c for c in candidates if c.source.startswith("detector")]
        independent_cands = [c for c in candidates if not c.source.startswith("detector")]

        allocated_crops: List[CropRegion] = []
        retained_det = 0
        admitted_indep = 0

        # Step 1: Allocate detector proposals first (if any exist)
        if detector_cands:
            det_boxes = [(c.x1, c.y1, c.x2, c.y2) for c in detector_cands if c.score >= 0.15]
            if det_boxes:
                det_crops = cluster_candidate_boxes(
                    boxes=det_boxes,
                    image_shape=image_shape,
                    max_distance_px=self.cluster_distance_px,
                    context_margin=self.context_margin,
                    min_crop_size=self.min_crop_size,
                    max_crops=max_k,
                )
                allocated_crops.extend(det_crops)
                retained_det = len(det_boxes)
                for c in detector_cands:
                    c.priority_score = 1.0 + c.score
                    c.allocation_status = "allocated"

        # Step 2: Adaptive check — do we have remaining budget?
        remaining_budget = max_k - len(allocated_crops)

        if remaining_budget > 0 and independent_cands:
            # Step 3: Filter independent proposals through strict admission gates
            eligible_indep: List[AllocatedCandidate] = []
            for ic in independent_cands:
                # Horizon check
                cy = ic.center[1]
                if not ((horizon_y_min - 30.0) <= cy <= (horizon_y_max + 40.0)):
                    ic.allocation_status = "rejected"
                    ic.rejection_reason = "outside_horizon_band"
                    continue

                # Evidence threshold
                if ic.score < self.min_independent_score:
                    ic.allocation_status = "rejected"
                    ic.rejection_reason = "weak_evidence_score"
                    continue

                # Redundancy check with existing crops
                overlaps_existing = False
                for cr in allocated_crops:
                    iou = compute_iou(ic.box_xyxy, cr.box_xyxy)
                    if iou >= self.overlap_redundancy_thresh:
                        overlaps_existing = True
                        break

                if overlaps_existing:
                    ic.allocation_status = "rejected"
                    ic.rejection_reason = "redundant_with_allocated_crop"
                    continue

                # Aspect ratio filter for wiper / puddle reflection
                aspect = ic.width / max(1.0, ic.height)
                if aspect > 4.0 or aspect < 0.25:
                    ic.allocation_status = "rejected"
                    ic.rejection_reason = "nuisance_aspect_ratio"
                    continue

                ic.priority_score = 0.50 + ic.score * 0.50
                eligible_indep.append(ic)

            if eligible_indep:
                # Sort by priority score
                eligible_indep.sort(key=lambda c: c.priority_score, reverse=True)
                indep_boxes = [(c.x1, c.y1, c.x2, c.y2) for c in eligible_indep]
                indep_crops = cluster_candidate_boxes(
                    boxes=indep_boxes,
                    image_shape=image_shape,
                    max_distance_px=self.cluster_distance_px,
                    context_margin=self.context_margin,
                    min_crop_size=self.min_crop_size,
                    max_crops=remaining_budget,
                )
                allocated_crops.extend(indep_crops)
                admitted_indep = len(indep_crops)
                for c in eligible_indep[:admitted_indep]:
                    c.allocation_status = "allocated"

        # Adaptive Budget: If total allocated crops is 0 (clean frame, no small targets), budget used is 0!
        return CropAllocationDecision(
            frame_idx=self.frame_counter,
            initial_budget=max_k,
            used_budget=len(allocated_crops),
            allocated_crops=allocated_crops,
            all_candidates=candidates,
            retained_detector_count=retained_det,
            admitted_independent_count=admitted_indep,
            displaced_detector_count=0,
            policy_name="policy_j_best_combination",
        )
