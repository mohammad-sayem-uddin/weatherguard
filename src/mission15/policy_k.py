"""WeatherGuard Policy K — Temporal-Horizon Priority Allocation Controller.

Module: src/mission15/policy_k.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-03
Mission: Mission 15 — Temporal Evidence Accumulation & Horizon-Anchored Candidate Generation

Design Principles:
1. Strict Reference Preservation: Leaves historical Policy J untouched as the frozen benchmark.
2. Multi-Tier Priority-Gated Admission:
   - Tier 1: Detector proposals (high and low confidence) - Rank 1.
   - Tier 2: Dual-Validated Candidates (Temporal + Horizon Intersection) - Rank 2.
   - Tier 3: Causal Temporal Evidence Accumulator Candidates - Rank 3.
   - Tier 4: Adaptive Horizon-Anchored Candidates - Rank 4.
3. Anti-Dilution Safeguards:
   - Capacity Reservation: Independent proposals never displace validated Tier 1 detector proposals.
   - Spatial Redundancy Suppression: Proposals overlapping existing detector boxes (IoU > 0.20) are pruned.
   - Admission Thresholding: Low-evidence proposals are rejected before crop placement.
4. Auditable Logging: Every admitted and rejected proposal is recorded with an explicit reason tag.
5. Edge-Optimized Clustering: Aggregates candidates into at most K crops (448x448) within budget limits.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.mission15.temporal_evidence import (
    CausalTemporalEvidenceAccumulator,
    TemporalEvidenceConfig,
)
from src.mission15.horizon_anchor import (
    AdaptiveHorizonAnchor,
    HorizonAnchorConfig,
)
from src.priority_allocation import (
    AllocatedCandidate,
    CandidateProposal,
    DetectionBox,
    PriorityAllocationController,
    compute_iou,
)


@dataclass
class PolicyKConfig:
    """Hyperparameters for WeatherGuard Policy K."""
    max_crops: int = 2
    crop_size: int = 448
    cluster_distance_px: float = 120.0
    context_margin: float = 0.20
    min_independent_score: float = 0.25
    redundancy_iou_thresh: float = 0.20
    dual_validation_iou_thresh: float = 0.15  # IoU overlap between temporal and horizon candidates
    dual_validation_boost: float = 1.35      # Score boost for candidates supported by both
    temporal_config: TemporalEvidenceConfig = field(default_factory=TemporalEvidenceConfig)
    horizon_config: HorizonAnchorConfig = field(default_factory=HorizonAnchorConfig)


@dataclass
class CandidateAdmissionRecord:
    """Detailed record of admission or rejection for each evaluated proposal."""
    frame_index: int
    candidate_id: str
    source: str
    box: Tuple[float, float, float, float]
    score: float
    admitted: bool
    rejection_reason: Optional[str]
    assigned_tier: int
    cluster_id: Optional[int] = None
    crop_allocated: bool = False


class PolicyKController:
    """WeatherGuard Policy K — Temporal-Horizon Priority Allocation Controller."""

    def __init__(self, config: Optional[PolicyKConfig] = None):
        self.config = config or PolicyKConfig()
        self.temporal_acc = CausalTemporalEvidenceAccumulator(self.config.temporal_config)
        self.horizon_anchor = AdaptiveHorizonAnchor(self.config.horizon_config)
        self.frame_counter: int = 0
        self.current_sequence_id: Optional[str] = None
        self.admission_logs: List[CandidateAdmissionRecord] = []

    def reset(self) -> None:
        """Resets all internal temporal states across video sequence boundaries."""
        self.temporal_acc.reset()
        self.horizon_anchor.reset()
        self.frame_counter = 0
        self.current_sequence_id = None
        self.admission_logs.clear()

    def process_and_allocate(
        self,
        frame_rgb: np.ndarray,
        sequence_id: str,
        detector_boxes: Optional[Sequence[DetectionBox]] = None,
        frame_id: Optional[str] = None,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[AllocatedCandidate], Dict[str, float]]:
        """Processes frame, gathers multi-tier candidates, gates admissions, and allocates crops.
        
        Args:
            frame_rgb: RGB frame array (H x W x 3).
            sequence_id: Video sequence identifier.
            detector_boxes: Initial base detector detections (e.g. from YOLOv8n-640).
            frame_id: Frame name or timestamp.
            
        Returns:
            crops: List of (x1, y1, x2, y2) integer crop coordinates (max K crops).
            admitted_candidates: List of admitted AllocatedCandidate objects.
            timings: Execution timing breakdown in milliseconds.
        """
        # Strict sequence isolation
        if self.current_sequence_id != sequence_id:
            self.reset()
            self.current_sequence_id = sequence_id

        h, w = frame_rgb.shape[:2]
        timings: Dict[str, float] = {}
        all_candidates: List[AllocatedCandidate] = []
        cand_idx = 0

        # Helper scale classification
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

        # -------------------------------------------------------------
        # Tier 1: Detector Proposals (High and Low Confidence)
        # -------------------------------------------------------------
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
                        metadata={"class_name": d.class_name, "raw_confidence": d.confidence, "tier": 1},
                        max_detector_overlap_iou=1.0,
                        is_new_spatial_coverage=True,
                    )
                    all_candidates.append(cand)
                    detector_cand_boxes.append((d.x1, d.y1, d.x2, d.y2))
                    cand_idx += 1

                    self.admission_logs.append(CandidateAdmissionRecord(
                        frame_index=self.frame_counter,
                        candidate_id=cand.candidate_id,
                        source=cand.source,
                        box=(cand.x1, cand.y1, cand.x2, cand.y2),
                        score=cand.score,
                        admitted=True,
                        rejection_reason=None,
                        assigned_tier=1,
                    ))
        timings["tier1_detector_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Tier 3: Causal Temporal Evidence Accumulator
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        temporal_props = self.temporal_acc.process_frame(frame_rgb, sequence_id=sequence_id, frame_id=frame_id)
        timings["tier3_temporal_acc_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Tier 4: Adaptive Horizon Anchor
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        horizon_props = self.horizon_anchor.process_frame(frame_rgb, sequence_id=sequence_id, frame_id=frame_id)
        timings["tier4_horizon_anchor_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Tier 2: Dual-Validated Intersection (Temporal + Horizon)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        dual_matched_temporal_indices = set()
        dual_matched_horizon_indices = set()

        for t_idx, tp in enumerate(temporal_props):
            t_box = (tp.x1, tp.y1, tp.x2, tp.y2)
            for h_idx, hp in enumerate(horizon_props):
                h_box = (hp.x1, hp.y1, hp.x2, hp.y2)
                iou = compute_iou(t_box, h_box)
                if iou >= self.config.dual_validation_iou_thresh:
                    dual_matched_temporal_indices.add(t_idx)
                    dual_matched_horizon_indices.add(h_idx)

                    # Enclose both in unified box
                    ux1 = min(tp.x1, hp.x1)
                    uy1 = min(tp.y1, hp.y1)
                    ux2 = max(tp.x2, hp.x2)
                    uy2 = max(tp.y2, hp.y2)
                    u_area = (ux2 - ux1) * (uy2 - uy1)
                    joint_score = min(1.0, max(tp.score, hp.score) * self.config.dual_validation_boost)

                    # Redundancy check with Tier 1 detector proposals
                    max_det_iou = max([compute_iou((ux1, uy1, ux2, uy2), db) for db in detector_cand_boxes], default=0.0)
                    if max_det_iou > self.config.redundancy_iou_thresh:
                        self.admission_logs.append(CandidateAdmissionRecord(
                            frame_index=self.frame_counter,
                            candidate_id=f"dual_{cand_idx}",
                            source="dual_temporal_horizon",
                            box=(ux1, uy1, ux2, uy2),
                            score=joint_score,
                            admitted=False,
                            rejection_reason="redundant_with_detector",
                            assigned_tier=2,
                        ))
                    elif joint_score < self.config.min_independent_score:
                        self.admission_logs.append(CandidateAdmissionRecord(
                            frame_index=self.frame_counter,
                            candidate_id=f"dual_{cand_idx}",
                            source="dual_temporal_horizon",
                            box=(ux1, uy1, ux2, uy2),
                            score=joint_score,
                            admitted=False,
                            rejection_reason="below_admission_threshold",
                            assigned_tier=2,
                        ))
                    else:
                        cand = AllocatedCandidate(
                            candidate_id=f"dual_{cand_idx}",
                            source="dual_temporal_horizon",
                            x1=ux1,
                            y1=uy1,
                            x2=ux2,
                            y2=uy2,
                            score=joint_score,
                            scale_bucket=get_scale(u_area),
                            frame_idx=self.frame_counter,
                            timestamp=time.time(),
                            metadata={"tier": 2, "temporal_score": tp.score, "horizon_score": hp.score},
                            max_detector_overlap_iou=max_det_iou,
                            is_new_spatial_coverage=True,
                        )
                        all_candidates.append(cand)
                        cand_idx += 1
                        self.admission_logs.append(CandidateAdmissionRecord(
                            frame_index=self.frame_counter,
                            candidate_id=cand.candidate_id,
                            source=cand.source,
                            box=(cand.x1, cand.y1, cand.x2, cand.y2),
                            score=cand.score,
                            admitted=True,
                            rejection_reason=None,
                            assigned_tier=2,
                        ))

        # -------------------------------------------------------------
        # Remaining Tier 3 Temporal Proposals
        # -------------------------------------------------------------
        for t_idx, tp in enumerate(temporal_props):
            if t_idx in dual_matched_temporal_indices:
                continue
            t_box = (tp.x1, tp.y1, tp.x2, tp.y2)
            max_det_iou = max([compute_iou(t_box, db) for db in detector_cand_boxes], default=0.0)

            if max_det_iou > self.config.redundancy_iou_thresh:
                self.admission_logs.append(CandidateAdmissionRecord(
                    frame_index=self.frame_counter,
                    candidate_id=f"temp_{cand_idx}",
                    source="temporal_accumulator",
                    box=t_box,
                    score=tp.score,
                    admitted=False,
                    rejection_reason="redundant_with_detector",
                    assigned_tier=3,
                ))
            elif tp.score < self.config.min_independent_score:
                self.admission_logs.append(CandidateAdmissionRecord(
                    frame_index=self.frame_counter,
                    candidate_id=f"temp_{cand_idx}",
                    source="temporal_accumulator",
                    box=t_box,
                    score=tp.score,
                    admitted=False,
                    rejection_reason="below_admission_threshold",
                    assigned_tier=3,
                ))
            else:
                cand = AllocatedCandidate(
                    candidate_id=f"temp_{cand_idx}",
                    source="temporal_accumulator",
                    x1=tp.x1,
                    y1=tp.y1,
                    x2=tp.x2,
                    y2=tp.y2,
                    score=tp.score,
                    scale_bucket=get_scale(tp.area),
                    frame_idx=self.frame_counter,
                    timestamp=time.time(),
                    metadata={"tier": 3, "metadata": tp.metadata},
                    max_detector_overlap_iou=max_det_iou,
                    is_new_spatial_coverage=True,
                )
                all_candidates.append(cand)
                cand_idx += 1
                self.admission_logs.append(CandidateAdmissionRecord(
                    frame_index=self.frame_counter,
                    candidate_id=cand.candidate_id,
                    source=cand.source,
                    box=(cand.x1, cand.y1, cand.x2, cand.y2),
                    score=cand.score,
                    admitted=True,
                    rejection_reason=None,
                    assigned_tier=3,
                ))

        # -------------------------------------------------------------
        # Remaining Tier 4 Horizon Proposals
        # -------------------------------------------------------------
        for h_idx, hp in enumerate(horizon_props):
            if h_idx in dual_matched_horizon_indices:
                continue
            h_box = (hp.x1, hp.y1, hp.x2, hp.y2)
            max_det_iou = max([compute_iou(h_box, db) for db in detector_cand_boxes], default=0.0)

            if max_det_iou > self.config.redundancy_iou_thresh:
                self.admission_logs.append(CandidateAdmissionRecord(
                    frame_index=self.frame_counter,
                    candidate_id=f"horiz_{cand_idx}",
                    source="horizon_anchor",
                    box=h_box,
                    score=hp.score,
                    admitted=False,
                    rejection_reason="redundant_with_detector",
                    assigned_tier=4,
                ))
            elif hp.score < self.config.min_independent_score:
                self.admission_logs.append(CandidateAdmissionRecord(
                    frame_index=self.frame_counter,
                    candidate_id=f"horiz_{cand_idx}",
                    source="horizon_anchor",
                    box=h_box,
                    score=hp.score,
                    admitted=False,
                    rejection_reason="below_admission_threshold",
                    assigned_tier=4,
                ))
            else:
                cand = AllocatedCandidate(
                    candidate_id=f"horiz_{cand_idx}",
                    source="horizon_anchor",
                    x1=hp.x1,
                    y1=hp.y1,
                    x2=hp.x2,
                    y2=hp.y2,
                    score=hp.score,
                    scale_bucket=get_scale(hp.area),
                    frame_idx=self.frame_counter,
                    timestamp=time.time(),
                    metadata={"tier": 4, "metadata": hp.metadata},
                    max_detector_overlap_iou=max_det_iou,
                    is_new_spatial_coverage=True,
                )
                all_candidates.append(cand)
                cand_idx += 1
                self.admission_logs.append(CandidateAdmissionRecord(
                    frame_index=self.frame_counter,
                    candidate_id=cand.candidate_id,
                    source=cand.source,
                    box=(cand.x1, cand.y1, cand.x2, cand.y2),
                    score=cand.score,
                    admitted=True,
                    rejection_reason=None,
                    assigned_tier=4,
                ))
        timings["tier2_dual_and_gating_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Priority-Aware Crop Allocation & Spatial Clustering
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        crops = self._allocate_crops(all_candidates, w, h)
        timings["crop_allocation_ms"] = (time.perf_counter() - t0) * 1000.0

        # Mark in admission logs which candidates received crop coverage
        for record in self.admission_logs[-len(all_candidates):]:
            if record.admitted:
                for c in crops:
                    cx1, cy1, cx2, cy2 = c
                    bx1, by1, bx2, by2 = record.box
                    # Check if candidate is enclosed inside crop
                    if cx1 <= bx1 and cy1 <= by1 and cx2 >= bx2 and cy2 >= by2:
                        record.crop_allocated = True
                        break

        self.frame_counter += 1
        return crops, all_candidates, timings

    def _allocate_crops(
        self,
        candidates: List[AllocatedCandidate],
        img_w: int,
        img_h: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Allocates up to K high-resolution crops honoring priority ranks."""
        if not candidates:
            return []

        # Sort candidates hierarchically:
        # Tier 1 (rank 1) -> Tier 2 (rank 2) -> Tier 3 (rank 3) -> Tier 4 (rank 4)
        # Within tier: sort by score descending
        def get_priority_key(cand: AllocatedCandidate) -> Tuple[int, float]:
            tier = cand.metadata.get("tier", 4)
            return (tier, -cand.score)

        sorted_cands = sorted(candidates, key=get_priority_key)

        crops: List[Tuple[int, int, int, int]] = []
        cluster_dist = self.config.cluster_distance_px
        crop_sz = self.config.crop_size

        for cand in sorted_cands:
            if len(crops) >= self.config.max_crops:
                break

            cand_cx = (cand.x1 + cand.x2) / 2.0
            cand_cy = (cand.y1 + cand.y2) / 2.0

            # Check if this candidate is already covered by an existing crop
            already_covered = False
            for c in crops:
                cx1, cy1, cx2, cy2 = c
                if cx1 <= cand.x1 and cy1 <= cand.y1 and cx2 >= cand.x2 and cy2 >= cand.y2:
                    already_covered = True
                    break
                # Check proximity to existing crop center
                crop_center_x = (cx1 + cx2) / 2.0
                crop_center_y = (cy1 + cy2) / 2.0
                if np.hypot(cand_cx - crop_center_x, cand_cy - crop_center_y) < cluster_dist:
                    already_covered = True
                    break

            if already_covered:
                continue

            # Place square crop of size crop_sz centered at (cand_cx, cand_cy)
            half_sz = crop_sz / 2.0
            x1 = int(np.clip(cand_cx - half_sz, 0, img_w - crop_sz))
            y1 = int(np.clip(cand_cy - half_sz, 0, img_h - crop_sz))
            x2 = x1 + crop_sz
            y2 = y1 + crop_sz

            crops.append((x1, y1, x2, y2))

        return crops

    def export_admission_logs(self, output_path: str) -> int:
        """Exports all recorded admission and rejection logs to a JSONL file."""
        count = 0
        with open(output_path, "w") as f:
            for rec in self.admission_logs:
                entry = dataclasses.asdict(rec)
                f.write(json.dumps(entry) + "\n")
                count += 1
        return count
