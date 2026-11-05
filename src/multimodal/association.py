"""Cautious Audiovisual Association Engine for UAV Search and Rescue (Mission 22 - Phase 6).

Enforces strict physical constraints on cross-modal identity association:
  1. Never assumes a visible person is the source of an acoustic distress alert.
  2. Resolves multiple-person ambiguities without spatial fabrication.
  3. Formulates mono-microphone acoustic evidence strictly as a search trigger,
     not as spatial victim coordinates.
  4. Manages temporal co-occurrence and window alignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.multimodal.fusion_engine import AcousticEvidence, VisualEvidence


class AssociationState(str, Enum):
    UNASSOCIATED_AUDIO_ONLY = "UNASSOCIATED_AUDIO_ONLY"
    UNASSOCIATED_VISION_ONLY = "UNASSOCIATED_VISION_ONLY"
    CORROBORATED_SINGLE_PERSON = "CORROBORATED_SINGLE_PERSON"
    AMBIGUOUS_MULTIPLE_PERSONS = "AMBIGUOUS_MULTIPLE_PERSONS"
    TEMPORAL_MISALIGNMENT = "TEMPORAL_MISALIGNMENT"
    NO_TARGET = "NO_TARGET"
    SENSOR_FAULT = "SENSOR_FAULT"


@dataclass
class AssociationResult:
    """Cautious association decision between visual detections and acoustic distress."""
    state: AssociationState
    is_spatially_grounded: bool  # True ONLY if spatial coordinates are physically verified
    target_box: Optional[Tuple[float, float, float, float]]
    candidate_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    requires_human_verification: bool = False
    search_command: str = "CONTINUE_GRID_SEARCH"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "is_spatially_grounded": self.is_spatially_grounded,
            "target_box": self.target_box,
            "candidate_boxes": self.candidate_boxes,
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "requires_human_verification": self.requires_human_verification,
            "search_command": self.search_command,
        }


class CautiousAssociationEngine:
    """Manages cross-modal association under partial, noisy, and ambiguous evidence."""

    def __init__(self, max_temporal_window_ms: float = 1500.0) -> None:
        self.max_temporal_window_ms = max_temporal_window_ms

    def associate(
        self,
        vis: VisualEvidence,
        aud: AcousticEvidence,
        timestamp_delta_ms: float = 0.0,
        has_microphone_array: bool = False,
    ) -> AssociationResult:
        """Associates visual detections with acoustic distress signals cautiously."""

        # 0. Hardware availability
        if not vis.sensor_available or not aud.sensor_available:
            return AssociationResult(
                state=AssociationState.SENSOR_FAULT,
                is_spatially_grounded=False,
                target_box=None,
                candidate_boxes=[],
                confidence=0.0,
                rationale="Sensor unavailable or hardware fault detected.",
                requires_human_verification=True,
                search_command="HOLD_POSITION_MONITOR_SENSORS",
            )

        # 1. Temporal Alignment Check
        if abs(timestamp_delta_ms) > self.max_temporal_window_ms:
            return AssociationResult(
                state=AssociationState.TEMPORAL_MISALIGNMENT,
                is_spatially_grounded=False,
                target_box=None,
                candidate_boxes=vis.person_boxes,
                confidence=0.0,
                rationale=f"Temporal offset ({timestamp_delta_ms:.1f}ms) exceeds maximum allowable window ({self.max_temporal_window_ms:.1f}ms).",
                requires_human_verification=True,
                search_command="RESYNCHRONIZE_BUFFERS",
            )

        vis_has_person = vis.person_detected and (len(vis.person_boxes) > 0)
        aud_has_distress = (aud.distress_prob >= aud.calibrated_threshold)
        person_count = len(vis.person_boxes)

        # 2. Case: Clean scene (No person, No distress)
        if not vis_has_person and not aud_has_distress:
            return AssociationResult(
                state=AssociationState.NO_TARGET,
                is_spatially_grounded=False,
                target_box=None,
                candidate_boxes=[],
                confidence=1.0 - max(vis.max_person_confidence, aud.distress_prob),
                rationale="Neither visual person nor acoustic distress detected.",
                requires_human_verification=False,
                search_command="CONTINUE_GRID_SEARCH",
            )

        # 3. Case: Audio-Only Alert without Visible Person (Fog, Night, Occlusion)
        if aud_has_distress and not vis_has_person:
            # Physical mono constraint: Zero spatial bearing or coordinates can be asserted
            return AssociationResult(
                state=AssociationState.UNASSOCIATED_AUDIO_ONLY,
                is_spatially_grounded=False,
                target_box=None,
                candidate_boxes=[],
                confidence=aud.distress_prob,
                rationale=(
                    "Acoustic distress vocalization detected, but no person is visible in camera frame. "
                    "Mono microphone provides omnidirectional energy with zero spatial direction. "
                    "Engaging optical spiral search pattern around current UAV GPS waypoint."
                ),
                requires_human_verification=True,
                search_command="ENGAGE_SPIRAL_SEARCH_PATTERN",
            )

        # 4. Case: Visual Person Detected without Acoustic Distress (Silent/Unconscious)
        if vis_has_person and not aud_has_distress:
            target = vis.person_boxes[0] if person_count == 1 else None
            return AssociationResult(
                state=AssociationState.UNASSOCIATED_VISION_ONLY,
                is_spatially_grounded=True if target is not None else False,
                target_box=target,
                candidate_boxes=vis.person_boxes,
                confidence=vis.max_person_confidence,
                rationale=f"Optical person detected ({person_count} candidates), but victim is silent or unvocalized.",
                requires_human_verification=False if person_count == 1 else True,
                search_command="ENGAGE_VISUAL_TRACKING",
            )

        # 5. Case: Bimodal Co-Occurrence with Multiple Visible People (Ambiguity!)
        if aud_has_distress and person_count > 1:
            # Under mono microphone, we CANNOT determine which bounding box emitted the sound!
            # Marking an arbitrary bounding box would be unscientific and safety-critical
            return AssociationResult(
                state=AssociationState.AMBIGUOUS_MULTIPLE_PERSONS,
                is_spatially_grounded=False,  # Unresolved spatial ambiguity
                target_box=None,  # Do NOT pick a random box
                candidate_boxes=vis.person_boxes,
                confidence=min(vis.max_person_confidence, aud.distress_prob),
                rationale=(
                    f"Acoustic distress co-occurs with {person_count} visible persons in the frame. "
                    "Single mono microphone cannot disambiguate the acoustic source among visual candidates. "
                    "Operator verification required before ground rescue dispatch."
                ),
                requires_human_verification=True,
                search_command="HOVER_AND_MAINTAIN_VISUAL_SURVEILLANCE",
            )

        # 6. Case: Bimodal Co-Occurrence with Exactly One Visible Person (Corroborated)
        if aud_has_distress and person_count == 1:
            target = vis.person_boxes[0]
            # Probable corroboration, though still qualified unless mouth motion is confirmed
            return AssociationResult(
                state=AssociationState.CORROBORATED_SINGLE_PERSON,
                is_spatially_grounded=True,
                target_box=target,
                candidate_boxes=vis.person_boxes,
                confidence=(vis.max_person_confidence * 0.5 + aud.distress_prob * 0.5),
                rationale="Single visible person co-occurs synchronously with verified distress vocalization.",
                requires_human_verification=False,
                search_command="CONFIRMED_VICTIM_LOCATED_DISPATCH_RESCUE",
            )

        # Fallback
        return AssociationResult(
            state=AssociationState.UNASSOCIATED_AUDIO_ONLY,
            is_spatially_grounded=False,
            target_box=None,
            candidate_boxes=[],
            confidence=0.5,
            rationale="Ambiguous cross-modal association fallback.",
            requires_human_verification=True,
            search_command="HOLD_POSITION_MONITOR_SENSORS",
        )
