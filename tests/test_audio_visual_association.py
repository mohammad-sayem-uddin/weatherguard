"""Unit tests for Cautious Audiovisual Association Engine (Mission 22 - Phase 6)."""
import pytest
from src.multimodal.association import (
    AssociationResult,
    AssociationState,
    CautiousAssociationEngine,
)
from src.multimodal.fusion_engine import AcousticEvidence, VisualEvidence


def test_clean_scene():
    engine = CautiousAssociationEngine()
    vis = VisualEvidence(person_detected=False)
    aud = AcousticEvidence(distress_prob=0.10)

    res = engine.associate(vis, aud)
    assert res.state == AssociationState.NO_TARGET
    assert res.target_box is None
    assert res.is_spatially_grounded is False
    assert res.requires_human_verification is False


def test_audio_only_no_visible_person():
    engine = CautiousAssociationEngine()
    vis = VisualEvidence(person_detected=False, person_boxes=[])
    aud = AcousticEvidence(distress_prob=0.90, calibrated_threshold=0.54)

    res = engine.associate(vis, aud)
    assert res.state == AssociationState.UNASSOCIATED_AUDIO_ONLY
    assert res.target_box is None  # Strict rule: never fabricate coordinates
    assert res.is_spatially_grounded is False
    assert res.search_command == "ENGAGE_SPIRAL_SEARCH_PATTERN"
    assert res.requires_human_verification is True


def test_single_person_corroborated():
    engine = CautiousAssociationEngine()
    box = (100, 150, 200, 300)
    vis = VisualEvidence(person_detected=True, max_person_confidence=0.88, person_boxes=[box])
    aud = AcousticEvidence(distress_prob=0.92, calibrated_threshold=0.54)

    res = engine.associate(vis, aud, timestamp_delta_ms=50.0)
    assert res.state == AssociationState.CORROBORATED_SINGLE_PERSON
    assert res.is_spatially_grounded is True
    assert res.target_box == box
    assert res.requires_human_verification is False
    assert res.search_command == "CONFIRMED_VICTIM_LOCATED_DISPATCH_RESCUE"


def test_multiple_people_ambiguity_no_arbitrary_box():
    engine = CautiousAssociationEngine()
    box1 = (50, 50, 100, 150)
    box2 = (200, 200, 250, 320)
    vis = VisualEvidence(person_detected=True, max_person_confidence=0.85, person_boxes=[box1, box2])
    aud = AcousticEvidence(distress_prob=0.91, calibrated_threshold=0.54)

    res = engine.associate(vis, aud)
    # MUST NOT pick box1 or box2 arbitrarily!
    assert res.state == AssociationState.AMBIGUOUS_MULTIPLE_PERSONS
    assert res.target_box is None
    assert res.is_spatially_grounded is False
    assert len(res.candidate_boxes) == 2
    assert res.requires_human_verification is True
    assert res.search_command == "HOVER_AND_MAINTAIN_VISUAL_SURVEILLANCE"


def test_temporal_misalignment_rejection():
    engine = CautiousAssociationEngine(max_temporal_window_ms=1000.0)
    vis = VisualEvidence(person_detected=True, person_boxes=[(10, 10, 50, 50)])
    aud = AcousticEvidence(distress_prob=0.85)

    # 2500 ms delta exceeds 1000 ms limit
    res = engine.associate(vis, aud, timestamp_delta_ms=2500.0)
    assert res.state == AssociationState.TEMPORAL_MISALIGNMENT
    assert res.is_spatially_grounded is False
    assert res.search_command == "RESYNCHRONIZE_BUFFERS"


def test_silent_person_vision_only():
    engine = CautiousAssociationEngine()
    box = (50, 60, 100, 200)
    vis = VisualEvidence(person_detected=True, max_person_confidence=0.80, person_boxes=[box])
    aud = AcousticEvidence(distress_prob=0.12, calibrated_threshold=0.54)

    res = engine.associate(vis, aud)
    assert res.state == AssociationState.UNASSOCIATED_VISION_ONLY
    assert res.is_spatially_grounded is True
    assert res.target_box == box
