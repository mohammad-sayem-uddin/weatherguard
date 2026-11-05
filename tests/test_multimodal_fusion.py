"""Unit tests for Multi-Modal Acoustic-Visual Fusion Engine (Mission 21)."""
import pytest
from src.multimodal.fusion_engine import (
    AcousticEvidence,
    MultiModalDecision,
    MultiModalDecisionEngine,
    MultiModalFusionConfig,
    VisualEvidence,
)


def test_fusion_engine_initialization():
    engine = MultiModalDecisionEngine()
    assert engine.config.visual_conf_threshold == 0.25
    assert engine.config.audio_calibrated_threshold == 0.54


def test_vision_only_baseline():
    engine = MultiModalDecisionEngine()

    # Person detected above threshold
    vis_alert = VisualEvidence(person_detected=True, max_person_confidence=0.82, person_boxes=[(10, 20, 50, 100)])
    dec = engine.evaluate_vision_only(vis_alert)
    assert dec.alert_state == "ALERT"
    assert dec.alert_score == 0.82
    assert dec.victim_coordinates is not None
    assert dec.spatial_bearing_valid is False

    # No person detected
    vis_neg = VisualEvidence(person_detected=False, max_person_confidence=0.0)
    dec_neg = engine.evaluate_vision_only(vis_neg)
    assert dec_neg.alert_state == "NO_ALERT"
    assert dec_neg.victim_coordinates is None


def test_audio_only_baseline():
    engine = MultiModalDecisionEngine()

    # Distress cry above threshold (0.54)
    aud_alert = AcousticEvidence(distress_prob=0.88, calibrated_threshold=0.54)
    dec = engine.evaluate_audio_only(aud_alert)
    assert dec.alert_state == "ALERT"
    assert dec.alert_score == 0.88
    assert dec.recommended_action == "ACOUSTIC_CUE_DISPATCH_OPTICAL_SEARCH"
    assert dec.victim_coordinates is None
    assert dec.spatial_bearing_valid is False

    # Normal speech below threshold
    aud_neg = AcousticEvidence(distress_prob=0.15, calibrated_threshold=0.54)
    dec_neg = engine.evaluate_audio_only(aud_neg)
    assert dec_neg.alert_state == "NO_ALERT"


def test_concordant_fusion():
    engine = MultiModalDecisionEngine()

    vis = VisualEvidence(person_detected=True, max_person_confidence=0.75, person_boxes=[(10, 20, 50, 100)])
    aud = AcousticEvidence(distress_prob=0.85, calibrated_threshold=0.54)

    dec = engine.evaluate_combined_fusion(vis, aud)
    assert dec.alert_state == "ALERT"
    assert dec.modality_agreement is True
    assert dec.dominant_modality == "concordant"
    assert dec.alert_score > 0.75  # Boosted by concordance bonus
    assert dec.recommended_action == "CONFIRMED_VICTIM_LOCATED"


def test_sensor_complementarity_night_or_fog():
    engine = MultiModalDecisionEngine()

    # Darkness / Night: Vision is blind (low illumination), Audio hears cry
    vis_night = VisualEvidence(
        person_detected=False,
        max_person_confidence=0.0,
        illumination_level=0.10,
        weather_domain="night",
    )
    aud_cry = AcousticEvidence(distress_prob=0.92, calibrated_threshold=0.54, snr_db=15.0)

    dec = engine.evaluate_combined_fusion(vis_night, aud_cry)
    assert dec.alert_state == "ALERT"
    assert dec.dominant_modality == "audio"
    assert dec.recommended_action == "ACOUSTIC_CUE_DISPATCH_OPTICAL_SEARCH"


def test_sensor_complementarity_heavy_rain_or_rotor_wash():
    engine = MultiModalDecisionEngine()

    # Severe acoustic noise (rain splatter), Vision clearly sees person
    vis_clear = VisualEvidence(
        person_detected=True,
        max_person_confidence=0.80,
        person_boxes=[(50, 60, 120, 200)],
        illumination_level=0.70,
        weather_domain="clear",
    )
    aud_noisy = AcousticEvidence(
        distress_prob=0.30,
        calibrated_threshold=0.54,
        snr_db=0.0,
        noise_condition="heavy_rain",
    )

    dec = engine.evaluate_combined_fusion(vis_clear, aud_noisy)
    assert dec.alert_state == "ALERT"
    assert dec.dominant_modality == "vision"
    assert dec.recommended_action == "VISUAL_SURVEILLANCE_VERIFIED"


def test_explicit_abstention_state():
    engine = MultiModalDecisionEngine()

    # Borderline optical candidate with no acoustic corroboration under quiet clear conditions
    vis_borderline = VisualEvidence(
        person_detected=True,
        max_person_confidence=0.28,  # Barely above 0.25
        person_boxes=[(10, 10, 25, 25)],
        illumination_level=0.80,
        contrast_index=0.60,
    )
    aud_silent = AcousticEvidence(
        distress_prob=0.10,
        calibrated_threshold=0.54,
        snr_db=20.0,
        noise_condition="clean_speech",
    )

    dec = engine.evaluate_combined_fusion(vis_borderline, aud_silent)
    assert dec.alert_state == "ABSTAIN_UNCERTAIN"
    assert dec.recommended_action == "UNCERTAIN_MAINTAIN_SEARCH"
