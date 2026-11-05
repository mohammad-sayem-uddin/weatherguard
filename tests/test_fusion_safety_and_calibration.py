"""Safety and Calibration Unit Tests for Multi-Modal Fusion Engine (Mission 22 - Phase 4)."""
import pytest
from src.multimodal.fusion_engine import (
    AcousticEvidence,
    MultiModalDecision,
    MultiModalDecisionEngine,
    MultiModalFusionConfig,
    VisualEvidence,
)


def test_missing_or_failed_camera():
    engine = MultiModalDecisionEngine()

    # Camera offline, audio hears genuine cry
    vis_offline = VisualEvidence(sensor_available=False)
    aud_cry = AcousticEvidence(distress_prob=0.90, calibrated_threshold=0.54)

    dec = engine.evaluate_combined_fusion(vis_offline, aud_cry)
    assert dec.alert_state == "ALERT"
    assert dec.operational_state == "ACOUSTIC_DISTRESS_DETECTED"
    assert dec.is_victim_visually_confirmed is False
    assert dec.requires_optical_search is True
    assert dec.victim_coordinates is None

    # Camera offline, audio reports negative
    aud_neg = AcousticEvidence(distress_prob=0.10, calibrated_threshold=0.54)
    dec_neg = engine.evaluate_combined_fusion(vis_offline, aud_neg)
    assert dec_neg.alert_state == "SENSOR_FAULT"
    assert dec_neg.operational_state == "SENSOR_UNAVAILABLE"


def test_missing_or_failed_microphone():
    engine = MultiModalDecisionEngine()

    # Microphone offline, camera sees person
    vis_person = VisualEvidence(person_detected=True, max_person_confidence=0.85, person_boxes=[(10, 20, 50, 100)])
    aud_offline = AcousticEvidence(sensor_available=False)

    dec = engine.evaluate_combined_fusion(vis_person, aud_offline)
    assert dec.alert_state == "ALERT"
    assert dec.operational_state == "VISUAL_PERSON_DETECTED"
    assert dec.is_victim_visually_confirmed is True
    assert dec.victim_coordinates is not None
    assert dec.requires_optical_search is False

    # Microphone offline, camera reports negative
    vis_neg = VisualEvidence(person_detected=False)
    dec_neg = engine.evaluate_combined_fusion(vis_neg, aud_offline)
    assert dec_neg.alert_state == "SENSOR_FAULT"
    assert dec_neg.operational_state == "SENSOR_UNAVAILABLE"


def test_dual_sensor_failure():
    engine = MultiModalDecisionEngine()
    vis_fail = VisualEvidence(sensor_fault=True)
    aud_fail = AcousticEvidence(sensor_fault=True)

    dec = engine.evaluate_combined_fusion(vis_fail, aud_fail)
    assert dec.alert_state == "SENSOR_FAULT"
    assert dec.operational_state == "SENSOR_UNAVAILABLE"
    assert dec.recommended_action == "ABORT_MISSION_RETURN_TO_HOME"


def test_silent_or_unconscious_victim():
    engine = MultiModalDecisionEngine()

    # Clear image of person, but victim is silent/unconscious (low audio prob)
    vis_victim = VisualEvidence(
        person_detected=True,
        max_person_confidence=0.88,
        person_boxes=[(100, 150, 200, 350)],
        illumination_level=0.75,
    )
    aud_silent = AcousticEvidence(distress_prob=0.08, calibrated_threshold=0.54, snr_db=25.0)

    dec = engine.evaluate_combined_fusion(vis_victim, aud_silent)
    assert dec.alert_state == "ALERT"
    assert dec.operational_state == "VISUAL_PERSON_DETECTED"
    assert dec.is_victim_visually_confirmed is True
    assert dec.victim_coordinates == [(100, 150, 200, 350)]
    assert dec.requires_optical_search is False


def test_acoustic_alert_without_visible_person():
    engine = MultiModalDecisionEngine()

    # Foggy scene with no person in sight, but acoustic distress cry heard
    vis_fog = VisualEvidence(
        person_detected=False,
        max_person_confidence=0.0,
        weather_domain="dense_fog",
        contrast_index=0.10,
    )
    aud_cry = AcousticEvidence(distress_prob=0.87, calibrated_threshold=0.54)

    dec = engine.evaluate_combined_fusion(vis_fog, aud_cry)
    assert dec.alert_state == "ALERT"
    assert dec.operational_state == "ACOUSTIC_DISTRESS_DETECTED"
    assert dec.is_victim_visually_confirmed is False
    assert dec.requires_optical_search is True
    assert dec.victim_coordinates is None  # Never fabricate coordinates


def test_repeated_acoustic_alerts_priority_boost():
    engine = MultiModalDecisionEngine()

    vis_clear_empty = VisualEvidence(person_detected=False, max_person_confidence=0.0, weather_domain="clear")

    # Single alert
    aud_1 = AcousticEvidence(distress_prob=0.65, consecutive_alerts=1)
    dec_1 = engine.evaluate_combined_fusion(vis_clear_empty, aud_1)

    # 3 consecutive alerts from stationary victim
    aud_3 = AcousticEvidence(distress_prob=0.65, consecutive_alerts=3)
    dec_3 = engine.evaluate_combined_fusion(vis_clear_empty, aud_3)

    assert dec_3.alert_score >= dec_1.alert_score
    assert dec_3.requires_optical_search is True


def test_multiple_visible_people():
    engine = MultiModalDecisionEngine()

    # Two persons visible in image
    vis_multi = VisualEvidence(
        person_detected=True,
        max_person_confidence=0.82,
        person_boxes=[(10, 20, 50, 100), (200, 220, 260, 320)],
    )
    aud_cry = AcousticEvidence(distress_prob=0.85, calibrated_threshold=0.54)

    dec = engine.evaluate_combined_fusion(vis_multi, aud_cry)
    assert dec.alert_state == "ALERT"
    assert dec.operational_state == "PERSON_VISUALLY_CORROBORATED"
    assert dec.is_victim_visually_confirmed is True
    assert len(dec.victim_coordinates) == 2
    assert "Multiple persons visible: 2" in dec.fusion_rationale


def test_long_duration_environmental_noise():
    engine = MultiModalDecisionEngine()

    # 15 seconds of sustained noise lowers acoustic reliability
    aud_sustained = AcousticEvidence(
        distress_prob=0.45,
        calibrated_threshold=0.54,
        noise_condition="drone_propeller",
        ambient_noise_duration_sec=15.0,
    )
    assert aud_sustained.acoustic_reliability_score < 0.80


def test_platt_probability_calibration():
    config = MultiModalFusionConfig(platt_a=1.2, platt_b=-0.3)
    engine = MultiModalDecisionEngine(config=config)

    # Monotonicity test
    p_low = engine.calibrate_probability(0.2)
    p_mid = engine.calibrate_probability(0.5)
    p_high = engine.calibrate_probability(0.8)

    assert p_low < p_mid < p_high
    assert 0.0 <= p_low <= 1.0
    assert 0.0 <= p_high <= 1.0
