"""Decision-Level Multi-Modal Fusion Architecture for Search and Rescue (SAR).

Integrates optical evidence from WeatherGuard (object detection, visual contrast,
illumination, weather degradation) with acoustic distress evidence from Rescue Drone
(emergency vocalization probability, SNR, environmental noise condition).

Safety and Operational States:
  - ACOUSTIC_DISTRESS_DETECTED: Acoustic distress vocalization confirmed; no visual
    confirmation yet. Triggers localized optical spiral search; does NOT assert visual
    victim coordinates.
  - VISUAL_PERSON_DETECTED: Optical person detection confirmed; no acoustic distress
    (silent, unconscious, or acoustic masking).
  - PERSON_VISUALLY_CORROBORATED: Both optical and acoustic sensors confirm victim presence.
  - SEARCH_REQUIRED: Ambiguous or weak acoustic/optical cue requiring search pattern engagement.
  - UNCERTAIN_EVIDENCE: Conflicting sensor evidence or low confidence margin; abstention state.
  - SENSOR_UNAVAILABLE: Camera or microphone offline, disconnected, or degraded.
  - NO_TARGET: Both sensors confirm absence of human distress or presence.

Approaches Supported:
  Approach A: Vision-Only Decision Baseline
  Approach B: Audio-Only Distress Baseline
  Approach C: Adaptive Uncertainty-Aware Multi-Modal Decision Fusion
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VisualEvidence:
    """Optical detection evidence extracted from camera frames."""
    person_detected: bool = False
    max_person_confidence: float = 0.0
    person_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    illumination_level: float = 0.5  # Mean brightness in [0, 1]
    contrast_index: float = 0.5     # Grayscale std-dev normalized to [0, 1]
    weather_domain: str = "clear"    # 'clear', 'low_light', 'dense_fog', 'heavy_rain', 'night', 'snow'
    ultra_small_detected: bool = False
    small_detected: bool = False
    sensor_available: bool = True    # Hardware health flag
    sensor_fault: bool = False        # Hardware malfunction / stream dropped
    camera_id: str = "cam_0"

    @property
    def optical_reliability_score(self) -> float:
        """Estimates environmental optical sensor reliability in [0, 1]."""
        if not self.sensor_available or self.sensor_fault:
            return 0.0

        score = 1.0
        # Darkness penalty
        if self.illumination_level < 0.20:
            score *= 0.35
        elif self.illumination_level < 0.35:
            score *= 0.65

        # Fog / contrast attenuation penalty
        if self.contrast_index < 0.12:
            score *= 0.40
        elif self.contrast_index < 0.20:
            score *= 0.70

        # Domain penalty
        if self.weather_domain == "dense_fog":
            score *= 0.60
        elif self.weather_domain == "night":
            score *= 0.45
        elif self.weather_domain == "heavy_rain":
            score *= 0.80

        return max(0.05, min(1.0, score))


@dataclass
class AcousticEvidence:
    """Acoustic distress evidence extracted from microphone audio."""
    distress_prob: float = 0.0
    calibrated_threshold: float = 0.54
    snr_db: float = float("inf")
    noise_condition: str = "clean_speech"  # 'clean_speech', 'strong_wind', 'drone_propeller', 'heavy_rain', 'flood_water'
    is_multi_channel: bool = False         # Physical constraint: mono vs array
    spatial_doa_available: bool = False    # Direction of arrival flag
    sensor_available: bool = True          # Hardware health flag
    sensor_fault: bool = False             # Hardware malfunction / buffer underrun
    mic_id: str = "mic_0"
    consecutive_alerts: int = 1            # Counter for repeated acoustic alerts
    ambient_noise_duration_sec: float = 0.0  # Duration of sustained high noise

    @property
    def acoustic_reliability_score(self) -> float:
        """Estimates acoustic sensor reliability in [0, 1] under physical noise."""
        if not self.sensor_available or self.sensor_fault:
            return 0.0

        score = 1.0
        # SNR acoustic floor attenuation
        if math.isinf(self.snr_db):
            score = 1.0
        elif self.snr_db <= 0.0:
            score *= 0.55
        elif self.snr_db <= 5.0:
            score *= 0.75
        elif self.snr_db <= 10.0:
            score *= 0.88

        # Noise interference penalty
        if self.noise_condition == "heavy_rain":
            score *= 0.75
        elif self.noise_condition == "drone_propeller":
            score *= 0.80
        elif self.noise_condition == "strong_wind":
            score *= 0.85
        elif self.noise_condition == "flood_water":
            score *= 0.88

        # Long duration noise penalty
        if self.ambient_noise_duration_sec > 10.0:
            score *= 0.85

        return max(0.10, min(1.0, score))

    @property
    def acoustic_confidence(self) -> float:
        """Distance from calibrated decision threshold."""
        return abs(self.distress_prob - self.calibrated_threshold) / max(self.calibrated_threshold, 1.0 - self.calibrated_threshold)


@dataclass
class MultiModalDecision:
    """Standardized multi-modal decision output."""
    alert_state: str  # 'ALERT', 'NO_ALERT', 'ABSTAIN_UNCERTAIN', 'SENSOR_FAULT'
    alert_score: float  # Composite alert probability in [0, 1]
    modality_agreement: bool
    dominant_modality: str  # 'vision', 'audio', 'concordant', 'none', 'fault'
    fusion_rationale: str
    recommended_action: str
    operational_state: str = "NO_TARGET"  # ACOUSTIC_DISTRESS_DETECTED, VISUAL_PERSON_DETECTED, PERSON_VISUALLY_CORROBORATED, etc.
    victim_coordinates: Optional[List[Tuple[float, float, float, float]]] = None
    spatial_bearing_valid: bool = False
    is_victim_visually_confirmed: bool = False
    requires_optical_search: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_state": self.alert_state,
            "alert_score": round(self.alert_score, 4),
            "modality_agreement": self.modality_agreement,
            "dominant_modality": self.dominant_modality,
            "fusion_rationale": self.fusion_rationale,
            "recommended_action": self.recommended_action,
            "operational_state": self.operational_state,
            "victim_coordinates": self.victim_coordinates,
            "spatial_bearing_valid": self.spatial_bearing_valid,
            "is_victim_visually_confirmed": self.is_victim_visually_confirmed,
            "requires_optical_search": self.requires_optical_search,
        }


@dataclass
class MultiModalFusionConfig:
    """Configuration parameters for multi-modal decision-level fusion."""
    visual_conf_threshold: float = 0.25
    audio_calibrated_threshold: float = 0.54
    optical_weight_scale: float = 1.0
    acoustic_weight_scale: float = 1.0
    concordance_bonus: float = 0.15
    abstention_margin: float = 0.10
    min_optical_reliability_for_solo_alert: float = 0.25
    min_acoustic_reliability_for_solo_alert: float = 0.30
    platt_a: float = 1.0  # Platt scaling slope calibrated on validation split
    platt_b: float = 0.0  # Platt scaling intercept calibrated on validation split


class MultiModalDecisionEngine:
    """Adaptive, uncertainty-aware decision-level fusion engine for UAV SAR missions."""

    def __init__(self, config: Optional[MultiModalFusionConfig] = None) -> None:
        self.config = config or MultiModalFusionConfig()

    def calibrate_probability(self, raw_prob: float) -> float:
        """Applies validation-calibrated Platt scaling to convert model logits/scores to calibrated probability."""
        if raw_prob <= 0.0:
            return 0.0
        if raw_prob >= 1.0:
            return 1.0
        # Logit transformation
        eps = 1e-7
        p = max(eps, min(1.0 - eps, raw_prob))
        logit = math.log(p / (1.0 - p))
        calibrated_logit = self.config.platt_a * logit + self.config.platt_b
        return 1.0 / (1.0 + math.exp(-calibrated_logit))

    def evaluate_vision_only(self, vis: VisualEvidence) -> MultiModalDecision:
        """Approach A: Vision-Only Decision Baseline."""
        if not vis.sensor_available or vis.sensor_fault:
            return MultiModalDecision(
                alert_state="SENSOR_FAULT",
                alert_score=0.0,
                modality_agreement=False,
                dominant_modality="fault",
                fusion_rationale="Optical sensor camera offline or malfunctioning.",
                recommended_action="CAMERA_SENSOR_FAULT",
                operational_state="SENSOR_UNAVAILABLE",
                victim_coordinates=None,
                spatial_bearing_valid=False,
                is_victim_visually_confirmed=False,
                requires_optical_search=False,
            )

        is_alert = vis.person_detected and (vis.max_person_confidence >= self.config.visual_conf_threshold)
        return MultiModalDecision(
            alert_state="ALERT" if is_alert else "NO_ALERT",
            alert_score=vis.max_person_confidence if is_alert else 0.0,
            modality_agreement=False,
            dominant_modality="vision",
            fusion_rationale=f"Vision baseline alert: person_detected={vis.person_detected}, conf={vis.max_person_confidence:.2f}",
            recommended_action="VISUAL_TRACKING" if is_alert else "CONTINUE_GRID_SEARCH",
            operational_state="VISUAL_PERSON_DETECTED" if is_alert else "NO_TARGET",
            victim_coordinates=vis.person_boxes if is_alert else None,
            spatial_bearing_valid=False,
            is_victim_visually_confirmed=is_alert,
            requires_optical_search=False,
        )

    def evaluate_audio_only(self, aud: AcousticEvidence) -> MultiModalDecision:
        """Approach B: Audio-Only Distress Baseline."""
        if not aud.sensor_available or aud.sensor_fault:
            return MultiModalDecision(
                alert_state="SENSOR_FAULT",
                alert_score=0.0,
                modality_agreement=False,
                dominant_modality="fault",
                fusion_rationale="Acoustic sensor microphone offline or stream dropped.",
                recommended_action="MICROPHONE_SENSOR_FAULT",
                operational_state="SENSOR_UNAVAILABLE",
                victim_coordinates=None,
                spatial_bearing_valid=False,
                is_victim_visually_confirmed=False,
                requires_optical_search=False,
            )

        is_alert = (aud.distress_prob >= aud.calibrated_threshold)
        return MultiModalDecision(
            alert_state="ALERT" if is_alert else "NO_ALERT",
            alert_score=aud.distress_prob if is_alert else 0.0,
            modality_agreement=False,
            dominant_modality="audio",
            fusion_rationale=f"Audio baseline alert: distress_prob={aud.distress_prob:.2f} >= threshold={aud.calibrated_threshold:.2f}",
            recommended_action="ACOUSTIC_CUE_DISPATCH_OPTICAL_SEARCH" if is_alert else "CONTINUE_GRID_SEARCH",
            operational_state="ACOUSTIC_DISTRESS_DETECTED" if is_alert else "NO_TARGET",
            victim_coordinates=None,
            spatial_bearing_valid=False,
            is_victim_visually_confirmed=False,
            requires_optical_search=is_alert,
        )

    def evaluate_combined_fusion(
        self,
        vis: VisualEvidence,
        aud: AcousticEvidence,
    ) -> MultiModalDecision:
        """Approach C: Adaptive Uncertainty-Aware Multi-Modal Decision Fusion."""

        # 0. Hardware Sensor Availability & Health Guards
        cam_ok = vis.sensor_available and not vis.sensor_fault
        mic_ok = aud.sensor_available and not aud.sensor_fault

        if not cam_ok and not mic_ok:
            return MultiModalDecision(
                alert_state="SENSOR_FAULT",
                alert_score=0.0,
                modality_agreement=False,
                dominant_modality="fault",
                fusion_rationale="CRITICAL FAULT: Both camera and microphone streams are offline.",
                recommended_action="ABORT_MISSION_RETURN_TO_HOME",
                operational_state="SENSOR_UNAVAILABLE",
                victim_coordinates=None,
                spatial_bearing_valid=False,
                is_victim_visually_confirmed=False,
                requires_optical_search=False,
            )

        if not cam_ok and mic_ok:
            # Fall back to audio-only with explicit sensor degraded warning
            aud_dec = self.evaluate_audio_only(aud)
            if aud_dec.alert_state == "ALERT":
                return MultiModalDecision(
                    alert_state="ALERT",
                    alert_score=aud_dec.alert_score,
                    modality_agreement=False,
                    dominant_modality="audio",
                    fusion_rationale="Camera offline; acoustic distress detected. Optical search cannot be verified.",
                    recommended_action="ACOUSTIC_ALERT_CAMERA_DEGRADED",
                    operational_state="ACOUSTIC_DISTRESS_DETECTED",
                    victim_coordinates=None,
                    spatial_bearing_valid=False,
                    is_victim_visually_confirmed=False,
                    requires_optical_search=True,
                )
            else:
                return MultiModalDecision(
                    alert_state="SENSOR_FAULT",
                    alert_score=0.0,
                    modality_agreement=False,
                    dominant_modality="audio",
                    fusion_rationale="Camera offline; audio reports negative.",
                    recommended_action="CAMERA_OFFLINE_MONITOR_AUDIO",
                    operational_state="SENSOR_UNAVAILABLE",
                    victim_coordinates=None,
                    spatial_bearing_valid=False,
                    is_victim_visually_confirmed=False,
                    requires_optical_search=False,
                )

        if cam_ok and not mic_ok:
            # Fall back to vision-only
            vis_dec = self.evaluate_vision_only(vis)
            if vis_dec.alert_state == "ALERT":
                return MultiModalDecision(
                    alert_state="ALERT",
                    alert_score=vis_dec.alert_score,
                    modality_agreement=False,
                    dominant_modality="vision",
                    fusion_rationale="Microphone offline; optical person detected.",
                    recommended_action="VISUAL_SURVEILLANCE_VERIFIED",
                    operational_state="VISUAL_PERSON_DETECTED",
                    victim_coordinates=vis.person_boxes,
                    spatial_bearing_valid=False,
                    is_victim_visually_confirmed=True,
                    requires_optical_search=False,
                )
            else:
                return MultiModalDecision(
                    alert_state="SENSOR_FAULT",
                    alert_score=0.0,
                    modality_agreement=False,
                    dominant_modality="vision",
                    fusion_rationale="Microphone offline; camera reports negative.",
                    recommended_action="MIC_OFFLINE_MONITOR_CAMERA",
                    operational_state="SENSOR_UNAVAILABLE",
                    victim_coordinates=None,
                    spatial_bearing_valid=False,
                    is_victim_visually_confirmed=False,
                    requires_optical_search=False,
                )

        # 1. Evaluate Individual Signals
        vis_alert = vis.person_detected and (vis.max_person_confidence >= self.config.visual_conf_threshold)
        aud_alert = (aud.distress_prob >= aud.calibrated_threshold)

        w_vis = vis.optical_reliability_score * self.config.optical_weight_scale
        w_aud = aud.acoustic_reliability_score * self.config.acoustic_weight_scale

        # Boost acoustic weight if repeated consecutive distress alerts are confirmed
        if aud.consecutive_alerts >= 2:
            w_aud *= min(1.5, 1.0 + 0.15 * aud.consecutive_alerts)

        # Base Weighted Decision Score
        denom = max(1e-6, w_vis + w_aud)
        composite_score = (w_vis * vis.max_person_confidence + w_aud * aud.distress_prob) / denom

        # Multiple visible people notice
        person_count = len(vis.person_boxes)
        multi_person_note = f" (Multiple persons visible: {person_count})" if person_count > 1 else ""

        # 2. Case: Sensor Concordance — Both Modalities Agree on Alert
        if vis_alert and aud_alert:
            boosted_score = min(1.0, composite_score + self.config.concordance_bonus)
            return MultiModalDecision(
                alert_state="ALERT",
                alert_score=boosted_score,
                modality_agreement=True,
                dominant_modality="concordant",
                fusion_rationale=f"Concordant multi-modal corroboration: Optical ({vis.max_person_confidence:.2f}) + Acoustic ({aud.distress_prob:.2f}){multi_person_note}.",
                recommended_action="CONFIRMED_VICTIM_LOCATED",
                operational_state="PERSON_VISUALLY_CORROBORATED",
                victim_coordinates=vis.person_boxes,
                spatial_bearing_valid=False,
                is_victim_visually_confirmed=True,
                requires_optical_search=False,
            )

        # 3. Case: Both Modalities Agree on Negative (No Target)
        if not vis_alert and not aud_alert:
            return MultiModalDecision(
                alert_state="NO_ALERT",
                alert_score=composite_score,
                modality_agreement=True,
                dominant_modality="none",
                fusion_rationale="Both modalities confirm absence of victim cues.",
                recommended_action="CONTINUE_GRID_SEARCH",
                operational_state="NO_TARGET",
                victim_coordinates=None,
                spatial_bearing_valid=False,
                is_victim_visually_confirmed=False,
                requires_optical_search=False,
            )

        # 4. Disagreement: Audio Alert, Vision No Alert
        # Distinguishes genuine acoustic cue (dispatch search) from confirmed visual victim
        if aud_alert and not vis_alert:
            # Check optical degradation (night, dense fog, distant target)
            if w_vis < self.config.min_optical_reliability_for_solo_alert or vis.weather_domain in ["night", "dense_fog"]:
                return MultiModalDecision(
                    alert_state="ALERT",
                    alert_score=aud.distress_prob,
                    modality_agreement=False,
                    dominant_modality="audio",
                    fusion_rationale=f"Optical sensor degraded (reliability={w_vis:.2f}, {vis.weather_domain}); acoustic distress cue accepted.",
                    recommended_action="ACOUSTIC_CUE_DISPATCH_OPTICAL_SEARCH",
                    operational_state="ACOUSTIC_DISTRESS_DETECTED",
                    victim_coordinates=None,  # NEVER fabricate visual coordinates for mono audio
                    spatial_bearing_valid=False,
                    is_victim_visually_confirmed=False,
                    requires_optical_search=True,
                )
            elif aud.acoustic_confidence >= self.config.abstention_margin:
                return MultiModalDecision(
                    alert_state="ALERT",
                    alert_score=composite_score,
                    modality_agreement=False,
                    dominant_modality="audio",
                    fusion_rationale="Strong acoustic distress cue overrides negative optical scan.",
                    recommended_action="ACOUSTIC_CUE_DISPATCH_OPTICAL_SEARCH",
                    operational_state="ACOUSTIC_DISTRESS_DETECTED",
                    victim_coordinates=None,
                    spatial_bearing_valid=False,
                    is_victim_visually_confirmed=False,
                    requires_optical_search=True,
                )
            else:
                # Borderline acoustic cue under clear optical visibility -> Abstain / Uncertain
                return MultiModalDecision(
                    alert_state="ABSTAIN_UNCERTAIN",
                    alert_score=composite_score,
                    modality_agreement=False,
                    dominant_modality="audio",
                    fusion_rationale="Borderline acoustic cue contradicts clear optical non-detection; awaiting corroboration.",
                    recommended_action="UNCERTAIN_MAINTAIN_SEARCH",
                    operational_state="UNCERTAIN_EVIDENCE",
                    victim_coordinates=None,
                    spatial_bearing_valid=False,
                    is_victim_visually_confirmed=False,
                    requires_optical_search=True,
                )

        # 5. Disagreement: Vision Alert, Audio No Alert
        # Silent/unconscious victim or acoustic masking (rotor wash/rain)
        if vis_alert and not aud_alert:
            if w_aud < self.config.min_acoustic_reliability_for_solo_alert or aud.noise_condition in ["heavy_rain", "drone_propeller"]:
                return MultiModalDecision(
                    alert_state="ALERT",
                    alert_score=vis.max_person_confidence,
                    modality_agreement=False,
                    dominant_modality="vision",
                    fusion_rationale=f"Acoustic channel noisy (SNR={aud.snr_db}dB, {aud.noise_condition}); optical person detection accepted{multi_person_note}.",
                    recommended_action="VISUAL_SURVEILLANCE_VERIFIED",
                    operational_state="VISUAL_PERSON_DETECTED",
                    victim_coordinates=vis.person_boxes,
                    spatial_bearing_valid=False,
                    is_victim_visually_confirmed=True,
                    requires_optical_search=False,
                )
            elif vis.max_person_confidence >= (self.config.visual_conf_threshold + self.config.abstention_margin):
                return MultiModalDecision(
                    alert_state="ALERT",
                    alert_score=composite_score,
                    modality_agreement=False,
                    dominant_modality="vision",
                    fusion_rationale=f"High optical confidence detects person despite silent/unvocalized state{multi_person_note}.",
                    recommended_action="VISUAL_SURVEILLANCE_VERIFIED",
                    operational_state="VISUAL_PERSON_DETECTED",
                    victim_coordinates=vis.person_boxes,
                    spatial_bearing_valid=False,
                    is_victim_visually_confirmed=True,
                    requires_optical_search=False,
                )
            else:
                # Borderline optical candidate without acoustic corroboration in quiet environment -> Abstain
                return MultiModalDecision(
                    alert_state="ABSTAIN_UNCERTAIN",
                    alert_score=composite_score,
                    modality_agreement=False,
                    dominant_modality="vision",
                    fusion_rationale=f"Borderline optical candidate lacks acoustic corroboration in quiet environment{multi_person_note}.",
                    recommended_action="UNCERTAIN_MAINTAIN_SEARCH",
                    operational_state="UNCERTAIN_EVIDENCE",
                    victim_coordinates=vis.person_boxes,
                    spatial_bearing_valid=False,
                    is_victim_visually_confirmed=False,
                    requires_optical_search=True,
                )

        # Default fallback
        return MultiModalDecision(
            alert_state="ABSTAIN_UNCERTAIN",
            alert_score=composite_score,
            modality_agreement=False,
            dominant_modality="none",
            fusion_rationale="Inconclusive multi-modal evidence.",
            recommended_action="UNCERTAIN_MAINTAIN_SEARCH",
            operational_state="UNCERTAIN_EVIDENCE",
            victim_coordinates=None,
            spatial_bearing_valid=False,
            is_victim_visually_confirmed=False,
            requires_optical_search=True,
        )
