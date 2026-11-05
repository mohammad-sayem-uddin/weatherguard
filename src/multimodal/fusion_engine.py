"""Decision-Level Multi-Modal Fusion Architecture for Search and Rescue.

Integrates optical evidence from WeatherGuard (object detection, visual contrast,
illumination, weather degradation) with acoustic distress evidence from Rescue Drone
(emergency vocalization probability, SNR, environmental noise condition).

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
    weather_domain: str = "clear"    # 'clear', 'low_light', 'dense_fog', 'heavy_rain', 'night'
    ultra_small_detected: bool = False
    small_detected: bool = False

    @property
    def optical_reliability_score(self) -> float:
        """Estimates environmental optical sensor reliability in [0, 1]."""
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

    @property
    def acoustic_reliability_score(self) -> float:
        """Estimates acoustic sensor reliability in [0, 1] under physical noise."""
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

        return max(0.10, min(1.0, score))

    @property
    def acoustic_confidence(self) -> float:
        """Distance from calibrated decision threshold."""
        return abs(self.distress_prob - self.calibrated_threshold) / max(self.calibrated_threshold, 1.0 - self.calibrated_threshold)


@dataclass
class MultiModalDecision:
    """Standardized multi-modal decision output."""
    alert_state: str  # 'ALERT', 'NO_ALERT', 'ABSTAIN_UNCERTAIN'
    alert_score: float  # Composite alert probability in [0, 1]
    modality_agreement: bool
    dominant_modality: str  # 'vision', 'audio', 'concordant', 'none'
    fusion_rationale: str
    recommended_action: str
    victim_coordinates: Optional[List[Tuple[float, float, float, float]]] = None
    spatial_bearing_valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_state": self.alert_state,
            "alert_score": round(self.alert_score, 4),
            "modality_agreement": self.modality_agreement,
            "dominant_modality": self.dominant_modality,
            "fusion_rationale": self.fusion_rationale,
            "recommended_action": self.recommended_action,
            "victim_coordinates": self.victim_coordinates,
            "spatial_bearing_valid": self.spatial_bearing_valid,
        }


@dataclass
class MultiModalFusionConfig:
    """Configuration parameters for multi-modal decision-level fusion."""
    visual_conf_threshold: float = 0.25
    audio_calibrated_threshold: float = 0.54
    abstention_margin: float = 0.12  # Range around conflict boundary where system abstains
    concordance_bonus: float = 0.15
    min_optical_reliability_for_solo_alert: float = 0.30
    min_acoustic_reliability_for_solo_alert: float = 0.40


class MultiModalDecisionEngine:
    """Decision-Level Fusion Engine for Search and Rescue Operations."""

    def __init__(self, config: Optional[MultiModalFusionConfig] = None) -> None:
        self.config = config or MultiModalFusionConfig()

    def evaluate_vision_only(self, vis: VisualEvidence) -> MultiModalDecision:
        """Approach A: Vision-Only Decision Baseline."""
        is_alert = vis.person_detected and (vis.max_person_confidence >= self.config.visual_conf_threshold)
        score = vis.max_person_confidence if vis.person_detected else 0.0

        return MultiModalDecision(
            alert_state="ALERT" if is_alert else "NO_ALERT",
            alert_score=score,
            modality_agreement=True,
            dominant_modality="vision",
            fusion_rationale="Vision-only evaluation based strictly on optical person detections.",
            recommended_action="CONFIRMED_VICTIM_LOCATED" if is_alert else "CONTINUE_VISUAL_SEARCH",
            victim_coordinates=vis.person_boxes if is_alert else None,
            spatial_bearing_valid=False,
        )

    def evaluate_audio_only(self, aud: AcousticEvidence) -> MultiModalDecision:
        """Approach B: Audio-Only Distress Baseline."""
        is_alert = aud.distress_prob >= aud.calibrated_threshold
        score = aud.distress_prob

        return MultiModalDecision(
            alert_state="ALERT" if is_alert else "NO_ALERT",
            alert_score=score,
            modality_agreement=True,
            dominant_modality="audio",
            fusion_rationale="Audio-only evaluation based strictly on vocal distress classification.",
            recommended_action="ACOUSTIC_CUE_DISPATCH_OPTICAL_SEARCH" if is_alert else "CONTINUE_ACOUSTIC_MONITORING",
            victim_coordinates=None,
            spatial_bearing_valid=False,  # Mono hardware constraint
        )

    def evaluate_combined_fusion(
        self,
        vis: VisualEvidence,
        aud: AcousticEvidence,
    ) -> MultiModalDecision:
        """Approach C: Combined Audio-Visual Decision-Level Fusion with Explicit Abstention."""
        vis_alert = vis.person_detected and (vis.max_person_confidence >= self.config.visual_conf_threshold)
        aud_alert = aud.distress_prob >= aud.calibrated_threshold

        w_vis = vis.optical_reliability_score
        w_aud = aud.acoustic_reliability_score

        # Normalize weights
        w_sum = w_vis + w_aud
        norm_w_vis = w_vis / w_sum
        norm_w_aud = w_aud / w_sum

        vis_signal = vis.max_person_confidence if vis.person_detected else 0.0
        aud_signal = aud.distress_prob

        # Joint score calculation
        composite_score = (norm_w_vis * vis_signal) + (norm_w_aud * aud_signal)

        # 1. Concordant Positive (Both modalities detect target)
        if vis_alert and aud_alert:
            boosted_score = min(1.0, composite_score + self.config.concordance_bonus)
            return MultiModalDecision(
                alert_state="ALERT",
                alert_score=boosted_score,
                modality_agreement=True,
                dominant_modality="concordant",
                fusion_rationale="High-confidence multi-modal concordance: visual person detected and Bengali distress verified.",
                recommended_action="CONFIRMED_VICTIM_LOCATED",
                victim_coordinates=vis.person_boxes,
                spatial_bearing_valid=False,
            )

        # 2. Concordant Negative (Neither modality detects target)
        if (not vis_alert) and (not aud_alert):
            return MultiModalDecision(
                alert_state="NO_ALERT",
                alert_score=composite_score,
                modality_agreement=True,
                dominant_modality="none",
                fusion_rationale="Both modalities confirm absence of victim cues.",
                recommended_action="CONTINUE_GRID_SEARCH",
                victim_coordinates=None,
                spatial_bearing_valid=False,
            )

        # 3. Disagreement / Sensor Complementarity: Audio Alert, Vision No Alert
        # Common in nighttime, dense fog, or when victim is distant/occluded (<16px)
        if aud_alert and not vis_alert:
            # Check if optical channel is compromised
            if w_vis < self.config.min_optical_reliability_for_solo_alert:
                # Vision is blind (e.g. night/fog); trust acoustic alert
                return MultiModalDecision(
                    alert_state="ALERT",
                    alert_score=aud.distress_prob,
                    modality_agreement=False,
                    dominant_modality="audio",
                    fusion_rationale=f"Optical sensor degraded (reliability={w_vis:.2f}, {vis.weather_domain}); acoustic distress cue accepted.",
                    recommended_action="ACOUSTIC_CUE_DISPATCH_OPTICAL_SEARCH",
                    victim_coordinates=None,
                    spatial_bearing_valid=False,
                )
            elif aud.acoustic_confidence >= self.config.abstention_margin:
                # Audio is very confident; issue alert to redirect cameras
                return MultiModalDecision(
                    alert_state="ALERT",
                    alert_score=composite_score,
                    modality_agreement=False,
                    dominant_modality="audio",
                    fusion_rationale="Strong acoustic distress cue overrides negative optical scan.",
                    recommended_action="ACOUSTIC_CUE_DISPATCH_OPTICAL_SEARCH",
                    victim_coordinates=None,
                    spatial_bearing_valid=False,
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
                    victim_coordinates=None,
                    spatial_bearing_valid=False,
                )

        # 4. Disagreement / Sensor Complementarity: Vision Alert, Audio No Alert
        # Common when victim is silent/unconscious, or acoustic channel has heavy rotor/rain noise
        if vis_alert and not aud_alert:
            # Check if acoustic channel is compromised
            if w_aud < self.config.min_acoustic_reliability_for_solo_alert or aud.noise_condition == "heavy_rain":
                return MultiModalDecision(
                    alert_state="ALERT",
                    alert_score=vis.max_person_confidence,
                    modality_agreement=False,
                    dominant_modality="vision",
                    fusion_rationale=f"Acoustic channel noisy (SNR={aud.snr_db}dB, {aud.noise_condition}); optical person detection accepted.",
                    recommended_action="VISUAL_SURVEILLANCE_VERIFIED",
                    victim_coordinates=vis.person_boxes,
                    spatial_bearing_valid=False,
                )
            elif vis.max_person_confidence >= (self.config.visual_conf_threshold + self.config.abstention_margin):
                return MultiModalDecision(
                    alert_state="ALERT",
                    alert_score=composite_score,
                    modality_agreement=False,
                    dominant_modality="vision",
                    fusion_rationale="High optical confidence detects person despite silent/unvocalized state.",
                    recommended_action="VISUAL_SURVEILLANCE_VERIFIED",
                    victim_coordinates=vis.person_boxes,
                    spatial_bearing_valid=False,
                )
            else:
                # Borderline visual candidate without acoustic corroboration -> Abstain
                return MultiModalDecision(
                    alert_state="ABSTAIN_UNCERTAIN",
                    alert_score=composite_score,
                    modality_agreement=False,
                    dominant_modality="vision",
                    fusion_rationale="Borderline optical candidate lacks acoustic corroboration in quiet environment.",
                    recommended_action="UNCERTAIN_MAINTAIN_SEARCH",
                    victim_coordinates=vis.person_boxes,
                    spatial_bearing_valid=False,
                )
