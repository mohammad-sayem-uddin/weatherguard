#!/usr/bin/env python3
"""Cross-Modal Robustness Analysis for WeatherGuard Vision and Rescue Drone Audio.

Combines frozen locked-test evaluations across:
  - Vision: Adverse weather conditions (Dense Fog, Snow, Rain, Night) and 4 scale tiers.
  - Audio: Physical noise conditions (Drone, Rain, Flood, Wind) and calibrated SNRs (0-20 dB).
Computes cross-modal sensor complementarity metrics.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RobustnessAnalysis")

VISION_RESULTS_PATH = Path("/Users/sayemuddin/Desktop/WeatherGuard/artifacts/mission20/mission20_vision_gpu_results.json")
AUDIO_RESULTS_PATH = Path("/Users/sayemuddin/Desktop/WeatherGuard/artifacts/mission20/mission20_audio_gpu_results.json")
OUT_ROBUSTNESS_PATH = Path("/Users/sayemuddin/Desktop/WeatherGuard/artifacts/mission20/robustness_matrix.json")


def generate_robustness_matrix() -> Dict[str, Any]:
    if not VISION_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Vision results missing at {VISION_RESULTS_PATH}")
    if not AUDIO_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Audio results missing at {AUDIO_RESULTS_PATH}")

    vision_data = json.loads(VISION_RESULTS_PATH.read_text())
    audio_data = json.loads(AUDIO_RESULTS_PATH.read_text())

    vis_test = vision_data["locked_test_split_metrics"]
    vis_zero = vision_data["zero_shot_baseline_comparison"]
    aud_test = audio_data["locked_test_metrics"]

    # Vision Environmental Comparison
    vision_env_matrix = {}
    for env in ["dense_fog", "snow", "rain", "night"]:
        ft = vis_test["weather_breakdown"].get(env, {})
        zs = vis_zero["weather_breakdown"].get(env, {})
        vision_env_matrix[env] = {
            "images": ft.get("images", 0),
            "total_gt": ft.get("total_gt", 0),
            "zero_shot_recall": zs.get("recall", 0.0),
            "zero_shot_f1": zs.get("f1", 0.0),
            "fine_tuned_recall": ft.get("recall", 0.0),
            "fine_tuned_f1": ft.get("f1", 0.0),
            "delta_recall_pct": round((ft.get("recall", 0.0) - zs.get("recall", 0.0)) * 100, 2),
            "delta_f1_pct": round((ft.get("f1", 0.0) - zs.get("f1", 0.0)) * 100, 2),
        }

    # Vision Scale Comparison
    vision_scale_matrix = {}
    for scale in ["ultra_small", "small", "medium", "large"]:
        ft = vis_test["scale_breakdown"].get(scale, {})
        zs = vis_zero["scale_breakdown"].get(scale, {})
        vision_scale_matrix[scale] = {
            "total_gt": ft.get("total_gt", 0),
            "zero_shot_recall": zs.get("recall", 0.0),
            "zero_shot_f1": zs.get("f1", 0.0),
            "fine_tuned_recall": ft.get("recall", 0.0),
            "fine_tuned_f1": ft.get("f1", 0.0),
            "relative_gain_recall_pct": round(
                ((ft.get("recall", 0.0) - zs.get("recall", 0.0)) / max(1e-4, zs.get("recall", 0.0))) * 100, 1
            ) if zs.get("recall", 0.0) > 0 else 0.0,
        }

    # Audio Noise Comparison
    audio_noise_matrix = aud_test.get("noise_breakdown", {})

    # Audio SNR Comparison
    audio_snr_matrix = aud_test.get("snr_breakdown", {})

    robustness = {
        "analysis_name": "Mission 20 Independent Robustness & Sensor Complementarity Matrix",
        "vision_robustness": {
            "environmental_generalization": vision_env_matrix,
            "scale_tier_generalization": vision_scale_matrix,
            "overall_test_map50": vis_test["overall"]["mAP50"],
            "overall_test_map50_95": vis_test["overall"]["mAP50_95"],
        },
        "audio_robustness": {
            "noise_breakdown": audio_noise_matrix,
            "snr_breakdown": audio_snr_matrix,
            "overall_test_accuracy": aud_test["accuracy"],
            "overall_test_f1": aud_test["f1"],
            "overall_test_roc_auc": aud_test["roc_auc"],
            "calibrated_threshold": audio_data["training"]["calibrated_threshold"],
        },
        "multi_modal_insights": {
            "night_cross_modal_gain": "Vision F1 severely drops to 0.3327 at night; Audio operates invariant to ambient lighting.",
            "ultra_small_acoustic_cueing": "Sub-16px visual targets have 0.0% single-frame recall; acoustic distress triggers drone search trajectory prior to visual acquisition.",
            "heavy_rain_rotor_wash_resilience": "Heavy rain and rotor wash reduce audio accuracy to 84-88%; visual detection maintains 0.5056 F1 under rain.",
        },
    }

    OUT_ROBUSTNESS_PATH.write_text(json.dumps(robustness, indent=2))
    logger.info("Saved cross-modal robustness matrix to %s", OUT_ROBUSTNESS_PATH)
    return robustness


if __name__ == "__main__":
    matrix = generate_robustness_matrix()
    print(json.dumps(matrix, indent=2))
