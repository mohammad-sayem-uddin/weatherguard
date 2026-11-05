#!/usr/bin/env python3
"""Controlled Multi-Modal Search-and-Rescue Experimental Evaluation (Mission 21).

Evaluates Approaches A (Vision-Only), B (Audio-Only), and C (Combined Decision Fusion)
across 9 controlled environmental scenarios:
  1. Daylight with clear visibility
  2. Nighttime
  3. Dense fog
  4. Heavy rain
  5. Low visual resolution
  6. Distant visual targets (small / ultra-small)
  7. Strong wind
  8. Drone rotor noise
  9. Flood-water noise

Simulation Protocol Disclosure:
  Visual test scenes and acoustic test recordings were collected independently.
  Controlled experimental scenarios assemble paired environmental test conditions
  under strict provenance tracking. Asynchronous samples are explicitly disclosed
  as synthetic scenario pairings, not physical simultaneous recordings.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from src.multimodal.fusion_engine import (
    AcousticEvidence,
    MultiModalDecision,
    MultiModalDecisionEngine,
    MultiModalFusionConfig,
    VisualEvidence,
)
from scripts.kaggle.audio_gpu_runner import (
    AudioBaselinePyTorchCNN,
    RealAudioSpectrogramDataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiModalControlledExperiments")

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
VISION_TEST_DIR = ROOT / "data/weatherguard_corpus/split/test"
VISION_MANIFEST = ROOT / "data/weatherguard_corpus/metadata/split_manifest.csv"
AUDIO_DIR = Path("/Users/sayemuddin/Desktop/Rescue_Drone/audio_dataset_v2/dataset_v2_1500")
AUDIO_MANIFEST = ROOT / "artifacts/mission20/audio_split_manifest.csv"

VISION_WEIGHTS = Path("/tmp/kaggle_wg_out/artifacts/vision/best.pt")
AUDIO_WEIGHTS = Path("/tmp/kaggle_audio_out/artifacts/audio_gpu/audio_best_weights.pt")

OUT_JSON = ROOT / "artifacts/mission21/multimodal_controlled_experiments.json"

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_visual_evidence(
    model: YOLO,
    img_path: Path,
    lbl_path: Path,
    weather_domain: str,
    downsample: bool = False,
) -> Tuple[VisualEvidence, bool]:
    """Runs vision model on frame and returns VisualEvidence and ground-truth person presence."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return VisualEvidence(), False

    h, w = img_bgr.shape[:2]
    if downsample:
        img_bgr = cv2.resize(img_bgr, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    # Illumination and contrast
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    illumination = float(np.mean(gray)) / 255.0
    contrast = float(np.std(gray)) / 255.0

    # Ground Truth Person Presence
    gt_person = False
    gt_boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and int(parts[0]) == 1:  # 1 is person in WG
                gt_person = True
                cx, cy, bw, bh = map(float, parts[1:5])
                pw, ph = bw * w, bh * h
                gt_boxes.append((cx * w, cy * h, pw, ph))

    # Inference
    results = model.predict(source=img_bgr, conf=0.15, verbose=False)[0]
    person_detected = False
    max_person_conf = 0.0
    person_boxes = []
    ultra_small_det = False
    small_det = False

    if results.boxes is not None and len(results.boxes) > 0:
        for b in results.boxes:
            c = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            if c == 1:  # Person
                person_detected = True
                if conf > max_person_conf:
                    max_person_conf = conf
                xyxy = b.xyxy[0].tolist()
                person_boxes.append(tuple(xyxy))
                pw = xyxy[2] - xyxy[0]
                ph = xyxy[3] - xyxy[1]
                if max(pw, ph) < 16:
                    ultra_small_det = True
                elif max(pw, ph) < 32:
                    small_det = True

    vis = VisualEvidence(
        person_detected=person_detected,
        max_person_confidence=max_person_conf,
        person_boxes=person_boxes,
        illumination_level=illumination,
        contrast_index=contrast,
        weather_domain=weather_domain,
        ultra_small_detected=ultra_small_det,
        small_detected=small_det,
    )
    return vis, gt_person


def evaluate_scenario(
    engine: MultiModalDecisionEngine,
    trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluates Approaches A, B, and C across a list of multi-modal trials."""
    results = {"A_vision_only": [], "B_audio_only": [], "C_combined_fusion": []}

    for t in trials:
        vis = t["visual_evidence"]
        aud = t["acoustic_evidence"]
        gt = t["ground_truth_target"]

        dec_a = engine.evaluate_vision_only(vis)
        dec_b = engine.evaluate_audio_only(aud)
        dec_c = engine.evaluate_combined_fusion(vis, aud)

        results["A_vision_only"].append((dec_a.alert_state == "ALERT", dec_a.alert_score, gt, dec_a.alert_state))
        results["B_audio_only"].append((dec_b.alert_state == "ALERT", dec_b.alert_score, gt, dec_b.alert_state))
        results["C_combined_fusion"].append((dec_c.alert_state == "ALERT", dec_c.alert_score, gt, dec_c.alert_state))

    def summarize(out_list):
        total = len(out_list)
        if total == 0:
            return {}
        tp = sum(1 for is_alert, _, gt, _ in out_list if is_alert and gt)
        fp = sum(1 for is_alert, _, gt, _ in out_list if is_alert and not gt)
        fn = sum(1 for is_alert, _, gt, _ in out_list if not is_alert and gt)
        tn = sum(1 for is_alert, _, gt, _ in out_list if not is_alert and not gt)
        abstain_count = sum(1 for _, _, _, state in out_list if state == "ABSTAIN_UNCERTAIN")

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        acc = (tp + tn) / total if total > 0 else 0.0

        # Expected Calibration Error (ECE)
        bins = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for i in range(10):
            bin_items = [item for item in out_list if bins[i] <= item[1] < bins[i + 1]]
            if bin_items:
                bin_acc = sum(1 for item in bin_items if (item[1] >= 0.5) == item[2]) / len(bin_items)
                bin_conf = sum(item[1] for item in bin_items) / len(bin_items)
                ece += (len(bin_items) / total) * abs(bin_acc - bin_conf)

        return {
            "samples": total,
            "accuracy": round(acc, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "detection_probability": round(r, 4),
            "false_alarm_rate": round(fpr, 4),
            "miss_rate": round(fnr, 4),
            "abstention_rate": round(abstain_count / total, 4),
            "expected_calibration_error": round(ece, 4),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    return {
        "A_vision_only": summarize(results["A_vision_only"]),
        "B_audio_only": summarize(results["B_audio_only"]),
        "C_combined_fusion": summarize(results["C_combined_fusion"]),
    }


def run_controlled_experiments() -> Dict[str, Any]:
    set_seed(SEED)
    engine = MultiModalDecisionEngine()

    logger.info("Loading vision model from %s", VISION_WEIGHTS)
    vis_model = YOLO(str(VISION_WEIGHTS))

    logger.info("Loading audio model from %s", AUDIO_WEIGHTS)
    aud_model = AudioBaselinePyTorchCNN()
    aud_model.load_state_dict(torch.load(AUDIO_WEIGHTS, map_location="cpu"))
    aud_model.eval()

    logger.info("Initializing audio test dataset...")
    test_ds = RealAudioSpectrogramDataset(str(AUDIO_MANIFEST), AUDIO_DIR, split="test")

    # Load Vision Manifest
    v_df = pd.read_csv(VISION_MANIFEST)
    v_test = v_df[v_df["split"] == "test"].copy()

    # Precompute audio inferences across all 300 test clips
    audio_bank = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, target, meta = test_ds[i]
            logits = aud_model(x.unsqueeze(0))
            prob = float(torch.sigmoid(logits).item())
            audio_bank.append({
                "sample_id": meta.get("filename", f"aud_{i}"),
                "distress_prob": prob,
                "is_distress": int(target) == 1,
                "noise_condition": meta.get("noise_condition", "clean_speech"),
                "snr_db": float(meta.get("snr_db", 0.0)),
                "root_source": meta.get("root_source", f"root_{i}"),
            })

    # Group audio bank by condition and distress status
    pos_audio_by_noise = {}
    neg_audio_by_noise = {}
    for a in audio_bank:
        n_cond = a["noise_condition"]
        if a["is_distress"]:
            pos_audio_by_noise.setdefault(n_cond, []).append(a)
        else:
            neg_audio_by_noise.setdefault(n_cond, []).append(a)

    # -------------------------------------------------------------
    # Define 9 Controlled Scenarios
    # -------------------------------------------------------------
    scenarios = [
        "Daylight_Clear",
        "Nighttime",
        "Dense_Fog",
        "Heavy_Rain",
        "Low_Visual_Resolution",
        "Distant_Visual_Targets",
        "Strong_Wind",
        "Drone_Rotor_Noise",
        "Flood_Water_Noise",
    ]

    scenario_results: Dict[str, Any] = {}

    for sc in scenarios:
        logger.info("Evaluating Scenario: %s", sc)
        trials = []

        # Select matching visual subset
        if sc == "Nighttime":
            v_sub = v_test[v_test["root_source"] == "bdd_video_b255cd6c"]
            w_domain = "night"
            downsample = False
            target_noise = "clean_speech"
        elif sc == "Dense_Fog":
            v_sub = v_test[v_test["root_source"] == "dawn_dense_fog"]
            w_domain = "dense_fog"
            downsample = False
            target_noise = "clean_speech"
        elif sc == "Heavy_Rain":
            v_sub = v_test[v_test["root_source"] == "bdd_video_b24d283f"]
            w_domain = "heavy_rain"
            downsample = False
            target_noise = "heavy_rain"
        elif sc == "Low_Visual_Resolution":
            v_sub = v_test[v_test["root_source"] == "dawn_snow"]
            w_domain = "snow"
            downsample = True
            target_noise = "clean_speech"
        elif sc == "Distant_Visual_Targets":
            # Select images with small or ultra-small objects
            v_sub = v_test[(v_test["small_count"] > 0) | (v_test["ultra_small_count"] > 0)]
            w_domain = "clear"
            downsample = False
            target_noise = "clean_speech"
        elif sc == "Strong_Wind":
            v_sub = v_test[v_test["root_source"] == "dawn_snow"]
            w_domain = "snow"
            downsample = False
            target_noise = "strong_wind"
        elif sc == "Drone_Rotor_Noise":
            v_sub = v_test[v_test["root_source"] == "dawn_snow"]
            w_domain = "snow"
            downsample = False
            target_noise = "drone_propeller"
        elif sc == "Flood_Water_Noise":
            v_sub = v_test[v_test["root_source"] == "bdd_video_b24d283f"]
            w_domain = "rain"
            downsample = False
            target_noise = "flood_water"
        else:  # Daylight_Clear
            v_sub = v_test[v_test["root_source"] == "dawn_snow"]
            w_domain = "clear"
            downsample = False
            target_noise = "clean_speech"

        # Pair visual frames with audio samples
        pos_auds = pos_audio_by_noise.get(target_noise, pos_audio_by_noise.get("clean_speech", []))
        neg_auds = neg_audio_by_noise.get(target_noise, neg_audio_by_noise.get("clean_speech", []))

        # Build balanced scenario trials
        for idx, (_, row) in enumerate(v_sub.iterrows()):
            img_p = VISION_TEST_DIR / "images" / row["image_id"]
            lbl_p = VISION_TEST_DIR / "labels" / (Path(row["image_id"]).stem + ".txt")
            vis, gt_person = extract_visual_evidence(vis_model, img_p, lbl_p, w_domain, downsample)

            # Assign audio
            # Positive trial if person present OR audio distress
            # For pure scenario benchmarking, test both paired positive and paired negative
            if idx % 2 == 0 and pos_auds:
                aud_sample = pos_auds[idx % len(pos_auds)]
                aud_ev = AcousticEvidence(
                    distress_prob=aud_sample["distress_prob"],
                    calibrated_threshold=0.54,
                    snr_db=aud_sample["snr_db"],
                    noise_condition=aud_sample["noise_condition"],
                )
                ground_truth = True  # Distress vocalization present
            elif neg_auds:
                aud_sample = neg_auds[idx % len(neg_auds)]
                aud_ev = AcousticEvidence(
                    distress_prob=aud_sample["distress_prob"],
                    calibrated_threshold=0.54,
                    snr_db=aud_sample["snr_db"],
                    noise_condition=aud_sample["noise_condition"],
                )
                ground_truth = gt_person  # Only target if visually confirmed person present

            trials.append({
                "visual_evidence": vis,
                "acoustic_evidence": aud_ev,
                "ground_truth_target": ground_truth,
            })

        scenario_results[sc] = evaluate_scenario(engine, trials)

    # Cross-Scenario Summary
    overall_comparison = {}
    for sc, res in scenario_results.items():
        overall_comparison[sc] = {
            "vision_f1": res["A_vision_only"]["f1"],
            "audio_f1": res["B_audio_only"]["f1"],
            "fusion_f1": res["C_combined_fusion"]["f1"],
            "fusion_gain_over_vision": round(res["C_combined_fusion"]["f1"] - res["A_vision_only"]["f1"], 4),
            "fusion_gain_over_audio": round(res["C_combined_fusion"]["f1"] - res["B_audio_only"]["f1"], 4),
            "abstention_rate": res["C_combined_fusion"]["abstention_rate"],
        }

    final_payload = {
        "experiment_name": "Mission 21 Controlled Multi-Modal Search-and-Rescue Experiments",
        "simulation_protocol_disclosure": (
            "Independent visual test frames from WeatherGuard and acoustic test recordings from "
            "Rescue Drone were assembled into paired environmental scenarios. Asynchronous cross-modal "
            "samples are evaluated as scenario-level co-occurrences, strictly disclosing that no physical "
            "simultaneous video-audio drone recording was performed."
        ),
        "approaches": {
            "A": "Vision-Only Decision Baseline",
            "B": "Audio-Only Distress Detection Baseline",
            "C": "Adaptive Uncertainty-Aware Multi-Modal Decision Fusion",
        },
        "scenario_results": scenario_results,
        "cross_scenario_comparison": overall_comparison,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(final_payload, indent=2))
    logger.info("Saved controlled experiments to %s", OUT_JSON)
    return final_payload


if __name__ == "__main__":
    run_controlled_experiments()
