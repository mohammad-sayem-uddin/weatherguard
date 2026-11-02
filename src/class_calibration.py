"""
WeatherGuard Class-Calibrated Confidence Module
Implements mathematically rigorous temperature scaling on YOLOv8 sigmoid logits,
per-class temperature scaling with sample-size shrinkage, and class-specific confidence thresholds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Sequence, Optional
import numpy as np

CANONICAL_CLASSES = ["Car", "Person", "Truck", "Bus", "Bicycle", "Motorcycle"]

def probability_to_logit(p: float, eps: float = 1e-7) -> float:
    """Inverts the sigmoid activation to obtain the exact pre-sigmoid classification logit."""
    p_clipped = min(max(p, eps), 1.0 - eps)
    return float(math.log(p_clipped / (1.0 - p_clipped)))

def logit_to_probability(z: float) -> float:
    """Applies sigmoid activation to logit."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

def apply_temperature_scaling(prob: float, temperature: float) -> float:
    """Scales logit by temperature T and maps back through sigmoid."""
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    z = probability_to_logit(prob)
    z_scaled = z / temperature
    return logit_to_probability(z_scaled)

def compute_ece(
    confidences: Sequence[float],
    accuracies: Sequence[int],
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Computes Expected Calibration Error (ECE) and reliability diagram bins."""
    if not confidences or len(confidences) != len(accuracies):
        return 0.0, []

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confs = np.array(confidences)
    accs = np.array(accuracies)
    n_samples = len(confs)

    ece = 0.0
    bins_data = []

    for bl, bu in zip(bin_lowers, bin_uppers):
        in_bin = (confs > bl) & (confs <= bu)
        prop_in_bin = float(np.mean(in_bin))
        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(accs[in_bin]))
            bin_conf = float(np.mean(confs[in_bin]))
            bin_err = abs(bin_acc - bin_conf)
            ece += bin_err * prop_in_bin
            bins_data.append({
                "bin_lower": float(bl),
                "bin_upper": float(bu),
                "count": bin_count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "error": bin_err,
            })
        else:
            bins_data.append({
                "bin_lower": float(bl),
                "bin_upper": float(bu),
                "count": 0,
                "accuracy": 0.0,
                "confidence": (bl + bu) / 2.0,
                "error": 0.0,
            })

    return float(ece), bins_data

@dataclass
class Calibrator:
    """Class-aware confidence calibrator."""
    global_temperature: float = 1.0
    class_temperatures: Dict[str, float] = field(default_factory=dict)
    class_thresholds: Dict[str, float] = field(default_factory=dict)
    shrinkage_prior_n: int = 30  # Bayesian shrinkage threshold for rare classes

    def calibrate_detection(self, class_name: str, raw_score: float) -> Tuple[float, bool]:
        """Returns (calibrated_score, passes_threshold)."""
        temp = self.class_temperatures.get(class_name, self.global_temperature)
        calibrated_score = apply_temperature_scaling(raw_score, temp)
        threshold = self.class_thresholds.get(class_name, 0.20)
        passes = calibrated_score >= threshold
        return calibrated_score, passes

    def fit_temperatures(
        self,
        class_samples: Dict[str, List[Tuple[float, int]]],
        candidate_temps: Sequence[float] = np.linspace(0.5, 2.5, 41),
    ):
        """Fits temperature per class by minimizing NLL with Bayesian shrinkage for rare classes."""
        # 1. Global temperature across all samples
        all_samples = []
        for s in class_samples.values():
            all_samples.extend(s)

        if all_samples:
            best_g_temp = 1.0
            best_g_nll = float("inf")
            for t in candidate_temps:
                nll = 0.0
                for prob, is_tp in all_samples:
                    p_scaled = apply_temperature_scaling(prob, t)
                    p_c = max(min(p_scaled, 1.0 - 1e-7), 1e-7)
                    nll -= math.log(p_c) if is_tp else math.log(1.0 - p_c)
                if nll < best_g_nll:
                    best_g_nll = nll
                    best_g_temp = float(t)
            self.global_temperature = best_g_temp

        # 2. Per-class temperature
        for c_name, samples in class_samples.items():
            n = len(samples)
            if n < 5:
                # Completely default to global temperature
                self.class_temperatures[c_name] = self.global_temperature
                continue

            best_c_temp = self.global_temperature
            best_c_nll = float("inf")
            for t in candidate_temps:
                nll = 0.0
                for prob, is_tp in samples:
                    p_scaled = apply_temperature_scaling(prob, t)
                    p_c = max(min(p_scaled, 1.0 - 1e-7), 1e-7)
                    nll -= math.log(p_c) if is_tp else math.log(1.0 - p_c)
                if nll < best_c_nll:
                    best_c_nll = nll
                    best_c_temp = float(t)

            # Bayesian shrinkage towards global temperature based on sample size
            shrinkage = min(1.0, n / float(self.shrinkage_prior_n))
            final_temp = (shrinkage * best_c_temp) + ((1.0 - shrinkage) * self.global_temperature)
            self.class_temperatures[c_name] = float(final_temp)

    def fit_thresholds(
        self,
        class_detections: Dict[str, List[Tuple[float, int, int]]],  # (score, is_tp, n_gt)
        candidate_thresholds: Sequence[float] = np.linspace(0.10, 0.60, 26),
        default_thresh: float = 0.20,
    ):
        """Finds F1-optimal operating threshold per class on development data."""
        for c_name, data in class_detections.items():
            if not data:
                self.class_thresholds[c_name] = default_thresh
                continue
            n_gt = data[0][2]
            if n_gt < 5:
                # Insufficient ground truth to tune threshold
                self.class_thresholds[c_name] = default_thresh
                continue

            best_th = default_thresh
            best_f1 = -1.0
            for th in candidate_thresholds:
                tp = sum(1 for s, is_tp, _ in data if s >= th and is_tp)
                fp = sum(1 for s, is_tp, _ in data if s >= th and not is_tp)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / n_gt if n_gt > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = float(th)

            # Rare-class protection
            if n_gt < self.shrinkage_prior_n:
                # Shrink towards default 0.20
                alpha = n_gt / float(self.shrinkage_prior_n)
                best_th = float(alpha * best_th + (1.0 - alpha) * default_thresh)

            self.class_thresholds[c_name] = best_th
