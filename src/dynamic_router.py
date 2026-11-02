"""
WeatherGuard Dynamic Router Module
Extracts lightweight routing signals from base detector predictions and implements
deterministic and learned uncertainty routing policies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Sequence, Optional
import numpy as np

@dataclass
class RoutingFeatures:
    """Lightweight signals extracted in < 0.2 ms from base detections."""
    num_detections: int = 0
    num_candidates_low: int = 0  # detections in score range [0.10, 0.30]
    num_small: int = 0           # bbox area < 1024
    num_medium: int = 0          # bbox area in [1024, 9216)
    num_large: int = 0           # bbox area >= 9216
    max_conf: float = 0.0
    mean_conf: float = 0.0
    std_conf: float = 0.0
    conf_margin: float = 0.0     # max_conf - 0.20 (distance from operating threshold)
    density: float = 0.0         # total candidate count / 10.0
    small_ratio: float = 0.0     # num_small / max(num_detections, 1)
    spatial_spread: float = 0.0  # normalized variance of box centers across image

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.num_detections,
            self.num_candidates_low,
            self.num_small,
            self.num_medium,
            self.num_large,
            self.max_conf,
            self.mean_conf,
            self.std_conf,
            self.conf_margin,
            self.density,
            self.small_ratio,
            self.spatial_spread,
        ], dtype=np.float32)

def extract_routing_features(
    raw_predictions: Sequence[Dict[str, Any]],
    score_threshold: float = 0.20,
    low_band: Tuple[float, float] = (0.10, 0.30),
    img_shape: Tuple[int, int] = (720, 1280),
) -> RoutingFeatures:
    """Extracts non-causal routing signals using only inference output from base pass."""
    if not raw_predictions:
        return RoutingFeatures()

    # Detections above operating threshold
    op_preds = [p for p in raw_predictions if p.get("score", 0.0) >= score_threshold]
    all_scores = [p.get("score", 0.0) for p in raw_predictions]
    op_scores = [p.get("score", 0.0) for p in op_preds]

    num_det = len(op_preds)
    num_low = sum(1 for s in all_scores if low_band[0] <= s <= low_band[1])

    num_small = 0
    num_med = 0
    num_lrg = 0
    centers_x = []
    centers_y = []

    for p in op_preds:
        b = p.get("box_xyxy", [0, 0, 0, 0])
        w = max(0.0, b[2] - b[0])
        h = max(0.0, b[3] - b[1])
        area = w * h
        if area < 1024:
            num_small += 1
        elif area < 9216:
            num_med += 1
        else:
            num_lrg += 1
        centers_x.append((b[0] + b[2]) / 2.0 / max(img_shape[1], 1))
        centers_y.append((b[1] + b[3]) / 2.0 / max(img_shape[0], 1))

    max_c = max(op_scores) if op_scores else (max(all_scores) if all_scores else 0.0)
    mean_c = float(np.mean(op_scores)) if op_scores else 0.0
    std_c = float(np.std(op_scores)) if len(op_scores) > 1 else 0.0
    conf_margin = max_c - score_threshold

    density = len(all_scores) / 10.0
    small_ratio = num_small / max(num_det, 1)

    if len(centers_x) > 1:
        spread = float(np.var(centers_x) + np.var(centers_y))
    else:
        spread = 0.0

    return RoutingFeatures(
        num_detections=num_det,
        num_candidates_low=num_low,
        num_small=num_small,
        num_medium=num_med,
        num_large=num_lrg,
        max_conf=max_c,
        mean_conf=mean_c,
        std_conf=std_c,
        conf_margin=conf_margin,
        density=density,
        small_ratio=small_ratio,
        spatial_spread=spread,
    )

class BaseRoutingPolicy:
    """Base class for all test-time routing policies."""
    name: str = "base_policy"

    def route(self, features: RoutingFeatures) -> bool:
        """Returns True if the policy decides to escalate to second stage, False otherwise."""
        raise NotImplementedError

class PolicyFixedBaseline(BaseRoutingPolicy):
    """Policy 1: Always execute base 640px detector (0% escalation)."""
    name = "policy1_fixed_baseline"
    def route(self, features: RoutingFeatures) -> bool:
        return False

class PolicyLowConfidence(BaseRoutingPolicy):
    """Policy 2: Escalate if number of low-confidence detections >= threshold."""
    name = "policy2_low_confidence"
    def __init__(self, k_low: int = 1):
        self.k_low = k_low
    def route(self, features: RoutingFeatures) -> bool:
        return features.num_candidates_low >= self.k_low

class PolicyConfidenceMargin(BaseRoutingPolicy):
    """Policy 3: Escalate when top detection confidence margin is below threshold."""
    name = "policy3_confidence_margin"
    def __init__(self, min_margin: float = 0.40):
        self.min_margin = min_margin
    def route(self, features: RoutingFeatures) -> bool:
        return features.conf_margin < self.min_margin

class PolicyDetectionDensity(BaseRoutingPolicy):
    """Policy 4: Escalate when candidate density indicates a complex scene."""
    name = "policy4_detection_density"
    def __init__(self, min_candidates: int = 15):
        self.min_candidates = min_candidates
    def route(self, features: RoutingFeatures) -> bool:
        return (features.density * 10.0) >= self.min_candidates

class PolicySmallObject(BaseRoutingPolicy):
    """Policy 5: Escalate when small object candidates are detected."""
    name = "policy5_small_object"
    def __init__(self, k_small: int = 1):
        self.k_small = k_small
    def route(self, features: RoutingFeatures) -> bool:
        return features.num_small >= self.k_small

class PolicyCombinedUncertainty(BaseRoutingPolicy):
    """Policy 6: Composite score balancing small candidates, low confidence, and peak confidence."""
    name = "policy6_combined_uncertainty"
    def __init__(
        self,
        w_small: float = 1.5,
        w_low: float = 1.0,
        w_density: float = 0.5,
        w_max_conf: float = 1.0,
        threshold: float = 2.0,
    ):
        self.w_small = w_small
        self.w_low = w_low
        self.w_density = w_density
        self.w_max_conf = w_max_conf
        self.threshold = threshold

    def compute_score(self, features: RoutingFeatures) -> float:
        return (
            self.w_small * features.num_small
            + self.w_low * features.num_candidates_low
            + self.w_density * features.density
            - self.w_max_conf * features.max_conf
        )

    def route(self, features: RoutingFeatures) -> bool:
        return self.compute_score(features) >= self.threshold

class PolicyLearnedRouter(BaseRoutingPolicy):
    """Policy 7: Logistic Regression classifier trained on Group B Dev."""
    name = "policy7_learned_router"
    def __init__(self, weights: np.ndarray, bias: float, threshold: float = 0.50):
        self.weights = np.array(weights, dtype=np.float32)
        self.bias = float(bias)
        self.threshold = float(threshold)

    def predict_proba(self, features: RoutingFeatures) -> float:
        x = features.to_vector()
        logit = float(np.dot(self.weights, x) + self.bias)
        # Numerically stable sigmoid
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        else:
            exp_z = math.exp(logit)
            return exp_z / (1.0 + exp_z)

    def route(self, features: RoutingFeatures) -> bool:
        return self.predict_proba(features) >= self.threshold
