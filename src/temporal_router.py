"""
WeatherGuard Temporal Router Module (Mission 8)
Implements temporal smoothing and hysteresis routing policies for continuous video streams:
1. Frame-Independent Policy 5a
2. Moving-Average Routing (SMA)
3. Exponential Moving Average Routing (EMA)
4. Hysteresis-Based Routing (Dual Thresholds)
5. Minimum Escalation Hold-Time Gating
6. Scene-Change-Triggered Dynamic Routing
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Sequence, Optional, Any
from dataclasses import dataclass, field
import numpy as np

@dataclass
class TemporalDecision:
    escalate: bool
    raw_score: float
    smoothed_score: float
    state: str  # "hold", "switched_on", "switched_off", "stable"
    hold_remaining: int

class TemporalRouter:
    """
    Manages temporal routing state across sequential video frames.
    Prevents high-frequency switching oscillations while preserving reaction speed.
    """
    def __init__(
        self,
        method: str = "hysteresis",
        window_size: int = 5,
        ema_alpha: float = 0.35,
        tau_high: float = 1.0,   # threshold to escalate (e.g. k_small >= 1)
        tau_low: float = 0.3,    # threshold to de-escalate
        min_hold_frames: int = 3, # minimum frames to remain escalated once triggered
    ):
        self.method = method
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.min_hold_frames = min_hold_frames

        # State tracking
        self.history_scores: List[float] = []
        self.ema_score: Optional[float] = None
        self.current_escalated: bool = False
        self.hold_counter: int = 0
        self.decision_history: List[bool] = []

    def reset(self) -> None:
        """Resets temporal router state at sequence boundaries."""
        self.history_scores.clear()
        self.ema_score = None
        self.current_escalated = False
        self.hold_counter = 0
        self.decision_history.clear()

    def update(self, raw_score: float) -> TemporalDecision:
        """
        Receives raw frame routing score (e.g. small candidate count k_small)
        and computes the temporally stabilized escalation decision.
        """
        self.history_scores.append(raw_score)
        if len(self.history_scores) > self.window_size:
            self.history_scores.pop(0)

        # 1. Compute smoothed score
        if self.method == "frame_independent":
            smoothed = raw_score
            should_escalate = smoothed >= self.tau_high

        elif self.method == "moving_average":
            smoothed = float(np.mean(self.history_scores))
            should_escalate = smoothed >= self.tau_high

        elif self.method == "ema":
            if self.ema_score is None:
                self.ema_score = raw_score
            else:
                self.ema_score = self.ema_alpha * raw_score + (1.0 - self.ema_alpha) * self.ema_score
            smoothed = self.ema_score
            should_escalate = smoothed >= self.tau_high

        elif self.method == "hysteresis":
            # Dual-threshold Schmitt trigger
            smoothed = raw_score
            if self.current_escalated:
                # Need to fall below tau_low to drop escalation
                should_escalate = smoothed >= self.tau_low
            else:
                # Need to exceed tau_high to trigger escalation
                should_escalate = smoothed >= self.tau_high

        elif self.method == "hold_time":
            # Hold time: once escalated, hold for at least min_hold_frames
            smoothed = raw_score
            if smoothed >= self.tau_high:
                should_escalate = True
                self.hold_counter = self.min_hold_frames
            elif self.hold_counter > 0:
                should_escalate = True
                self.hold_counter -= 1
            else:
                should_escalate = False

        else:
            raise ValueError(f"Unknown temporal routing method: {self.method}")

        # State transition analysis
        prev_state = self.current_escalated
        self.current_escalated = should_escalate
        self.decision_history.append(should_escalate)

        if not prev_state and should_escalate:
            state_str = "switched_on"
        elif prev_state and not should_escalate:
            state_str = "switched_off"
        elif should_escalate and self.hold_counter > 0:
            state_str = "hold"
        else:
            state_str = "stable"

        return TemporalDecision(
            escalate=should_escalate,
            raw_score=raw_score,
            smoothed_score=smoothed,
            state=state_str,
            hold_remaining=self.hold_counter,
        )


def compute_temporal_stability_metrics(decisions: Sequence[bool]) -> Dict[str, float]:
    """
    Computes switching frequency, escalation rate, and average run lengths.
    """
    if not decisions:
        return {"escalation_rate": 0.0, "switching_frequency": 0.0, "mean_escalated_run": 0.0}

    n = len(decisions)
    escalated_count = sum(1 for d in decisions if d)
    escalation_rate = escalated_count / float(n)

    # Count state transitions
    switches = sum(1 for i in range(1, n) if decisions[i] != decisions[i-1])
    switching_freq = switches / float(max(n - 1, 1))

    # Compute run lengths of consecutive escalated frames
    runs = []
    current_run = 0
    for d in decisions:
        if d:
            current_run += 1
        elif current_run > 0:
            runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    mean_run = float(np.mean(runs)) if runs else 0.0

    return {
        "escalation_rate": escalation_rate,
        "switching_frequency": switching_freq,
        "total_switches": switches,
        "mean_escalated_run_length": mean_run,
        "total_runs": len(runs),
    }
