"""Unit tests for Class Calibration (src/class_calibration.py)."""

import math
import pytest
from src.class_calibration import (
    probability_to_logit,
    logit_to_probability,
    apply_temperature_scaling,
    compute_ece,
    Calibrator,
)


def test_sigmoid_and_logit_inversion():
    probs = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    for p in probs:
        z = probability_to_logit(p)
        p_rec = logit_to_probability(z)
        assert abs(p - p_rec) < 1e-5


def test_temperature_scaling_softening():
    # An overconfident probability (0.90) scaled by T = 2.0 should drop towards 0.50
    p = 0.90
    p_scaled = apply_temperature_scaling(p, temperature=2.0)
    assert p_scaled < p
    assert p_scaled > 0.50

    # A low probability (0.10) scaled by T = 2.0 should rise towards 0.50
    p_low = 0.10
    p_low_scaled = apply_temperature_scaling(p_low, temperature=2.0)
    assert p_low_scaled > p_low
    assert p_low_scaled < 0.50


def test_calibrator_class_routing():
    temperatures = {
        "Car": 1.15,
        "Person": 1.25,
        "Truck": 1.10,
    }
    thresholds = {
        "Car": 0.16,
        "Person": 0.22,
        "Truck": 0.12,
    }
    calibrator = Calibrator(
        global_temperature=1.15,
        class_temperatures=temperatures,
        class_thresholds=thresholds,
    )

    # Car with score 0.18 -> maps to ~0.227 >= 0.16 -> passes
    score, passes = calibrator.calibrate_detection("Car", 0.18)
    assert passes is True  # passes Car threshold 0.16

    # Person with raw score 0.12 -> maps to ~0.168 < 0.22 -> fails
    score_p, passes_p = calibrator.calibrate_detection("Person", 0.12)
    assert passes_p is False  # fails Person threshold 0.22


def test_compute_ece_perfect():
    # If confidence equals accuracy, ECE should be near 0
    confs = [0.1] * 10 + [0.9] * 10
    accs = [0] * 9 + [1] * 1 + [1] * 9 + [0] * 1  # 1/10 = 0.1, 9/10 = 0.9
    ece, bins = compute_ece(confs, accs, n_bins=10)
    assert ece < 0.05
