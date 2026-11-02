"""Tests for physics-based and photometric weather augmentations."""

import numpy as np
from PIL import Image
import pytest

from src.weather_augmentation import (
    apply_atmospheric_fog,
    apply_rain_streaks,
    apply_defocus_blur,
    apply_photometric_jitter,
    weather_augment_policy,
)


def test_atmospheric_fog_shape_and_bounds():
    img = Image.new("RGB", (100, 80), color=(120, 100, 80))
    fogged = apply_atmospheric_fog(img, beta=0.8, seed=42)
    assert fogged.size == (100, 80)
    arr = np.array(fogged)
    assert arr.shape == (80, 100, 3)
    assert arr.dtype == np.uint8
    assert arr.min() >= 0
    assert arr.max() <= 255


def test_rain_streaks_shape_and_bounds():
    img = Image.new("RGB", (100, 80), color=(50, 50, 50))
    rained = apply_rain_streaks(img, density=100, seed=42)
    assert rained.size == (100, 80)
    arr = np.array(rained)
    assert arr.shape == (80, 100, 3)
    assert arr.dtype == np.uint8
    # Rain should add bright pixels
    assert arr.mean() >= np.array(img).mean()


def test_defocus_blur():
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    blurred = apply_defocus_blur(img, radius=1.0)
    assert blurred.size == (64, 64)


def test_photometric_jitter():
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    jittered = apply_photometric_jitter(img, 1.2, 1.1, 0.9)
    assert jittered.size == (64, 64)


def test_weather_augment_policies():
    img = Image.new("RGB", (80, 80), color=(100, 120, 140))
    for policy in ["photo", "weather", "combined"]:
        res = weather_augment_policy(img, policy=policy, seed=123)
        assert res.size == (80, 80)
        assert isinstance(res, Image.Image)

    with pytest.raises(ValueError):
        weather_augment_policy(img, policy="invalid_policy")
