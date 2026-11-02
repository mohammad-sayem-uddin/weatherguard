"""Physics-based and photometric adverse-weather augmentations for WeatherGuard.

Implements reproducible transforms for simulating weather degradations:
  - Atmospheric Fog (Koschmieder's law with vertical road-depth gradient)
  - Directional Rain Streaks
  - Defocus / Haze Blur
  - Photometric Jitter (brightness, contrast, HSV)
"""

from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def apply_atmospheric_fog(
    image: Image.Image,
    beta: float = 0.8,
    atmospheric_light: float = 0.85,
    seed: int | None = None,
) -> Image.Image:
    """Simulates atmospheric fog using Koschmieder's law:
        I(x) = J(x) * t(x) + A * (1 - t(x))
    where t(x) = exp(-beta * d(x)) with a realistic road-scene vertical depth prior.
    """
    if seed is not None:
        np.random.seed(seed)

    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = arr.shape

    # Vertical depth prior: distant objects (near horizon / upper 40%) have larger depth
    y_coords = np.linspace(0, 1, h)[:, None]
    # Invert so horizon (near top) has higher distance, foreground (bottom) has lower distance
    depth = np.clip(1.2 - 0.9 * y_coords, 0.2, 1.2)
    # Broadcast across width
    depth = np.repeat(depth, w, axis=1)

    # Transmission map
    transmission = np.exp(-beta * depth)[:, :, None]
    fogged = arr * transmission + atmospheric_light * (1.0 - transmission)
    fogged = np.clip(fogged * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(fogged)


def apply_rain_streaks(
    image: Image.Image,
    density: int = 400,
    length: int = 25,
    angle_deg: float = 15.0,
    intensity: float = 0.6,
    seed: int | None = None,
) -> Image.Image:
    """Simulates directional rain streaks overlaying the driving scene."""
    rng = random.Random(seed)
    w, h = image.size

    # Create rain streak mask
    rain_mask = np.zeros((h, w), dtype=np.float32)
    rad = math.radians(angle_deg)
    dx = math.sin(rad) * length
    dy = math.cos(rad) * length

    for _ in range(density):
        x0 = rng.randint(0, w - 1)
        y0 = rng.randint(0, h - 1)
        x1 = int(round(x0 + dx))
        y1 = int(round(y0 + dy))

        # Bresenham-like linear interpolation
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, steps).astype(int)
        ys = np.linspace(y0, y1, steps).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        rain_mask[ys[valid], xs[valid]] = rng.uniform(0.4, 0.8) * intensity

    arr = np.array(image.convert("RGB"), dtype=np.float32)
    # Additive rain streak blending
    rain_layer = rain_mask[:, :, None] * 255.0
    blended = np.clip(arr + rain_layer, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended)


def apply_defocus_blur(image: Image.Image, radius: float = 1.2) -> Image.Image:
    """Applies Gaussian defocus blur to simulate lens water condensation / haze diffusion."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_photometric_jitter(
    image: Image.Image,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0,
    color_factor: float = 1.0,
) -> Image.Image:
    """Applies standard photometric transformations (brightness, contrast, saturation)."""
    img = image.convert("RGB")
    if brightness_factor != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    if contrast_factor != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    if color_factor != 1.0:
        img = ImageEnhance.Color(img).enhance(color_factor)
    return img


def weather_augment_policy(
    image: Image.Image,
    policy: str,
    seed: int | None = None,
) -> Image.Image:
    """Applies the specified adverse weather or photometric augmentation policy:
      - 'photo': Photometric jitter (brightness +/-20%, contrast +/-20%, saturation +/-20%)
      - 'weather': Physical weather transforms (fog, rain, or blur)
      - 'combined': Photometric jitter + adverse weather transform
    """
    rng = random.Random(seed)
    img = image.copy()

    if policy == "photo":
        b_fac = rng.uniform(0.8, 1.2)
        c_fac = rng.uniform(0.8, 1.2)
        s_fac = rng.uniform(0.7, 1.3)
        return apply_photometric_jitter(img, b_fac, c_fac, s_fac)

    elif policy == "weather":
        effect = rng.choice(["fog", "rain", "fog_rain", "blur"])
        if effect in ("fog", "fog_rain"):
            beta = rng.uniform(0.5, 1.0)
            img = apply_atmospheric_fog(img, beta=beta, seed=seed)
        if effect in ("rain", "fog_rain"):
            dens = rng.randint(300, 600)
            ang = rng.uniform(-20.0, 20.0)
            img = apply_rain_streaks(img, density=dens, angle_deg=ang, seed=seed)
        if effect == "blur":
            img = apply_defocus_blur(img, radius=rng.uniform(0.8, 1.5))
        return img

    elif policy == "combined":
        # First photometric
        b_fac = rng.uniform(0.85, 1.15)
        c_fac = rng.uniform(0.85, 1.15)
        s_fac = rng.uniform(0.8, 1.2)
        img = apply_photometric_jitter(img, b_fac, c_fac, s_fac)
        # Then adverse weather
        beta = rng.uniform(0.4, 0.8)
        img = apply_atmospheric_fog(img, beta=beta, seed=seed)
        if rng.random() > 0.5:
            dens = rng.randint(250, 450)
            img = apply_rain_streaks(img, density=dens, seed=seed)
        return img

    else:
        raise ValueError(f"Unknown augmentation policy: {policy}")
