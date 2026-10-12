"""Controlled robustness experiments.

We define a small set of photometric corruptions and apply them to the
*same* source images, *same* annotations (because photometric transforms
preserve object geometry), *same* model, and *same* evaluation protocol.

These corruptions are not equivalent to real adverse weather. They are
useful as a controlled robustness benchmark to expose simple failure
modes (e.g. detector sensitivity to contrast or blur) that may also
appear under real weather.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class CorruptionConfig:
    name: str
    description: str
    # All parameters are stored verbatim and reported in results.
    params: dict


def corrupt_image(image: Image.Image, cfg: CorruptionConfig, seed: int = 0) -> Image.Image:
    """Apply a single corruption. Photometric only — never resize/crop."""
    if cfg.name == "identity":
        return image.copy()
    if cfg.name == "brightness_reduce":
        factor = float(cfg.params.get("factor", 0.5))
        return ImageEnhance.Brightness(image).enhance(factor)
    if cfg.name == "contrast_reduce":
        factor = float(cfg.params.get("factor", 0.5))
        return ImageEnhance.Contrast(image).enhance(factor)
    if cfg.name == "gaussian_noise":
        sigma = float(cfg.params.get("sigma", 25.0))
        rng = np.random.default_rng(seed)
        arr = np.asarray(image).astype(np.float32)
        noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode=image.mode)
    if cfg.name == "blur":
        radius = float(cfg.params.get("radius", 4.0))
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    raise ValueError(f"Unknown corruption: {cfg.name!r}")


def default_corruption_grid() -> list[CorruptionConfig]:
    """A small, documented grid of corruptions and severities."""
    return [
        CorruptionConfig("identity", "Uncorrupted baseline.", {}),
        CorruptionConfig("brightness_reduce", "Reduce brightness to 50%.", {"factor": 0.5}),
        CorruptionConfig("contrast_reduce", "Reduce contrast to 50%.", {"factor": 0.5}),
        CorruptionConfig("blur", "Gaussian blur with radius 4 px.", {"radius": 4.0}),
        CorruptionConfig(
            "gaussian_noise", "Additive Gaussian noise, sigma=25 in [0,255].",
            {"sigma": 25.0},
        ),
    ]


def encode_pil(img: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()
