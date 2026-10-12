"""Visualization helpers — kept lightweight, no plotly/bokeh."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .dataset import BBox, DawnRecord
from .model import Prediction


def _try_load_default_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[tuple[float, float, float, float]],
    labels: Sequence[str],
    color: tuple[int, int, int] = (255, 0, 0),
    width: int = 2,
    font_size: int = 12,
) -> Image.Image:
    """Draw axis-aligned boxes on a copy of `image` and return the copy."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = _try_load_default_font(font_size)
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if label:
            # background box for label readability
            try:
                tb = draw.textbbox((x1, y1), label, font=font)
            except Exception:
                tb = (x1, y1, x1 + 7 * len(label), y1 + font_size + 2)
            draw.rectangle([tb[0], tb[1], tb[2], tb[3]], fill=color)
            draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    return out


def render_record(
    record: DawnRecord,
    *,
    box_color: tuple[int, int, int] = (0, 180, 0),
) -> Image.Image:
    img = record.to_pil()
    boxes = [b.to_xyxy() for b in record.objects]
    labels = [b.class_name for b in record.objects]
    return draw_boxes(img, boxes, labels, color=box_color)


def render_prediction(
    record: DawnRecord,
    prediction: Prediction,
    *,
    score_threshold: float = 0.0,
    pred_color: tuple[int, int, int] = (220, 30, 30),
    gt_color: tuple[int, int, int] = (0, 160, 0),
) -> Image.Image:
    img = record.to_pil()
    # ground truth
    gt_boxes = [b.to_xyxy() for b in record.objects]
    gt_labels = [f"GT:{b.class_name}" for b in record.objects]
    img = draw_boxes(img, gt_boxes, gt_labels, color=gt_color, width=2)
    # predictions
    pred_boxes = [d.box_xyxy for d in prediction.detections if d.score >= score_threshold]
    pred_labels = [
        f"{d.common_label or d.label}:{d.score:.2f}"
        for d in prediction.detections
        if d.score >= score_threshold
    ]
    return draw_boxes(img, pred_boxes, pred_labels, color=pred_color, width=2)


def render_comparison_grid(
    panels: Sequence[tuple[str, Image.Image]],
    cols: int = 2,
) -> Image.Image:
    """Build a simple grid of labelled panels (used by the dashboard)."""
    if not panels:
        return Image.new("RGB", (320, 240), color=(255, 255, 255))
    # Normalise panel sizes for the grid.
    target_w = max(p[1].width for p in panels)
    target_h = max(p[1].height for p in panels)
    rows = (len(panels) + cols - 1) // cols
    cell_w = target_w
    cell_h = target_h + 28
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(grid)
    font = _try_load_default_font(14)
    for idx, (label, img) in enumerate(panels):
        r, c = divmod(idx, cols)
        x, y = c * cell_w, r * cell_h
        grid.paste(img, (x, y + 24))
        draw.rectangle([x, y, x + cell_w, y + 24], fill=(40, 40, 40))
        draw.text((x + 6, y + 4), label, fill=(255, 255, 255), font=font)
    return grid


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
