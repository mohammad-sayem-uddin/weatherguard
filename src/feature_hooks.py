"""
WeatherGuard Feature Hooks Module (Mission 8)
Provides clean, memory-safe forward hooks to capture intermediate
backbone and neck feature maps from YOLOv8 models.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple, Sequence, Optional, Any
import torch
import torch.nn as nn
from ultralytics import YOLO

class FeatureHookManager:
    """Manages forward hooks for capturing multi-scale feature maps from YOLOv8n."""

    def __init__(self, model: YOLO, target_layers: Optional[Sequence[int]] = None):
        """
        target_layers: list of layer indices to hook.
        Default: [4 (Backbone P3), 6 (Backbone P4), 9 (Backbone P5),
                  15 (Neck P3), 18 (Neck P4), 21 (Neck P5)]
        """
        self.yolo_model = model
        self.nn_model = model.model
        if target_layers is None:
            self.target_layers = [4, 6, 9, 15, 18, 21]
        else:
            self.target_layers = list(target_layers)

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.features: Dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook_fn(module: nn.Module, inp: Any, out: Any):
            # Store cloned or detached tensor
            if isinstance(out, torch.Tensor):
                self.features[layer_idx] = out
            elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                self.features[layer_idx] = out[0]
        return hook_fn

    def register_hooks(self) -> None:
        """Registers hooks on target layers."""
        self.remove_hooks()
        for idx in self.target_layers:
            layer = self.nn_model.model[idx]
            handle = layer.register_forward_hook(self._make_hook(idx))
            self.hooks.append(handle)

    def remove_hooks(self) -> None:
        """Removes all registered hooks to avoid memory leaks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def clear_features(self) -> None:
        """Clears captured feature maps."""
        self.features.clear()

    def extract_features(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Runs a forward pass and returns captured features."""
        self.clear_features()
        if not self.hooks:
            self.register_hooks()
        with torch.no_grad():
            _ = self.nn_model(x)
        return dict(self.features)

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        self.clear_features()


def compute_feature_similarity(feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
    """
    Computes mean cosine similarity across channel vectors after spatial alignment.
    Both tensors must have shape (B, C, H, W). If H and W differ, feat_a is interpolated to match feat_b.
    """
    if feat_a.shape[2:] != feat_b.shape[2:]:
        feat_a = nn.functional.interpolate(feat_a, size=feat_b.shape[2:], mode="bilinear", align_corners=False)

    # Flatten spatial dimensions: (B, C, H*W)
    b, c, h, w = feat_a.shape
    v_a = feat_a.view(b, c, -1)
    v_b = feat_b.view(b, c, -1)

    # Cosine similarity along channel dimension: shape (B, H*W)
    cos = nn.functional.cosine_similarity(v_a, v_b, dim=1)
    return float(cos.mean().item())


def compute_linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes Linear CKA between two feature representations flattened to (N, C).
    """
    if x.shape[2:] != y.shape[2:]:
        x = nn.functional.interpolate(x, size=y.shape[2:], mode="bilinear", align_corners=False)

    # Reshape to (H*W, C)
    x_flat = x.view(x.size(1), -1).t().float()
    y_flat = y.view(y.size(1), -1).t().float()

    # Center representations
    x_c = x_flat - x_flat.mean(dim=0, keepdim=True)
    y_c = y_flat - y_flat.mean(dim=0, keepdim=True)

    # HSIC
    hsic_xy = torch.norm(torch.mm(x_c.t(), y_c), p="fro") ** 2
    hsic_xx = torch.norm(torch.mm(x_c.t(), x_c), p="fro") ** 2
    hsic_yy = torch.norm(torch.mm(y_c.t(), y_c), p="fro") ** 2

    denom = torch.sqrt(hsic_xx * hsic_yy) + 1e-10
    return float((hsic_xy / denom).item())
