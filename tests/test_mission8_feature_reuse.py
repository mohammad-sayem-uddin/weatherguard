"""
Unit tests for WeatherGuard Mission 8 Multi-Scale Feature Reuse & Refinement
"""

import pytest
import torch
import torch.nn as nn
from src.feature_hooks import FeatureHookManager
from src.feature_reuse import FeatureReuseDetector
from src.lightweight_refinement import (
    LightweightResidualRefiner,
    P3DetailRefiner,
    CrossScaleGuidanceRefiner,
)
from src.feature_distillation import MultiScaleDistillationLoss


class MockYOLOModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.Conv2d(32, 64, 3, padding=1),
        )

    def forward(self, x):
        return self.model(x)


class MockYOLOWrapper:
    def __init__(self):
        self.model = MockYOLOModel()


def test_feature_hook_manager_lifecycle():
    wrapper = MockYOLOWrapper()
    mgr = FeatureHookManager(wrapper, target_layers=[0, 2])
    assert len(mgr.target_layers) == 2

    mgr.register_hooks()
    assert len(mgr.hooks) == 2

    # Forward pass
    dummy = torch.randn(1, 3, 32, 32)
    _ = wrapper.model(dummy)

    assert 0 in mgr.features
    assert 2 in mgr.features
    assert mgr.features[0].shape == (1, 16, 32, 32)
    assert mgr.features[2].shape == (1, 64, 32, 32)

    mgr.clear_features()
    assert len(mgr.features) == 0

    mgr.remove_hooks()
    assert len(mgr.hooks) == 0


def test_lightweight_residual_refiner_forward():
    refiner = LightweightResidualRefiner()
    p3 = torch.randn(1, 64, 104, 104)
    p4 = torch.randn(1, 128, 52, 52)
    p5 = torch.randn(1, 256, 26, 26)

    out = refiner(p3, p4, p5)
    assert len(out) == 3
    assert out[0].shape == p3.shape
    assert out[1].shape == p4.shape
    assert out[2].shape == p5.shape

    params = sum(p.numel() for p in refiner.parameters())
    assert params == 91840


def test_p3_detail_refiner_forward():
    refiner = P3DetailRefiner()
    p3 = torch.randn(2, 64, 104, 104)
    p4 = torch.randn(2, 128, 52, 52)
    p5 = torch.randn(2, 256, 26, 26)

    out = refiner(p3, p4, p5)
    assert len(out) == 3
    assert out[0].shape == p3.shape
    params = sum(p.numel() for p in refiner.parameters())
    assert params == 5376


def test_cross_scale_refiner_forward():
    refiner = CrossScaleGuidanceRefiner()
    p3 = torch.randn(1, 64, 104, 104)
    p4 = torch.randn(1, 128, 52, 52)
    p5 = torch.randn(1, 256, 26, 26)

    out = refiner(p3, p4, p5)
    assert len(out) == 3
    assert out[0].shape == p3.shape
    assert out[1].shape == p4.shape
    assert out[2].shape == p5.shape
    params = sum(p.numel() for p in refiner.parameters())
    assert params == 45248


def test_feature_distillation_loss():
    loss_fn = MultiScaleDistillationLoss(w_mse=1.0, w_cos=0.5, p3_weight=2.0)
    s_feats = (
        torch.zeros(1, 64, 104, 104),
        torch.zeros(1, 128, 52, 52),
        torch.zeros(1, 256, 26, 26),
    )
    t_feats = (
        torch.ones(1, 64, 104, 104),
        torch.ones(1, 128, 52, 52),
        torch.ones(1, 256, 26, 26),
    )

    loss, details = loss_fn(s_feats, t_feats)
    assert isinstance(loss.item(), float)
    assert loss.item() > 0
    assert "mse_p3" in details
    assert "mse_p4" in details
    assert "mse_p5" in details


def test_feature_reuse_detector_upsampling_mode():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    detector = FeatureReuseDetector(model, target_resolution=832, base_resolution=640)

    # Dummy neck feature maps from 640px pass
    features = {
        15: torch.randn(1, 64, 80, 80),
        18: torch.randn(1, 128, 40, 40),
        21: torch.randn(1, 256, 20, 20),
    }

    # Test bilinear forward pass
    preds, lat = detector.predict_from_neck_features(features, orig_shape=(720, 1280), mode="bilinear")
    assert isinstance(preds, list)
    assert isinstance(lat, float)
    assert lat > 0

    # Test nearest forward pass
    preds_nn, lat_nn = detector.predict_from_neck_features(features, orig_shape=(720, 1280), mode="nearest")
    assert isinstance(preds_nn, list)
    assert isinstance(lat_nn, float)
