"""Unit tests for src/regularized_finetuning.py."""

import pytest
import torch
import torch.nn as nn

from src.regularized_finetuning import (
    compute_l2sp_loss,
    apply_l2sp_gradients,
)


def test_l2sp_identical_weights():
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )
    pre_weights = {k: v.clone().detach() for k, v in model.named_parameters()}
    loss = compute_l2sp_loss(model, pre_weights, alpha=0.01)
    assert abs(loss.item()) < 1e-6


def test_l2sp_perturbed_weights():
    model = nn.Sequential(nn.Linear(4, 2, bias=False))
    pre_weights = {"0.weight": torch.zeros(2, 4)}

    # Set model weights to 1.0
    with torch.no_grad():
        model[0].weight.fill_(1.0)

    # 8 elements with value 1.0. Loss = 0.5 * alpha * sum((1 - 0)^2) = 0.5 * 0.1 * 8 = 0.4
    loss = compute_l2sp_loss(model, pre_weights, alpha=0.1)
    assert abs(loss.item() - 0.4) < 1e-5


def test_l2sp_frozen_parameters_ignored():
    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.Linear(4, 2),
    )
    # Freeze layer 0
    for p in model[0].parameters():
        p.requires_grad = False

    pre_weights = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2.0)

    # Only layer 1 parameters should contribute to loss
    l1_weight_elems = model[1].weight.numel()  # 4 * 2 = 8
    l1_bias_elems = model[1].bias.numel()      # 2
    expected = 0.5 * 0.1 * (8 * 4.0 + 2 * 4.0)  # 0.05 * 40 = 2.0
    loss = compute_l2sp_loss(model, pre_weights, alpha=0.1)
    assert abs(loss.item() - expected) < 1e-4


def test_apply_l2sp_gradients():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    pre_weights = {"weight": torch.ones(2, 2)}

    # Synthesize initial grad
    model.weight.grad = torch.zeros(2, 2)
    # apply penalty: grad += alpha * (w - w0) = 0.05 * (2 - 1) = 0.05
    apply_l2sp_gradients(model, pre_weights, alpha=0.05)
    assert torch.allclose(model.weight.grad, torch.full((2, 2), 0.05))
