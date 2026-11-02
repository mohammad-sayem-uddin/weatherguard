"""WeatherGuard Mission 6 — Regularized Fine-Tuning & Representation Preservation (L2-SP).

Implements:
  1. L2-SP (L2-Starting Point) regularization penalty:
     Penalizes deviation from pretrained reference weights:
       L_reg = (alpha / 2) * sum_l ||w_l - w_l^0||_2^2
  2. Weight anchoring optimizer wrapper or gradient adjustment.
  3. Conservative Head-Only parameter freezing (freeze=22).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


def compute_l2sp_loss(
    model: nn.Module,
    pretrained_weights: Dict[str, torch.Tensor],
    alpha: float = 0.01,
    beta: float = 0.001,
) -> torch.Tensor:
    """Computes the explicit L2-SP regularization loss.
    
    Args:
        model: Active PyTorch model.
        pretrained_weights: Dictionary of target parameter tensors from pretrained checkpoint.
        alpha: Weight penalty for pretrained parameters: (alpha / 2) * ||w - w0||^2.
        beta: Standard L2 weight decay penalty for newly initialized / unanchored weights: (beta / 2) * ||w||^2.
    """
    l2sp_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name in pretrained_weights:
            ref_param = pretrained_weights[name].to(param.device)
            if param.shape == ref_param.shape:
                diff = param - ref_param
                l2sp_loss = l2sp_loss + 0.5 * alpha * torch.sum(diff ** 2)
            else:
                # Shape mismatch (e.g. head adapted from 80 classes to 6 classes)
                l2sp_loss = l2sp_loss + 0.5 * beta * torch.sum(param ** 2)
        else:
            l2sp_loss = l2sp_loss + 0.5 * beta * torch.sum(param ** 2)

    return l2sp_loss


def extract_pretrained_weight_dict(model_path: str) -> Dict[str, torch.Tensor]:
    """Loads state_dict and returns cloned detached parameter tensors on CPU."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"].state_dict() if "model" in ckpt else ckpt
    return {k: v.clone().detach().cpu() for k, v in state_dict.items()}


def apply_l2sp_gradients(
    model: nn.Module,
    pretrained_weights: Dict[str, torch.Tensor],
    alpha: float = 0.01,
    beta: float = 0.001,
) -> None:
    """Directly adds L2-SP gradient penalty: grad += alpha * (w - w0) to each trainable parameter."""
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue

        if name in pretrained_weights:
            ref_param = pretrained_weights[name].to(param.device)
            if param.shape == ref_param.shape:
                param.grad.add_(alpha * (param.data - ref_param))
            else:
                param.grad.add_(beta * param.data)
        else:
            param.grad.add_(beta * param.data)
