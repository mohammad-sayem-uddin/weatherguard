"""
WeatherGuard Feature Distillation & Teacher Guidance Module (Mission 8)
Implements feature-level distillation where the native 832px YOLOv8n acts as the Teacher
and a lightweight refinement module on 640px features acts as the Student.
"""
from __future__ import annotations

import time
import math
from typing import Dict, List, Tuple, Sequence, Optional
import torch
import torch.nn as nn
import torch.optim as optim

class MultiScaleDistillationLoss(nn.Module):
    """
    Computes normalized multi-scale feature alignment loss between
    refined student feature maps and native 832px teacher feature maps.
    Combines Normalized MSE and Cosine Feature Alignment.
    """
    def __init__(self, w_mse: float = 1.0, w_cos: float = 0.5, p3_weight: float = 2.0):
        super().__init__()
        self.w_mse = w_mse
        self.w_cos = w_cos
        self.p3_weight = p3_weight  # higher weight on P3 for small-object sensitivity

    def forward(
        self,
        student_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        teacher_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        s_p3, s_p4, s_p5 = student_feats
        t_p3, t_p4, t_p5 = teacher_feats

        # P3 alignment
        mse_p3 = nn.functional.mse_loss(s_p3, t_p3)
        cos_p3 = 1.0 - nn.functional.cosine_similarity(s_p3, t_p3, dim=1).mean()

        # P4 alignment
        mse_p4 = nn.functional.mse_loss(s_p4, t_p4)
        cos_p4 = 1.0 - nn.functional.cosine_similarity(s_p4, t_p4, dim=1).mean()

        # P5 alignment
        mse_p5 = nn.functional.mse_loss(s_p5, t_p5)
        cos_p5 = 1.0 - nn.functional.cosine_similarity(s_p5, t_p5, dim=1).mean()

        loss_p3 = self.p3_weight * (self.w_mse * mse_p3 + self.w_cos * cos_p3)
        loss_p4 = self.w_mse * mse_p4 + self.w_cos * cos_p4
        loss_p5 = self.w_mse * mse_p5 + self.w_cos * cos_p5

        total_loss = loss_p3 + loss_p4 + loss_p5

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "mse_p3": float(mse_p3.item()),
            "cos_p3": float(cos_p3.item()),
            "mse_p4": float(mse_p4.item()),
            "mse_p5": float(mse_p5.item()),
        }
        return total_loss, loss_dict


def train_distillation_refiner(
    refiner_module: nn.Module,
    training_data: Sequence[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Trains a lightweight refiner module on pre-extracted (student_input, teacher_target) feature pairs.
    training_data: list of ((f15_640_up, f18_640_up, f21_640_up), (t15_832, t18_832, t21_832))
    """
    refiner_module.to(device).train()
    optimizer = optim.AdamW(refiner_module.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = MultiScaleDistillationLoss()

    history = []
    t0 = time.perf_counter()

    for epoch in range(num_epochs):
        epoch_losses = []
        for s_feats, t_feats in training_data:
            s_p3, s_p4, s_p5 = [x.to(device) for x in s_feats]
            t_p3, t_p4, t_p5 = [x.to(device) for x in t_feats]

            optimizer.zero_grad()
            refined_feats = refiner_module(s_p3, s_p4, s_p5)
            loss, loss_dict = criterion(refined_feats, (t_p3, t_p4, t_p5))
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss_dict["total_loss"])

        mean_ep_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        history.append({"epoch": epoch + 1, "mean_loss": mean_ep_loss})

    elapsed = time.perf_counter() - t0
    refiner_module.eval()

    return {
        "epochs": num_epochs,
        "elapsed_seconds": elapsed,
        "history": history,
        "final_loss": history[-1]["mean_loss"] if history else 0.0,
    }
