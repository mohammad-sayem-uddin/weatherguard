#!/usr/bin/env python3
"""Audio Emergency Vocalization Detection — Reusable Kaggle GPU Runner.

Target Environment: Kaggle Cloud, NVIDIA Tesla T4 GPU (16 GB VRAM).
Classification Task:
  - Class 1: Positive Emergency Cry ('bachao' / বাঁচাও)
  - Class 0: Negative Normal Speech & Environmental Background Acoustics
Feature Representation:
  - 16 kHz Mono, 2.0s (32,000 samples)
  - Log-Mel Spectrogram (64 mel bins x 126 frames, 1 channel)
Model:
  - Deep 2D Convolutional Neural Network with BatchNorm, ReLU, Dropout, and Linear Head.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# Optional evaluation metrics
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Audio-KaggleGPU")

SEED = 42


def set_reproducibility(seed: int = SEED) -> None:
    """Sets deterministic seeds across Python, NumPy, PyTorch, and CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Deterministic seed %d initialized for Audio GPU Pipeline.", seed)


class AudioBaselinePyTorchCNN(nn.Module):
    """PyTorch 2D CNN matching the AudioBaselineCNN architecture."""

    def __init__(self, n_mels: int = 64, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(dropout_rate)
        # Compute flattened feature dimension after 3 poolings on (64, 126):
        # 64 // 8 = 8, 126 // 8 = 15 -> 128 * 8 * 15 = 15360
        self.fc1 = nn.Linear(128 * 8 * 15, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 64, 126)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.flatten(1)
        x = self.dropout(F.relu(self.bn_fc(self.fc1(x))))
        logits = self.fc_out(x)
        return logits.squeeze(1)


def evaluate_audio_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics."""
    model.eval()
    all_preds: List[int] = []
    all_probs: List[float] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    if HAS_SKLEARN:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
    else:
        # Pure numpy implementation fallback
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        cm = [[tn, fp], [fn, tp]]
        total = max(len(y_true), 1)
        acc = float((tp + tn) / total)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        auc = 0.5

    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "sample_count": len(y_true),
    }
