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


class SyntheticSpectrogramDataset(Dataset):
    """Generates synthetic Log-Mel spectrogram tensors for testing GPU training."""

    def __init__(self, num_samples: int = 200, n_mels: int = 64, n_frames: int = 126, seed: int = 42) -> None:
        np.random.seed(seed)
        self.features = np.random.randn(num_samples, 1, n_mels, n_frames).astype(np.float32)
        # Class 1 has slightly elevated energy in low mels (simulating distress cry formants)
        self.labels = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32)
        for i in range(num_samples):
            if self.labels[i] == 1.0:
                self.features[i, 0, :20, :] += 1.5

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


def train_audio_classifier(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
    output_dir: str = "/kaggle/working/artifacts/audio_gpu",
) -> Dict[str, Any]:
    """Trains AudioBaselinePyTorchCNN with GPU acceleration and returns evaluation metrics."""
    set_reproducibility(SEED)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training AudioBaselinePyTorchCNN on device: %s", device)

    train_ds = SyntheticSpectrogramDataset(num_samples=300, seed=SEED)
    val_ds = SyntheticSpectrogramDataset(num_samples=100, seed=SEED + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AudioBaselinePyTorchCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        train_loss /= len(train_ds)
        val_metrics = evaluate_audio_model(model, val_loader, device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = round(train_loss, 4)
        history.append(val_metrics)
        logger.info("Epoch %d/%d — Train Loss: %.4f | Val F1: %.4f | Val AUC: %.4f",
                    epoch, epochs, train_loss, val_metrics["f1"], val_metrics["roc_auc"])

    elapsed = time.perf_counter() - start_time
    best_weights_path = out_p / "audio_best_weights.pt"
    torch.save(model.state_dict(), best_weights_path)

    final_metrics = evaluate_audio_model(model, val_loader, device)
    summary = {
        "status": "COMPLETED",
        "device": str(device),
        "training_time_seconds": round(elapsed, 2),
        "epochs": epochs,
        "history": history,
        "final_metrics": final_metrics,
        "best_weights": str(best_weights_path),
    }

    metrics_json_path = out_p / "audio_gpu_training_metrics.json"
    metrics_json_path.write_text(json.dumps(summary, indent=2))
    logger.info("Audio training summary saved to %s", metrics_json_path)
    return summary


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
