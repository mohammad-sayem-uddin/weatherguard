#!/usr/bin/env python3
"""Audio Emergency Vocalization Detection — Reusable Kaggle GPU Runner.

Target Environment: Kaggle Cloud, NVIDIA Tesla T4 GPU (16 GB VRAM).
Classification Task:
  - Class 1: Positive Emergency Distress Cry ('bachao' / বাঁচাও)
  - Class 0: Negative Normal Bengali Conversational Speech & Ambient Noise
Acoustic Features:
  - 16 kHz Mono, 2.0s (32,000 samples)
  - Pure PyTorch STFT + Mel Filterbank -> Log-Mel Spectrogram (64 mel bins x 126 frames, 1 channel)
Model:
  - Deep 2D Convolutional Neural Network with BatchNorm, ReLU, Dropout, and Linear Head.
Evaluation:
  - Checkpoint selection strictly on Validation split (300 samples).
  - Decision threshold tau* calibrated on Validation split.
  - Frozen independent evaluation on Locked Test split (300 samples, 0 root leakage).
  - Granular breakdown across 5 noise conditions and 6 SNR tiers.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import shutil
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        precision_recall_curve,
        auc,
        confusion_matrix,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Audio-KaggleGPU")

SEED = 42


def set_reproducibility(seed: int = SEED) -> None:
    """Sets random seeds across all libraries for deterministic execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Deterministic seed %d initialized.", seed)


def audit_gpu_environment() -> Dict[str, Any]:
    """Inspects and returns the hardware runtime profile."""
    is_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    total_mem_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if is_cuda else 0.0
    )

    profile = {
        "platform": "Kaggle Cloud",
        "cuda_available": is_cuda,
        "device_name": device_name,
        "gpu_memory_gb": round(total_mem_gb, 2),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if is_cuda else None,
        "is_nvidia_t4": "T4" in device_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    logger.info("GPU Audit: %s (%.2f GB VRAM, CUDA %s)",
                device_name, total_mem_gb, profile["cuda_version"])
    return profile


def create_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 64,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    """Constructs triangular Mel filterbank matrix in pure PyTorch."""
    def hz_to_mel(hz: float) -> float:
        return float(2595.0 * np.log10(1.0 + hz / 700.0))

    def mel_to_hz(mel: float) -> float:
        return float(700.0 * (10.0 ** (mel / 2595.0) - 1.0))

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_prev = bin_points[m - 1]
        f_curr = bin_points[m]
        f_next = bin_points[m + 1]
        for k in range(f_prev, f_curr):
            fbank[m - 1, k] = (k - f_prev) / max(1, (f_curr - f_prev))
        for k in range(f_curr, f_next):
            fbank[m - 1, k] = (f_next - k) / max(1, (f_next - f_curr))

    return torch.from_numpy(fbank)


class RealAudioSpectrogramDataset(Dataset):
    """Loads 16 kHz WAV files and computes normalized Log-Mel spectrograms."""

    def __init__(
        self,
        manifest_csv: Path,
        dataset_root: Path,
        split: str = "train",
        n_mels: int = 64,
        target_length: int = 32000,
    ) -> None:
        self.split = split
        self.target_length = target_length
        self.dataset_root = Path(dataset_root)
        self.n_mels = n_mels
        self.mel_fb = create_mel_filterbank(sr=16000, n_fft=1024, n_mels=n_mels)
        self.window = torch.hann_window(1024)

        # Read manifest rows
        self.records: List[Dict[str, Any]] = []
        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") == split:
                    self.records.append(row)

        logger.info("RealAudioDataset initialized for split '%s': %d samples.", split, len(self.records))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.records[idx]
        cls_sub = "positive" if row["class_label"] == "bachao" else "negative"
        wav_path = self.dataset_root / "audio" / cls_sub / row["filename"]

        # Read PCM
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found at {wav_path}")

        with wave.open(str(wav_path), "rb") as w:
            n_frames = w.getnframes()
            frames = w.readframes(n_frames)
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        else:
            audio = audio[:self.target_length]

        x = torch.from_numpy(audio).float()

        # Compute STFT & Mel Spectrogram
        stft = torch.stft(
            x,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=self.window,
            return_complex=True,
        )
        linear_spec = torch.abs(stft)  # (513, 126)
        mel_spec = torch.matmul(self.mel_fb, linear_spec)  # (64, 126)
        log_mel = torch.log(mel_spec + 1e-6)
        norm_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        feature_tensor = norm_mel.unsqueeze(0)  # (1, 64, 126)

        label = 1.0 if row["class_label"] == "bachao" else 0.0
        label_tensor = torch.tensor(label, dtype=torch.float32)

        meta = {
            "filename": row["filename"],
            "class_label": row["class_label"],
            "noise_condition": row.get("noise_condition", "unknown"),
            "snr_db": float(row.get("snr_db", 0.0)),
            "distance_simulation": row.get("distance_simulation", "unknown"),
            "root_source": row.get("root_source", "unknown"),
        }
        return feature_tensor, label_tensor, meta


class AudioBaselinePyTorchCNN(nn.Module):
    """Deep 2D Convolutional Neural Network for Spectrogram Distress Classification."""

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
        # Dimensions: 64//8 = 8, 126//8 = 15 -> 128 * 8 * 15 = 15,360
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


def evaluate_audio_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes full evaluation metrics including noise and SNR breakdowns."""
    model.eval()
    all_targets: List[int] = []
    all_probs: List[float] = []
    all_meta: List[Dict[str, Any]] = []

    with torch.no_grad():
        for features, labels, meta in loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            targets = labels.numpy().astype(int).tolist()

            all_probs.extend(probs)
            all_targets.extend(targets)

            # Unpack metadata batch
            batch_size = len(targets)
            for i in range(batch_size):
                all_meta.append({
                    "noise_condition": meta["noise_condition"][i],
                    "snr_db": float(meta["snr_db"][i]),
                    "distance_simulation": meta["distance_simulation"][i],
                })

    preds = [1 if p >= threshold else 0 for p in all_probs]

    acc = float(accuracy_score(all_targets, preds)) if HAS_SKLEARN else 0.0
    prec = float(precision_score(all_targets, preds, zero_division=0)) if HAS_SKLEARN else 0.0
    rec = float(recall_score(all_targets, preds, zero_division=0)) if HAS_SKLEARN else 0.0
    f1 = float(f1_score(all_targets, preds, zero_division=0)) if HAS_SKLEARN else 0.0
    roc_auc = float(roc_auc_score(all_targets, all_probs)) if HAS_SKLEARN else 0.0

    cm = confusion_matrix(all_targets, preds) if HAS_SKLEARN else np.array([[0, 0], [0, 0]])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
    fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0

    # Noise Breakdown
    noise_stats: Dict[str, Dict[str, Any]] = {}
    for noise in ["clean_speech", "drone_propeller", "flood_water", "heavy_rain", "strong_wind"]:
        indices = [i for i, m in enumerate(all_meta) if m["noise_condition"] == noise]
        if indices:
            sub_t = [all_targets[i] for i in indices]
            sub_p = [preds[i] for i in indices]
            sub_acc = float(accuracy_score(sub_t, sub_p)) if HAS_SKLEARN else 0.0
            sub_f1 = float(f1_score(sub_t, sub_p, zero_division=0)) if HAS_SKLEARN else 0.0
            noise_stats[noise] = {
                "samples": len(indices),
                "accuracy": round(sub_acc, 4),
                "f1": round(sub_f1, 4),
            }

    # SNR Breakdown
    snr_stats: Dict[str, Dict[str, Any]] = {}
    for snr in sorted(list(set(m["snr_db"] for m in all_meta))):
        indices = [i for i, m in enumerate(all_meta) if m["snr_db"] == snr]
        if indices:
            sub_t = [all_targets[i] for i in indices]
            sub_p = [preds[i] for i in indices]
            sub_acc = float(accuracy_score(sub_t, sub_p)) if HAS_SKLEARN else 0.0
            sub_f1 = float(f1_score(sub_t, sub_p, zero_division=0)) if HAS_SKLEARN else 0.0
            snr_stats[str(snr)] = {
                "samples": len(indices),
                "accuracy": round(sub_acc, 4),
                "f1": round(sub_f1, 4),
            }

    return {
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "noise_breakdown": noise_stats,
        "snr_breakdown": snr_stats,
        "all_probs": all_probs,
        "all_targets": all_targets,
    }


def train_real_audio_classifier(
    dataset_dir: str = "/kaggle/input/rescue-drone-audio-1500",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    output_dir: str = "/kaggle/working/artifacts/audio_gpu",
) -> Dict[str, Any]:
    """Orchestrates end-to-end real audio dataset training and evaluation on Kaggle GPU."""
    set_reproducibility(SEED)
    gpu_profile = audit_gpu_environment()

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    data_p = Path(dataset_dir)
    manifest_p = data_p / "manifest.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing AudioBaselinePyTorchCNN on device: %s", device)

    train_ds = RealAudioSpectrogramDataset(manifest_p, data_p, split="train")
    val_ds = RealAudioSpectrogramDataset(manifest_p, data_p, split="val")
    test_ds = RealAudioSpectrogramDataset(manifest_p, data_p, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = AudioBaselinePyTorchCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_weights_path = out_p / "audio_best_weights.pt"

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for features, labels, _ in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            total_train += len(labels)

        scheduler.step()
        train_loss /= total_train
        train_acc = train_correct / total_train

        # Validation evaluation
        val_eval = evaluate_audio_split(model, val_loader, device, threshold=0.5)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": val_eval["accuracy"],
            "val_f1": val_eval["f1"],
            "val_roc_auc": val_eval["roc_auc"],
        }
        history.append(epoch_record)
        logger.info("Epoch %2d/%2d - Loss: %.4f - TrainAcc: %.4f - ValAcc: %.4f - ValF1: %.4f - ValAUC: %.4f",
                    epoch, epochs, train_loss, train_acc, val_eval["accuracy"], val_eval["f1"], val_eval["roc_auc"])

        if val_eval["f1"] > best_val_f1:
            best_val_f1 = val_eval["f1"]
            torch.save(model.state_dict(), best_weights_path)
            logger.info("  --> Saved new best checkpoint with Val F1: %.4f", best_val_f1)

    elapsed_s = time.perf_counter() - start_time
    logger.info("Training complete in %.2f seconds.", elapsed_s)

    # Load best checkpoint
    model.load_state_dict(torch.load(best_weights_path, map_location=device))

    # Calibrate decision threshold on validation set
    val_preds_dict = evaluate_audio_split(model, val_loader, device, threshold=0.5)
    best_tau = 0.5
    best_tau_f1 = -1.0
    for tau in np.arange(0.1, 0.92, 0.02):
        tau = round(float(tau), 2)
        eval_t = evaluate_audio_split(model, val_loader, device, threshold=tau)
        if eval_t["f1"] > best_tau_f1:
            best_tau_f1 = eval_t["f1"]
            best_tau = tau

    logger.info("Calibrated optimal decision threshold on Validation: tau* = %.2f (Val F1 = %.4f)",
                best_tau, best_tau_f1)

    # Frozen Independent Evaluation on Locked Test Split
    test_eval = evaluate_audio_split(model, test_loader, device, threshold=best_tau)

    # Strip large raw arrays from JSON export
    del test_eval["all_probs"]
    del test_eval["all_targets"]

    summary = {
        "experiment_id": "audio_gpu_real_distress_benchmark",
        "dataset_dir": dataset_dir,
        "gpu_profile": gpu_profile,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "training_time_seconds": round(elapsed_s, 2),
            "best_checkpoint": str(best_weights_path),
            "calibrated_threshold": best_tau,
        },
        "training_history": history,
        "locked_test_metrics": test_eval,
    }

    out_json = out_p / "mission20_audio_gpu_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    logger.info("Saved complete audio benchmark results to %s", out_json)

    # Save results CSV
    with open(out_p / "audio_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_accuracy", "val_accuracy", "val_f1", "val_roc_auc"])
        writer.writeheader()
        writer.writerows(history)

    return summary


if __name__ == "__main__":
    profile = audit_gpu_environment()
    print(json.dumps(profile, indent=2))
