"""WeatherGuard Mission 6 — Representation Drift & Activation Analysis.

Provides mathematical and diagnostic tools to measure:
  1. Layer-wise parameter drift across model checkpoints (L2 norm, relative change, unchanged %).
  2. Intermediate feature representation alignment (Linear CKA, Cosine Similarity).
  3. Domain separability diagnostics across DAWN and BDD100K feature spaces.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO


def compute_layerwise_parameter_drift(
    model_pre_path: str | Path,
    model_ft_path: str | Path,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Computes exact parameter-by-parameter drift between two YOLO checkpoints."""
    try:
        pre_ckpt = torch.load(model_pre_path, map_location="cpu", weights_only=False)
        ft_ckpt = torch.load(model_ft_path, map_location="cpu", weights_only=False)
    except Exception:
        yolo_pre = YOLO(str(model_pre_path))
        yolo_ft = YOLO(str(model_ft_path))
        pre_ckpt = {"model": yolo_pre.model}
        ft_ckpt = {"model": yolo_ft.model}

    # Extract state_dict
    sd_pre = pre_ckpt["model"].state_dict() if "model" in pre_ckpt else pre_ckpt
    sd_ft = ft_ckpt["model"].state_dict() if "model" in ft_ckpt else ft_ckpt

    layers_drift = {}
    total_params = 0
    total_unchanged = 0
    total_l2_diff_sq = 0.0
    total_l2_pre_sq = 0.0

    block_stats = {
        "backbone": {"params": 0, "unchanged": 0, "diff_sq": 0.0, "pre_sq": 0.0},
        "neck": {"params": 0, "unchanged": 0, "diff_sq": 0.0, "pre_sq": 0.0},
        "head": {"params": 0, "unchanged": 0, "diff_sq": 0.0, "pre_sq": 0.0},
        "other": {"params": 0, "unchanged": 0, "diff_sq": 0.0, "pre_sq": 0.0},
    }

    common_keys = sorted(set(sd_pre.keys()) & set(sd_ft.keys()))

    for key in common_keys:
        t_pre = sd_pre[key].float()
        t_ft = sd_ft[key].float()

        if t_pre.shape != t_ft.shape:
            continue

        n_p = t_pre.numel()
        diff = t_ft - t_pre
        l2_diff = float(torch.norm(diff).item())
        l2_pre = float(torch.norm(t_pre).item())
        l2_ft = float(torch.norm(t_ft).item())
        rel_diff = float(l2_diff / (l2_pre + eps))

        # Check exact equality / threshold closeness
        is_close = torch.isclose(t_pre, t_ft, atol=1e-7, rtol=1e-5)
        n_unchanged = int(is_close.sum().item())
        pct_unchanged = (n_unchanged / n_p * 100.0) if n_p > 0 else 100.0

        # Classify block
        # YOLOv8 naming: model.0..model.9 = backbone, model.10..model.21 = neck, model.22 = head
        block = "other"
        if key.startswith("model."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_idx = int(parts[1])
                if layer_idx <= 9:
                    block = "backbone"
                elif layer_idx <= 21:
                    block = "neck"
                elif layer_idx == 22:
                    block = "head"

        layers_drift[key] = {
            "param_count": n_p,
            "block": block,
            "l2_diff": l2_diff,
            "l2_pre": l2_pre,
            "l2_ft": l2_ft,
            "rel_diff": rel_diff,
            "unchanged_count": n_unchanged,
            "unchanged_pct": pct_unchanged,
        }

        total_params += n_p
        total_unchanged += n_unchanged
        total_l2_diff_sq += l2_diff ** 2
        total_l2_pre_sq += l2_pre ** 2

        block_stats[block]["params"] += n_p
        block_stats[block]["unchanged"] += n_unchanged
        block_stats[block]["diff_sq"] += l2_diff ** 2
        block_stats[block]["pre_sq"] += l2_pre ** 2

    # Overall summary
    overall_l2_diff = float(np.sqrt(total_l2_diff_sq))
    overall_l2_pre = float(np.sqrt(total_l2_pre_sq))
    overall_rel = float(overall_l2_diff / (overall_l2_pre + eps))

    summary_by_block = {}
    for blk, b_data in block_stats.items():
        if b_data["params"] > 0:
            b_l2_diff = float(np.sqrt(b_data["diff_sq"]))
            b_l2_pre = float(np.sqrt(b_data["pre_sq"]))
            summary_by_block[blk] = {
                "param_count": b_data["params"],
                "l2_diff": b_l2_diff,
                "l2_pre": b_l2_pre,
                "rel_diff": float(b_l2_diff / (b_l2_pre + eps)),
                "unchanged_pct": float(b_data["unchanged"] / b_data["params"] * 100.0),
            }

    return {
        "total_params": total_params,
        "overall_l2_diff": overall_l2_diff,
        "overall_rel_diff": overall_rel,
        "overall_unchanged_pct": float(total_unchanged / total_params * 100.0) if total_params > 0 else 100.0,
        "by_block": summary_by_block,
        "layers": layers_drift,
    }


def linear_cka(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Calculates Linear Centered Kernel Alignment (CKA) between feature representations.
    
    Args:
        x: Array of shape (N, D1), N samples, D1 feature dimensions.
        y: Array of shape (N, D2), N samples, D2 feature dimensions.
    Returns:
        Scalar CKA similarity score in [0.0, 1.0].
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample count mismatch: {x.shape[0]} vs {y.shape[0]}")

    n = x.shape[0]
    if n < 2:
        return 1.0

    # Center representations along sample dimension
    x_c = x - np.mean(x, axis=0, keepdims=True)
    y_c = y - np.mean(y, axis=0, keepdims=True)

    # Fast Linear CKA via Frobenius norm of cross-covariance matrix
    # HSIC(K, L) = ||Y_c^T X_c||_F^2 / (n-1)^2
    cross_cov = np.dot(x_c.T, y_c)
    hsic_xy = np.sum(cross_cov ** 2)

    cov_xx = np.dot(x_c.T, x_c)
    hsic_xx = np.sum(cov_xx ** 2)

    cov_yy = np.dot(y_c.T, y_c)
    hsic_yy = np.sum(cov_yy ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy) + eps
    cka_score = float(hsic_xy / denom)
    return float(np.clip(cka_score, 0.0, 1.0))


def mean_cosine_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> Optional[float]:
    """Computes mean cosine similarity between corresponding sample vectors.
    
    Returns None if channel dimensions D1 and D2 differ (e.g. COCO 80 classes vs DAWN 6 classes).
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample count mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[1] != y.shape[1]:
        return None

    norm_x = np.linalg.norm(x, axis=1, keepdims=True) + eps
    norm_y = np.linalg.norm(y, axis=1, keepdims=True) + eps
    x_norm = x / norm_x
    y_norm = y / norm_y
    sims = np.sum(x_norm * y_norm, axis=1)
    return float(np.mean(sims))


class YOLOFeatureExtractor:
    """Extracts intermediate layer activations from a YOLOv8 PyTorch model via forward hooks."""

    def __init__(self, model_path: str | Path, target_layers: List[int] = [9, 15, 21, 22]):
        self.model_path = str(model_path)
        self.target_layers = target_layers
        self.yolo = YOLO(self.model_path)
        self.nn_model = self.yolo.model
        self.nn_model.eval()
        self.features: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for l_idx in self.target_layers:
            if hasattr(self.nn_model, "model") and l_idx < len(self.nn_model.model):
                layer = self.nn_model.model[l_idx]
                hook = layer.register_forward_hook(self._make_hook(l_idx))
                self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # If output is tuple/list (e.g. Detect head), take first tensor
            out_tensor = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(out_tensor, torch.Tensor):
                # Global Average Pool: (B, C, H, W) -> (B, C)
                if out_tensor.ndim == 4:
                    pooled = torch.mean(out_tensor, dim=(2, 3)).detach().cpu().numpy()
                elif out_tensor.ndim == 3:
                    pooled = torch.mean(out_tensor, dim=2).detach().cpu().numpy()
                else:
                    pooled = out_tensor.detach().cpu().numpy()
                self.features[layer_idx].append(pooled)
        return hook_fn

    def extract_from_batch(self, imgs_np: np.ndarray) -> Dict[int, np.ndarray]:
        """Runs forward pass on preprocessed images (B, 3, H, W) in [0, 1] and returns pooled features."""
        self.features.clear()
        with torch.no_grad():
            t_input = torch.from_numpy(imgs_np).float()
            _ = self.nn_model(t_input)

        out = {}
        for l_idx in self.target_layers:
            if self.features[l_idx]:
                out[l_idx] = np.concatenate(self.features[l_idx], axis=0)
        return out

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def evaluate_domain_separability(
    dawn_features: np.ndarray,
    bdd_features: np.ndarray,
    n_splits: int = 5,
    epochs: int = 50,
    lr: float = 0.05,
) -> Dict[str, float]:
    """Evaluates whether intermediate representations linearly separate DAWN from BDD100K.
    
    Uses stratified K-fold cross-validation with PyTorch Logistic Regression.
    """
    n_dawn = len(dawn_features)
    n_bdd = len(bdd_features)
    x_all = np.vstack([dawn_features, bdd_features])
    y_all = np.array([0.0] * n_dawn + [1.0] * n_bdd, dtype=np.float32)

    # Deterministic stratified split
    rng = np.random.default_rng(42)
    dawn_perm = rng.permutation(n_dawn)
    bdd_perm = rng.permutation(n_bdd) + n_dawn

    dawn_folds = np.array_split(dawn_perm, n_splits)
    bdd_folds = np.array_split(bdd_perm, n_splits)

    accs = []
    aucs = []

    for k in range(n_splits):
        val_idx = np.concatenate([dawn_folds[k], bdd_folds[k]])
        train_mask = np.ones(len(x_all), dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.where(train_mask)[0]

        # Standardize train
        x_tr = x_all[train_idx]
        mean = np.mean(x_tr, axis=0, keepdims=True)
        std = np.std(x_tr, axis=0, keepdims=True) + 1e-7
        x_tr_norm = (x_tr - mean) / std
        x_val_norm = (x_all[val_idx] - mean) / std

        # PyTorch logistic regression
        d_in = x_all.shape[1]
        model = nn.Linear(d_in, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        t_x_tr = torch.from_numpy(x_tr_norm).float()
        t_y_tr = torch.from_numpy(y_all[train_idx]).unsqueeze(1)
        t_x_val = torch.from_numpy(x_val_norm).float()
        t_y_val = y_all[val_idx]

        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(t_x_tr)
            loss = criterion(logits, t_y_tr)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_logits = model(t_x_val).squeeze(1).numpy()
            probs = 1.0 / (1.0 + np.exp(-val_logits))
            preds = (probs >= 0.5).astype(np.float32)

        acc = float(np.mean(preds == t_y_val))
        accs.append(acc)

        # Simplified AUC computation (Mann-Whitney U statistic)
        pos_scores = probs[t_y_val == 1.0]
        neg_scores = probs[t_y_val == 0.0]
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            u_stat = sum(float(np.sum(p > neg_scores) + 0.5 * np.sum(p == neg_scores)) for p in pos_scores)
            auc = u_stat / (len(pos_scores) * len(neg_scores))
        else:
            auc = 0.5
        aucs.append(float(auc))

    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_roc_auc": float(np.mean(aucs)),
        "std_roc_auc": float(np.std(aucs)),
    }
