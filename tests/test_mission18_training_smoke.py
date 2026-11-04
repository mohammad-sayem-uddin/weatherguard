"""Mission 18 — M1 Training Safety, Memory Measurement, and Pipeline Smoke Tests.

Validates training configuration integrity across all 12 Mission 16 configs,
leakage-safe split usage in dataset YAML generation, small-object oversampling reality,
and empirical RAM / peak RSS consumption during CPU training on Apple Silicon M1.

Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""

import json
import resource
import shutil
import tempfile
import time
from pathlib import Path
from PIL import Image
import pytest
import yaml
from ultralytics import YOLO

from scripts.train_weatherguard_d1 import (
    BASE_WEIGHTS_PATH,
    ROOT,
    TrainingExperimentConfig,
    generate_yolo_dataset_yaml,
    load_training_config_from_yaml,
)

CONFIGS_M16_DIR = ROOT / "configs" / "mission16"


# ==============================================================================
# 1. Configuration Mapping & Hyperparameter Audit
# ==============================================================================

def test_all_12_configs_hyperparameter_mapping():
    """Verifies that all 12 Mission 16 experiment YAML files map to valid training configs."""
    yaml_files = sorted(list(CONFIGS_M16_DIR.glob("*.yaml")))
    assert len(yaml_files) == 12, f"Expected 12 configs, found {len(yaml_files)}"

    for yf in yaml_files:
        cfg = load_training_config_from_yaml(yf)
        assert cfg.experiment_id.lower() == yf.stem.lower()
        assert cfg.imgsz in (640, 832)
        assert 1 <= cfg.batch_size <= 16
        assert cfg.freeze_layers >= 0
        assert cfg.random_seed == 42


def test_baseline_configs_have_training_disabled():
    """Baselines E1, E3, and E5 are frozen evaluation baselines and must have training disabled."""
    for exp_id in ("e1_baseline_yolov8n", "e3_baseline_policy_j", "e5_baseline_policy_k"):
        cfg_file = CONFIGS_M16_DIR / f"{exp_id}.yaml"
        assert cfg_file.exists()
        cfg = load_training_config_from_yaml(cfg_file)
        assert cfg.enabled is False, f"Baseline {exp_id} must have training.enabled=False"


# ==============================================================================
# 2. Split Usage & Test Image Leakage Protection
# ==============================================================================

def test_generate_yolo_dataset_yaml_strict_test_isolation(tmp_path: Path):
    """Verifies that generate_yolo_dataset_yaml writes ONLY train and val images,
    completely excluding test split images.
    """
    split_manifest = tmp_path / "d1_split_manifest.json"
    dataset_root = tmp_path / "WeatherGuard-D1"
    output_yaml = dataset_root / "dataset.yaml"

    mock_splits = {
        "manifest_hash": "test_hash_123",
        "train": {"image_ids": ["img_tr_01", "img_tr_02"]},
        "validation": {"image_ids": ["img_va_01"]},
        "test": {"image_ids": ["img_te_01", "img_te_02"]},
    }
    with open(split_manifest, "w") as f:
        json.dump(mock_splits, f)

    res_yaml = generate_yolo_dataset_yaml(
        split_manifest_path=split_manifest,
        dataset_root_dir=dataset_root,
        output_yaml_path=output_yaml,
        oversample_small=False,
    )
    assert res_yaml.exists()

    train_txt = dataset_root / "split_file_lists" / "train.txt"
    val_txt = dataset_root / "split_file_lists" / "val.txt"

    train_content = train_txt.read_text()
    val_content = val_txt.read_text()

    # Train assertions
    assert "img_tr_01" in train_content
    assert "img_tr_02" in train_content
    assert "img_va_01" not in train_content
    assert "img_te_01" not in train_content
    assert "img_te_02" not in train_content

    # Val assertions
    assert "img_va_01" in val_content
    assert "img_tr_01" not in val_content
    assert "img_te_01" not in val_content


# ==============================================================================
# 3. Small-Object Oversampling Reality Check
# ==============================================================================

def test_small_object_oversampling_genuinely_changes_sampling(tmp_path: Path):
    """Verifies that small_object_oversampling does not merely parse a key, but
    actively duplicates instances containing small targets (< 32px) in train.txt.
    """
    dataset_root = tmp_path / "WeatherGuard-D1"
    img_dir = dataset_root / "images"
    lbl_dir = dataset_root / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Image 1: Large object (w=200, h=200) -> NOT oversampled
    # Image 2: Small object (w=20, h=20 -> max_dim=20 < 32px) -> OVERSAMPLED
    for img_id in ("img_large", "img_small"):
        (img_dir / f"{img_id}.jpg").write_bytes(b"\x00" * 10)

    # YOLO labels: class x_c y_c w h (normalized to 1280x720)
    # img_large: w = 200/1280 = 0.15625, h = 200/720 = 0.27778
    (lbl_dir / "img_large.txt").write_text("3 0.5 0.5 0.15625 0.27778\n")
    # img_small: w = 20/1280 = 0.015625, h = 20/720 = 0.02778
    (lbl_dir / "img_small.txt").write_text("1 0.5 0.5 0.015625 0.02778\n")

    split_manifest = tmp_path / "manifest.json"
    with open(split_manifest, "w") as f:
        json.dump({
            "train": {"image_ids": ["img_large", "img_small"]},
            "validation": {"image_ids": []},
            "test": {"image_ids": []},
        }, f)

    # 1. Without oversampling
    generate_yolo_dataset_yaml(
        split_manifest_path=split_manifest,
        dataset_root_dir=dataset_root,
        output_yaml_path=dataset_root / "ds_normal.yaml",
        oversample_small=False,
    )
    lines_normal = (dataset_root / "split_file_lists" / "train.txt").read_text().strip().split("\n")
    assert len(lines_normal) == 2

    # 2. With 2x oversampling (oversample_factor = 2.0)
    generate_yolo_dataset_yaml(
        split_manifest_path=split_manifest,
        dataset_root_dir=dataset_root,
        output_yaml_path=dataset_root / "ds_oversample.yaml",
        oversample_small=True,
        oversample_factor=2.0,
    )
    lines_oversampled = (dataset_root / "split_file_lists" / "train.txt").read_text().strip().split("\n")
    # img_large appears once, img_small appears twice -> total 3 entries!
    assert len(lines_oversampled) == 3
    assert lines_oversampled.count(str((img_dir / "img_large.jpg").resolve())) == 1
    assert lines_oversampled.count(str((img_dir / "img_small.jpg").resolve())) == 2


# ==============================================================================
# 4. Empirical M1 CPU Memory & Execution Smoke Test
# ==============================================================================

def test_m1_cpu_training_memory_and_execution_smoke(tmp_path: Path):
    """Executes a real 1-epoch training pass on Apple Silicon M1 CPU and verifies
    that peak RSS memory remains strictly under practical limits (< 2.5 GB).
    """
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Create 2 synthetic RGB images
    for name in ("smoke_img_01", "smoke_img_02"):
        im = Image.new("RGB", (640, 640), color=(120, 140, 160))
        im.save(img_dir / f"{name}.jpg")
        with open(lbl_dir / f"{name}.txt", "w") as f:
            f.write("0 0.5 0.5 0.25 0.25\n")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(f"path: {tmp_path}\ntrain: images\nval: images\nnames:\n  0: truck\n")

    base_mtime_before = Path("yolov8n.pt").stat().st_mtime
    start_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    t0 = time.perf_counter()

    # Load YOLOv8n and run 1 epoch on CPU with batch_size=2, workers=0
    model = YOLO("yolov8n.pt")
    runs_dir = tmp_path / "runs"
    results = model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        project=str(runs_dir),
        name="m1_smoke_run",
        verbose=False,
    )
    duration_s = time.perf_counter() - t0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)

    # Memory assertions
    # Peak RSS must be well below our 5.5 GB headroom limit on an 8 GB M1 system
    assert peak_rss_mb < 2500.0, f"Peak RSS {peak_rss_mb:.1f} MB exceeded 2.5 GB safety threshold!"
    assert duration_s < 30.0, f"Smoke test took {duration_s:.1f}s, exceeding 30s threshold"

    # Weight checkpoint integrity: best.pt and last.pt must exist in isolated output dir
    best_weights = runs_dir / "m1_smoke_run" / "weights" / "best.pt"
    assert best_weights.exists(), "best.pt checkpoint was not created"
    assert best_weights.stat().st_size > 1_000_000, "Checkpoint file is suspiciously small"

    # Verify original base weights file was untouched
    base_mtime_after = Path("yolov8n.pt").stat().st_mtime
    assert base_mtime_after == base_mtime_before, "FATAL: Base weights yolov8n.pt was modified during training!"
