"""Unit tests for WeatherGuard-D1 training infrastructure and leakage safeguards.

Module: tests/test_d1_training_infra.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""
import json
import shutil
import tempfile
from pathlib import Path
import pytest
import yaml

from scripts.train_weatherguard_d1 import (
    BASE_WEIGHTS_PATH,
    TrainingExperimentConfig,
    WeatherGuardTrainer,
    generate_yolo_dataset_yaml,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "weatherguard_d1.yaml"


@pytest.fixture
def temp_workspace():
    tmp_dir = Path(tempfile.mkdtemp(prefix="test_d1_train_"))
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_training_config_dataclass():
    cfg = TrainingExperimentConfig(
        experiment_id="E2_test_config",
        base_model_weights="yolov8n.pt",
        imgsz=832,
        batch_size=8,
        epochs=20,
    )
    assert cfg.experiment_id == "E2_test_config"
    assert cfg.imgsz == 832
    assert cfg.batch_size == 8
    assert cfg.workers == 4
    assert cfg.mosaic == 0.5


def test_generate_yolo_dataset_yaml_strict_exclusion(temp_workspace):
    split_manifest_path = temp_workspace / "d1_split_manifest.json"
    dataset_root = temp_workspace / "WeatherGuard-D1"
    output_yaml = dataset_root / "dataset.yaml"

    mock_splits = {
        "manifest_hash": "abc123hash",
        "train": {"image_ids": ["train_001", "train_002"]},
        "validation": {"image_ids": ["val_001"]},
        "test": {"image_ids": ["test_001", "test_002"]},
    }
    with open(split_manifest_path, "w") as f:
        json.dump(mock_splits, f)

    yaml_path = generate_yolo_dataset_yaml(
        split_manifest_path=split_manifest_path,
        dataset_root_dir=dataset_root,
        output_yaml_path=output_yaml,
        d1_config_path=CONFIG_PATH,
    )

    assert yaml_path.exists()
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Verify metadata
    assert data["metadata"]["total_train_images"] == 2
    assert data["metadata"]["total_val_images"] == 1
    assert data["metadata"]["test_images_excluded"] == 2
    assert data["names"][0] == "truck"
    assert data["names"][3] == "car"

    # Verify file lists
    train_txt = Path(data["train"])
    val_txt = Path(data["val"])
    assert train_txt.exists()
    assert val_txt.exists()

    train_lines = train_txt.read_text()
    assert "train_001" in train_lines
    assert "test_001" not in train_lines
    assert "test_002" not in train_lines


def test_trainer_preflight_dry_run(temp_workspace):
    split_manifest_path = temp_workspace / "d1_split_manifest.json"
    dataset_root = temp_workspace / "WeatherGuard-D1"
    output_yaml = dataset_root / "dataset.yaml"

    mock_splits = {
        "manifest_hash": "hash789",
        "train": {"image_ids": ["img_tr1"]},
        "validation": {"image_ids": ["img_va1"]},
        "test": {"image_ids": ["img_te1"]},
    }
    with open(split_manifest_path, "w") as f:
        json.dump(mock_splits, f)

    generate_yolo_dataset_yaml(
        split_manifest_path=split_manifest_path,
        dataset_root_dir=dataset_root,
        output_yaml_path=output_yaml,
        d1_config_path=CONFIG_PATH,
    )

    cfg = TrainingExperimentConfig(
        experiment_id="E2_preflight_unit_test",
        base_model_weights=str(BASE_WEIGHTS_PATH),
    )
    trainer = WeatherGuardTrainer(cfg)
    run_dir = temp_workspace / "runs"
    res = trainer.run_training(dataset_yaml_path=output_yaml, output_dir=run_dir, dry_run=True)

    assert res["status"] == "PREFLIGHT_VERIFIED"
    assert "preflight_audit.json" in [f.name for f in (run_dir / "E2_preflight_unit_test").glob("*.json")]
    assert res["hardware"]["torch_device"] == "cpu"
