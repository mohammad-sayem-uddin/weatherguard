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
    load_training_config_from_yaml,
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


def test_load_training_config_from_yaml_experiments():
    e1_path = ROOT / "configs" / "mission16" / "e1_baseline_yolov8n.yaml"
    e2_path = ROOT / "configs" / "mission16" / "e2_d1_finetune_yolov8n.yaml"
    e7_path = ROOT / "configs" / "mission16" / "e7_small_object_sampling.yaml"

    cfg1 = load_training_config_from_yaml(e1_path)
    assert cfg1.experiment_id == "E1_baseline_yolov8n"
    assert cfg1.enabled is False

    cfg2 = load_training_config_from_yaml(e2_path)
    assert cfg2.experiment_id == "E2_d1_finetune_yolov8n"
    assert cfg2.enabled is True
    assert cfg2.epochs == 30
    assert cfg2.batch_size == 8
    assert cfg2.optimizer == "AdamW"

    cfg7 = load_training_config_from_yaml(e7_path)
    assert cfg7.experiment_id == "E7_small_object_sampling"
    assert cfg7.small_object_oversampling is True
    assert cfg7.oversample_factor == 2.0


def test_training_disabled_execution(temp_workspace):
    cfg = TrainingExperimentConfig(
        experiment_id="E1_test_disabled",
        base_model_weights="yolov8n.pt",
        enabled=False,
    )
    trainer = WeatherGuardTrainer(cfg)
    run_dir = temp_workspace / "runs"
    res = trainer.run_training(dataset_yaml_path=Path("dummy.yaml"), output_dir=run_dir)
    assert res["status"] == "TRAINING_DISABLED"


def test_generate_yolo_dataset_yaml_with_oversampling(temp_workspace):
    split_manifest_path = temp_workspace / "d1_split_manifest.json"
    dataset_root = temp_workspace / "WeatherGuard-D1"
    labels_dir = dataset_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    output_yaml = dataset_root / "dataset.yaml"

    mock_splits = {
        "manifest_hash": "hash_sample",
        "train": {"image_ids": ["img_small", "img_large"]},
        "validation": {"image_ids": ["img_val"]},
        "test": {"image_ids": ["img_test"]},
    }
    with open(split_manifest_path, "w") as f:
        json.dump(mock_splits, f)

    # Write small object label (<32px) for img_small: 0.02 * 1280 = 25.6px
    with open(labels_dir / "img_small.txt", "w") as f:
        f.write("3 0.5 0.5 0.02 0.02\n")

    # Write large object label (>96px) for img_large: 0.20 * 1280 = 256px
    with open(labels_dir / "img_large.txt", "w") as f:
        f.write("3 0.5 0.5 0.20 0.20\n")

    generate_yolo_dataset_yaml(
        split_manifest_path=split_manifest_path,
        dataset_root_dir=dataset_root,
        output_yaml_path=output_yaml,
        d1_config_path=CONFIG_PATH,
        oversample_small=True,
        oversample_factor=2.0,
    )

    train_txt = dataset_root / "split_file_lists" / "train.txt"
    lines = train_txt.read_text().strip().split("\n")
    # img_small should be duplicated (appears twice) while img_large appears once
    small_occurrences = sum(1 for l in lines if "img_small" in l)
    large_occurrences = sum(1 for l in lines if "img_large" in l)
    assert small_occurrences == 2
    assert large_occurrences == 1
