"""Unit tests for Mission 16 experiment configuration and standardized registry.

Module: tests/test_mission16_experiment_prep.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""
import json
import shutil
import tempfile
from pathlib import Path
import pytest
import yaml

from scripts.manage_experiment_registry import (
    REGISTRY_CSV_PATH,
    SCHEMA_FIELDS,
    ExperimentRegistry,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_M16_DIR = ROOT / "configs" / "mission16"


def test_all_12_experiment_configs_exist_and_valid():
    assert CONFIGS_M16_DIR.exists(), "configs/mission16/ directory must exist."
    config_files = sorted(list(CONFIGS_M16_DIR.glob("*.yaml")))
    assert len(config_files) == 12, f"Expected exactly 12 experiment configurations, found {len(config_files)}"

    expected_ids = [
        "E1_baseline_yolov8n",
        "E2_d1_finetune_yolov8n",
        "E3_baseline_policy_j",
        "E4_d1_policy_j",
        "E5_baseline_policy_k",
        "E6_d1_policy_k",
        "E7_small_object_sampling",
        "E8_adverse_weather_aug",
        "E9_condition_ablation",
        "E10_mixed_condition",
        "E11_layer_freezing",
        "E12_synthetic_real_mixed",
    ]

    found_ids = []
    for cf in config_files:
        with open(cf) as f:
            data = yaml.safe_load(f)
        assert "experiment_id" in data
        assert "mission_id" in data
        assert "research_question" in data
        assert "hypothesis" in data
        assert "model" in data
        assert "status" in data
        assert len(data["research_question"]) > 15
        assert len(data["hypothesis"]) > 15
        found_ids.append(data["experiment_id"])

    for exp_id in expected_ids:
        assert exp_id in found_ids, f"Experiment {exp_id} not found in configurations!"


def test_experiment_registry_csv_integrity():
    assert REGISTRY_CSV_PATH.exists(), "experiments/experiment_registry.csv must exist."
    reg = ExperimentRegistry(csv_path=REGISTRY_CSV_PATH)
    records = reg.load_records()
    assert len(records) >= 12

    # Check schema consistency
    for r in records:
        for field in SCHEMA_FIELDS:
            assert field in r, f"Record {r.get('experiment_id')} missing field {field}"


def test_registry_upsert_and_json_sync(tmp_path):
    csv_file = tmp_path / "test_registry.csv"
    reg = ExperimentRegistry(csv_path=csv_file)

    test_rec = {
        "experiment_id": "TEST_EXP_01",
        "mission_id": "TestMission",
        "research_question": "Does testing work?",
        "hypothesis": "Yes it works.",
        "status": "COMPLETED",
    }
    reg.upsert_experiment(test_rec)

    records = reg.load_records()
    assert len(records) == 1
    assert records[0]["experiment_id"] == "TEST_EXP_01"
    assert records[0]["status"] == "COMPLETED"

    # Verify JSON export sync
    json_file = tmp_path / "test_registry.json"
    assert json_file.exists()
    with open(json_file) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["experiment_id"] == "TEST_EXP_01"

    # Update existing
    test_rec["status"] = "UPDATED_STATUS"
    reg.upsert_experiment(test_rec)
    records_updated = reg.load_records()
    assert len(records_updated) == 1
    assert records_updated[0]["status"] == "UPDATED_STATUS"
