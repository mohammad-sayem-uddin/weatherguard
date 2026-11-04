"""Tests for Mission 16 Preflight Verification and Experiment Readiness Tool."""

import json
from pathlib import Path
import tempfile
import yaml
import pytest

from scripts.mission16_preflight import (
    ROOT,
    audit_mission16_configs,
    estimate_m1_ram_footprint,
    validate_config,
)


def test_estimate_m1_ram_footprint_safe():
    est, safe, warns = estimate_m1_ram_footprint(imgsz=640, batch_size=8, policy="raw_detector", training_enabled=False)
    assert safe is True
    assert 2.0 <= est <= 3.5
    assert len(warns) == 0


def test_estimate_m1_ram_footprint_warnings():
    # Large batch size and excessive resolution
    est, safe, warns = estimate_m1_ram_footprint(imgsz=1280, batch_size=32, policy="policy_k", training_enabled=True)
    assert any("Batch size" in w for w in warns)
    assert any("Resolution" in w for w in warns)


def test_validate_config_broken_yaml(tmp_path: Path):
    broken_file = tmp_path / "broken.yaml"
    broken_file.write_text("experiment_id: [broken yaml", encoding="utf-8")

    res = validate_config(broken_file, root_dir=tmp_path)
    assert res.status == "CONFIGURATION_ERROR"
    assert len(res.errors) > 0


def test_validate_config_synthetic_valid(tmp_path: Path):
    cfg_data = {
        "experiment_id": "test_exp_valid",
        "mission_id": "Mission 16",
        "model": {"architecture": "YOLOv8n", "weights": "yolov8n.pt"},
        "training": {"enabled": False},
        "evaluation": {"policy": "raw_detector", "imgsz": 640},
    }
    cfg_file = tmp_path / "valid.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(cfg_data, f)

    res = validate_config(cfg_file, root_dir=tmp_path)
    assert res.status == "READY_FOR_EXECUTION"
    assert res.experiment_id == "test_exp_valid"
    assert res.m1_hardware_safe is True
    assert len(res.errors) == 0


def test_audit_all_mission16_configs_zero_errors():
    cfg_dir = ROOT / "configs" / "mission16"
    assert cfg_dir.exists(), "configs/mission16 directory must exist"

    report = audit_mission16_configs(config_dir=cfg_dir, root_dir=ROOT)
    assert report["total_configs"] == 12
    assert report["configuration_errors"] == 0, f"Found unexpected configuration errors: {report}"
    assert report["ready_for_execution"] + report["pending_data_ingestion"] + report["pending_weights"] == 12
    assert "system_profile" in report
    assert report["system_profile"]["memory_limit_gb"] == 8.0
