"""Tests for Mission 5 data protocol, manifest integrity, and dataset preparation."""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data" / "dawn_yolo"


def test_mission5_split_manifest_integrity():
    manifest_path = ARTIFACTS / "mission5_split_manifest.json"
    assert manifest_path.exists(), "Manifest artifact missing"
    with open(manifest_path) as f:
        data = json.load(f)

    groups = data["groups"]
    assert "Group_A_DAWN_Train" in groups
    assert "Group_A_DAWN_Val" in groups
    assert "Group_B_BDD100K_Development" in groups
    assert "Group_C_BDD100K_Locked_Test" in groups
    assert "Group_D_BDD100K_Train" in groups

    assert groups["Group_A_DAWN_Train"]["n_images"] == 821
    assert groups["Group_A_DAWN_Val"]["n_images"] == 206
    assert groups["Group_B_BDD100K_Development"]["n_images"] == 160
    assert groups["Group_C_BDD100K_Locked_Test"]["n_images"] == 203
    assert groups["Group_D_BDD100K_Train"]["status"] == "BLOCKED_UNAVAILABLE"

    # Verify disjointness
    dev_ids = set(groups["Group_B_BDD100K_Development"]["image_ids"])
    test_ids = set(groups["Group_C_BDD100K_Locked_Test"]["image_ids"])
    assert len(dev_ids.intersection(test_ids)) == 0, "Group B and Group C must have zero overlap"


def test_mission5_prepared_dataset_structure():
    yaml_200 = DATA_DIR / "dawn_200.yaml"
    assert yaml_200.exists(), "dawn_200.yaml missing"

    train_200_txt = DATA_DIR / "train_200.txt"
    assert train_200_txt.exists(), "train_200.txt missing"
    with open(train_200_txt) as f:
        train_lines = [line.strip() for line in f if line.strip()]
    assert len(train_lines) == 200, f"Expected 200 training images, got {len(train_lines)}"

    val_txt = DATA_DIR / "val.txt"
    assert val_txt.exists(), "val.txt missing"
    with open(val_txt) as f:
        val_lines = [line.strip() for line in f if line.strip()]
    assert len(val_lines) == 206, f"Expected 206 val images, got {len(val_lines)}"


def test_mission5_domain_results_blocked():
    dom_path = ARTIFACTS / "mission5_domain_results.json"
    assert dom_path.exists(), "mission5_domain_results.json missing"
    with open(dom_path) as f:
        data = json.load(f)
    assert data["status"] == "BLOCKED_UNAVAILABLE"
    assert "risk_assessment" in data["cohort_audit"]["group_d_bdd100k_train"]
