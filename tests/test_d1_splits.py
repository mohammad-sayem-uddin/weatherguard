"""Unit tests for WeatherGuard-D1 Leakage-Safe Dataset Splitting.

Module: tests/test_d1_splits.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""
import json
import shutil
import tempfile
from pathlib import Path
import pytest

from scripts.create_d1_splits import D1SplitGenerator, compute_manifest_hash

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "weatherguard_d1.yaml"


@pytest.fixture
def temp_workspace():
    tmp_dir = Path(tempfile.mkdtemp(prefix="test_d1_splits_"))
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_split_generation_disjointness():
    gen = D1SplitGenerator(config_path=CONFIG_PATH, random_seed=42)

    # Mock 100 ingested images across 4 conditions
    mock_records = []
    conditions = ["day", "night", "rain", "fog"]
    for i in range(100):
        cond = conditions[i % len(conditions)]
        mock_records.append({
            "image_id": f"img_{i:03d}",
            "condition": cond,
            "status": "ACCEPTED",
            "scenario_family_id": f"fam_{cond}_{i // 10}",
            "flags": [],
        })

    manifest_in = {"records": mock_records}
    splits = gen.generate_splits(manifest_in, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20)

    assert splits["status"] == "SPLITS_FROZEN"
    train_ids = set(splits["train"]["image_ids"])
    val_ids = set(splits["validation"]["image_ids"])
    test_ids = set(splits["test"]["image_ids"])

    # Strict disjointness
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

    # Total conservation
    assert len(train_ids) + len(val_ids) + len(test_ids) == 100
    assert 55 <= len(train_ids) <= 65
    assert 15 <= len(val_ids) <= 25
    assert 15 <= len(test_ids) <= 25


def test_scenario_family_atomic_isolation():
    gen = D1SplitGenerator(config_path=CONFIG_PATH, random_seed=123)

    mock_records = []
    # Create a specific tightly related scenario family with 8 images
    for i in range(8):
        mock_records.append({
            "image_id": f"fam_member_{i}",
            "condition": "rain",
            "status": "ACCEPTED",
            "scenario_family_id": "atomic_family_007",
            "flags": [],
        })
    # Add other filler families
    for i in range(40):
        mock_records.append({
            "image_id": f"filler_{i}",
            "condition": "rain",
            "status": "ACCEPTED",
            "scenario_family_id": f"fam_other_{i // 4}",
            "flags": [],
        })

    manifest_in = {"records": mock_records}
    splits = gen.generate_splits(manifest_in)

    train_ids = set(splits["train"]["image_ids"])
    val_ids = set(splits["validation"]["image_ids"])
    test_ids = set(splits["test"]["image_ids"])

    fam_members = {f"fam_member_{i}" for i in range(8)}
    # The entire family must be wholly within ONE split
    in_train = fam_members.issubset(train_ids)
    in_val = fam_members.issubset(val_ids)
    in_test = fam_members.issubset(test_ids)

    assert in_train or in_val or in_test, "Scenario family was fragmented across split boundaries!"


def test_near_duplicate_cluster_atomic_isolation():
    gen = D1SplitGenerator(config_path=CONFIG_PATH, random_seed=42)

    mock_records = [
        {
            "image_id": "seed_a",
            "condition": "fog",
            "status": "ACCEPTED",
            "scenario_family_id": "default",
            "flags": [],
        },
        {
            "image_id": "seed_b",
            "condition": "fog",
            "status": "NEAR_DUPLICATE_FLAGGED",
            "scenario_family_id": "default",
            "flags": ["Near-duplicate candidate of: seed_a(dist=2)"],
        },
    ]
    # Add fillers
    for i in range(30):
        mock_records.append({
            "image_id": f"filler_{i}",
            "condition": "fog",
            "status": "ACCEPTED",
            "scenario_family_id": f"fam_{i // 3}",
            "flags": [],
        })

    splits = gen.generate_splits({"records": mock_records})
    train_ids = set(splits["train"]["image_ids"])
    val_ids = set(splits["validation"]["image_ids"])
    test_ids = set(splits["test"]["image_ids"])

    pair = {"seed_a", "seed_b"}
    assert pair.issubset(train_ids) or pair.issubset(val_ids) or pair.issubset(test_ids), \
        "Near-duplicate pair was separated across splits!"


def test_split_reproducibility_and_hashing():
    gen1 = D1SplitGenerator(config_path=CONFIG_PATH, random_seed=999)
    gen2 = D1SplitGenerator(config_path=CONFIG_PATH, random_seed=999)

    mock_records = [
        {"image_id": f"im_{i}", "condition": "day" if i % 2 == 0 else "night",
         "status": "ACCEPTED", "scenario_family_id": f"f_{i // 5}", "flags": []}
        for i in range(40)
    ]

    s1 = gen1.generate_splits({"records": mock_records})
    s2 = gen2.generate_splits({"records": mock_records})

    assert s1["train"]["image_ids"] == s2["train"]["image_ids"]
    assert s1["validation"]["image_ids"] == s2["validation"]["image_ids"]
    assert s1["test"]["image_ids"] == s2["test"]["image_ids"]
    assert s1["manifest_hash"] == s2["manifest_hash"]
    assert len(s1["manifest_hash"]) == 64


def test_empty_manifest_handling():
    gen = D1SplitGenerator(config_path=CONFIG_PATH)
    res = gen.generate_splits({"records": []})
    assert res["status"] == "NOT_AVAILABLE"
