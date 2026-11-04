"""Mission 18 — Independent Leakage-Safe Dataset Split Integration Tests.

Validates the global Disjoint Set Union (DSU) clustering and split partitioning engine
against adversarial fixtures: multi-condition scenario families, transitive duplicate chains,
missing references, and large atomic clusters.

Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""

from pathlib import Path
import pytest

from scripts.create_d1_splits import (
    DEFAULT_CONFIG_PATH,
    D1SplitGenerator,
    DisjointSetUnion,
)


@pytest.fixture
def split_generator():
    return D1SplitGenerator(config_path=DEFAULT_CONFIG_PATH, random_seed=42)


# ==============================================================================
# 1. Transitive Duplicate Chains (A -> B -> C -> D)
# ==============================================================================

def test_transitive_near_duplicate_chains_never_split(split_generator):
    """Verifies that an extended transitive near-duplicate chain A -> B -> C -> D
    is unified into a single atomic connected component and placed into exactly one split.
    """
    records = [
        # Chain 1: img_001 -> img_002 -> img_003 -> img_004 across 4 different weather conditions!
        {"image_id": "img_001", "condition": "day", "status": "ACCEPTED", "scenario_family_id": "fam_default", "flags": ["Near-duplicate candidate of: img_002(dist=2)"]},
        {"image_id": "img_002", "condition": "fog", "status": "ACCEPTED", "scenario_family_id": "fam_default", "flags": ["Near-duplicate candidate of: img_003(dist=3)"]},
        {"image_id": "img_003", "condition": "rain", "status": "ACCEPTED", "scenario_family_id": "fam_default", "flags": ["Near-duplicate candidate of: img_004(dist=1)"]},
        {"image_id": "img_004", "condition": "night", "status": "ACCEPTED", "scenario_family_id": "fam_default", "flags": []},

        # Independent filler records (20 independent images) to enable train/val/test allocation
        *[
            {"image_id": f"filler_{i:03d}", "condition": "day", "status": "ACCEPTED", "scenario_family_id": "fam_default", "flags": []}
            for i in range(20)
        ]
    ]

    manifest = {"records": records}
    split_res = split_generator.generate_splits(manifest, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    assert split_res["status"] == "SPLITS_FROZEN"

    train_ids = set(split_res["train"]["image_ids"])
    val_ids = set(split_res["validation"]["image_ids"])
    test_ids = set(split_res["test"]["image_ids"])

    chain_ids = {"img_001", "img_002", "img_003", "img_004"}

    # The entire chain must reside in exactly one split
    in_train = chain_ids.issubset(train_ids)
    in_val = chain_ids.issubset(val_ids)
    in_test = chain_ids.issubset(test_ids)

    assert in_train or in_val or in_test, "Transitive chain was split across partitions!"
    # Ensure no member leaked into another split
    assert len(chain_ids.intersection(train_ids)) in (0, 4)
    assert len(chain_ids.intersection(val_ids)) in (0, 4)
    assert len(chain_ids.intersection(test_ids)) in (0, 4)


# ==============================================================================
# 2. Multi-Condition Scenario Family Across Day, Fog, Rain, Night
# ==============================================================================

def test_multi_condition_scenario_family_atomic_isolation(split_generator):
    """Verifies that images generated from the same prompt scenario family across
    different weather conditions are never separated across Train, Val, and Test.
    """
    records = [
        {"image_id": "scene_highway_day", "condition": "day", "scenario_family_id": "fam_highway_01", "status": "ACCEPTED"},
        {"image_id": "scene_highway_fog", "condition": "fog", "scenario_family_id": "fam_highway_01", "status": "ACCEPTED"},
        {"image_id": "scene_highway_rain", "condition": "rain", "scenario_family_id": "fam_highway_01", "status": "ACCEPTED"},
        {"image_id": "scene_highway_night", "condition": "night", "scenario_family_id": "fam_highway_01", "status": "ACCEPTED"},
        *[
            {"image_id": f"other_{i:03d}", "condition": "day", "scenario_family_id": f"fam_distinct_{i}", "status": "ACCEPTED"}
            for i in range(25)
        ]
    ]

    manifest = {"records": records}
    split_res = split_generator.generate_splits(manifest, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    assert split_res["status"] == "SPLITS_FROZEN"

    train_set = set(split_res["train"]["image_ids"])
    val_set = set(split_res["validation"]["image_ids"])
    test_set = set(split_res["test"]["image_ids"])

    fam_members = {"scene_highway_day", "scene_highway_fog", "scene_highway_rain", "scene_highway_night"}
    resides_together = (
        fam_members.issubset(train_set) or
        fam_members.issubset(val_set) or
        fam_members.issubset(test_set)
    )
    assert resides_together is True, "Multi-condition scenario family was leaked across splits!"


# ==============================================================================
# 3. Missing Referenced Duplicate IDs & Corrupted Flags
# ==============================================================================

def test_missing_referenced_duplicate_id_graceful_handling(split_generator):
    """Verifies that if a flag points to a non-existent image ID, DSU handles it gracefully
    without raising KeyError or dropping valid images.
    """
    records = [
        {"image_id": "img_valid_01", "condition": "day", "status": "ACCEPTED", "flags": ["Near-duplicate candidate of: non_existent_image_999(dist=1)"]},
        {"image_id": "img_valid_02", "condition": "day", "status": "ACCEPTED", "flags": []},
        {"image_id": "img_valid_03", "condition": "fog", "status": "ACCEPTED", "flags": []},
    ]
    manifest = {"records": records}
    split_res = split_generator.generate_splits(manifest, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    assert split_res["status"] == "SPLITS_FROZEN"
    assert split_res["summary"]["total_images"] == 3


# ==============================================================================
# 4. Large Atomic Connected Component & Ratio Deviation
# ==============================================================================

def test_large_atomic_cluster_preserves_isolation_over_ratios(split_generator):
    """When a single massive scenario family encompasses 70% of the dataset, the engine
    must prioritize zero leakage over exact 60/20/20 target ratios.
    """
    # 14 images in one cluster, 6 images in separate clusters (Total = 20)
    big_cluster = [
        {"image_id": f"cluster_big_{i:02d}", "condition": "day", "scenario_family_id": "fam_massive", "status": "ACCEPTED"}
        for i in range(14)
    ]
    small_clusters = [
        {"image_id": f"cluster_small_{i:02d}", "condition": "night", "scenario_family_id": f"fam_small_{i}", "status": "ACCEPTED"}
        for i in range(6)
    ]

    manifest = {"records": big_cluster + small_clusters}
    split_res = split_generator.generate_splits(manifest, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    assert split_res["status"] == "SPLITS_FROZEN"

    train_set = set(split_res["train"]["image_ids"])
    val_set = set(split_res["validation"]["image_ids"])
    test_set = set(split_res["test"]["image_ids"])

    big_cluster_ids = {r["image_id"] for r in big_cluster}
    # All 14 must be together in train
    assert big_cluster_ids.issubset(train_set)
    # Zero overlap with val or test
    assert len(big_cluster_ids.intersection(val_set)) == 0
    assert len(big_cluster_ids.intersection(test_set)) == 0

    # Ratios deviate (14/20 = 70% train) but leakage is strictly 0
    assert split_res["allocation_ratios"]["actual"]["train"] == 0.70
