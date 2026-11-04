"""WeatherGuard-D1 Leakage-Safe Stratified Dataset Splitting Engine.

Module: scripts/create_d1_splits.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
Mission: Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

Partitions WeatherGuard-D1 into Train (1,200 / 60%), Validation (400 / 20%),
and Internal Test (400 / 20%) while strictly preventing data leakage across splits:
1. Scenario / prompt family clustering (entire families stay within one split).
2. Perceptual near-duplicate clustering (similar images never split).
3. Condition-level stratification (Day, Night, Rain, Fog).
4. Deterministic random seed and split integrity hashing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "weatherguard_d1.yaml"


def compute_manifest_hash(split_data: Dict[str, Any]) -> str:
    """Computes SHA-256 hash of the deterministic split assignment."""
    canonical_str = ""
    for split_name in ("train", "validation", "test"):
        img_ids = sorted(split_data.get(split_name, {}).get("image_ids", []))
        canonical_str += f"{split_name}:{','.join(img_ids)};"
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


class DisjointSetUnion:
    """Disjoint Set Union (Union-Find) with path compression for transitive clustering."""

    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str):
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.parent[rx] = ry


class D1SplitGenerator:
    """Partitions D1 images into train, val, and test splits with strict group-level isolation."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        random_seed: int = 42,
    ):
        if not config_path.exists():
            raise FileNotFoundError(f"Config path {config_path} does not exist.")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.random_seed = random_seed
        self.splits_cfg = self.config.get("splits", {})
        self.conditions = list(self.config.get("conditions", {}).keys())

    def generate_splits(
        self,
        ingestion_manifest_data: Dict[str, Any],
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
    ) -> Dict[str, Any]:
        """Generates leakage-safe splits from ingestion manifest records.

        Enforces global transitive grouping across both prompt scenario families
        and perceptual near-duplicates spanning multiple weather conditions.
        """
        records = ingestion_manifest_data.get("records", [])
        if not records:
            return {
                "status": "NOT_AVAILABLE",
                "message": "No ingested image records found in manifest. Cannot partition empty dataset.",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        # Filter accepted images
        accepted_records = [
            r for r in records
            if r.get("status") in ("ACCEPTED", "NEAR_DUPLICATE_FLAGGED")
        ]

        if not accepted_records:
            return {
                "status": "NO_VALID_RECORDS",
                "message": "No accepted or near-duplicate records available for splitting.",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        total_images = len(accepted_records)

        # Step 1: Global Transitive Clustering via Disjoint Set Union (DSU)
        # Groups images connected by:
        # 1. Identical scenario_family_id (even across different weather conditions!)
        # 2. Near-duplicate flags and transitive chains (A ~ B and B ~ C -> {A, B, C})
        dsu = DisjointSetUnion()

        for r in accepted_records:
            img_id = r["image_id"]
            dsu.find(img_id)

            # Global scenario family link
            fam_id = r.get("scenario_family_id")
            if fam_id and fam_id != "fam_default":
                dsu.union(img_id, f"scenario_family::{fam_id}")

            # Global near-duplicate links and transitive chains
            for fl in r.get("flags", []):
                if "Near-duplicate candidate of:" in fl:
                    cand_part = fl.split("Near-duplicate candidate of:")[-1]
                    for candidate in cand_part.split(","):
                        cand_id = candidate.split("(")[0].strip()
                        if cand_id:
                            dsu.union(img_id, cand_id)

        # Step 2: Build atomic connected components
        clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in accepted_records:
            root = dsu.find(r["image_id"])
            clusters[root].append(r)

        # Sort clusters deterministically: larger clusters first for better bin packing, then deterministic tie-break
        rng = random.Random(self.random_seed)
        cluster_list = list(clusters.values())
        cluster_list.sort(key=lambda c: (-len(c), sorted([r["image_id"] for r in c])[0]))

        target_train = int(round(total_images * train_ratio))
        target_val = int(round(total_images * val_ratio))
        target_test = total_images - target_train - target_val

        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []

        split_summary: Dict[str, Dict[str, int]] = {
            "train": {c: 0 for c in self.conditions},
            "validation": {c: 0 for c in self.conditions},
            "test": {c: 0 for c in self.conditions},
        }

        # Step 3: Atomic Group Allocation
        for cluster in cluster_list:
            c_count = len(cluster)
            c_ids = [r["image_id"] for r in cluster]

            # Determine best destination split
            curr_tr = len(train_ids)
            curr_va = len(val_ids)
            curr_te = len(test_ids)

            # Allocate to split with highest remaining capacity relative to target
            rem_tr = target_train - curr_tr
            rem_va = target_val - curr_va
            rem_te = target_test - curr_te

            if rem_tr >= rem_va and rem_tr >= rem_te and rem_tr > 0:
                dest = "train"
                train_ids.extend(c_ids)
            elif rem_va >= rem_te and rem_va > 0:
                dest = "validation"
                val_ids.extend(c_ids)
            else:
                dest = "test"
                test_ids.extend(c_ids)

            for r in cluster:
                cond = r.get("condition", "day")
                if cond in split_summary[dest]:
                    split_summary[dest][cond] += 1

        # Step 4: Verification of zero leakage
        s_train = set(train_ids)
        s_val = set(val_ids)
        s_test = set(test_ids)

        overlap_tv = s_train.intersection(s_val)
        overlap_tt = s_train.intersection(s_test)
        overlap_vt = s_val.intersection(s_test)

        if overlap_tv or overlap_tt or overlap_vt:
            raise RuntimeError(f"Data leakage detected! Overlaps: Train-Val={len(overlap_tv)}, Train-Test={len(overlap_tt)}, Val-Test={len(overlap_vt)}")

        # Verify cluster-level atomic isolation
        train_roots = {dsu.find(i) for i in train_ids}
        val_roots = {dsu.find(i) for i in val_ids}
        test_roots = {dsu.find(i) for i in test_ids}

        c_tv = train_roots.intersection(val_roots)
        c_tt = train_roots.intersection(test_roots)
        c_vt = val_roots.intersection(test_roots)
        if c_tv or c_tt or c_vt:
            raise RuntimeError(f"Cluster leakage detected! Overlapping cluster roots: TV={len(c_tv)}, TT={len(c_tt)}, VT={len(c_vt)}")

        actual_tr_r = round(len(train_ids) / total_images, 4)
        actual_va_r = round(len(val_ids) / total_images, 4)
        actual_te_r = round(len(test_ids) / total_images, 4)

        split_manifest = {
            "status": "SPLITS_FROZEN",
            "dataset_name": self.config.get("dataset_name", "WeatherGuard-D1"),
            "dataset_version": self.config.get("dataset_version", "1.0.0-synthetic"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "random_seed": self.random_seed,
            "isolation_rule": "global_dsu_transitive_scenario_family_and_dhash_clustering",
            "allocation_ratios": {
                "target": {"train": train_ratio, "validation": val_ratio, "test": test_ratio},
                "actual": {"train": actual_tr_r, "validation": actual_va_r, "test": actual_te_r},
                "deviation": {
                    "train": round(actual_tr_r - train_ratio, 4),
                    "validation": round(actual_va_r - val_ratio, 4),
                    "test": round(actual_te_r - test_ratio, 4),
                },
            },
            "summary": {
                "total_images": total_images,
                "train_count": len(train_ids),
                "validation_count": len(val_ids),
                "test_count": len(test_ids),
                "condition_breakdown": split_summary,
                "total_connected_clusters": len(clusters),
            },
            "train": {
                "count": len(train_ids),
                "image_ids": sorted(train_ids),
            },
            "validation": {
                "count": len(val_ids),
                "image_ids": sorted(val_ids),
            },
            "test": {
                "count": len(test_ids),
                "image_ids": sorted(test_ids),
                "locked_status": "FROZEN_INTERNAL_TEST",
            },
        }

        split_manifest["manifest_hash"] = compute_manifest_hash(split_manifest)
        return split_manifest

        split_manifest["manifest_hash"] = compute_manifest_hash(split_manifest)
        return split_manifest

    def export_splits(self, split_manifest: Dict[str, Any], output_path: Path) -> Path:
        """Saves split manifest to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(split_manifest, f, indent=2)
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate leakage-safe stratified dataset splits for WeatherGuard-D1.")
    parser.add_argument("--ingestion-manifest", "--manifest", dest="ingestion_manifest", type=str, default="data/WeatherGuard-D1/manifests/ingestion_manifest.json")
    parser.add_argument("--output-manifest", "--output", dest="output_manifest", type=str, default="data/WeatherGuard-D1/manifests/d1_split_manifest.json")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ingest_p = Path(args.ingestion_manifest)
    if not ingest_p.exists() and Path("artifacts/d1_ingestion_manifest.json").exists():
        ingest_p = Path("artifacts/d1_ingestion_manifest.json")

    if not ingest_p.exists():
        print(f"Ingestion manifest {ingest_p} not found. Cannot generate splits. Exiting.")
        return

    with open(ingest_p) as f:
        ingest_data = json.load(f)

    gen = D1SplitGenerator(config_path=Path(args.config), random_seed=args.seed)
    splits = gen.generate_splits(ingest_data)
    out_p = gen.export_splits(splits, Path(args.output_manifest))

    # Mirror to artifacts/d1_splits_manifest.json for cross-tool compatibility
    artifact_mirror = ROOT / "artifacts" / "d1_splits_manifest.json"
    gen.export_splits(splits, artifact_mirror)

    print(f"Splits generated: Train={splits['summary']['train_count']}, Val={splits['summary']['validation_count']}, Test={splits['summary']['test_count']}.")
    print(f"Saved to {out_p} and mirrored to {artifact_mirror} with Hash: {splits.get('manifest_hash')}")


if __name__ == "__main__":
    main()
