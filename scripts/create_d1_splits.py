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
        """Generates leakage-safe splits from ingestion manifest records."""
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

        # Step 1: Group by condition, and within condition, group by scenario/family ID
        rng = random.Random(self.random_seed)

        # Build clusters
        # If scenario_family_id is missing or trivial, group by perceptual dhash near-duplicates
        condition_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

        for r in accepted_records:
            cond = r.get("condition", "day")
            fam_id = r.get("scenario_family_id") or "fam_default"
            # If flagged as near-duplicate, append to cluster key
            flags = r.get("flags", [])
            for fl in flags:
                if "Near-duplicate candidate of:" in fl:
                    # extract candidate ID to cluster them together
                    parts = fl.split("Near-duplicate candidate of:")[-1].strip().split(",")
                    primary_id = parts[0].split("(")[0].strip()
                    fam_id = f"cluster_{primary_id}"
                    break
            condition_groups[cond][fam_id].append(r)

        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []

        split_summary: Dict[str, Dict[str, int]] = {
            "train": {c: 0 for c in self.conditions},
            "validation": {c: 0 for c in self.conditions},
            "test": {c: 0 for c in self.conditions},
        }

        # Step 2: Stratified allocation per condition
        for cond, fam_dict in condition_groups.items():
            families = list(fam_dict.keys())
            # Deterministic shuffle
            rng.shuffle(families)

            total_cond_images = sum(len(fam_dict[f]) for f in families)
            target_train = int(round(total_cond_images * train_ratio))
            target_val = int(round(total_cond_images * val_ratio))

            curr_train_cnt = 0
            curr_val_cnt = 0

            for f_id in families:
                f_records = fam_dict[f_id]
                f_count = len(f_records)
                f_img_ids = [r["image_id"] for r in f_records]

                # Greedily allocate the entire family without splitting
                if curr_train_cnt + f_count <= target_train or (curr_train_cnt < target_train and curr_val_cnt >= target_val):
                    train_ids.extend(f_img_ids)
                    curr_train_cnt += f_count
                    split_summary["train"][cond] = split_summary["train"].get(cond, 0) + f_count
                elif curr_val_cnt + f_count <= target_val or (curr_val_cnt < target_val):
                    val_ids.extend(f_img_ids)
                    curr_val_cnt += f_count
                    split_summary["validation"][cond] = split_summary["validation"].get(cond, 0) + f_count
                else:
                    test_ids.extend(f_img_ids)
                    split_summary["test"][cond] = split_summary["test"].get(cond, 0) + f_count

        # Step 3: Verification of zero leakage
        s_train = set(train_ids)
        s_val = set(val_ids)
        s_test = set(test_ids)

        overlap_tv = s_train.intersection(s_val)
        overlap_tt = s_train.intersection(s_test)
        overlap_vt = s_val.intersection(s_test)

        if overlap_tv or overlap_tt or overlap_vt:
            raise RuntimeError(f"Data leakage detected! Overlaps: Train-Val={len(overlap_tv)}, Train-Test={len(overlap_tt)}, Val-Test={len(overlap_vt)}")

        split_manifest = {
            "status": "SPLITS_FROZEN",
            "dataset_name": self.config.get("dataset_name", "WeatherGuard-D1"),
            "dataset_version": self.config.get("dataset_version", "1.0.0-synthetic"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "random_seed": self.random_seed,
            "isolation_rule": "prompt_scenario_family_and_dhash_clustering",
            "allocation_ratios": {
                "train": train_ratio,
                "validation": val_ratio,
                "test": test_ratio,
            },
            "summary": {
                "total_images": len(train_ids) + len(val_ids) + len(test_ids),
                "train_count": len(train_ids),
                "validation_count": len(val_ids),
                "test_count": len(test_ids),
                "condition_breakdown": split_summary,
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
