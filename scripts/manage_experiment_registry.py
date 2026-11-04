"""WeatherGuard Standardized Experiment Registry Management.

Module: scripts/manage_experiment_registry.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
Mission: Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

Maintains a machine-readable CSV and JSON registry of all research experiments
recording hypotheses, seeds, hardware, metric results, artifact paths, and status.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_CSV_PATH = ROOT / "experiments" / "experiment_registry.csv"
REGISTRY_JSON_PATH = ROOT / "experiments" / "experiment_registry.json"

SCHEMA_FIELDS = [
    "experiment_id",
    "mission_id",
    "research_question",
    "hypothesis",
    "dataset_version",
    "dataset_manifest_hash",
    "model_architecture",
    "pretrained_weights_id",
    "training_config",
    "evaluation_config",
    "random_seed",
    "software_versions",
    "hardware",
    "start_time",
    "end_time",
    "overall_f1",
    "small_f1",
    "ultra_small_f1",
    "mean_latency_ms",
    "fps",
    "status",  # "PLANNED", "PREFLIGHT_VERIFIED", "RUNNING", "COMPLETED", "FAILED"
    "artifact_locations",
    "known_limitations",
]


class ExperimentRegistry:
    """Manages reading, inserting, and updating experiment records."""

    def __init__(self, csv_path: Path = REGISTRY_CSV_PATH):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            self._initialize_csv()

    def _initialize_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(SCHEMA_FIELDS)

    def load_records(self) -> List[Dict[str, str]]:
        if not self.csv_path.exists():
            return []
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def upsert_experiment(self, record: Dict[str, Any]):
        """Inserts or updates an experiment record keyed by experiment_id."""
        existing = self.load_records()
        exp_id = record.get("experiment_id")
        if not exp_id:
            raise ValueError("Record must have 'experiment_id'")

        updated = False
        formatted_record = {}
        for k in SCHEMA_FIELDS:
            val = record.get(k, "")
            if val is None:
                val = ""
            formatted_record[k] = str(val)

        for i, row in enumerate(existing):
            if row["experiment_id"] == exp_id:
                existing[i] = formatted_record
                updated = True
                break

        if not updated:
            existing.append(formatted_record)

        # Write back CSV
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEMA_FIELDS)
            writer.writeheader()
            writer.writerows(existing)

        # Sync JSON export
        json_path = self.csv_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(existing, f, indent=2)

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, str]]:
        records = self.load_records()
        for r in records:
            if r["experiment_id"] == experiment_id:
                return r
        return None


def populate_initial_mission16_matrix(registry: ExperimentRegistry):
    """Initializes the 12 planned Mission 16 experiments into the registry."""
    configs_dir = ROOT / "configs" / "mission16"
    import yaml

    for cfg_file in sorted(list(configs_dir.glob("*.yaml"))):
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)

        exp_id = cfg.get("experiment_id")
        if not exp_id:
            continue

        rec = {
            "experiment_id": exp_id,
            "mission_id": cfg.get("mission_id", "Mission 16"),
            "research_question": cfg.get("research_question", ""),
            "hypothesis": cfg.get("hypothesis", ""),
            "dataset_version": "1.0.0-synthetic",
            "dataset_manifest_hash": "pending_data_ingestion",
            "model_architecture": cfg.get("model", {}).get("architecture", "YOLOv8n"),
            "pretrained_weights_id": cfg.get("model", {}).get("weights") or cfg.get("model", {}).get("base_weights", "yolov8n.pt"),
            "training_config": json.dumps(cfg.get("training", {})),
            "evaluation_config": json.dumps(cfg.get("evaluation", {})),
            "random_seed": cfg.get("training", {}).get("seed", 42),
            "software_versions": f"Python {platform.python_version()}",
            "hardware": "Apple M1 8-core CPU, 8 GB Unified Memory",
            "start_time": "",
            "end_time": "",
            "overall_f1": "",
            "small_f1": "",
            "ultra_small_f1": "",
            "mean_latency_ms": "",
            "fps": "",
            "status": cfg.get("status", "PLANNED"),
            "artifact_locations": f"configs/mission16/{cfg_file.name}",
            "known_limitations": "Pending synthetic data ingestion and split freezing",
        }
        registry.upsert_experiment(rec)


def main():
    parser = argparse.ArgumentParser(description="Manage WeatherGuard standardized experiment registry.")
    parser.add_argument("--init-m16", action="store_true", help="Populate initial Mission 16 experiment matrix")
    parser.add_argument("--list", action="store_true", help="List all registered experiments")
    args = parser.parse_args()

    reg = ExperimentRegistry()
    if args.init_m16:
        populate_initial_mission16_matrix(reg)
        print(f"Populated Mission 16 experiment matrix into {REGISTRY_CSV_PATH}")

    if args.list:
        records = reg.load_records()
        print(f"\nTotal Registered Experiments: {len(records)}")
        for r in records:
            print(f"- [{r['status']}] {r['experiment_id']}: {r['research_question'][:80]}...")


if __name__ == "__main__":
    main()
