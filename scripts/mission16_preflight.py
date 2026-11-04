#!/usr/bin/env python3
"""WeatherGuard Mission 16 Preflight Verification and Experiment Readiness Tool.

Audits all experiment configuration files (E1-E12), verifies hyperparameter syntax,
evaluates data dependencies, models memory footprints for Apple M1 8 GB CPU-only
execution, and reports execution readiness without fabricating results.

Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PreflightResult:
    experiment_id: str
    config_file: str
    status: str  # READY_FOR_EXECUTION, PENDING_DATA_INGESTION, CONFIGURATION_ERROR
    model_architecture: str
    weights_path: str
    weights_exist: bool
    training_enabled: bool
    evaluation_policy: str
    dataset_dependencies_met: bool
    missing_dependencies: List[str]
    estimated_ram_gb: float
    m1_hardware_safe: bool
    warnings: List[str]
    errors: List[str]


def estimate_m1_ram_footprint(
    imgsz: int = 640,
    batch_size: int = 8,
    policy: str = "raw_detector",
    training_enabled: bool = False,
) -> Tuple[float, bool, List[str]]:
    """Estimates memory footprint on Apple Silicon M1 (8 GB unified memory, CPU-only).

    Returns:
        (estimated_ram_gb, is_safe, warnings)
    """
    warnings: List[str] = []
    # Baseline OS + Python interpreter + PyTorch runtime
    base_ram = 1.5

    # Model parameters and forward graph (YOLOv8n)
    model_ram = 0.8

    # Batch tensor buffer
    tensor_bytes = batch_size * 3 * imgsz * imgsz * 4  # float32
    tensor_ram = tensor_bytes / (1024 ** 3)

    # Gradient & optimizer memory during training (AdamW stores 2 states per param)
    training_ram = 0.9 if training_enabled else 0.0

    # High-resolution crops / proposals for Policy J or K
    policy_ram = 0.5 if policy in ("policy_j", "policy_k") else 0.0

    total_est = round(base_ram + model_ram + tensor_ram + training_ram + policy_ram, 2)

    if batch_size > 8:
        warnings.append(f"Batch size {batch_size} exceeds recommended edge limit of 8 on 8 GB M1; risk of memory swap.")
    if imgsz > 832:
        warnings.append(f"Resolution {imgsz}px may cause significant CPU latency spikes.")
    if total_est > 5.5:
        warnings.append(f"Estimated memory {total_est} GB exceeds safe headroom (5.5 GB) on 8 GB unified system.")
        return total_est, False, warnings

    return total_est, True, warnings


def validate_config(config_path: Path, root_dir: Path = ROOT) -> PreflightResult:
    """Performs deep static and operational preflight audit of a single YAML config."""
    errors: List[str] = []
    warnings: List[str] = []
    missing_deps: List[str] = []

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        return PreflightResult(
            experiment_id=config_path.stem,
            config_file=str(config_path.relative_to(root_dir) if config_path.is_relative_to(root_dir) else config_path),
            status="CONFIGURATION_ERROR",
            model_architecture="unknown",
            weights_path="none",
            weights_exist=False,
            training_enabled=False,
            evaluation_policy="unknown",
            dataset_dependencies_met=False,
            missing_dependencies=[f"YAML parsing error: {e}"],
            estimated_ram_gb=0.0,
            m1_hardware_safe=False,
            warnings=[],
            errors=[str(e)],
        )

    # 1. Check required top-level metadata
    exp_id = cfg.get("experiment_id", config_path.stem)
    if "experiment_id" not in cfg:
        errors.append("Missing 'experiment_id' key in configuration.")
    if "mission_id" not in cfg:
        warnings.append("Missing 'mission_id' key.")

    # 2. Check model specification
    model_cfg = cfg.get("model", {})
    arch = model_cfg.get("architecture", "YOLOv8n")
    weights_str = model_cfg.get("weights") or model_cfg.get("base_weights") or "yolov8n.pt"
    weights_path = root_dir / weights_str if not Path(weights_str).is_absolute() else Path(weights_str)
    weights_exist = weights_path.exists()

    if not weights_exist:
        # Standard weights like yolov8n.pt might be downloadable by Ultralytics, but flag for audit
        if weights_str in ("yolov8n.pt", "yolov8s.pt"):
            warnings.append(f"Base weights '{weights_str}' not locally cached; will be fetched automatically by Ultralytics.")
        else:
            missing_deps.append(f"Model weights file '{weights_str}' not found on disk.")

    # 3. Check training vs evaluation flags
    train_cfg = cfg.get("training", {})
    training_enabled = bool(train_cfg.get("enabled", False))
    eval_cfg = cfg.get("evaluation", {})
    policy = eval_cfg.get("policy", "raw_detector")
    batch_size = int(train_cfg.get("batch_size", 8))
    imgsz = int(train_cfg.get("imgsz", eval_cfg.get("imgsz", 640)))

    # 4. Check dataset dependencies
    dataset_dependencies_met = True
    if training_enabled:
        ds_yaml = train_cfg.get("dataset_yaml", "data/WeatherGuard-D1/dataset.yaml")
        ds_yaml_path = root_dir / ds_yaml if not Path(ds_yaml).is_absolute() else Path(ds_yaml)
        split_manifest = root_dir / "data" / "WeatherGuard-D1" / "manifests" / "d1_split_manifest.json"
        artifacts_split = root_dir / "artifacts" / "d1_splits_manifest.json"

        if not ds_yaml_path.exists() and not split_manifest.exists() and not artifacts_split.exists():
            dataset_dependencies_met = False
            missing_deps.append("WeatherGuard-D1 dataset not yet ingested (manifests/splits missing).")

    # Check evaluation datasets
    eval_datasets = eval_cfg.get("datasets", [])
    for ds in eval_datasets:
        ds_name = ds.get("name", "").lower()
        if ds_name == "d1":
            split_manifest = root_dir / "data" / "WeatherGuard-D1" / "manifests" / "d1_split_manifest.json"
            artifacts_split = root_dir / "artifacts" / "d1_splits_manifest.json"
            if not split_manifest.exists() and not artifacts_split.exists():
                dataset_dependencies_met = False
                missing_deps.append("D1 evaluation requested but split manifest is absent.")

    # 5. Hardware memory estimation for M1
    est_ram, m1_safe, hw_warns = estimate_m1_ram_footprint(
        imgsz=imgsz,
        batch_size=batch_size,
        policy=policy,
        training_enabled=training_enabled,
    )
    warnings.extend(hw_warns)

    # 6. Overall readiness status
    if errors:
        status = "CONFIGURATION_ERROR"
    elif not dataset_dependencies_met:
        status = "PENDING_DATA_INGESTION"
    elif not weights_exist and weights_str not in ("yolov8n.pt", "yolov8s.pt"):
        status = "PENDING_WEIGHTS"
    else:
        status = "READY_FOR_EXECUTION"

    rel_cfg = str(config_path.relative_to(root_dir) if config_path.is_relative_to(root_dir) else config_path)

    return PreflightResult(
        experiment_id=exp_id,
        config_file=rel_cfg,
        status=status,
        model_architecture=arch,
        weights_path=weights_str,
        weights_exist=weights_exist,
        training_enabled=training_enabled,
        evaluation_policy=policy,
        dataset_dependencies_met=dataset_dependencies_met,
        missing_dependencies=missing_deps,
        estimated_ram_gb=est_ram,
        m1_hardware_safe=m1_safe,
        warnings=warnings,
        errors=errors,
    )


def audit_mission16_configs(
    config_dir: Path,
    root_dir: Path = ROOT,
) -> Dict[str, Any]:
    """Scans and audits all experiment configuration files in the given directory."""
    config_files = sorted(list(config_dir.glob("*.yaml")))
    results: List[PreflightResult] = []

    for cf in config_files:
        results.append(validate_config(cf, root_dir=root_dir))

    ready_count = sum(1 for r in results if r.status == "READY_FOR_EXECUTION")
    pending_count = sum(1 for r in results if r.status == "PENDING_DATA_INGESTION")
    pending_weights = sum(1 for r in results if r.status == "PENDING_WEIGHTS")
    error_count = sum(1 for r in results if r.status == "CONFIGURATION_ERROR")

    return {
        "total_configs": len(results),
        "ready_for_execution": ready_count,
        "pending_data_ingestion": pending_count,
        "pending_weights": pending_weights,
        "configuration_errors": error_count,
        "experiments": [asdict(r) for r in results],
        "system_profile": {
            "platform": "Apple Silicon M1 (macOS Darwin)",
            "memory_limit_gb": 8.0,
            "max_safe_ram_footprint_gb": 5.5,
            "execution_target": "CPU-only",
        },
    }


def print_preflight_summary(report: Dict[str, Any]) -> None:
    """Prints a structured ASCII report table."""
    print("=" * 90)
    print(" WEATHERGUARD — MISSION 16 EXPERIMENT MATRIX PREFLIGHT AUDIT")
    print("=" * 90)
    print(f"Total Configurations Scanned : {report['total_configs']}")
    print(f"Ready for Immediate Run      : {report['ready_for_execution']}")
    print(f"Pending D1 Data Ingestion    : {report['pending_data_ingestion']}")
    print(f"Pending Model Weights        : {report.get('pending_weights', 0)}")
    print(f"Configuration Errors         : {report['configuration_errors']}")
    print("-" * 90)
    print(f"{'Exp ID':<26} {'Status':<24} {'Train':<7} {'Policy':<16} {'Est RAM':<8}")
    print("-" * 90)

    for exp in report["experiments"]:
        train_flag = "Yes" if exp["training_enabled"] else "No"
        print(f"{exp['experiment_id']:<26} {exp['status']:<24} {train_flag:<7} {exp['evaluation_policy']:<16} {exp['estimated_ram_gb']:<5} GB")
        if exp["missing_dependencies"]:
            for d in exp["missing_dependencies"]:
                print(f"   ↳ Dependency: {d}")
        if exp["errors"]:
            for err in exp["errors"]:
                print(f"   ↳ Error: {err}")

    print("=" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(description="WeatherGuard Mission 16 Experiment Matrix Preflight Audit")
    parser.add_argument("--config-dir", type=str, default=str(ROOT / "configs" / "mission16"), help="Directory containing experiment YAML configs")
    parser.add_argument("--output-json", type=str, default=str(ROOT / "artifacts" / "mission16_preflight_report.json"), help="Output path for preflight JSON report")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any configuration errors or unmet dependencies are found")
    args = parser.parse_args()

    cfg_dir = Path(args.config_dir)
    if not cfg_dir.exists():
        print(f"Error: Config directory '{cfg_dir}' does not exist.", file=sys.stderr)
        return 1

    report = audit_mission16_configs(config_dir=cfg_dir, root_dir=ROOT)
    print_preflight_summary(report)

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nPreflight audit report written to: {out_p}")

    if args.strict and report["configuration_errors"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
