"""Workstream H — experiment registry.

Appends a record for every successful run to `artifacts/experiment_registry.jsonl`
(one JSON object per line) and `artifacts/experiment_registry.csv`. Each
entry has a unique ID, timestamp, git commit, detector, dataset,
configuration, and pointers to result artifacts.

Existing results are not overwritten: the registry only *appends*.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, cwd=ROOT,
        )
        return out.stdout.strip()[:12]
    except Exception:
        return "unknown"


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def make_entry(
    experiment_id: str,
    description: str,
    detector: str,
    dataset_split: str,
    image_ids: list[str],
    config: dict,
    result_artifacts: list[str],
    status: str = "completed",
    error: str | None = None,
    runtime_seconds: float | None = None,
) -> dict:
    artifact_paths = [str(Path(p)) for p in result_artifacts]
    return {
        "experiment_id": experiment_id,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_commit": _git_commit(),
        "description": description,
        "detector": detector,
        "dataset_split": dataset_split,
        "n_images": len(image_ids),
        "image_ids_sample": image_ids[:5],
        "config": config,
        "result_artifacts": artifact_paths,
        "result_hashes": {p: _file_hash(Path(p)) for p in artifact_paths},
        "runtime_seconds": runtime_seconds,
        "status": status,
        "error": error,
    }


def append_entry(entry: dict) -> None:
    jsonl = ARTIFACTS / "experiment_registry.jsonl"
    csv_path = ARTIFACTS / "experiment_registry.csv"
    with jsonl.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    # CSV (header if missing)
    flat = {
        "experiment_id": entry["experiment_id"],
        "timestamp": entry["timestamp"],
        "git_commit": entry["git_commit"],
        "description": entry["description"],
        "detector": entry["detector"],
        "dataset_split": entry["dataset_split"],
        "n_images": entry["n_images"],
        "status": entry["status"],
        "runtime_seconds": entry["runtime_seconds"],
        "result_artifacts": ";".join(entry["result_artifacts"]),
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if write_header:
            w.writeheader()
        w.writerow(flat)


def list_entries() -> list[dict]:
    jsonl = ARTIFACTS / "experiment_registry.jsonl"
    if not jsonl.exists():
        return []
    return [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List existing entries and exit.")
    parser.add_argument("--experiment-id", type=str)
    parser.add_argument("--description", type=str)
    parser.add_argument("--detector", type=str, default="")
    parser.add_argument("--dataset-split", type=str, default="val")
    parser.add_argument("--image-ids", type=str, default="")
    parser.add_argument("--config-json", type=str, default="{}")
    parser.add_argument("--result-artifacts", type=str, default="")
    parser.add_argument("--status", type=str, default="completed")
    parser.add_argument("--error", type=str, default="")
    parser.add_argument("--runtime-seconds", type=float, default=0.0)
    args = parser.parse_args()

    if args.list:
        for e in list_entries():
            print(json.dumps(e, indent=2))
        return

    if not args.experiment_id or not args.description:
        raise SystemExit("--experiment-id and --description are required.")

    ids = [s for s in args.image_ids.split(",") if s]
    config = json.loads(args.config_json) if args.config_json else {}
    artifacts_list = [s for s in args.result_artifacts.split(",") if s]
    entry = make_entry(
        experiment_id=args.experiment_id,
        description=args.description,
        detector=args.detector,
        dataset_split=args.dataset_split,
        image_ids=ids,
        config=config,
        result_artifacts=artifacts_list,
        status=args.status,
        error=args.error or None,
        runtime_seconds=args.runtime_seconds,
    )
    append_entry(entry)
    print(f"Appended {entry['experiment_id']} to {ARTIFACTS/'experiment_registry.jsonl'}")


if __name__ == "__main__":
    main()
