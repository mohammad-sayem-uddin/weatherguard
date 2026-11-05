#!/usr/bin/env python3
"""WeatherGuard — Dataset Versioning & Cryptographic Manifest Generator.

Generates immutable, versioned JSON manifests for WeatherGuard V2 (Vision) and Audio V2 datasets:
  - artifacts/dataset_manifests/weatherguard_v2_manifest.json
  - artifacts/dataset_manifests/audio_v2_manifest.json
Each manifest includes dataset-level cryptographic checksums, full provenance attribution,
class distributions, and sample-level SHA-256 digests.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetVersioning")

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "weatherguard_corpus"
METADATA_DIR = CORPUS_DIR / "metadata"
MANIFESTS_DIR = ROOT / "artifacts" / "dataset_manifests"


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_vision_v2_manifest() -> Path:
    """Generates weatherguard_v2_manifest.json from corpus_manifest.csv."""
    csv_path = METADATA_DIR / "corpus_manifest.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded vision corpus manifest: %d rows", len(df))

    records = []
    for _, row in df.iterrows():
        img_name = row["image_id"]
        img_path = CORPUS_DIR / "images" / img_name
        lbl_path = CORPUS_DIR / "labels" / f"{Path(img_name).stem}.txt"

        img_hash = compute_file_sha256(img_path) if img_path.exists() else row["duplicate_hash"]

        records.append({
            "image_id": img_name,
            "source_category": row["source_category"],
            "original_source_ref": row["original_source_ref"],
            "collection_date": str(row["collection_date"]),
            "weather_condition": row["weather_condition"],
            "lighting_condition": row["lighting_condition"],
            "camera_viewpoint": row["camera_viewpoint"],
            "scenario_id": row["scenario_id"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "sha256": img_hash,
            "object_count": int(row["object_count"]),
            "ultra_small_count": int(row["ultra_small_count"]),
            "small_count": int(row["small_count"]),
            "medium_count": int(row["medium_count"]),
            "large_count": int(row["large_count"]),
        })

    # Combined master digest
    concat_hashes = "".join(r["sha256"] for r in records)
    dataset_checksum = hashlib.sha256(concat_hashes.encode("utf-8")).hexdigest()

    manifest = {
        "dataset_name": "WeatherGuard Adverse-Weather Road Perception Corpus",
        "dataset_version": "v2.0.0",
        "release_date": "2026-11-05",
        "author": "Mohammad Sayem Uddin <sayemwork41@gmail.com>",
        "repository": "https://github.com/mohammad-sayem-uddin/weatherguard",
        "dataset_checksum_sha256": dataset_checksum,
        "total_images": len(records),
        "total_annotated_objects": int(df["object_count"].sum()),
        "provenance_policy": "Owner-curated multi-source dataset retaining explicit source provenance across public, augmented, and real-world captures.",
        "summary_statistics": {
            "source_category_breakdown": df["source_category"].value_counts().to_dict(),
            "weather_condition_breakdown": df["weather_condition"].value_counts().to_dict(),
            "lighting_condition_breakdown": df["lighting_condition"].value_counts().to_dict(),
            "scale_distribution": {
                "ultra_small": int(df["ultra_small_count"].sum()),
                "small": int(df["small_count"].sum()),
                "medium": int(df["medium_count"].sum()),
                "large": int(df["large_count"].sum()),
            },
        },
        "records": records,
    }

    out_path = MANIFESTS_DIR / "weatherguard_v2_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Saved vision V2 manifest to %s (%d records, master hash: %s)",
                out_path, len(records), dataset_checksum[:12])
    return out_path


def generate_audio_v2_manifest() -> Path:
    """Generates audio_v2_manifest.json from audio_corpus_manifest.csv."""
    csv_path = MANIFESTS_DIR / "audio_corpus_manifest.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded audio corpus manifest: %d rows", len(df))

    records = []
    for _, row in df.iterrows():
        records.append({
            "sample_id": row["sample_id"],
            "filename": row["filename"],
            "class_label": row["class_label"],
            "class_id": int(row["class_id"]),
            "transcript": row["transcript"],
            "source_category": row["source_category"],
            "original_source_ref": row["original_source_ref"],
            "noise_condition": row["noise_condition"],
            "snr_db": str(row["snr_db"]),
            "distance_simulation": row["distance_simulation"],
            "pitch_shift_semitones": float(row["pitch_shift_semitones"]),
            "speed_factor": float(row["speed_factor"]),
            "rms_linear": float(row["rms_linear"]),
            "peak_linear": float(row["peak_linear"]),
            "clipping_count": int(row["clipping_count"]),
            "sha256": str(row["sha256_hash"]),
        })

    concat_hashes = "".join(r["sha256"] for r in records)
    dataset_checksum = hashlib.sha256(concat_hashes.encode("utf-8")).hexdigest()

    manifest = {
        "dataset_name": "Rescue Drone & WeatherGuard Expanded Bengali Distress Vocalization Corpus",
        "dataset_version": "v2.0.0",
        "release_date": "2026-11-05",
        "author": "Mohammad Sayem Uddin <sayemwork41@gmail.com>",
        "repository": "https://github.com/mohammad-sayem-uddin/weatherguard",
        "dataset_checksum_sha256": dataset_checksum,
        "total_samples": len(records),
        "target_keyword": "বাঁচাও (bachao)",
        "audio_specification": {
            "sample_rate_hz": 16000,
            "channels": 1,
            "duration_sec": 2.0,
            "total_samples": 32000,
            "bit_depth": 16,
            "target_rms_linear": 0.0600,
            "target_rms_dbfs": -24.44,
            "peak_ceiling_linear": 0.9500,
            "boundary_fade_ms": 20.0,
        },
        "summary_statistics": {
            "positive_count": int((df["class_label"] == "bachao").sum()),
            "negative_count": int((df["class_label"] == "normal_bengali").sum()),
            "source_category_breakdown": df["source_category"].value_counts().to_dict(),
            "noise_condition_breakdown": df["noise_condition"].value_counts().to_dict(),
            "distance_simulation_breakdown": df["distance_simulation"].value_counts().to_dict(),
            "snr_distribution": df["snr_db"].astype(str).value_counts().to_dict(),
            "mean_rms_linear": float(df["rms_linear"].mean()),
            "mean_peak_linear": float(df["peak_linear"].mean()),
            "total_clipped_samples": int(df["clipping_count"].sum()),
            "unique_sha256_count": int(df["sha256_hash"].nunique()),
        },
        "records": records,
    }

    out_path = MANIFESTS_DIR / "audio_v2_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Saved audio V2 manifest to %s (%d records, master hash: %s)",
                out_path, len(records), dataset_checksum[:12])
    return out_path


if __name__ == "__main__":
    generate_vision_v2_manifest()
    generate_audio_v2_manifest()
