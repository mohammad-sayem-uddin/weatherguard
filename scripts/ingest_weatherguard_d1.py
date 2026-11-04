"""WeatherGuard-D1 Dataset Ingestion and Image Integrity Pipeline.

Module: scripts/ingest_weatherguard_d1.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
Mission: Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

Responsibilities:
1. Discover incoming raw synthetic dashcam images from Google Flow.
2. Verify image file integrity (no corruption/truncation).
3. Validate exact image dimensions (1280x720). Never silently resize or crop.
4. Calculate cryptographic SHA-256 and perceptual difference hashes (dHash).
5. Detect exact duplicate images and flag near-duplicate clusters.
6. Record complete metadata into structured manifests.
7. Produce an ingestion report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG_PATH = ROOT / "configs" / "weatherguard_d1.yaml"


@dataclass
class ImageRecord:
    image_id: str
    relative_path: str
    condition: str
    width: int
    height: int
    format: str
    sha256: str
    dhash: str
    status: str  # "ACCEPTED", "DIMENSION_MISMATCH", "CORRUPTED", "EXACT_DUPLICATE", "NEAR_DUPLICATE_FLAGGED"
    flags: List[str]
    scenario_family_id: str
    prompt_seed: Optional[int] = None
    original_source_file: Optional[str] = None


def compute_sha256(file_path: Path) -> str:
    """Calculates cryptographic SHA-256 digest of a file in 64KB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Calculates perceptual difference hash (dHash) for near-duplicate detection.

    Steps:
    1. Convert to grayscale.
    2. Resize to (hash_size + 1, hash_size) using bilinear interpolation.
    3. Compare adjacent pixels horizontally to form a binary bit array.
    4. Format bit array as a hexadecimal string.
    """
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(gray, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Pack boolean bits into integer
    bit_string = "".join(["1" if b else "0" for b in diff.flatten()])
    int_val = int(bit_string, 2)
    return f"{int_val:0{hash_size * hash_size // 4}x}"


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Computes Hamming distance between two hex-encoded hashes."""
    n1 = int(hex_hash1, 16)
    n2 = int(hex_hash2, 16)
    return bin(n1 ^ n2).count("1")


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Loads and validates central dataset YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_d1_directory_structure(base_dir: Path) -> Dict[str, Path]:
    """Initializes standard maintainable dataset directory tree."""
    dirs = {
        "base": base_dir,
        "images": base_dir / "images",
        "day": base_dir / "images" / "day",
        "night": base_dir / "images" / "night",
        "rain": base_dir / "images" / "rain",
        "fog": base_dir / "images" / "fog",
        "annotations": base_dir / "annotations",
        "metadata": base_dir / "metadata",
        "manifests": base_dir / "manifests",
        "quality_control": base_dir / "quality_control",
        "reports": base_dir / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


class WeatherGuardD1IngestionEngine:
    """Engine for scanning, validating, hashing, and cataloging incoming D1 images."""

    def __init__(self, config: Dict[str, Any], target_dir: Path):
        self.config = config
        self.target_dir = target_dir
        self.dirs = setup_d1_directory_structure(target_dir)

        # Config properties
        img_props = config.get("image_properties", {})
        self.target_width = img_props.get("target_width", 1280)
        self.target_height = img_props.get("target_height", 720)
        self.allowed_formats = [ext.lower() for ext in img_props.get("allowed_formats", [".jpg", ".jpeg", ".png"])]

        qc = config.get("quality_control", {})
        self.hamming_thresh = qc.get("near_duplicate_hamming_threshold", 4)
        self.dhash_size = qc.get("dhash_size", 8)

        # Internal tracking
        self.seen_sha256: Dict[str, str] = {}  # sha256 -> image_id
        self.dhash_index: List[Tuple[str, str]] = []  # [(dhash_hex, image_id), ...]
        self.records: List[ImageRecord] = []

    def ingest_single_image(
        self,
        src_path: Path,
        inferred_condition: str,
        scenario_family_id: str = "generic_scenario",
        prompt_seed: Optional[int] = None,
        copy_file: bool = False,
    ) -> ImageRecord:
        """Inspects, hashes, and validates a single incoming image."""
        flags: List[str] = []
        status = "ACCEPTED"
        image_id = src_path.stem

        # Step 1: Check extension
        ext = src_path.suffix.lower()
        if ext not in self.allowed_formats:
            status = "FORMAT_REJECTED"
            flags.append(f"Disallowed extension: {ext}")
            rec = ImageRecord(
                image_id=image_id,
                relative_path=str(src_path),
                condition=inferred_condition,
                width=0,
                height=0,
                format=ext,
                sha256="",
                dhash="",
                status=status,
                flags=flags,
                scenario_family_id=scenario_family_id,
                prompt_seed=prompt_seed,
                original_source_file=str(src_path),
            )
            self.records.append(rec)
            return rec

        # Step 2: Cryptographic SHA-256
        sha256_val = compute_sha256(src_path)
        if sha256_val in self.seen_sha256:
            existing_id = self.seen_sha256[sha256_val]
            status = "EXACT_DUPLICATE"
            flags.append(f"Identical byte hash to existing image: {existing_id}")

        # Step 3: Image opening & dimension verification
        try:
            with Image.open(src_path) as im:
                im.verify()  # detects truncation/corruption
            # Re-open for decoding dimensions and dHash
            with Image.open(src_path) as im:
                w, h = im.size
                img_fmt = (im.format or ext.replace(".", "")).upper()
                dhash_val = compute_dhash(im, hash_size=self.dhash_size)
        except Exception as e:
            status = "CORRUPTED"
            flags.append(f"Image load failure: {type(e).__name__}: {str(e)}")
            rec = ImageRecord(
                image_id=image_id,
                relative_path=str(src_path),
                condition=inferred_condition,
                width=0,
                height=0,
                format="UNKNOWN",
                sha256=sha256_val,
                dhash="",
                status=status,
                flags=flags,
                scenario_family_id=scenario_family_id,
                prompt_seed=prompt_seed,
                original_source_file=str(src_path),
            )
            self.records.append(rec)
            return rec

        # Step 4: Dimension validation (strict: no silent resizing)
        if w != self.target_width or h != self.target_height:
            if status == "ACCEPTED":
                status = "DIMENSION_MISMATCH"
            flags.append(f"Dimensions {w}x{h} != expected {self.target_width}x{self.target_height}")

        # Step 5: Near-duplicate check via dHash
        near_dups: List[str] = []
        for existing_dhash, existing_id in self.dhash_index:
            dist = hamming_distance(dhash_val, existing_dhash)
            if dist <= self.hamming_thresh:
                near_dups.append(f"{existing_id}(dist={dist})")
        if near_dups:
            flags.append(f"Near-duplicate candidate of: {', '.join(near_dups)}")
            if status == "ACCEPTED":
                status = "NEAR_DUPLICATE_FLAGGED"

        # Register accepted hashes
        if status in ("ACCEPTED", "NEAR_DUPLICATE_FLAGGED"):
            self.seen_sha256[sha256_val] = image_id
            self.dhash_index.append((dhash_val, image_id))

        # Destination placement
        dest_rel = f"images/{inferred_condition}/{src_path.name}"
        if copy_file and status in ("ACCEPTED", "NEAR_DUPLICATE_FLAGGED"):
            dest_abs = self.target_dir / dest_rel
            if not dest_abs.exists():
                shutil.copy2(src_path, dest_abs)

        record = ImageRecord(
            image_id=image_id,
            relative_path=dest_rel,
            condition=inferred_condition,
            width=w,
            height=h,
            format=img_fmt,
            sha256=sha256_val,
            dhash=dhash_val,
            status=status,
            flags=flags,
            scenario_family_id=scenario_family_id,
            prompt_seed=prompt_seed,
            original_source_file=str(src_path),
        )
        self.records.append(record)
        return record

    def ingest_directory(
        self,
        raw_source_dir: Path,
        manifest_file: Optional[Path] = None,
        copy_files: bool = False,
    ) -> Dict[str, Any]:
        """Discovers and ingests all images from an incoming directory."""
        if not raw_source_dir.exists():
            return {
                "status": "NOT_AVAILABLE",
                "message": f"Source directory does not exist: {raw_source_dir}",
                "records_processed": 0,
            }

        # Optional manifest mapping
        meta_by_filename: Dict[str, Dict[str, Any]] = {}
        if manifest_file and manifest_file.exists():
            try:
                with open(manifest_file, "r") as mf:
                    raw_meta = json.load(mf)
                    if isinstance(raw_meta, list):
                        for item in raw_meta:
                            fname = item.get("filename") or item.get("image_id")
                            if fname:
                                meta_by_filename[fname] = item
                    elif isinstance(raw_meta, dict):
                        meta_by_filename = raw_meta
            except Exception as e:
                print(f"Warning: Failed to load manifest {manifest_file}: {e}")

        # Scan for images
        discovered_files = []
        for ext in self.allowed_formats:
            discovered_files.extend(list(raw_source_dir.rglob(f"*{ext}")))
            discovered_files.extend(list(raw_source_dir.rglob(f"*{ext.upper()}")))
        discovered_files = sorted(list(set(discovered_files)))

        if not discovered_files:
            return {
                "status": "EMPTY_SOURCE",
                "message": f"No image files matching {self.allowed_formats} found in {raw_source_dir}",
                "records_processed": 0,
            }

        valid_conditions = set(self.config.get("conditions", {}).keys())

        for file_path in discovered_files:
            # Infer condition
            condition = "day"  # fallback default
            parent_name = file_path.parent.name.lower()
            stem_lower = file_path.stem.lower()
            for c in valid_conditions:
                if c in parent_name or stem_lower.startswith(c) or f"_{c}_" in stem_lower:
                    condition = c
                    break

            meta = meta_by_filename.get(file_path.name, meta_by_filename.get(file_path.stem, {}))
            if "condition" in meta and meta["condition"] in valid_conditions:
                condition = meta["condition"]

            fam_id = meta.get("scenario_family_id", meta.get("prompt_id", f"{condition}_batch_01"))
            seed = meta.get("seed", meta.get("prompt_seed"))

            self.ingest_single_image(
                src_path=file_path,
                inferred_condition=condition,
                scenario_family_id=fam_id,
                prompt_seed=seed,
                copy_file=copy_files,
            )

        return self.generate_reports()

    def generate_reports(self) -> Dict[str, Any]:
        """Summarizes ingestion results and writes JSON manifest and markdown report."""
        summary = {
            "total_images_processed": len(self.records),
            "status_counts": {},
            "condition_counts": {},
            "dimension_mismatch_count": 0,
            "corrupted_count": 0,
            "exact_duplicate_count": 0,
            "near_duplicate_count": 0,
            "accepted_count": 0,
        }

        for r in self.records:
            summary["status_counts"][r.status] = summary["status_counts"].get(r.status, 0) + 1
            if r.status in ("ACCEPTED", "NEAR_DUPLICATE_FLAGGED"):
                summary["condition_counts"][r.condition] = summary["condition_counts"].get(r.condition, 0) + 1

            if r.status == "ACCEPTED":
                summary["accepted_count"] += 1
            elif r.status == "DIMENSION_MISMATCH":
                summary["dimension_mismatch_count"] += 1
            elif r.status == "CORRUPTED":
                summary["corrupted_count"] += 1
            elif r.status == "EXACT_DUPLICATE":
                summary["exact_duplicate_count"] += 1
            elif r.status == "NEAR_DUPLICATE_FLAGGED":
                summary["near_duplicate_count"] += 1

        # Write machine JSON manifest
        manifest_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_name": self.config.get("dataset_name", "WeatherGuard-D1"),
            "dataset_version": self.config.get("dataset_version", "1.0.0-synthetic"),
            "target_resolution": f"{self.target_width}x{self.target_height}",
            "summary": summary,
            "records": [asdict(r) for r in self.records],
        }

        manifest_path = self.dirs["manifests"] / "ingestion_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_payload, f, indent=2)

        # Write markdown report
        report_path = self.dirs["reports"] / "ingestion_report.md"
        report_lines = [
            "# WeatherGuard-D1 Image Ingestion & Integrity Report",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
            f"**Target Resolution:** {self.target_width} × {self.target_height} (Strict, no silent resizing)  ",
            "",
            "## 1. Ingestion Summary",
            f"- **Total Discovered:** {summary['total_images_processed']}",
            f"- **Accepted:** {summary['accepted_count']}",
            f"- **Near-Duplicate Suspects (Flagged):** {summary['near_duplicate_count']}",
            f"- **Exact Duplicates (Rejected):** {summary['exact_duplicate_count']}",
            f"- **Dimension Mismatches (Rejected):** {summary['dimension_mismatch_count']}",
            f"- **Corrupted / Invalid Files:** {summary['corrupted_count']}",
            "",
            "## 2. Condition Breakdown (Accepted & Flagged)",
        ]
        for cond, count in summary["condition_counts"].items():
            report_lines.append(f"- **{cond.title()}:** {count}")

        if summary["dimension_mismatch_count"] > 0 or summary["corrupted_count"] > 0:
            report_lines.extend([
                "",
                "## 3. Discrepancies & Review Flags",
            ])
            for r in self.records:
                if r.status in ("DIMENSION_MISMATCH", "CORRUPTED", "EXACT_DUPLICATE"):
                    report_lines.append(f"- `{r.image_id}` [{r.status}]: {'; '.join(r.flags)}")

        with open(report_path, "w") as f:
            f.write("\n".join(report_lines) + "\n")

        return manifest_payload


def main():
    parser = argparse.ArgumentParser(description="Ingest WeatherGuard-D1 images with integrity and duplicate verification.")
    parser.add_argument("--source-dir", type=str, default="data/WeatherGuard-D1/raw_incoming", help="Path to incoming images")
    parser.add_argument("--target-dir", type=str, default="data/WeatherGuard-D1", help="Path to dataset root")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="YAML configuration path")
    parser.add_argument("--manifest", type=str, default=None, help="Optional generation metadata manifest")
    parser.add_argument("--copy", action="store_true", help="Copy accepted files into structured condition folders")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    engine = WeatherGuardD1IngestionEngine(config=cfg, target_dir=Path(args.target_dir))
    res = engine.ingest_directory(
        raw_source_dir=Path(args.source_dir),
        manifest_file=Path(args.manifest) if args.manifest else None,
        copy_files=args.copy,
    )
    print(f"Ingestion finished: {res.get('summary', {}).get('accepted_count', 0)} accepted out of {res.get('records_processed', len(engine.records))}.")


if __name__ == "__main__":
    main()
