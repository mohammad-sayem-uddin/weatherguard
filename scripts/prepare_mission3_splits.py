"""Prepares and validates the Mission 3 development split and split manifest.

Samples a disjoint 160-image BDD100K development cohort from the remaining
validation pool, downloads image assets, performs QA checks, and generates
the split manifest and development summary artifacts.
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.external_dataset import (
    BDD100K_TO_CANONICAL,
    CANONICAL_CLASSES,
    Bdd100kDataset,
    ExternalRecord,
)

ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data" / "bdd100k"
IMAGES_DIR = DATA_DIR / "images"
SAMPLES_JSON = DATA_DIR / "samples.json"
LOCKED_TEST_SUMMARY = ARTIFACTS / "external_dataset_summary.json"


def download_image(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read()
                if len(content) > 0:
                    with open(dest_path, "wb") as f:
                        f.write(content)
                    return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to download {url}: {e}")
                return False
            time.sleep(1.0)
    return False


def main() -> None:
    print("=" * 70)
    print("WEATHERGUARD MISSION 3 — SPLIT PREPARATION & MANIFEST GENERATION")
    print("=" * 70)

    # 1. Load locked test cohort
    with open(LOCKED_TEST_SUMMARY) as f:
        locked_test_data = json.load(f)
    locked_test_ids = set(locked_test_data["cohort_image_ids"])
    print(f"Loaded Group C: Locked Test Cohort ({len(locked_test_ids)} images).")

    # 2. Load all available BDD100K validation samples
    print(f"Loading candidate pool from {SAMPLES_JSON}...")
    full_dataset = Bdd100kDataset.from_fiftyone_json(SAMPLES_JSON, images_dir=IMAGES_DIR)

    # 3. Filter candidate pool to disjoint daytime adverse & control images
    candidate_records = [
        r for r in full_dataset
        if r.image_id not in locked_test_ids and r.timeofday == "daytime"
    ]
    print(f"Disjoint daytime candidates: {len(candidate_records)} images.")

    # 4. Deterministic Stratified Sampling for Development Cohort (Group B)
    dev_strata_targets = {
        "rainy": 50,
        "snowy": 50,
        "overcast": 30,
        "clear": 30,
    }
    rng = random.Random(101)  # Distinct seed for dev sampling
    dev_records: list[ExternalRecord] = []

    for cond, target_cnt in dev_strata_targets.items():
        cond_candidates = [r for r in candidate_records if r.condition == cond]
        cond_candidates.sort(key=lambda r: r.image_id)
        assert len(cond_candidates) >= target_cnt, f"Insufficient candidates for {cond}"
        sampled_indices = sorted(rng.sample(range(len(cond_candidates)), target_cnt))
        chosen = [cond_candidates[i] for i in sampled_indices]
        dev_records.extend(chosen)

    dev_image_ids = [r.image_id for r in dev_records]
    print(f"Selected Group B: Development Cohort ({len(dev_image_ids)} images).")

    # 5. Strict Disjointness Verification
    overlap = set(dev_image_ids).intersection(locked_test_ids)
    assert len(overlap) == 0, f"FATAL: Overlap detected between dev and test: {overlap}"
    print("Disjointness check PASSED: Zero overlap between Group B (Dev) and Group C (Test).")

    # 6. Download Dev Images
    print("Downloading Group B image assets...")
    download_successes = 0
    total = len(dev_records)
    for i, rec in enumerate(dev_records, 1):
        if rec.image_url and rec.image_path:
            success = download_image(rec.image_url, rec.image_path)
            if success:
                download_successes += 1
            print(f"Downloading [{i}/{total}] {rec.image_id}... ({download_successes}/{i} ok)", end="\r")
    print(f"\nCompleted Group B downloads: {download_successes}/{total} successful.")

    # 7. Compute Dev Statistics
    target_box_counts = Counter()
    scale_counts = {"small": 0, "medium": 0, "large": 0}
    empty_images = 0

    for rec in dev_records:
        if len(rec.objects) == 0:
            empty_images += 1
        for b in rec.objects:
            target_box_counts[b.class_name] += 1
            area = b.width * b.height
            if area < 1024:
                scale_counts["small"] += 1
            elif area < 9216:
                scale_counts["medium"] += 1
            else:
                scale_counts["large"] += 1

    total_target_boxes = sum(target_box_counts.values())

    # 8. Save Development Summary Artifact
    dev_summary = {
        "dataset_name": "BDD100K Development Cohort (Group B)",
        "provenance": "Official BDD100K validation split (disjoint from locked test cohort)",
        "sample_size": len(dev_records),
        "resolution": "1280x720 (uniform)",
        "sampling_seed": 101,
        "weather_breakdown": dict(Counter(r.condition for r in dev_records)),
        "timeofday": "daytime",
        "target_class_counts": dict(target_box_counts),
        "total_target_boxes": total_target_boxes,
        "average_target_boxes_per_image": float(total_target_boxes / len(dev_records)),
        "scale_distribution": {
            "counts": scale_counts,
            "percentages": {
                k: float(v / total_target_boxes * 100) if total_target_boxes else 0.0
                for k, v in scale_counts.items()
            },
        },
        "empty_images_count": empty_images,
        "dev_image_ids": dev_image_ids,
    }
    dev_summary_path = ARTIFACTS / "mission3_development_summary.json"
    with open(dev_summary_path, "w") as f:
        json.dump(dev_summary, f, indent=2)
    print(f"Saved {dev_summary_path}")

    # 9. Save Split Manifest Artifact
    manifest = {
        "mission": "WeatherGuard Mission 3",
        "groups": {
            "Group_A_DAWN_Development": {
                "dataset": "DAWN",
                "source": "train-00000-of-00001.parquet",
                "n_images": 821,
                "role": "Historical development split for initial threshold calibration (0.60 FRCNN, 0.20 YOLO)",
            },
            "Group_B_BDD100K_Development": {
                "dataset": "BDD100K",
                "source": "Official BDD100K validation split (sampled with seed 101)",
                "n_images": len(dev_image_ids),
                "weather_breakdown": dev_summary["weather_breakdown"],
                "role": "Active development split for selecting scale-aware inference parameters, tiling ratios, and adaptive thresholds",
                "image_ids": dev_image_ids,
            },
            "Group_C_BDD100K_Locked_Test": {
                "dataset": "BDD100K",
                "source": "Official BDD100K validation split (sampled in Mission 2 with seed 42)",
                "n_images": len(locked_test_ids),
                "weather_breakdown": locked_test_data["weather_breakdown"],
                "role": "LOCKED external evaluation cohort for final unbiased testing of selected methods",
                "image_ids": sorted(list(locked_test_ids)),
            },
        },
        "integrity_checks": {
            "disjoint_B_and_C": len(overlap) == 0,
            "overlap_count": len(overlap),
            "locked_test_intact": len(locked_test_ids) == 203,
        },
    }
    manifest_path = ARTIFACTS / "mission3_split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
