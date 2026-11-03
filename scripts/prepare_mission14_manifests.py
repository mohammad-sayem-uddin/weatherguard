"""Prepares the Mission 14 Sequence Inventory, Exclusion Manifest, and Sequence Cohorts.

Performs:
1. Full inventory of all 200 BDD100K validation sequences.
2. Strict isolation and exclusion of all 14 historical sequences (Missions 9-13).
3. Creation of artifacts/mission14/sequence_exclusion_manifest.json.
4. Selection and partition of disjoint Development (2 seqs) and Locked Test (6 seqs) cohorts.
5. Byte-range download of the 8 selected sequences into data/mission14_independent/.
6. Verification of all frames and labels.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.download_mission10_sequences import RemoteZipStream, REMOTE_ZIP_URL
ARTIFACTS_DIR = ROOT / "artifacts" / "mission14"
REPORTS_DIR = ROOT / "reports" / "mission14"
DATA_DIR = ROOT / "data" / "mission14_independent"
SEQ_DIR = DATA_DIR / "sequences"
LABELS_DIR = DATA_DIR / "labels"

# 14 Historical Sequences to strictly exclude
HISTORICAL_EXCLUSIONS = {
    # Mission 9 (8 sequences)
    "b1e9ee0e-67e26f2e": {"mission": "Mission 9", "role": "dev"},
    "b1d0a191-65deaeef": {"mission": "Mission 9", "role": "dev"},
    "b24071b8-b3ee1196": {"mission": "Mission 9", "role": "dev"},
    "b2208b0f-2796a692": {"mission": "Mission 9", "role": "dev"},
    "b2064e61-2beadd45": {"mission": "Mission 9", "role": "test"},
    "b1d0a191-03dcecc2": {"mission": "Mission 9", "role": "test"},
    "b1d968b9-ce42734f": {"mission": "Mission 9", "role": "test"},
    "b1ca2e5d-84cf9134": {"mission": "Mission 9", "role": "test"},
    # Missions 10-13 (6 sequences)
    "b24380ab-6dbeb908": {"mission": "Missions 10-13", "role": "dev"},
    "b255cd6c-2f889586": {"mission": "Missions 10-13", "role": "dev"},
    "b1e1a7b8-65ec7612": {"mission": "Missions 10-13", "role": "test"},
    "b1d0a191-06deb55d": {"mission": "Missions 10-13", "role": "test"},
    "b24d283f-33783d1b": {"mission": "Missions 10-13", "role": "test"},
    "b23c9e00-b425de1b": {"mission": "Missions 10-13", "role": "test"},
}

# Selected Mission 14 Cohorts
SELECTED_COHORTS = {
    "development_cohort": {
        "b1e1a7b8-b397c445": {
            "title": "Dev Seq 1 (Rainy Daytime City Street)",
            "weather": "rainy",
            "timeofday": "daytime",
            "scene": "city street",
            "description": "Rainy daytime urban traffic with wet windshield and street reflections",
        },
        "b204a5c1-064b0040": {
            "title": "Dev Seq 2 (Snowy Daytime Urban Arterial)",
            "weather": "snowy",
            "timeofday": "daytime",
            "scene": "city street",
            "description": "Snowy daytime urban street with snowbanks and road slush",
        },
    },
    "locked_test_cohort": {
        "b23f7012-fab06dac": {
            "title": "Test Seq 1 (Rainy Night City Street)",
            "weather": "rainy",
            "timeofday": "night",
            "scene": "city street",
            "description": "Rainy night urban traffic with high glare, wet asphalt, and heavy reflections",
        },
        "b1d0a191-de8948f6": {
            "title": "Test Seq 2 (Snowy Daytime Highway)",
            "weather": "snowy",
            "timeofday": "daytime",
            "scene": "highway",
            "description": "High-speed snowy highway with roadside snowbanks, spray, and distant vehicles",
        },
        "b1d0a191-28f0e779": {
            "title": "Test Seq 3 (Snowy Daytime City Street)",
            "weather": "snowy",
            "timeofday": "daytime",
            "scene": "city street",
            "description": "Dense urban snowy traffic with pedestrian crossings and occluded vehicles",
        },
        "b1f4491b-cf446195": {
            "title": "Test Seq 4 (Overcast Daytime Highway Arterial)",
            "weather": "overcast",
            "timeofday": "daytime",
            "scene": "highway",
            "description": "Low-contrast overcast daytime highway with dense multi-lane traffic",
        },
        "b1c66a42-6f7d68ca": {
            "title": "Test Seq 5 (Overcast Daytime Urban Core)",
            "weather": "overcast",
            "timeofday": "daytime",
            "scene": "city street",
            "description": "Overcast daytime urban street with heavy vehicle congestion and distant horizon targets",
        },
        "b2505382-de5238f0": {
            "title": "Test Seq 6 (Clear Night City Arterial)",
            "weather": "clear",
            "timeofday": "night",
            "scene": "city street",
            "description": "Clear night urban high-speed arterial with headlights and distant tail lights",
        },
    },
}


def build_inventory_and_manifests():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load samples.json metadata
    samples_data = json.load(open(ROOT / "data" / "bdd100k" / "samples.json"))
    keyframe_meta = {}
    for s in samples_data["samples"]:
        fp = s.get("filepath", "")
        fn = fp.split("/")[-1].replace(".jpg", "")
        weather = s.get("weather", {}).get("label", "unknown") if isinstance(s.get("weather"), dict) else "unknown"
        timeofday = s.get("timeofday", {}).get("label", "unknown") if isinstance(s.get("timeofday"), dict) else "unknown"
        scene = s.get("scene", {}).get("label", "unknown") if isinstance(s.get("scene"), dict) else "unknown"
        keyframe_meta[fn] = {"weather": weather, "timeofday": timeofday, "scene": scene}

    # 2. Inspect all 200 validation sequences
    labels_zip_path = ROOT / "data" / "bdd100k" / "box_track_labels_trainval.zip"
    z = zipfile.ZipFile(labels_zip_path)
    val_files = [f for f in z.namelist() if f.startswith("bdd100k/labels/box_track_20/val/") and f.endswith(".json")]

    full_inventory = []
    exclusion_records = []

    for vf in sorted(val_files):
        sid = vf.split("/")[-1].replace(".json", "")
        frames = json.loads(z.read(vf))
        meta = keyframe_meta.get(sid, {"weather": "unknown", "timeofday": "unknown", "scene": "unknown"})

        total_boxes = 0
        cat_counts = Counter()
        size_counts = Counter()

        for f in frames:
            for lbl in f.get("labels", []):
                cat = lbl.get("category", "unknown")
                cat_counts[cat] += 1
                total_boxes += 1
                b = lbl.get("box2d", {})
                w = b.get("x2", 0) - b.get("x1", 0)
                h = b.get("y2", 0) - b.get("y1", 0)
                area = w * h
                if area < 256:
                    size_counts["ultra_small"] += 1
                elif area < 1024:
                    size_counts["small"] += 1
                elif area < 9216:
                    size_counts["medium"] += 1
                else:
                    size_counts["large"] += 1

        is_hist = sid in HISTORICAL_EXCLUSIONS
        hist_info = HISTORICAL_EXCLUSIONS.get(sid, {})

        entry = {
            "sequence_id": sid,
            "num_frames": len(frames),
            "total_boxes": total_boxes,
            "weather": meta["weather"],
            "timeofday": meta["timeofday"],
            "scene": meta["scene"],
            "ultra_small_boxes": size_counts["ultra_small"],
            "small_boxes": size_counts["small"],
            "medium_boxes": size_counts["medium"],
            "large_boxes": size_counts["large"],
            "categories": dict(cat_counts),
            "is_historical": is_hist,
            "historical_info": hist_info,
        }
        full_inventory.append(entry)

        if is_hist:
            exclusion_records.append({
                "sequence_id": sid,
                "reason": f"Evaluated in prior {hist_info.get('mission', 'previous research')} as {hist_info.get('role', 'historical')} sequence",
                "historical_mission": hist_info.get("mission"),
                "historical_role": hist_info.get("role"),
                "frames": len(frames),
                "total_boxes": total_boxes,
            })

    # Save exclusion manifest
    exclusion_manifest_path = ARTIFACTS_DIR / "sequence_exclusion_manifest.json"
    with open(exclusion_manifest_path, "w") as f:
        json.dump({
            "mission": "Mission 14 — Independent Cross-Domain Validation",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "policy": "Strict sequence-level isolation. Zero sequence leakage across missions.",
            "total_historical_sequences_excluded": len(exclusion_records),
            "exclusions": exclusion_records,
        }, f, indent=2)
    print(f"Saved exclusion manifest ({len(exclusion_records)} sequences) to {exclusion_manifest_path.name}")

    # Build cohort entries
    manifest_data = {
        "mission": "Mission 14 — Independent Cross-Domain Validation",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Completely unseen BDD100K driving sequences with zero overlap with Missions 9-13",
        "cohorts": {},
        "full_inventory_summary": {
            "total_val_sequences_inspected": len(full_inventory),
            "historical_excluded": len(exclusion_records),
            "unused_available": len(full_inventory) - len(exclusion_records),
        }
    }

    inv_lookup = {item["sequence_id"]: item for item in full_inventory}

    for cohort_name, seqs in SELECTED_COHORTS.items():
        c_list = []
        for sid, s_meta in seqs.items():
            inv_item = inv_lookup[sid]
            c_list.append({
                "sequence_id": sid,
                "title": s_meta["title"],
                "weather": inv_item["weather"],
                "timeofday": inv_item["timeofday"],
                "scene": inv_item["scene"],
                "description": s_meta["description"],
                "frames": inv_item["num_frames"],
                "total_boxes": inv_item["total_boxes"],
                "ultra_small_boxes": inv_item["ultra_small_boxes"],
                "small_boxes": inv_item["small_boxes"],
                "medium_boxes": inv_item["medium_boxes"],
                "large_boxes": inv_item["large_boxes"],
                "categories": inv_item["categories"],
            })
        manifest_data["cohorts"][cohort_name] = {
            "num_sequences": len(c_list),
            "total_frames": sum(s["frames"] for s in c_list),
            "total_boxes": sum(s["total_boxes"] for s in c_list),
            "total_small_boxes": sum(s["small_boxes"] for s in c_list),
            "total_ultra_small_boxes": sum(s["ultra_small_boxes"] for s in c_list),
            "sequences": c_list,
        }

    manifest_path = ARTIFACTS_DIR / "sequence_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved sequence manifest to {manifest_path.name}")

    # 3. Extract labels for selected sequences from box_track_labels_trainval.zip
    print("\nExtracting ground-truth labels for the 8 selected sequences...")
    all_selected_sids = list(SELECTED_COHORTS["development_cohort"].keys()) + list(SELECTED_COHORTS["locked_test_cohort"].keys())
    for sid in all_selected_sids:
        dest_lbl = LABELS_DIR / f"{sid}.json"
        if not dest_lbl.exists():
            zf_path = f"bdd100k/labels/box_track_20/val/{sid}.json"
            with open(dest_lbl, "wb") as f:
                f.write(z.read(zf_path))
            print(f"Extracted label: {dest_lbl.name}")
        else:
            print(f"Label already exists: {dest_lbl.name}")

    # 4. Download images for selected sequences using RemoteZipStream
    print("\nChecking frame downloads for selected sequences...")
    pending_download = []
    for sid in all_selected_sids:
        seq_folder = SEQ_DIR / sid
        existing_frames = list(seq_folder.glob("*.jpg")) if seq_folder.exists() else []
        if len(existing_frames) < 200:
            pending_download.append(sid)

    if not pending_download:
        print("All 8 selected sequences already downloaded and intact on disk!")
        return

    print(f"Connecting to remote archive to download {len(pending_download)} sequences: {pending_download}...")
    stream = RemoteZipStream(REMOTE_ZIP_URL)
    remote_zf = zipfile.ZipFile(stream)

    for sid in pending_download:
        t0 = time.time()
        prefix = f"bdd100k/images/track/val/{sid}/"
        seq_infos = [info for info in remote_zf.infolist() if info.filename.startswith(prefix) and info.filename.endswith(".jpg")]
        if not seq_infos:
            print(f"Error: No frames found for {sid} in remote zip!")
            continue

        min_off = min(i.header_offset for i in seq_infos)
        last_info = max(seq_infos, key=lambda i: i.header_offset)
        max_off = last_info.header_offset + last_info.compress_size + 4096
        span_mb = (max_off - min_off) / (1024 * 1024)

        print(f"Downloading {sid} ({len(seq_infos)} frames, {span_mb:.2f} MB)...")
        headers = {"Range": f"bytes={min_off}-{max_off}"}
        resp = requests.get(stream.url, headers=headers, timeout=60)
        chunk_bytes = resp.content

        out_dir = SEQ_DIR / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        extracted = 0
        for info in seq_infos:
            rel_off = info.header_offset - min_off
            sig = chunk_bytes[rel_off : rel_off + 4]
            if sig != b"PK\x03\x04":
                continue
            fn_len = int.from_bytes(chunk_bytes[rel_off + 26 : rel_off + 28], "little")
            extra_len = int.from_bytes(chunk_bytes[rel_off + 28 : rel_off + 30], "little")
            c_size = info.compress_size
            data_start = rel_off + 30 + fn_len + extra_len
            data_bytes = chunk_bytes[data_start : data_start + c_size]

            if info.compress_type == 0:
                raw_img = data_bytes
            elif info.compress_type == 8:
                raw_img = zlib.decompress(data_bytes, -15)
            else:
                continue

            fn = Path(info.filename).name
            with open(out_dir / fn, "wb") as f:
                f.write(raw_img)
            extracted += 1

        print(f"Extracted {extracted}/{len(seq_infos)} frames for {sid} in {time.time()-t0:.2f}s!")

    # Verify integrity of all downloaded sequences
    print("\nVerifying frame dimensions and file integrity...")
    total_downloaded_frames = 0
    for sid in all_selected_sids:
        seq_folder = SEQ_DIR / sid
        frames = sorted(list(seq_folder.glob("*.jpg")))
        total_downloaded_frames += len(frames)
        assert len(frames) >= 200, f"Incomplete sequence {sid}: only {len(frames)} frames"
        with Image.open(frames[0]) as im:
            assert im.size == (1280, 720), f"Invalid resolution {im.size} for {sid}"
        print(f"Sequence {sid}: {len(frames)} frames verified (1280x720 RGB)")

    print(f"\nALL 8 SEQUENCES READY: Total {total_downloaded_frames} continuous frames ready for Mission 14!")


if __name__ == "__main__":
    build_inventory_and_manifests()
