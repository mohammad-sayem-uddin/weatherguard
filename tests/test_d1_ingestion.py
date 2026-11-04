"""Unit tests for WeatherGuard-D1 image ingestion and integrity verification.

Module: tests/test_d1_ingestion.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from scripts.ingest_weatherguard_d1 import (
    WeatherGuardD1IngestionEngine,
    compute_dhash,
    compute_sha256,
    hamming_distance,
    load_config,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "weatherguard_d1.yaml"


@pytest.fixture
def temp_workspace():
    tmp_dir = Path(tempfile.mkdtemp(prefix="test_d1_ingestion_"))
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_d1_config_loading():
    assert CONFIG_PATH.exists(), "Central config file configs/weatherguard_d1.yaml must exist."
    cfg = load_config(CONFIG_PATH)
    assert cfg["dataset_name"] == "WeatherGuard-D1"
    assert cfg["image_properties"]["target_width"] == 1280
    assert cfg["image_properties"]["target_height"] == 720
    assert len(cfg["classes"]) == 6
    assert cfg["classes"][0] == "truck"
    assert cfg["classes"][3] == "car"
    assert "day" in cfg["conditions"]
    assert "fog" in cfg["conditions"]


def test_ingest_valid_fixture(temp_workspace):
    cfg = load_config(CONFIG_PATH)
    source_dir = temp_workspace / "incoming"
    target_dir = temp_workspace / "d1_dataset"
    source_dir.mkdir(parents=True)

    # Create synthetic test fixture (1280x720)
    arr = np.zeros((720, 1280, 3), dtype=np.uint8)
    arr[200:300, 400:600] = [200, 50, 50]  # distinct rectangle
    im = Image.fromarray(arr)
    img_file = source_dir / "day_scene_001.jpg"
    im.save(img_file, "JPEG")

    engine = WeatherGuardD1IngestionEngine(config=cfg, target_dir=target_dir)
    rec = engine.ingest_single_image(img_file, inferred_condition="day", copy_file=True)

    assert rec.status == "ACCEPTED"
    assert rec.width == 1280
    assert rec.height == 720
    assert rec.sha256 == compute_sha256(img_file)
    assert len(rec.dhash) == 16  # 64-bit hex
    assert (target_dir / rec.relative_path).exists()


def test_ingest_dimension_mismatch(temp_workspace):
    cfg = load_config(CONFIG_PATH)
    source_dir = temp_workspace / "incoming"
    target_dir = temp_workspace / "d1_dataset"
    source_dir.mkdir(parents=True)

    # Create wrong resolution image (e.g. 640x480)
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    im = Image.fromarray(arr)
    bad_img = source_dir / "bad_res.jpg"
    im.save(bad_img, "JPEG")

    engine = WeatherGuardD1IngestionEngine(config=cfg, target_dir=target_dir)
    rec = engine.ingest_single_image(bad_img, inferred_condition="rain", copy_file=True)

    assert rec.status == "DIMENSION_MISMATCH"
    assert "Dimensions 640x480 != expected 1280x720" in rec.flags[0]
    # Verify file was NOT copied to accepted images
    assert not (target_dir / rec.relative_path).exists()


def test_exact_duplicate_detection(temp_workspace):
    cfg = load_config(CONFIG_PATH)
    source_dir = temp_workspace / "incoming"
    target_dir = temp_workspace / "d1_dataset"
    source_dir.mkdir(parents=True)

    arr = np.zeros((720, 1280, 3), dtype=np.uint8)
    arr[100:150, 100:150] = 255
    im = Image.fromarray(arr)
    f1 = source_dir / "orig.jpg"
    f2 = source_dir / "copy.jpg"
    im.save(f1, "JPEG")
    im.save(f2, "JPEG")  # identical bytes

    engine = WeatherGuardD1IngestionEngine(config=cfg, target_dir=target_dir)
    r1 = engine.ingest_single_image(f1, inferred_condition="night")
    r2 = engine.ingest_single_image(f2, inferred_condition="night")

    assert r1.status == "ACCEPTED"
    assert r2.status == "EXACT_DUPLICATE"
    assert "Identical byte hash" in r2.flags[0]


def test_near_duplicate_detection(temp_workspace):
    cfg = load_config(CONFIG_PATH)
    source_dir = temp_workspace / "incoming"
    target_dir = temp_workspace / "d1_dataset"
    source_dir.mkdir(parents=True)

    arr1 = np.zeros((720, 1280, 3), dtype=np.uint8)
    arr1[200:300, 400:600] = 200
    im1 = Image.fromarray(arr1)

    arr2 = arr1.copy()
    arr2[200:202, 400:402] = 201  # microscopic 2x2 change, dhash will match or have hamming dist <= 2
    im2 = Image.fromarray(arr2)

    f1 = source_dir / "scene_a.png"
    f2 = source_dir / "scene_b.png"
    im1.save(f1, "PNG")
    im2.save(f2, "PNG")

    engine = WeatherGuardD1IngestionEngine(config=cfg, target_dir=target_dir)
    r1 = engine.ingest_single_image(f1, inferred_condition="fog")
    r2 = engine.ingest_single_image(f2, inferred_condition="fog")

    assert r1.status == "ACCEPTED"
    assert r2.status == "NEAR_DUPLICATE_FLAGGED"
    assert any("Near-duplicate candidate" in fl for fl in r2.flags)


def test_corrupted_file_handling(temp_workspace):
    cfg = load_config(CONFIG_PATH)
    source_dir = temp_workspace / "incoming"
    target_dir = temp_workspace / "d1_dataset"
    source_dir.mkdir(parents=True)

    corrupt_file = source_dir / "corrupted.jpg"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE_BYTES")

    engine = WeatherGuardD1IngestionEngine(config=cfg, target_dir=target_dir)
    rec = engine.ingest_single_image(corrupt_file, inferred_condition="day")

    assert rec.status == "CORRUPTED"
    assert any("Image load failure" in fl for fl in rec.flags)


def test_directory_ingestion_and_manifest_generation(temp_workspace):
    cfg = load_config(CONFIG_PATH)
    source_dir = temp_workspace / "raw_batch"
    target_dir = temp_workspace / "WeatherGuard-D1"
    (source_dir / "day").mkdir(parents=True)
    (source_dir / "rain").mkdir(parents=True)

    # 2 valid day images with distinct gradients (preventing accidental dhash collisions)
    arr0 = np.tile(np.linspace(0, 255, 1280, dtype=np.uint8), (720, 1))
    Image.fromarray(np.stack([arr0]*3, axis=-1)).save(source_dir / "day" / "day_0.jpg", "JPEG")

    arr1 = np.tile(np.linspace(255, 0, 1280, dtype=np.uint8), (720, 1))
    Image.fromarray(np.stack([arr1]*3, axis=-1)).save(source_dir / "day" / "day_1.jpg", "JPEG")

    # 1 valid rain image with alternating stripes
    arr_rain = np.tile((np.arange(1280, dtype=np.uint8) % 2) * 255, (720, 1))
    Image.fromarray(np.stack([arr_rain]*3, axis=-1)).save(source_dir / "rain" / "rain_0.png", "PNG")

    engine = WeatherGuardD1IngestionEngine(config=cfg, target_dir=target_dir)
    res = engine.ingest_directory(raw_source_dir=source_dir, copy_files=True)

    assert res["summary"]["accepted_count"] == 3
    assert res["summary"]["condition_counts"]["day"] == 2
    assert res["summary"]["condition_counts"]["rain"] == 1

    manifest_file = target_dir / "manifests" / "ingestion_manifest.json"
    report_file = target_dir / "reports" / "ingestion_report.md"
    assert manifest_file.exists()
    assert report_file.exists()

    with open(manifest_file) as f:
        data = json.load(f)
        assert len(data["records"]) == 3
