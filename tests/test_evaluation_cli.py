"""Unit tests for the end-to-end evaluation CLI and dataset loaders.

Module: tests/test_evaluation_cli.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""
import json
import pytest
from pathlib import Path

from scripts.evaluate_weatherguard import (
    WeatherGuardEvaluator,
    load_dataset_records,
)

ROOT = Path(__file__).resolve().parent.parent


def test_locked_test_split_raises_permission_error():
    with pytest.raises(PermissionError, match="Access to locked test split"):
        load_dataset_records(dataset_name="d1", split="test", allow_locked_test=False)

    with pytest.raises(PermissionError, match="Access to locked test split"):
        load_dataset_records(dataset_name="d1", split="locked", allow_locked_test=False)


def test_d1_unmanifested_raises_filenotfound():
    # If D1 manifests are absent or missing
    with pytest.raises(FileNotFoundError, match="split manifest not found"):
        load_dataset_records(dataset_name="d1", split="validation")


def test_dawn_dataset_loading_subset():
    records = load_dataset_records(dataset_name="dawn", split="val", max_samples=2)
    assert len(records) <= 2
    if records:
        r = records[0]
        assert "image_id" in r
        assert "file_path" in r
        assert Path(r["file_path"]).exists()
        assert "gt_boxes" in r


def test_evaluator_empty_records():
    evaluator = WeatherGuardEvaluator(weights_path=ROOT / "yolov8n.pt")
    res = evaluator.evaluate_image_stream([])
    assert "overall" in res
    assert res["overall"]["tp"] == 0
    assert res["overall"]["fp"] == 0
    assert res["overall"]["fn"] == 0
    assert res["latency"]["measured_images"] == 0
