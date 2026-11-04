"""Unit tests for WeatherGuard-D1 Quality Control, Uncertainty Review, and Statistics.

Module: tests/test_d1_quality_control.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""
import json
import shutil
import tempfile
from pathlib import Path
import pytest

from scripts.extract_uncertain_annotations import UncertaintyExtractor
from scripts.generate_annotation_review_report import generate_markdown_report
from scripts.analyze_weatherguard_d1 import (
    D1DatasetAnalyzer,
    compute_gini_coefficient,
    compute_shannon_entropy,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "weatherguard_d1.yaml"


@pytest.fixture
def temp_workspace():
    tmp_dir = Path(tempfile.mkdtemp(prefix="test_d1_qc_"))
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_gini_and_entropy_computations():
    # Perfectly equal distribution
    assert compute_gini_coefficient([10, 10, 10, 10]) == 0.0
    # Skewed distribution
    gini_skew = compute_gini_coefficient([0, 0, 0, 100])
    assert gini_skew > 0.70

    # Entropy
    ent_uniform = compute_shannon_entropy([10, 10])
    ent_skew = compute_shannon_entropy([1, 99])
    assert ent_uniform > ent_skew


def test_uncertainty_triage_three_queues():
    extractor = UncertaintyExtractor(low_confidence_threshold=0.50, tiny_ambiguous_px_threshold=8.0)

    mock_manifest = {
        "results": [
            {
                "image_id": "img_001",
                "condition": "day",
                "valid_boxes": [
                    {
                        "box_id": "b_accepted",
                        "class_name": "car",
                        "confidence": 0.95,
                        "max_dim_px": 50.0,
                        "pixel_width": 50.0,
                        "pixel_height": 30.0,
                        "recalculated_size": "medium",
                        "size_disagreement": False,
                    },
                    {
                        "box_id": "b_low_conf",
                        "class_name": "person",
                        "confidence": 0.35,  # < 0.50 -> needs review
                        "max_dim_px": 25.0,
                        "pixel_width": 10.0,
                        "pixel_height": 25.0,
                        "recalculated_size": "small",
                        "size_disagreement": False,
                    },
                    {
                        "box_id": "b_tiny",
                        "class_name": "car",
                        "confidence": 0.90,
                        "max_dim_px": 6.5,  # < 8.0px -> needs review
                        "pixel_width": 6.5,
                        "pixel_height": 4.0,
                        "recalculated_size": "ultra_small",
                        "size_disagreement": False,
                    }
                ],
                "rejected_boxes": [
                    {
                        "box_id": "b_inverted",
                        "rejection_reasons": ["Non-positive width: x_min >= x_max"]
                    }
                ],
                "is_empty": False,
            }
        ]
    }

    triage = extractor.triage_validation_manifest(mock_manifest)
    assert triage["summary"]["total_accepted_structural"] == 1
    assert triage["summary"]["total_needs_visual_review"] == 2
    assert triage["summary"]["total_rejected"] == 1

    rev_box_ids = [item["box_id"] for item in triage["needs_visual_review"]]
    assert "b_low_conf" in rev_box_ids
    assert "b_tiny" in rev_box_ids


def test_generate_review_report(temp_workspace):
    report_file = temp_workspace / "review_report.md"
    queue_data = {
        "summary": {
            "total_accepted_structural": 10,
            "total_needs_visual_review": 2,
            "total_rejected": 1,
        },
        "needs_visual_review": [
            {
                "image_id": "test_img",
                "condition": "fog",
                "class_name": "car",
                "pixel_dimensions": "7.0x5.0",
                "max_dim_px": 7.0,
                "confidence": 0.42,
                "review_reasons": ["Extremely small object", "Borderline confidence"],
            }
        ],
        "rejected": [
            {
                "image_id": "bad_img",
                "condition": "rain",
                "reason": "Structural error",
                "box": {"rejection_reasons": ["Non-positive height"]}
            }
        ]
    }

    out_p = generate_markdown_report(queue_data, report_file)
    assert out_p.exists()
    content = out_p.read_text()
    assert "Annotation Quality Control & Visual Review Report" in content
    assert "test_img" in content
    assert "Extremely small object" in content


def test_dataset_analysis_not_available_state(temp_workspace):
    analyzer = D1DatasetAnalyzer(config_path=CONFIG_PATH)

    non_existent_ingest = temp_workspace / "no_ingest.json"
    non_existent_valid = temp_workspace / "no_valid.json"

    res = analyzer.analyze_dataset(non_existent_ingest, non_existent_valid)
    assert res["status"] == "NOT_AVAILABLE"
    assert "Awaiting raw synthetic data" in res["message"]
    assert res["target_specification"]["target_images"] == 2000

    out_json = temp_workspace / "analysis.json"
    out_csv = temp_workspace / "dist.csv"
    out_rep = temp_workspace / "report.md"
    analyzer.export_reports(res, out_json, out_csv, out_rep)

    assert out_json.exists()
    assert out_rep.exists()
    assert "Status:** NOT_AVAILABLE" in out_rep.read_text()


def test_dataset_analysis_calculation(temp_workspace):
    analyzer = D1DatasetAnalyzer(config_path=CONFIG_PATH)

    ingest_path = temp_workspace / "ingest.json"
    valid_path = temp_workspace / "valid.json"

    # Synthetic ingestion records
    mock_ingest = {
        "records": [
            {"image_id": "img1", "condition": "day", "status": "ACCEPTED"},
            {"image_id": "img2", "condition": "night", "status": "ACCEPTED"},
            {"image_id": "img3", "condition": "rain", "status": "ACCEPTED"},
        ]
    }
    with open(ingest_path, "w") as f:
        json.dump(mock_ingest, f)

    # Synthetic validation records
    mock_valid = {
        "results": [
            {
                "image_id": "img1",
                "condition": "day",
                "valid_boxes": [
                    {"class_name": "car", "recalculated_size": "medium", "confidence": 0.90},
                    {"class_name": "car", "recalculated_size": "large", "confidence": 0.95},
                ],
                "is_empty": False,
            },
            {
                "image_id": "img2",
                "condition": "night",
                "valid_boxes": [
                    {"class_name": "truck", "recalculated_size": "small", "confidence": 0.85},
                    {"class_name": "person", "recalculated_size": "ultra_small", "confidence": 0.70},
                ],
                "is_empty": False,
            },
            {
                "image_id": "img3",
                "condition": "rain",
                "valid_boxes": [],
                "is_empty": True,
            }
        ]
    }
    with open(valid_path, "w") as f:
        json.dump(mock_valid, f)

    analysis = analyzer.analyze_dataset(ingest_path, valid_path)
    assert analysis["status"] == "AVAILABLE"
    assert analysis["image_summary"]["accepted_images"] == 3
    assert analysis["image_summary"]["empty_images_count"] == 1
    assert analysis["object_summary"]["total_valid_objects"] == 4
    assert analysis["object_summary"]["class_counts"]["car"] == 2
    assert analysis["object_summary"]["class_counts"]["truck"] == 1
    assert analysis["object_summary"]["class_counts"]["person"] == 1
    assert analysis["object_summary"]["object_size_distribution"]["ultra_small"] == 1
    assert analysis["object_summary"]["object_size_distribution"]["small"] == 1
    assert analysis["object_summary"]["object_size_distribution"]["medium"] == 1
    assert analysis["object_summary"]["object_size_distribution"]["large"] == 1

    out_json = temp_workspace / "analysis.json"
    out_csv = temp_workspace / "dist.csv"
    out_rep = temp_workspace / "report.md"
    analyzer.export_reports(analysis, out_json, out_csv, out_rep)

    assert out_csv.exists()
    assert out_rep.exists()
    rep_text = out_rep.read_text()
    assert "WeatherGuard-D1 Dataset Characterization" in rep_text
    assert "Class Balance & Diversity" in rep_text
