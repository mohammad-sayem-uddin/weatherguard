"""Unit tests for model-specific class mapping resolution in WeatherGuard evaluation.

Module: tests/test_model_class_mapping.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
"""
import pytest
from pathlib import Path

from scripts.evaluate_weatherguard import (
    CANONICAL_CLASSES,
    COCO_TO_CANONICAL,
    D1_TO_CANONICAL,
    ModelClassMappingError,
    WeatherGuardEvaluator,
    resolve_model_class_mapping,
)


def test_coco_model_class_mapping_resolution():
    # Construct standard COCO 80 names dict
    coco_names = {i: f"class_{i}" for i in range(80)}
    coco_names[0] = "person"
    coco_names[1] = "bicycle"
    coco_names[2] = "car"
    coco_names[3] = "motorcycle"
    coco_names[5] = "bus"
    coco_names[7] = "truck"

    mapping = resolve_model_class_mapping(coco_names)
    assert mapping == COCO_TO_CANONICAL
    assert mapping[0] == "person"
    assert mapping[1] == "bicycle"
    assert mapping[2] == "car"
    assert mapping[3] == "motorcycle"
    assert mapping[5] == "bus"
    assert mapping[7] == "truck"


def test_d1_six_class_mapping_resolution():
    # WeatherGuard-D1 custom 6-class taxonomy:
    # 0: truck, 1: person, 2: bicycle, 3: car, 4: motorcycle, 5: bus
    d1_names = {
        0: "truck",
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "bus",
    }
    mapping = resolve_model_class_mapping(d1_names)
    assert mapping == D1_TO_CANONICAL
    assert mapping[0] == "truck"
    assert mapping[1] == "person"
    assert mapping[2] == "bicycle"
    assert mapping[3] == "car"
    assert mapping[4] == "motorcycle"
    assert mapping[5] == "bus"


def test_d1_six_class_from_list():
    d1_list = ["truck", "person", "bicycle", "car", "motorcycle", "bus"]
    mapping = resolve_model_class_mapping(d1_list)
    assert mapping == D1_TO_CANONICAL


def test_model_type_hints():
    assert resolve_model_class_mapping(model_type_hint="coco") == COCO_TO_CANONICAL
    assert resolve_model_class_mapping(model_type_hint="d1") == D1_TO_CANONICAL
    assert resolve_model_class_mapping(model_type_hint="weatherguard_d1") == D1_TO_CANONICAL

    with pytest.raises(ModelClassMappingError, match="Unknown model_type_hint"):
        resolve_model_class_mapping(model_type_hint="imagenet")


def test_invalid_and_ambiguous_metadata():
    # Empty
    with pytest.raises(ModelClassMappingError):
        resolve_model_class_mapping({})

    # No model names or hint
    with pytest.raises(ModelClassMappingError):
        resolve_model_class_mapping(None)

    # Completely non-driving classes
    with pytest.raises(ModelClassMappingError, match="No canonical driving classes found"):
        resolve_model_class_mapping({0: "cat", 1: "dog", 2: "laptop"})

    # Ambiguous duplicate classes (two IDs map to 'car')
    with pytest.raises(ModelClassMappingError, match="Ambiguous model.names"):
        resolve_model_class_mapping({0: "car", 1: "Car", 2: "truck", 3: "bus"})

    # 6-class model missing some canonical classes
    with pytest.raises(ModelClassMappingError, match="6-class model is missing canonical classes"):
        resolve_model_class_mapping({0: "car", 1: "truck", 2: "bus", 3: "bicycle", 4: "boat", 5: "train"})


def test_evaluator_uses_d1_mapping_with_hint():
    evaluator = WeatherGuardEvaluator(
        weights_path=Path("non_existent.pt"),
        model_type_hint="d1",
    )
    assert evaluator.class_mapping == D1_TO_CANONICAL
    assert evaluator.class_mapping[0] == "truck"
    assert evaluator.class_mapping[1] == "person"
