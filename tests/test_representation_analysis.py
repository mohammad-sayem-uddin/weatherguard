"""Unit tests for src/representation_analysis.py."""

import numpy as np
import pytest
import torch

from src.representation_analysis import (
    linear_cka,
    mean_cosine_similarity,
    evaluate_domain_separability,
)


def test_linear_cka_identical():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((50, 64))
    cka_self = linear_cka(x, x)
    assert abs(cka_self - 1.0) < 1e-5


def test_linear_cka_orthogonal():
    # Construct zero-mean orthogonal blocks
    x = np.array([[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]])
    y = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]])
    cka_orth = linear_cka(x, y)
    assert abs(cka_orth) < 1e-4


def test_linear_cka_scale_invariance():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((40, 32))
    y = 5.5 * x
    cka_scaled = linear_cka(x, y)
    assert abs(cka_scaled - 1.0) < 1e-5


def test_mean_cosine_similarity():
    x = np.array([[1.0, 0.0], [0.0, 2.0]])
    y = np.array([[2.0, 0.0], [0.0, 1.0]])
    assert abs(mean_cosine_similarity(x, y) - 1.0) < 1e-5

    z = np.array([[-1.0, 0.0], [0.0, -2.0]])
    assert abs(mean_cosine_similarity(x, z) - (-1.0)) < 1e-5


def test_domain_separability_synthetic():
    rng = np.random.default_rng(123)
    # Distinct distributions
    dawn_synth = rng.normal(loc=-2.0, scale=1.0, size=(40, 16))
    bdd_synth = rng.normal(loc=2.0, scale=1.0, size=(40, 16))
    res = evaluate_domain_separability(dawn_synth, bdd_synth, n_splits=3)
    assert res["mean_accuracy"] > 0.90
    assert res["mean_roc_auc"] > 0.90
