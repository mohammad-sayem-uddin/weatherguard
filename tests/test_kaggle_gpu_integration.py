"""Unit and integration tests for Kaggle GPU environments and runners."""
import json
from pathlib import Path
import pytest
import torch

from scripts.kaggle.weatherguard_gpu_runner import (
    set_reproducibility,
    audit_gpu_environment,
)
from scripts.kaggle.audio_gpu_runner import (
    AudioBaselinePyTorchCNN,
    set_reproducibility as set_audio_reproducibility,
)
from scripts.kaggle.generate_kaggle_notebook import (
    generate_weatherguard_notebook,
    generate_audio_notebook,
)


def test_kaggle_gpu_audit_returns_valid_schema():
    profile = audit_gpu_environment()
    assert "platform" in profile
    assert profile["platform"] == "Kaggle Cloud"
    assert "cuda_available" in profile
    assert "device_name" in profile
    assert "pytorch_version" in profile
    assert isinstance(profile["gpu_memory_gb"], (int, float))


def test_reproducibility_seeding():
    set_reproducibility(1234)
    r1 = torch.rand(5)
    set_reproducibility(1234)
    r2 = torch.rand(5)
    assert torch.equal(r1, r2)


def test_audio_pytorch_cnn_forward_shape():
    model = AudioBaselinePyTorchCNN()
    model.eval()
    # Batch of 2 spectrograms: (B, C, n_mels, time_frames) = (2, 1, 64, 126)
    dummy_input = torch.randn(2, 1, 64, 126)
    logits = model(dummy_input)
    assert logits.shape == (2,)


def test_generate_kaggle_notebooks_valid_json():
    wg_nb_str = generate_weatherguard_notebook()
    nb = json.loads(wg_nb_str)
    assert "cells" in nb
    assert len(nb["cells"]) >= 4
    assert nb["metadata"]["language_info"]["name"] == "python"

    audio_nb_str = generate_audio_notebook()
    a_nb = json.loads(audio_nb_str)
    assert "cells" in a_nb
    assert len(a_nb["cells"]) >= 4
