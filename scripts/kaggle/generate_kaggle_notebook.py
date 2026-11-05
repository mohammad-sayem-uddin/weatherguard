#!/usr/bin/env python3
"""Compiles modular training runners into valid Kaggle Jupyter Notebook JSON files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


def create_kaggle_notebook_json(cells: List[dict]) -> str:
    """Wraps notebook cells into nbformat 4.4 JSON structure."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return json.dumps(nb, indent=2)


def make_markdown_cell(source_lines: List[str]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines],
    }


def make_code_cell(source_lines: List[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines],
    }


def generate_weatherguard_notebook() -> str:
    cells = [
        make_markdown_cell([
            "# WeatherGuard — Cloud GPU Training & Evaluation Harness",
            "**Platform**: Kaggle NVIDIA Tesla T4 GPU Runtime (16 GB VRAM)",
            "**Project Owner**: Mohammad Sayem Uddin",
            "**Repository**: https://github.com/mohammad-sayem-uddin/weatherguard",
        ]),
        make_code_cell([
            "!nvidia-smi",
            "import torch",
            "print('PyTorch Version:', torch.__version__)",
            "print('CUDA Available:', torch.cuda.is_available())",
            "if torch.cuda.is_available():",
            "    print('GPU Device:', torch.cuda.get_device_name(0))",
            "    print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))",
        ]),
        make_code_cell([
            "!git clone https://github.com/mohammad-sayem-uddin/weatherguard.git /kaggle/working/weatherguard",
            "%cd /kaggle/working/weatherguard",
            "!pip install -q ultralytics pyyaml pandas opencv-python pillow",
        ]),
        make_code_cell([
            "import sys",
            "sys.path.insert(0, '/kaggle/working/weatherguard')",
            "from scripts.kaggle.weatherguard_gpu_runner import audit_gpu_environment",
            "profile = audit_gpu_environment()",
            "print('Audited Environment:', profile)",
        ]),
    ]
    return create_kaggle_notebook_json(cells)


def generate_audio_notebook() -> str:
    cells = [
        make_markdown_cell([
            "# Audio Emergency Vocalization Classification — Cloud GPU Harness",
            "**Platform**: Kaggle NVIDIA Tesla T4 GPU Runtime (16 GB VRAM)",
            "**Project Owner**: Mohammad Sayem Uddin",
            "**Task**: Bengali Distress Cry ('bachao' / বাঁচাও) vs Normal Speech & Hard Negatives",
        ]),
        make_code_cell([
            "!nvidia-smi",
            "import torch",
            "print('CUDA Available:', torch.cuda.is_available())",
            "if torch.cuda.is_available():",
            "    print('GPU Device:', torch.cuda.get_device_name(0))",
        ]),
        make_code_cell([
            "!pip install -q torchaudio librosa soundfile scikit-learn",
        ]),
        make_code_cell([
            "import os, torch, torchaudio, librosa",
            "print('Torchaudio Version:', torchaudio.__version__)",
            "print('Librosa Version:', librosa.__version__)",
        ]),
    ]
    return create_kaggle_notebook_json(cells)


if __name__ == "__main__":
    out_dir = Path("/Users/sayemuddin/Desktop/WeatherGuard/artifacts/kaggle_gpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    wg_nb = generate_weatherguard_notebook()
    (out_dir / "weatherguard_training_kaggle.ipynb").write_text(wg_nb)

    audio_nb = generate_audio_notebook()
    (out_dir / "audio_training_kaggle.ipynb").write_text(audio_nb)

    print("Kaggle notebook templates compiled to:", out_dir)
