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
    runner_py_path = Path(__file__).resolve().parent / "weatherguard_gpu_runner.py"
    runner_code = runner_py_path.read_text()

    cells = [
        make_markdown_cell([
            "# WeatherGuard Mission 20 — Real Dataset GPU Training & Evaluation Harness",
            "**Platform**: Kaggle NVIDIA Tesla T4 GPU Runtime (16 GB VRAM)",
            "**Dataset**: `allfaz/weatherguard-v2-corpus` (1,000 Curated Frames)",
            "**Splits**: Train (667 frames), Val (173 frames), Locked Test (160 frames, 0 augmented)",
            "**Repository**: https://github.com/mohammad-sayem-uddin/weatherguard",
            "**Committer**: Mohammad Sayem Uddin <sayemwork41@gmail.com>",
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
            "!pip install -q ultralytics pyyaml pandas opencv-python pillow",
        ]),
        make_code_cell([
            "import os",
            "from pathlib import Path",
            "dataset_path = Path('/kaggle/input/weatherguard-v2-corpus')",
            "print('Dataset directory exists:', dataset_path.exists())",
            "if dataset_path.exists():",
            "    for split in ['train', 'val', 'test']:",
            "        img_cnt = len(list((dataset_path / split / 'images').glob('*.*')))",
            "        lbl_cnt = len(list((dataset_path / split / 'labels').glob('*.txt')))",
            "        print(f'  {split}: {img_cnt} images, {lbl_cnt} labels')",
        ]),
        make_code_cell([
            "# Write out standalone runner module to ensure exact version control",
            "from pathlib import Path",
            f"runner_content = {repr(runner_code)}",
            "Path('/kaggle/working/weatherguard_gpu_runner.py').write_text(runner_content)",
            "print('Runner written to /kaggle/working/weatherguard_gpu_runner.py')",
        ]),
        make_code_cell([
            "import sys",
            "sys.path.insert(0, '/kaggle/working')",
            "from weatherguard_gpu_runner import run_full_vision_benchmark",
            "",
            "# Execute end-to-end real dataset training and independent test evaluation",
            "benchmark_results = run_full_vision_benchmark(",
            "    data_yaml='/kaggle/input/weatherguard-v2-corpus/data.yaml',",
            "    weights_path='yolov8n.pt',",
            "    epochs=15,",
            "    imgsz=640,",
            "    batch_size=16,",
            "    output_dir='/kaggle/working/runs/train',",
            "    experiment_id='weatherguard_t4_real',",
            ")",
            "",
            "print('--- BENCHMARK RESULTS SUMMARY ---')",
            "print('Validation Split Metrics:', benchmark_results['validation_split_metrics'])",
            "print('Locked Test Overall Metrics:', benchmark_results['locked_test_split_metrics']['overall'])",
            "print('Locked Test Per-Class:', benchmark_results['locked_test_split_metrics']['per_class'])",
            "print('Locked Test Scale Breakdown:', benchmark_results['locked_test_split_metrics']['scale_breakdown'])",
            "print('Locked Test Weather Breakdown:', benchmark_results['locked_test_split_metrics']['weather_breakdown'])",
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
            "!git clone https://github.com/mohammad-sayem-uddin/weatherguard.git /kaggle/working/weatherguard",
            "%cd /kaggle/working/weatherguard",
            "!pip install -q torchaudio librosa soundfile scikit-learn",
        ]),
        make_code_cell([
            "import sys",
            "sys.path.insert(0, '/kaggle/working/weatherguard')",
            "from scripts.kaggle.audio_gpu_runner import train_audio_classifier",
            "",
            "# Execute Audio 2D CNN Spectrogram Training on Tesla T4",
            "audio_results = train_audio_classifier(",
            "    epochs=5,",
            "    batch_size=32,",
            "    lr=0.001,",
            "    output_dir='/kaggle/working/artifacts/audio_gpu',",
            ")",
            "print('Audio Training Results:', audio_results)",
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
