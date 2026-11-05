# WeatherGuard & Multimodal Research — Phase 1: Reusable Kaggle Training Environment Report

**Date:** 2026-11-05  
**Project Owner:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** [https://github.com/mohammad-sayem-uddin/weatherguard](https://github.com/mohammad-sayem-uddin/weatherguard)  
**Target Hardware:** Kaggle Cloud NVIDIA Tesla T4 GPU (16 GB VRAM)  
**Artifacts:**
- [`scripts/kaggle/weatherguard_gpu_runner.py`](file:///Users/sayemuddin/Desktop/WeatherGuard/scripts/kaggle/weatherguard_gpu_runner.py)
- [`scripts/kaggle/audio_gpu_runner.py`](file:///Users/sayemuddin/Desktop/WeatherGuard/scripts/kaggle/audio_gpu_runner.py)
- [`scripts/kaggle/generate_kaggle_notebook.py`](file:///Users/sayemuddin/Desktop/WeatherGuard/scripts/kaggle/generate_kaggle_notebook.py)
- [`artifacts/kaggle_gpu/weatherguard_training_kaggle.ipynb`](file:///Users/sayemuddin/Desktop/WeatherGuard/artifacts/kaggle_gpu/weatherguard_training_kaggle.ipynb)
- [`artifacts/kaggle_gpu/audio_training_kaggle.ipynb`](file:///Users/sayemuddin/Desktop/WeatherGuard/artifacts/kaggle_gpu/audio_training_kaggle.ipynb)

---

## 1. Executive Summary

Phase 1 established modular, reproducible, and decoupled GPU training harnesses for both research streams:
1. **WeatherGuard Stream:** Ultralytics YOLOv8 fine-tuning, mixed-precision AMP, AdamW optimization, small-object weighting, 101-point VOC/COCO mAP evaluation, and multi-scale resolution scaling.
2. **Audio Stream:** 2D CNN acoustic vocalization classification (`AudioBaselinePyTorchCNN`) operating on 16 kHz Log-Mel spectrograms ($64 \times 126$), with binary cross-entropy loss, dropout regularization, and ROC-AUC / F1 evaluation.

The environments and data pipelines remain strictly separated to prevent cross-project contamination while sharing standardized Kaggle GPU deployment infrastructure.

---

## 2. Environment Architecture & Tooling

| Dimension | WeatherGuard Perception Stream | Audio Vocalization Stream |
| :--- | :--- | :--- |
| **Framework** | PyTorch + Ultralytics YOLOv8 | PyTorch + Torchaudio |
| **Input Representation** | 3-Channel RGB / Thermal ($640 \times 640$, $832 \times 832$) | Single-Channel Log-Mel Spectrogram ($64 \times 126$) |
| **Loss Function** | Complete YOLOv8 Multi-Task Loss (CIoU + DFL + BCE) | Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) |
| **Precision** | Automated Mixed Precision (`torch.cuda.amp`) | FP32 / Mixed Precision |
| **Optimizer** | AdamW (`lr=0.001`, `weight_decay=0.0005`) | AdamW (`lr=0.001`, `weight_decay=0.0001`) |
| **Evaluation Metrics** | 101-point mAP@0.5, mAP@0.5:0.95, Scale Bins | Accuracy, Precision, Recall, F1, ROC-AUC, FPR/FNR |
| **Output Destination** | `/kaggle/working/artifacts/` | `/kaggle/working/artifacts/` |
| **Local Integration** | Synchronized with local `artifacts/` via Kaggle MCP | Synchronized with local `results/` via Kaggle MCP |

---

## 3. Verification & Test Suite

The runners and notebook compiler were verified via [`tests/test_kaggle_gpu_integration.py`](file:///Users/sayemuddin/Desktop/WeatherGuard/tests/test_kaggle_gpu_integration.py):
- GPU environment audit schema validation passed.
- Deterministic seeding verified across PyTorch and CUDA.
- Audio PyTorch CNN forward pass verified with shape `(B, 1, 64, 126) -> (B,)`.
- Automated notebook compilation produced valid `nbformat: 4` Jupyter JSON files.
