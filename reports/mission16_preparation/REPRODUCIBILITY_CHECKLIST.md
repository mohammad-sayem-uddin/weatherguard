# WeatherGuard Mission 16 — Reproducibility Checklist & Experimental Protocol
**Document:** `reports/mission16_preparation/REPRODUCIBILITY_CHECKLIST.md`  
**Date:** 2026-11-04  
**Author:** Mohammad Sayem Uddin (<sayemwork41@gmail.com>)  
**Mission:** Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

---

## 1. Computational Environment & Dependency Specifications

- **Hardware Platform:** Apple MacBook Air M1 (8 cores: 4 Firestorm @ 3.2 GHz + 4 Icestorm @ 2.06 GHz), 8 GB Unified Memory.
- **Operating System:** macOS Darwin 24.3.0 (arm64).
- **Execution Mode:** FP32 CPU-only (zero GPU / Neural Engine offloading).
- **Python Version:** Python 3.11.15.
- **Core Libraries:**
  - `torch==2.5.1`
  - `torchvision==0.20.1`
  - `ultralytics==8.3.40`
  - `opencv-python-headless==4.10.0.84`
  - `numpy==1.26.4`
  - `scipy==1.14.1`
  - `pandas==2.2.3`
  - `pyyaml==6.0.2`
  - `pytest==8.3.4`

---

## 2. Deterministic Seeds & Randomness Control

| Process | Deterministic Seed | Enforcing Code / Config |
| :--- | :---: | :--- |
| **Dataset Ingestion dHash** | `dhash_size=8` | `scripts/ingest_weatherguard_d1.py` |
| **Dataset Splitting (Train/Val/Test)** | `seed=42` | `scripts/create_d1_splits.py` |
| **YOLOv8 Fine-Tuning** | `seed=42` | `scripts/train_weatherguard_d1.py` |
| **Paired Sequence Bootstrap ($B=1,000$)** | `seed=42` | `scripts/run_mission15_locked_eval.py` |
| **Scenario Block Bootstrap ($B=1,000$)** | `seed=42` | `scripts/evaluate_weatherguard.py` |

---

## 3. Data Integrity & Cryptographic Manifests

1. **Ingestion Verification:**
   - Manifest: `data/WeatherGuard-D1/manifests/ingestion_manifest.json`
   - Records: SHA-256 cryptographic digest, dHash, and dimension verification for all 2,000 images.
2. **Split Verification:**
   - Manifest: `data/WeatherGuard-D1/manifests/d1_split_manifest.json`
   - Split Integrity Hash: SHA-256 hash computed over canonical image ID assignments.
3. **Locked Test Protection:**
   - BDD100K locked cohort: Exactly 6 sequences (1,214 frames) recorded in `artifacts/mission15/sequence_manifest.json`.
   - Historical exclusion: 22 sequences blocked via `artifacts/mission15/sequence_exclusion_manifest.json`.

---

## 4. Step-by-Step Pipeline Reproduction Instructions

```bash
# 1. Activate development environment
source .venv/bin/activate
export PYTHONPATH=.

# 2. Ingest WeatherGuard-D1 synthetic images (upon delivery)
python scripts/ingest_weatherguard_d1.py --source-dir /path/to/raw_d1_images --copy

# 3. Validate Gemini Flash annotations and produce YOLO ground truth
python scripts/validate_d1_annotations.py --input-dir /path/to/raw_annotations

# 4. Triage uncertainty queues and extract review reports
python scripts/extract_uncertain_annotations.py
python scripts/generate_annotation_review_report.py
python scripts/analyze_weatherguard_d1.py

# 5. Generate leakage-safe stratified splits (Train 1200 / Val 400 / Test 400)
python scripts/create_d1_splits.py --seed 42

# 6. Execute fine-tuning experiment (e.g. Experiment E2)
python scripts/train_weatherguard_d1.py --exp-id E2_d1_finetune_yolov8n --imgsz 640 --batch 8

# 7. Evaluate on held-out synthetic test set and real-world locked video
python scripts/evaluate_weatherguard.py --model runs/train_d1/E2_d1_finetune_yolov8n/weights/best.pt --dataset d1 --split test
python scripts/evaluate_weatherguard.py --model runs/train_d1/E2_d1_finetune_yolov8n/weights/best.pt --dataset bdd100k --split locked
```
