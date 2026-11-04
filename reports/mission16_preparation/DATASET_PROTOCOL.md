# WeatherGuard-D1 — Dataset Engineering & Ingestion Protocol
**Document:** `reports/mission16_preparation/DATASET_PROTOCOL.md`  
**Date:** 2026-11-04  
**Author:** Mohammad Sayem Uddin (<sayemwork41@gmail.com>)  
**Mission:** Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

---

## 1. Dataset Overview & Operational Scope

WeatherGuard-D1 is a synthetic photorealistic driving dataset designed to evaluate and enhance adverse-weather object detection on resource-constrained edge hardware.
- **Total Volume:** 2,000 independent still images.
- **Resolution:** Strict $1280 \times 720$ pixels ($16:9$ dashcam perspective).
- **Generation Engine:** Google Flow diffusion generation.
- **Annotation Method:** Gemini Flash vision-language model with multi-tier algorithmic verification.
- **Target Distribution:**
  - Daylight Clear / Urban: 600 images ($30\%$).
  - Night Low-Light / Headlight Bloom: 500 images ($25\%$).
  - Rain Precipitation / Wet Road Spray: 500 images ($25\%$).
  - Dense Fog / Low-Contrast Scattering: 400 images ($20\%$).

---

## 2. Ingestion Pipeline & File Integrity Checks

The ingestion pipeline (`scripts/ingest_weatherguard_d1.py`) enforces strict verification before admitting raw files:
1. **Dimension Verification:** Every image must be exactly $1280 \times 720\text{ px}$. Non-standard resolutions are rejected; the pipeline **never silently resizes or crops incoming images**.
2. **Cryptographic Deduplication:** SHA-256 hashes are computed for every file to prevent identical duplicate uploads.
3. **Perceptual Near-Duplicate Discovery:** 64-bit difference hashes (dHash) are computed across all accepted images. Pairs exhibiting Hamming distance $\le 4$ are flagged as near-duplicate clusters to prevent data leakage.
4. **Immutability of Raw Data:** Original source files are treated as read-only and never overwritten.

---

## 3. Directory Layout Standard

```
data/WeatherGuard-D1/
├── images/
│   ├── day/              # 600 accepted images
│   ├── night/            # 500 accepted images
│   ├── rain/             # 500 accepted images
│   └── fog/              # 400 accepted images
├── labels/               # Standardized YOLO format text files
├── annotations/          # Validated JSON annotation files
├── metadata/             # Prompt, seed, and generation parameter manifests
├── manifests/            # Ingestion, validation, and split manifests
├── quality_control/      # Triage and uncertainty review queues
└── reports/              # Ingestion, validation, and statistical characterization reports
```
*(All image, label, and cache directories under `data/WeatherGuard-D1/` are excluded from Git via `.gitignore`.)*
