# WeatherGuard-D1 — Dataset Split Protocol & Leakage Prevention Strategy
**Document:** `reports/mission16_preparation/D1_SPLIT_PROTOCOL.md`  
**Date:** 2026-11-04  
**Author:** Mohammad Sayem Uddin (<sayemwork41@gmail.com>)  
**Mission:** Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

---

## 1. Executive Summary & Split Target Specifications

WeatherGuard-D1 consists of 2,000 synthetic photorealistic dashcam scenes generated via Google Flow with AI-assisted Gemini Flash annotations.

The dataset is partitioned into three strictly disjoint cohorts:
- **Training Cohort:** 1,200 images ($60.0\%$) — Used exclusively for parameter fine-tuning and weight adaptation.
- **Validation Cohort:** 400 images ($20.0\%$) — Used for model selection, checkpoint early stopping, confidence threshold sweeping, and hyperparameter tuning.
- **Internal Test Cohort (Locked):** 400 images ($20.0\%$) — Held strictly frozen for unbiased internal evaluation.

### Condition-Stratified Target Matrix

| Condition | Target Total | Train Allocation (60%) | Validation Allocation (20%) | Test Allocation (20%) |
| :--- | :---: | :---: | :---: | :---: |
| **Day (Clear)** | 600 | 360 | 120 | 120 |
| **Night (Low-Light)** | 500 | 300 | 100 | 100 |
| **Rain (Precipitation)** | 500 | 300 | 100 | 100 |
| **Fog (Scattering)** | 400 | 240 | 80 | 80 |
| **Total** | **2,000** | **1,200** | **400** | **400** |

---

## 2. The Synthetic Data Leakage Problem & Scientific Mitigation

In real-world driving video (such as BDD100K), frames are temporally continuous; random frame-level splitting causes catastrophic temporal data leakage.

In synthetic image generation (such as Google Flow), images are generated independently. However, **naive uniform random splitting introduces an equally severe synthetic leakage flaw**:
1. **Prompt Template Overlap:** If a specific text prompt ("heavy rain on highway with distant red taillights, telephoto dashcam") is generated with multiple seeds, the resulting images share identical road layouts, lighting gradients, and asset arrangements.
2. **Visual Near-Duplicates:** Diffusion models frequently reproduce nearly identical background textures, vehicle silhouettes, or horizon trees across adjacent generation seeds.
3. **Overfitting Artifacts:** A network trained on an image with a specific diffusion generation artifact will trivially detect objects on a test image sharing that exact artifact, giving a false impression of generalized detector acuity.

### Mandatory Leakage Prevention Invariants:
1. **Scenario Family Clustering:** All images generated from the same prompt template, batch ID, or scenario descriptor are grouped into an atomic cluster ($F_k$).
2. **Perceptual dHash Cluster Grouping:** Images exhibiting perceptual difference hash (dHash) Hamming distance $\le 4$ are unified into the same cluster.
3. **Atomic Set Allocation:** An entire cluster $F_k$ is placed wholly into Train, wholly into Validation, or wholly into Test:
   $$\forall F_k: \quad (F_k \subseteq S_{train}) \lor (F_k \subseteq S_{val}) \lor (F_k \subseteq S_{test})$$
4. **Zero Intersection Guarantee:**
   $$S_{train} \cap S_{val} = \emptyset, \quad S_{train} \cap S_{test} = \emptyset, \quad S_{val} \cap S_{test} = \emptyset$$

---

## 3. Test Set Freezing & Research Governance

1. **Blinded Model Selection:** The internal test set ($400$ images) must **never be touched during model selection, learning rate tuning, augmentation search, or candidate threshold optimization**.
2. **One-Time Evaluation Rule:** The internal test set is evaluated once at the conclusion of model fine-tuning.
3. **External Real-World Primacy:** Superior performance on the D1 internal test set does not prove real-world generalization. The primary scientific benchmark remains the strictly external real-world BDD100K and FLIR ADAS evaluation splits.
4. **Reproducibility Manifest:** Every split assignment generates a cryptographic SHA-256 hash in `manifests/d1_split_manifest.json` locking the exact image IDs.
