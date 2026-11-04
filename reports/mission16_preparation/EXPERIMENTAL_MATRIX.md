# WeatherGuard Mission 16 — Standardized Experimental Matrix
**Document:** `reports/mission16_preparation/EXPERIMENTAL_MATRIX.md`  
**Date:** 2026-11-04  
**Author:** Mohammad Sayem Uddin (<sayemwork41@gmail.com>)  
**Mission:** Mission 16 — Photorealistic Synthetic Data Fine-Tuning & Real-World Domain Transfer  
**Execution Environment:** Apple MacBook Air M1, 8-core CPU, 8 GB Unified Memory, macOS Darwin 24.3.0, CPU-only.

---

## 1. Central Research Question

> **"Can photorealistic synthetic adverse-weather dashcam data improve small and distant road-user detection, and can these improvements transfer to real-world driving datasets without unacceptable degradation in generalization or CPU inference efficiency?"**

---

## 2. Controlled Factorial Experiment Matrix (E1 to E12)

| Experiment ID | Model / Weights | Training Strategy & Data | Inference Policy | Research Hypothesis | Primary Evaluation Sets | Target M1 CPU FPS | Initial Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **E1** | YOLOv8n (Pretrained) | None (Frozen baseline) | Raw Detector (640px) | High baseline precision on large objects; severe recall collapse on small objects ($< 32\text{px}$). | D1 Val/Test, BDD100K Locked, FLIR | 26.6 FPS | **READY_FOR_EXECUTION** |
| **E2** | YOLOv8n (D1 Fine-Tuned) | D1 Train (1,200 img, standard setup) | Raw Detector (640px) | Adapting feature heads to synthetic adverse weather improves small-object recall on BDD100K by $\ge +0.03\text{ }F_1$. | D1 Val/Test, BDD100K Locked, FLIR | 26.6 FPS | **PENDING_DATA_INGESTION** |
| **E3** | YOLOv8n (Pretrained) | None (Frozen baseline) | Policy J ($K=2$) | Static visual salience discovers distant objects but suffers candidate dilution on synthetic textures. | D1 Val/Test, BDD100K Locked | 16.1 FPS | **READY_FOR_EXECUTION** |
| **E4** | YOLOv8n (D1 Fine-Tuned) | D1 Train (from E2) | Policy J ($K=2$) | Fine-tuned weights improve secondary crop classification precision, suppressing false positives. | D1 Val/Test, BDD100K Locked | 16.1 FPS | **PENDING_E2_TRAINING** |
| **E5** | YOLOv8n (Pretrained) | None (Frozen baseline) | Policy K ($K=2$, M15) | Temporal evidence and horizon anchoring maintain reference video benchmark performance. | BDD100K Locked (Video) | 9.1 FPS | **READY_FOR_EXECUTION** |
| **E6** | YOLOv8n (D1 Fine-Tuned) | D1 Train (from E2) | Policy K ($K=2$, M15) | Combining synthetic contrast heads with temporal-horizon routing eliminates TF-12 crop misses. | BDD100K Locked (Video) | 9.1 FPS | **PENDING_E2_TRAINING** |
| **E7** | YOLOv8n (D1 Small-Oversample) | D1 Train (2.0x small-object weight) | Raw Detector (640px) | Biased sampling counteracts natural size imbalance, boosting small-object $F_1$ by $+0.04$. | D1 Val/Test, BDD100K Locked | 26.6 FPS | **PENDING_DATA_INGESTION** |
| **E8** | YOLOv8n (Adverse-Aug) | D1 Train (Targeted blur/contrast) | Raw Detector (640px) | Photometric atmospheric simulation narrows the synthetic-to-real domain gap. | D1 Val/Test, BDD100K Locked | 26.6 FPS | **PENDING_DATA_INGESTION** |
| **E9a** | YOLOv8n (Day-Only) | D1 Day Subset (360 img) | Raw Detector (640px) | Day-only synthetic training exhibits negative transfer when tested on real rain/fog. | D1 Test, BDD100K Subgroups | 26.6 FPS | **PENDING_DATA_INGESTION** |
| **E9b** | YOLOv8n (Adverse-Only) | D1 Rain/Night/Fog (840 img) | Raw Detector (640px) | Training exclusively on adverse scenes maximizes low-contrast feature extraction. | D1 Test, BDD100K Subgroups | 26.6 FPS | **PENDING_DATA_INGESTION** |
| **E10** | YOLOv8n (Condition-Balanced) | D1 Train (Equal batch sampling) | Raw Detector (640px) | Balanced condition batches prevent daylight loss gradients from dominating night/rain features. | D1 Val/Test, BDD100K Locked | 26.6 FPS | **PENDING_DATA_INGESTION** |
| **E11** | YOLOv8n (Layer-Freezing) | D1 Train (Freeze stages 0..4) | Raw Detector (640px) | Freezing backbone layers 0..10 reduces training time by $40\%$ while preserving low-level edge features. | D1 Val/Test, BDD100K Locked | 26.6 FPS | **PENDING_DATA_INGESTION** |
| **E12** | YOLOv8n (Synthetic + Real) | D1 Train (1,200) + BDD Train (1,200) | Raw Detector (640px) | Co-training regularizes diffusion artifacts while injecting synthetic small-object geometric priors. | D1 Test, BDD100K Locked, FLIR | 26.6 FPS | **PENDING_DATA_INGESTION** |

---

## 3. Disjoint Evaluation Cohorts & Scientific Boundaries

To preserve strict scientific honesty and prevent misleading aggregate claims:

1. **Synthetic Evaluation (D1 Internal Test, 400 Images):**
   - Assesses in-domain synthetic learning, fine-tuning convergence, and prompt-family memorization.
   - Evaluated exactly once; strictly held out during model selection.
2. **Real-World External Evaluation (BDD100K Locked Test, 1,214 Frames):**
   - The primary benchmark of synthetic-to-real transfer.
   - Preserves historical Mission 15 locked evaluation protocol (6 unseen sequences).
3. **Cross-Spectral Thermal Evaluation (FLIR ADAS, 200 Images):**
   - Tests whether synthetic RGB fine-tuning damages or preserves thermal infrared transfer.
4. **Non-Equivalence Rule:** Performance on D1 synthetic test data must **never be averaged with or substituted for real-world driving results**. All benchmark tables must report synthetic and real evaluations in separate, clearly labeled columns.
