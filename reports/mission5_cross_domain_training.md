# WeatherGuard Mission 5: Experiment C — Cross-Domain Training Audit

**Document:** `reports/mission5_cross_domain_training.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Preregistered & Formally Documented (`BLOCKED_UNAVAILABLE`)  

---

## 1. Research Hypothesis & Objective

Hypothesis H3 posited that joint supervised training across in-domain adverse weather (DAWN) and target-domain driving scenes (BDD100K) would reduce domain shift and maximize generalization across both datasets.

---

## 2. Technical Audit & Formal Block Justification

In accordance with the WeatherGuard project charter and scientific integrity guidelines, Experiment C has been formally designated as **`BLOCKED_UNAVAILABLE`** based on three non-negotiable considerations:

### 2.1 Storage & Hardware Budget Constraints
- **Available Local Disk Storage:** Approximately 17.8 GB.
- **BDD100K Full Training Set Archive:** ~18.0 GB compressed archive, expanding to >50 GB uncompressed for 100,000 driving frames.
- **Resource Rule Violation:** Non-negotiable constraint 0 states: *'Avoid unnecessarily downloading datasets or weights. Limited available disk storage.'* Downloading the full BDD100K training corpus would exhaust local disk resources and overwhelm the 8 GB unified memory on Apple M1 CPU execution.

### 2.2 Data Integrity & Leakage Prevention
- The local BDD100K cohort consists strictly of:
  - **Group B Development ($N=160$):** Used solely for validation ranking and threshold calibration.
  - **Group C Locked Test ($N=203$):** Strictly locked for frozen final evaluation.
- Training on either Group B or Group C would constitute **fatal target-domain data leakage**, completely invalidating the zero-shot cross-dataset generalization claims established across Missions 2–5.
- There are zero locally available, disjoint BDD100K training images (Group D count = 0).

### 2.3 Scientific Alternative: Synthetic Domain Adaptation
- Rather than relying on massive external data downloads, WeatherGuard employs **physics-based synthetic weather domain adaptation** (Experiment B).
- By simulating atmospheric scattering (Koschmieder's law), vertical road-depth gradients, directional precipitation streaks, and optical blur on DAWN source imagery, the model learns invariant semantic representations that transfer zero-shot to external driving scenes.

---

## 3. Summary

| Cohort | Image Count | Status | Role |
|---|---|---|---|
| **Group A (DAWN)** | 821 Train / 206 Val | Available | Source domain fine-tuning & validation |
| **Group B (BDD100K Dev)** | 160 Images | Available (Disjoint) | Target domain model selection |
| **Group C (BDD100K Test)** | 203 Images | Locked | Frozen external benchmark |
| **Group D (BDD100K Train)** | 0 Images Local | **BLOCKED_UNAVAILABLE** | Joint supervised cross-domain training |

The project rigorously maintains the isolation of Group B and Group C, ensuring that all cross-dataset evaluations remain strictly zero-shot.
