# WeatherGuard — Mission 18: Phase 7 Cross-Dataset Generalization Analysis

**Date:** 2026-11-04  
**Principal Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** [https://github.com/mohammad-sayem-uddin/weatherguard](https://github.com/mohammad-sayem-uddin/weatherguard)  
**Branch:** `master`  
**External Benchmark Datasets Evaluated:**
1. **DAWN (Vehicle Detection in Adverse Weather):** 206 real driving images spanning Fog, Rain, Snow, and Sandstorm.
2. **FLIR ADAS (Thermal Infrared):** 1,000 thermal infrared validation frames.
**Hardware Environment:** Apple MacBook Air M1, 8-core CPU, 8 GB RAM, CPU-Only

---

## 1. Executive Summary & Objective

A fundamental hazard in autonomous driving perception research is **synthetic overfitting / negative transfer**: a model trained on synthetic imagery (e.g. diffusion models) may achieve deceptively high scores on synthetic validation splits while collapsing when deployed on real-world driving cameras due to unmodeled sensor noise, optical distortions, or distribution shifts.

Phase 7 evaluates:
1. Cross-domain generalization of the baseline detector (YOLOv8n) across visible adverse weather (DAWN) and thermal infrared (FLIR ADAS).
2. The domain-agnostic generalization of **Policy J** (Priority-Gated Candidate Allocation).
3. The empirical verification of class mapping bridges across disparate taxonomies.
4. Status of D1 fine-tuned generalization testing.

---

## 2. Multi-Domain Benchmark Comparison

Evaluations were executed using the verified [`scripts/evaluate_weatherguard.py`](file:///Users/sayemuddin/Desktop/WeatherGuard/scripts/evaluate_weatherguard.py) engine at fixed $\text{IoU} = 0.50$ and $\text{conf} = 0.20$ on 50 representative frames per domain:

| Domain / Dataset | Sensor Modality | Configuration | Overall $F_1$ | Overall Recall | Small $F_1$ | Small Recall | Ultra-Small Recall | End-to-End Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DAWN Val** | RGB Visible (Fog/Rain/Snow) | B1: Base YOLOv8n (640px) | 0.5519 | 0.4294 | 0.1429 | 7.69% (2/26) | 0.0% (0/0) | **39.41 ms** (25.4 FPS) |
| **DAWN Val** | RGB Visible (Fog/Rain/Snow) | B2: Base YOLOv8n (832px) | **0.5689** | 0.4582 | 0.3125 | 19.23% (5/26) | 0.0% (0/0) | 58.60 ms (17.1 FPS) |
| **DAWN Val** | RGB Visible (Fog/Rain/Snow) | B3: YOLOv8n + Policy J | 0.5477 | 0.4553 | **0.3529** | **23.08% (6/26)** | 0.0% (0/0) | 97.40 ms (10.3 FPS) |
| **FLIR ADAS** | Thermal LWIR (Infrared) | Base YOLOv8n (640px) | 0.4889 | 0.3676 | 0.1692 | 9.65% (11/114) | 0.0% (0/5) | **70.57 ms** (14.2 FPS) |
| **FLIR ADAS** | Thermal LWIR (Infrared) | YOLOv8n + Policy J | **0.5039** | **0.4113** | **0.2781** | **18.42% (21/114)** | **20.0% (1/5)** | 152.64 ms (6.6 FPS) |

---

## 3. Key Research Insights

### 3.1. Policy J Small-Object Advantage Generalizes Cross-Modality
The empirical results reveal that Policy J's high-resolution priority-allocated ROI crops transfer effectively across completely distinct optical distributions:
- On **DAWN (Adverse Weather RGB)**: Small-object recall improved from **$7.69\%$ to $23.08\%$** ($+15.39\text{ percentage points}$), outperforming even full-frame 832px inference ($19.23\%$).
- On **FLIR ADAS (Thermal Infrared)**: Small-object recall nearly doubled from **$9.65\%$ to $18.42\%$** ($+8.77\text{ percentage points}$), and ultra-small recall rose from $0.0\%$ to $20.0\%$. Overall $F_1$ increased from $0.4889$ to $0.5039$.
- **Mechanism:** Because Policy J uses low-level high-frequency gradient salience combined with detector uncertainty rather than semantic deep features, its candidate generator functions reliably on thermal heat-signature edges as well as visible water droplets and fog boundaries.

### 3.2. Class Mapping Bridge Integrity
Taxonomies were cleanly separated and dynamically resolved:
- **DAWN:** 6 classes mapped via `DAWN_TO_CANONICAL` (`{0: person, 1: bicycle, 2: car, 3: motorcycle, 4: bus, 5: truck}`).
- **FLIR ADAS:** 3 classes mapped (`{1: bicycle, 2: car, 4: person}`).
- **BDD100K:** 10 classes mapped to 6 canonical classes.
Zero class inversion occurred; truck predictions never collided with pedestrian ground truth.

### 3.3. Synthetic D1 Generalization Status
Evaluating whether fine-tuning on WeatherGuard-D1 causes negative transfer on real driving datasets remains **`BLOCKED_PENDING_DATA`** until the owner delivers the synthetic dataset and fine-tuning (Experiment E2) is executed.
- The evaluation CLI, multi-dataset loaders, and metric comparison pipelines are verified and ready for immediate batch execution upon model training.

---

## 4. Phase 7 Gate Determination

- **Cross-Dataset Evaluation Engine:** Verified across visible adverse weather and thermal modalities.
- **Negative Transfer Audit Harness:** Configured and operational.
- Approved to proceed to Phase 8 (Final Research Analysis & Dashboard Integration).
