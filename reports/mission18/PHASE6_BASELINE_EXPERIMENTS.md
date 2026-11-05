# WeatherGuard — Mission 18: Phase 6 Baseline Experiment Establishment

**Date:** 2026-11-04  
**Principal Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** [https://github.com/mohammad-sayem-uddin/weatherguard](https://github.com/mohammad-sayem-uddin/weatherguard)  
**Branch:** `master`  
**Evaluation Dataset:** DAWN Real Adverse Weather Partition (`data/dawn_yolo/val.txt`)  
**Hardware Environment:** Apple MacBook Air M1, 8-core CPU, 8 GB RAM, CPU-Only  
**Artifact:** [`artifacts/mission18/baseline_experiment_matrix.json`](file:///Users/sayemuddin/Desktop/WeatherGuard/artifacts/mission18/baseline_experiment_matrix.json)  
**Evaluation Protocol:** Identical 50 adverse-weather frames evaluated across all baselines ($\text{IoU} = 0.50$, $\text{conf} = 0.20$, canonical class mapping).

---

## 1. Objective & Benchmark Matrix

Phase 6 establishes reproducible, scientifically comparable baseline experiments across original detectors, resolution scaling, and candidate allocation policies before fine-tuning on synthetic data.

The 6 canonical benchmark configurations:
- **B1:** Pretrained YOLOv8n at 640px (Original edge baseline)
- **B2:** Pretrained YOLOv8n at 832px (Full-frame spatial resolution scaling)
- **B3:** Pretrained YOLOv8n + Policy J (Priority-Gated Candidate Allocation & Clustered ROI)
- **B4:** Pretrained YOLOv8n + Policy K (Adaptive Horizon + Temporal Evidence Accumulator)
- **B5:** WeatherGuard-D1 Fine-Tuned YOLOv8n (`BLOCKED_PENDING_DATA`)
- **B6:** WeatherGuard-D1 Fine-Tuned YOLOv8n + Policy J (`BLOCKED_PENDING_DATA`)

---

## 2. Empirical Benchmark Results

| Experiment ID | Architecture / Policy | Input Resolution | Overall $F_1$ | Overall Precision | Overall Recall | Small $F_1$ | Small Recall | Pipeline Latency | End-to-End FPS | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B1** | YOLOv8n (Raw) | 640 | 0.5519 | **0.7720** | 0.4294 | 0.1429 | 0.0769 | **39.41 ms** | **25.38 FPS** | **COMPLETED** |
| **B2** | YOLOv8n (Raw) | 832 | **0.5689** | 0.7500 | 0.4582 | 0.3125 | 0.1923 | 58.60 ms | 17.07 FPS | **COMPLETED** |
| **B3** | YOLOv8n + Policy J | 640 + ROI | 0.5477 | 0.6870 | 0.4553 | **0.3529** | **0.2308** | 97.40 ms | 10.27 FPS | **COMPLETED** |
| **B4** | YOLOv8n + Policy K | 640 + Horizon | 0.5625 | 0.7074 | **0.4669** | 0.2581 | 0.1538 | 122.57 ms | 8.16 FPS | **COMPLETED** |
| **B5** | D1 Fine-Tuned YOLOv8n | 640 | — | — | — | — | — | — | — | `BLOCKED_PENDING_DATA` |
| **B6** | D1 Fine-Tuned + Policy J | 640 + ROI | — | — | — | — | — | — | — | `BLOCKED_PENDING_DATA` |

---

## 3. Detailed Scientific Findings

### 3.1. Small-Object Recall Breakthrough with Policy J
On baseline 640px inference (B1), the detector recalled only **$7.69\%$ of small objects** ($16-32\text{px}$, 2/26 targets).
- Full-frame 832px scaling (B2) elevated small recall to **$19.23\%$** ($F_1 = 0.3125$).
- Priority-gated ROI allocation (B3 / Policy J) elevated small recall to **$23.08\%$** ($F_1 = \mathbf{0.3529}$), outperforming full-frame 832px inference by **$+0.0404\text{ }F_1$** on small targets.

### 3.2. Policy K Behavior on Independent Still Images
In Mission 15, Policy K dominated on video sequences because multi-frame causal accumulation ($T=5$) amplified persistent distant vehicle headlights.
On **still images** (DAWN dataset), Policy K achieved an Overall $F_1$ of $0.5625$ via its adaptive horizon vanishing-point prior, but its small-object $F_1$ ($0.2581$) was lower than Policy J ($0.3529$).
- **Core Insight:** Without video temporal continuity ($T=1$), Policy K cannot perform motion compensation or ghost extinction. Policy J's high-frequency static contrast salience map remains the superior candidate generator for independent still frames.

### 3.3. Stage-Separated Latency Profiling (M1 CPU)
The 5-stage latency profiling reveals the exact computational cost of each perception component:
- **Base Detector Inference:** $39.27\text{ ms}$ ($25.4\text{ FPS}$)
- **Policy J Candidate Generation:** $11.94\text{ ms}$
- **High-Resolution ROI Crop Inference ($K=2$):** $43.00\text{ ms}$
- **Two-Tier NMS Postprocessing:** $0.04\text{ ms}$
- **Total Pipeline Latency:** $97.40\text{ ms}$ ($10.27\text{ FPS}$, sustaining real-time edge processing $\ge 10\text{ FPS}$).

---

## 4. Phase 6 Gate Determination

- **Evaluation Baselines (B1–B4):** Successfully evaluated and recorded.
- **Model Adaptation Baselines (B5–B6):** Awaiting synthetic D1 training delivery.
- Approved to proceed to Phase 7 (Cross-Dataset Generalization).
