# WeatherGuard — Initial Repository Audit & Mission 16 Architecture Baseline
**Document:** `reports/mission16_preparation/INITIAL_REPOSITORY_AUDIT.md`  
**Date:** 2026-11-04  
**Author:** Mohammad Sayem Uddin (<sayemwork41@gmail.com>)  
**Execution Environment:** Apple MacBook Air M1, 8-core CPU (4 Firestorm + 4 Icestorm), 8 GB Unified Memory, macOS Darwin 24.3.0, CPU-only.  
**Repository:** https://github.com/mohammad-sayem-uddin/weatherguard

---

## 1. Executive Summary

This audit establishes the comprehensive software, algorithmic, data, and experimental state of WeatherGuard prior to initiating the dataset engineering and research infrastructure for **WeatherGuard-D1** and **Mission 16**.

WeatherGuard has progressed through 15 research missions, advancing from single-image baseline evaluation through scale-aware routing, feature reuse, temporal video object detection, region-adaptive high-resolution inference, priority-gated candidate allocation (Policy J), to causal temporal evidence accumulation and horizon-anchored proposals (Policy K).

---

## 2. Repository Structure & Codebase Inventory

### 2.1 File Tree Summary
```
WeatherGuard/
├── app.py                         # Streamlit interactive analysis dashboard (Pages 1–28)
├── configs/                       # Configuration files for historical experiments
├── data/                          # Dataset stores (BDD100K, DAWN, FLIR ADAS, Mission cohorts)
├── models/                        # Model weights and torchscript exports (git-ignored)
├── artifacts/                     # Machine-readable evaluation JSON/CSV and plots
│   ├── mission14/                 # Mission 14 validation and taxonomy artifacts
│   ├── mission15/                 # Mission 15 ablation, locked eval, and timing artifacts
│   └── plots/                     # Global plot repository for publication and dashboard
├── reports/                       # Formal scientific markdown reports (Missions 1–15)
│   ├── mission14/                 # Mission 14 audit, validation, and failure reports
│   ├── mission15/                 # Mission 15 design, ablation, benchmark, and final reports
│   └── mission16_preparation/     # Mission 16 preparatory audits and protocol designs
├── scripts/                       # 100+ executable evaluation, benchmark, and research scripts
├── src/                           # Core research library packages
│   ├── candidate_generators.py    # Detector-independent proposal extractors (Salience, Flow)
│   ├── priority_allocation.py     # Policy J priority allocation and crop controller
│   └── mission15/                 # Policy K edge architecture
│       ├── temporal_evidence.py   # Causal temporal evidence accumulator
│       ├── horizon_anchor.py      # Adaptive vanishing-point and horizon anchor
│       └── policy_k.py            # Policy K priority allocation controller
├── tests/                         # Comprehensive automated test suite (212 unit/integration tests)
└── requirements.txt               # Pinned Python package dependencies
```

---

## 3. Existing Model Architectures & Inference Policies

1. **Static Detectors:**
   - **YOLOv8n (640×640):** Fast edge baseline ($37.56\text{ ms}$ on M1 CPU, $26.63\text{ FPS}$). High precision ($0.6654$ on locked test), but severely deficient on small objects ($F_1 = 0.2564$) and ultra-small objects ($F_1 = 0.0112$, $0.56\%$ recall).
   - **YOLOv8n (832×832):** Full-frame high-resolution baseline ($67.59\text{ ms}$, $14.80\text{ FPS}$). Elevates small-object $F_1$ to $0.3469$ and overall $F_1$ to $0.5347$, but incurs a $1.8\times$ latency penalty.
   - **Faster R-CNN (ResNet-50-FPN):** Evaluated in Missions 1–3; prohibitive CPU latency on M1 ($> 280\text{ ms}$/frame), establishing YOLOv8n as the edge standard.

2. **Adaptive Inference Policies:**
   - **Policies A–H (Missions 4–10):** Scale-aware routing, hold-time smoothing ($H=3$), confidence-gated cascades, and low-confidence ByteTrack association.
   - **Policy J (Missions 13–14):** Priority-gated candidate allocation combining single-frame visual salience with low-confidence detector proposals under capacity reservation. Achieves overall $F_1 = 0.5269$ and small-object $F_1 = 0.4141$, but suffers from candidate dilution ($40.61\%$ ultra-small precision, $5,115$ false positives) due to unconstrained salience scatter.
   - **Policy K (Mission 15):** Causal temporal evidence accumulation ($T=5$, $\alpha=0.75$, $\lambda_{ghost}=0.35$) with adaptive vanishing-point horizon anchoring ($\beta=0.85$, road trapezoid $220-480\text{px}$) and 4-tier priority gating. Achieves Overall $F_1 = 0.5394$ ($p = 0.028$ vs Policy J) and Small $F_1 = 0.4162$, lifting ultra-small precision to $48.54\%$ ($p = 0.004$) at $110.37\text{ ms}$/frame ($9.06\text{ FPS}$).

---

## 4. Dataset Inventory & Benchmark Protocols

1. **BDD100K Multi-Weather (100k Still Images & Video Clips):**
   - 10-class autonomous driving dataset; WeatherGuard maps classes into 6 canonical evaluation classes: `car`, `truck`, `bus`, `person`, `motorcycle`, `bicycle`.
   - 22 historical driving sequences utilized in Missions 9–14 are strictly cataloged in `artifacts/mission15/sequence_exclusion_manifest.json` to prevent contamination.
   - 10 virgin sequences (2,025 frames, 21,023 annotations) established in Mission 15: Development (2 seqs), Validation (2 seqs), Locked Test (6 seqs).
2. **DAWN (Vehicle Detection in Adverse Weather):**
   - 1,000 real images spanning Fog, Snow, Rain, Sandstorm. Used for image-level adverse-weather benchmarking.
3. **FLIR ADAS Thermal (Infrared Driving Corpus):**
   - Evaluated in Missions 11–14 for cross-spectral domain transfer.
4. **Statistical Protocols:**
   - Circular block bootstrap ($B=1,000$) for image datasets with spatial/temporal dependencies.
   - Paired sequence-level bootstrap ($B=1,000$) across independent driving sequences.
   - Object-scale categorization convention:
     - Ultra-small: $\text{area} < 256\text{ px}^2$ ($\max(w,h) < 16\text{px}$).
     - Small: $256 \le \text{area} < 1024\text{ px}^2$ ($16 \le \max(w,h) < 32\text{px}$).
     - Medium: $1024 \le \text{area} < 9216\text{ px}^2$ ($32 \le \max(w,h) < 96\text{px}$).
     - Large: $\text{area} \ge 9216\text{ px}^2$ ($\max(w,h) \ge 96\text{px}$).

---

## 5. Known Historical Inconsistencies & Resolutions

1. **FLIR Small-Object $F_1$ Reporting:** Reconciled in Mission 15 Phase 0; YOLO-832 holds peak Small $F_1$ ($0.1254$) on FLIR thermal, whereas Policy J holds peak overall $F_1$ ($0.3185$) and peak precision ($0.3370$).
2. **Failure Taxonomy Double-Counting:** Historical heuristic classifiers allowed multi-label attribution. Resolved by establishing a mutually exclusive primary-cause taxonomy with exact integer accounting ($3,778$ total misses in Mission 15 matched to machine records).
3. **Statistical Sample Size Bounds:** Inferences drawn on $N=6$ sequences cannot claim unbounded domain invariance. All claims are explicitly bounded by empirical 95% bootstrap confidence intervals.

---

## 6. Technical Limitations & Hardware Constraints

1. **Compute Environment:** Apple MacBook Air M1, 8 GB Unified Memory, strictly CPU-only FP32 execution. No CUDA GPUs or Apple Neural Engine (ANE) compilation utilized in baseline releases to maintain platform-agnostic reproducibility.
2. **Crop Budget Ceiling ($K=2$):** Latency constraints restrict high-resolution inference to $K=2$ crops per frame ($110.37\text{ ms}$). Expanding to $K=3$ incurs latency degradation ($156.61\text{ ms}$, $6.39\text{ FPS}$) and precision loss.
3. **Physical Contrast Scattering:** Extreme adverse weather (heavy blizzard, tire spray) extinguishes target contrast below sensor SNR floors, where algorithmic accumulation cannot reconstruct absent edges.

---

## 7. Current Test Coverage & Repository Hygiene

- **Automated Test Suite:** 212 passed tests across all test modules (`tests/test_*.py`) in $17.73\text{ seconds}$.
- **Artifact Verification:** 109 artifacts validated via `scripts/validate_artifacts.py`.
- **Git Hygiene:** No raw datasets (`.parquet`, `.jpg`, `.mp4`) or weights (`.pt`) tracked. `.gitignore` strictly protects repository boundaries.

---

## 8. Proposed Mission 16 Development Plan

The owner is independently synthesizing **WeatherGuard-D1** (2,000 synthetic photorealistic images generated via Google Flow with AI annotations generated via Gemini Flash).

To prepare the repository without waiting passively, the following infrastructure will be engineered:
1. **D1 Directory & Configuration Specification:** `configs/weatherguard_d1.yaml`.
2. **Rigorous Ingestion Pipeline:** `scripts/ingest_weatherguard_d1.py` with perceptual hashing and resolution validation.
3. **Independent Annotation Validator:** `scripts/validate_d1_annotations.py` to audit Gemini Flash coordinate conventions, bounding box validity, and object-size calculations without blind trust.
4. **Quality Control & Uncertainty Review:** `scripts/extract_uncertain_annotations.py` and `scripts/analyze_weatherguard_d1.py`.
5. **Leakage-Safe Stratified Splitting:** `scripts/create_d1_splits.py` (1,200 train, 400 val, 400 test) grouping related generation prompt families to prevent near-duplicate leakage.
6. **Configurable Training Pipeline:** `scripts/train_weatherguard_d1.py` with small-object aware sampling and M1-friendly resource limits.
7. **Unified Model Evaluation Framework:** `scripts/evaluate_weatherguard.py`.
8. **Mission 16 Experiment Registry & Matrix (E1–E12):** Standardized experiment configurations.
9. **Streamlit Dashboard Integration:** Page 29 providing full D1 readiness and annotation QC views.
10. **Research Documentation & Paper Preparation:** Formal protocols, methodology, and reproducibility checklists.
