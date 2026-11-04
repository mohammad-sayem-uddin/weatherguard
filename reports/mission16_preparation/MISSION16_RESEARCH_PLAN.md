# WeatherGuard — Mission 16 Research Plan & Methodology
## Synthetic-to-Real Domain Generalization, Small-Object Adaptation & Edge Inference

**Project:** WeatherGuard  
**Mission:** Mission 16 — Photorealistic Synthetic Data Fine-Tuning & Real-World Domain Transfer  
**Principal Investigator:** Mohammad Sayem Uddin (<sayemwork41@gmail.com>)  
**Date:** 2026-11-04  
**Hardware Environment:** Apple MacBook Air M1, 8-core CPU (4 Firestorm + 4 Icestorm), 8 GB Unified Memory, macOS Darwin 24.3.0, CPU-only.

---

## 1. Scientific Context & Motivation

Missions 11–15 established that selective high-resolution inference (Policy J, Policy K) dramatically improves small and ultra-small object detection under adverse weather while operating within edge CPU budgets. In Mission 15, Policy K achieved an Overall $F_1$ of **$0.5394$** and Small $F_1$ of **$0.4162$** on locked driving video.

However, Mission 15's diagnostic failure taxonomy revealed a critical insight:
- **TF-12 Crop Localization/Classification Misses (17.58% of misses):** Even when a high-resolution crop was placed squarely on a distant vehicle candidate, the underlying YOLOv8n detector frequently failed to detect the object due to physical contrast extinction and low adverse-weather training representation.

Real-world adverse-weather driving data is notoriously scarce, imbalanced, and expensive to annotate. Synthetic image generation (via Google Flow) offers the potential to synthesize thousands of photorealistic adverse-weather scenes (Night, Rain, Fog) enriched with distant road users.

---

## 2. Research Questions & Hypotheses

### Primary Research Question
> **Can photorealistic synthetic adverse-weather dashcam data improve small and distant road-user detection, and can these improvements transfer to real-world driving datasets without unacceptable degradation in generalization or CPU inference efficiency?**

### Secondary Research Questions
- **RQ1 (In-Domain Learning):** Does fine-tuning on WeatherGuard-D1 improve detection of distant vehicles ($< 32\text{px}$) on held-out synthetic test images?
- **RQ2 (Cross-Domain Transfer):** Do representations learned from synthetic diffusion data transfer positively to real-world BDD100K adverse-weather driving video, or does synthetic domain shift induce negative transfer?
- **RQ3 (Policy Interaction):** How does synthetic fine-tuning interact with edge candidate-generation policies (Policy J and Policy K)? Does fine-tuning resolve TF-12 crop classification failures?
- **RQ4 (Augmentation & Sampling):** Can small-object-aware oversampling (E7) or targeted photometric blur/contrast simulation (E8) narrow the synthetic-to-real domain gap?
- **RQ5 (Computational Overhead):** Does fine-tuning alter CPU forward-pass latency or memory footprint on the Apple M1 platform?

---

## 3. Strict Research Boundaries: Temporal Video vs Independent Still Images

A fundamental scientific principle of WeatherGuard is **never claiming temporal performance from independent still images**:

1. **WeatherGuard-D1 Nature:** WeatherGuard-D1 consists of **2,000 independently generated synthetic images**. They are not temporally contiguous video frames.
2. **Permitted Research Uses of D1:**
   - Image-level detection fine-tuning.
   - Weather condition contrast adaptation (Day, Night, Rain, Fog).
   - Small-object feature head calibration.
   - Static candidate generation (visual salience, horizon priors).
   - Synthetic-to-real transfer analysis.
3. **Prohibited Claims from D1:**
   - Multi-frame temporal evidence accumulation cannot be trained or evaluated on D1.
   - Tracking continuity, trajectory association, and temporal flicker cannot be evaluated on D1.
4. **Video Benchmarking Protocol:** Policy K evaluation remains strictly evaluated on continuous BDD100K driving video sequences (M15 locked cohort), never on stitched synthetic stills.

---

## 4. Phased Execution Roadmap

```
+-----------------------------------------------------------------------------------+
|                           MISSION 16 EXECUTION PIPELINE                           |
+-----------------------------------------------------------------------------------+
| Phase 1: Ingestion & Integrity Verification (scripts/ingest_weatherguard_d1.py)   |
|   - 1280x720 dimension enforcement, sha256/dhash, duplicate/near-duplicate audit |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 2: Independent Annotation Validation (scripts/validate_d1_annotations.py)   |
|   - Gemini Flash [0..1000] to YOLO format, object-size max(w,h) recalculation      |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 3: Uncertainty Triage & Quality Control (scripts/extract_uncertain_annot.py)|
|   - Accepted Structural vs Needs Visual Review vs Rejected queues                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 4: Leakage-Safe Stratified Splitting (scripts/create_d1_splits.py)          |
|   - Train (1,200), Val (400), Test (400); prompt family atomic isolation           |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 5: Controlled Model Fine-Tuning (scripts/train_weatherguard_d1.py)          |
|   - Execute Experiments E2, E7, E8, E9, E10, E11, E12 on Apple M1 CPU             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 6: Unified Multi-Domain Evaluation (scripts/evaluate_weatherguard.py)       |
|   - Evaluate D1 Test, BDD100K Locked Video, FLIR Thermal; Paired Bootstrap        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Phase 7: Synthesis, Dashboard (Page 29), and Publication Delivery                 |
+-----------------------------------------------------------------------------------+
```

---

## 5. Statistical Rigor & Reporting Invariants

- **Paired Sequence-Level Bootstrap ($B=1,000$):** Used for all real-world video comparisons on BDD100K.
- **Circular Block Bootstrap ($B=1,000$):** Used for D1 test evaluations to account for scenario family clustering.
- **Reporting Separation:** Synthetic internal test scores and real-world external test scores must appear in separate columns; never average synthetic and real scores into a single headline metric.
- **Zero Historical Overwrite:** Historical benchmark numbers from Missions 1–15 are frozen permanently.
