# MISSION 22 — PHASE 7: Locked Evaluation Protocol for Synchronized Multi-Modal Benchmarks

**Principal Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** `https://github.com/mohammad-sayem-uddin/weatherguard`  
**Branch:** `master`  
**Specification:** `artifacts/mission22/locked_evaluation_design.json`  
**Date:** November 5, 2026  

---

## 1. Executive Directive & Scope

This protocol establishes the rigorous, leak-free evaluation methodology that must govern all future benchmarks on physically synchronized audio-visual search-and-rescue data. It directly eliminates the data leakage, speaker reuse, and phantom-person scoring flaws identified during the Mission 21 forensic audit.

---

## 2. Partitioning Standards: Leak-Free Clustering

### 2.1 Speaker & Session Clustering (Zero Overlap Rule)
- **Speaker Isolation:** Every human participant and synthetic voice engine is assigned a unique identifier. If an individual participated in multiple recording sessions, all recordings from that speaker must reside in **exactly one split**.
  $$\text{Speakers}_{\text{train}} \cap \text{Speakers}_{\text{val}} = \emptyset, \quad \text{Speakers}_{\text{train}} \cap \text{Speakers}_{\text{test}} = \emptyset, \quad \text{Speakers}_{\text{val}} \cap \text{Speakers}_{\text{test}} = \emptyset$$
- **Flight-Sequence Isolation:** Video footage is partitioned by distinct flight sorties and geographical sites. Zero visual sequence leakage is permitted across splits.
- **Minimum Test Pool Requirements:**
  - Minimum **30 independent, unseen human distress speakers** in the locked test split.
  - Minimum **20 independent flight sequences** across varied terrain.
  - Minimum **1,000 synchronized test evaluation pairs**.

### 2.2 Split Allocation & Access Control
- **Training Set (60%):** Model weights and feature representations trained here.
- **Validation / Calibration Set (20%):** Hyperparameter tuning, Platt scaling calibration, and operational threshold selection ($\tau_{\text{val}}^*$).
- **Locked Test Set (20%):** **Hermetically sealed.** Evaluated exactly once after models and thresholds are frozen. Zero threshold sweeping or tuning on test data is permitted.

---

## 3. Disaggregated Evaluation Tasks

To avoid penalizing single-modality detectors on targets that physically do not exist in their modality, every synchronized evaluation set must be evaluated under four distinct, disaggregated tasks:

```
[ Synchronized Audiovisual Test Pair ] ---> (x_vis, x_aud)
                     |
       +-------------+-------------+-------------+
       |             |             |             |
       v             v             v             v
   [ Task 1 ]    [ Task 2 ]    [ Task 3 ]    [ Task 4 ]
    Bimodal       Rescue        Optical       Acoustic
  Corroboration   Trigger       Tracking       Alert
   (vis & aud)  (vis | aud)   (vis only)    (aud only)
```

1. **Task 1: Bimodal Corroboration:**  
   Ground truth is positive **only if BOTH** a visible person and distress vocalization co-occur. Evaluates multi-modal precision and immunity to false alarms.
2. **Task 2: Complementary Rescue Trigger:**  
   Ground truth is positive if **EITHER** a visible victim OR an acoustic distress cry is present. Evaluates comprehensive search coverage.
3. **Task 3: Optical Person Detection:**  
   Ground truth is positive if a person is visible in the frame, regardless of acoustic silence or distress. Evaluates the optical detector's true recall without phantom penalties.
4. **Task 4: Acoustic Distress Detection:**  
   Ground truth is positive if emergency cries occur, regardless of camera occlusion. Evaluates keyword spotting under acoustic propeller wash.

---

## 4. Benchmark Paradigms Evaluated on Identical Test Inputs

Every test trial must feed the exact same synchronized input pair $(x_{\text{vis}}, x_{\text{aud}})$ to all six paradigms:
1. **Vision-Only Baseline:** Single-frame YOLOv8n / RT-DETR.
2. **Audio-Only Baseline:** 2D Spectrogram CNN at validation-calibrated threshold $\tau_{\text{val}}^*$.
3. **Simple Heuristic OR:** Issues alert if $s_{\text{vis}} \ge \tau_{\text{vis}}$ or $s_{\text{aud}} \ge \tau_{\text{aud}}$.
4. **Simple Heuristic AND:** Issues alert only if $s_{\text{vis}} \ge \tau_{\text{vis}}$ and $s_{\text{aud}} \ge \tau_{\text{aud}}$.
5. **Confidence-Weighted Average:** Linear combination of sensor scores.
6. **Adaptive Uncertainty-Aware Fusion Engine:** Proposed dynamic reliability-weighted architecture with explicit abstention.

---

## 5. Statistical Inference & Reporting Standards

1. **Primary Test:** McNemar's exact paired test on matched trial predictions.
2. **Multiple Comparisons:** Holm-Bonferroni step-down correction at FWER $\alpha = 0.05$.
3. **Hierarchical Resampling:** 5,000-iteration cluster bootstrap resampling at the flight-sequence and speaker levels.
4. **Effect Sizes:** Reported as Odds Ratios, Cohen's $g$, and percentile confidence intervals for $\Delta F_1$.
5. **Prohibited:** Never report bootstrap z-ratios as Cohen's $d$, and never claim $p < 0.001$ without exact test computation.

Phase 7 is complete.
