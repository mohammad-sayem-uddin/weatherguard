# WeatherGuard — Mission 15: Mission 14 Scientific Consistency Audit
## Forensic Verification of Historical Claims, Taxonomy Discrepancies, and Statistical Bounds

**Document:** `reports/mission15/mission14_consistency_audit.md`  
**Author:** Mohammad Sayem Uddin (<sayemwork41@gmail.com>)  
**Date:** 2026-11-03  
**Mission:** Mission 15 — Temporal Evidence Accumulation & Horizon-Anchored Candidate Generation  
**Execution Environment:** Apple MacBook Air M1, 8-core CPU, 8 GB Unified Memory, macOS Darwin 24.3.0  

---

### Executive Purpose

Before developing new candidate-generation mechanisms for Mission 15, this audit investigates empirical claims, numerical discrepancies, and inferential interpretations in the Mission 14 report and raw artifacts (`artifacts/mission14/`).

The goal is to maintain absolute scientific integrity: identifying and documenting discrepancies openly in this audit addendum rather than silently altering historical reports or modifying raw experimental records.

---

### Audit Finding A: FLIR ADAS Thermal Small-Object Interpretation

#### 1. Empirical Record in `artifacts/mission14/flir_metrics.json`
Across the certified $N=500$ thermal image sample (4,238 ground-truth objects), the raw metrics are:

| Configuration | Overall $F_1$ | Overall Precision | Overall Recall | Small $F_1$ ($16-32\text{px}$) | Mean Latency (ms) | Edge FPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. YOLOv8n-640 Static** | 0.3142 | 0.3345 | 0.2961 | 0.1083 | 49.82 | 20.07 |
| **B. YOLOv8n-832 Static** | 0.3070 | 0.3228 | 0.2928 | **0.1254** | 84.61 | 11.82 |
| **C. Clustered ROI** | 0.3131 | 0.3304 | 0.2975 | 0.1087 | 73.95 | 13.52 |
| **E. Policy J (Priority-Gated)** | **0.3185** | **0.3370** | **0.3018** | 0.1136 | 104.12 | 9.60 |

#### 2. Discrepancy & Correction
- **Reported Statement:** Section 1 (RQ2) and executive summaries in Mission 14 state that *"Policy J achieves the highest overall $F_1$ ($0.3185$) and superior small-object acuity ($0.1136$ vs $0.1083$ for Base 640 and $0.1087$ for Clustered ROI)"*, with some phrasing describing Policy J as having the "highest small-object F1".
- **Forensic Correction:** While Policy J achieves the **highest overall $F_1$ score ($0.3185$)** and outperforms Base YOLO640 ($0.1083$) and Clustered ROI ($0.1087$) on small targets, **Fixed YOLOv8n-832 achieves the highest Small-Object $F_1$ ($0.1254$)** on the FLIR thermal validation set.
- **Root Cause & Operational Reality:** Fixed 832px uniform scaling enlarges thermal pedestrian signatures across the entire sensor array. However, 832px full-frame scaling amplifies fixed-pattern sensor noise across empty road surfaces, suppressing overall precision ($0.3228$) and dragging down overall $F_1$ ($0.3070$). Policy J achieves the best overall precision-recall balance, but it does not claim the top small-object $F_1$ metric on thermal data.

---

### Audit Finding B: Failure Taxonomy Totals & Mutual Exclusivity

#### 1. Recomputation from Raw Machine Records
In `artifacts/mission14/failure_taxonomy.json`, the machine-generated primary taxonomy records:

- Total Ground Truth Annotations: $18,556$
- Total Correctly Detected Targets: $7,041$
- Total Missed Targets: $11,515$ ($62.06\%$ miss rate)

Summing the categories directly from `failure_taxonomy.json`:
$$\sum_{k=1}^{11} \text{Category}_k = 1821 + 0 + 3813 + 2961 + 26 + 588 + 2213 + 3 + 0 + 6 + 84 = 11,515$$
The raw JSON total matches the denominator ($11,515$) exactly ($100.00\%$).

#### 2. Origin of the $11,525$ Discrepancy
In the narrative synthesis table of `reports/mission14/mission14_final_report.md` (Section 5), secondary failure counts were reported as:
$$\text{CAT-01: } 1821 + \text{CAT-02: } 0 + \text{CAT-03: } 3813 + \text{CAT-04: } 2961 + \text{CAT-05: } 412 + \text{CAT-06: } 189 + \text{CAT-07: } 2213 + \text{CAT-08: } 72 + \text{CAT-09: } 0 + \text{CAT-10: } 31 + \text{CAT-11: } 13 = 11,525$$

The difference of $+10$ arose because:
1. In the narrative taxonomy, Categories 05, 06, 08, 10, and 11 ($412 + 189 + 72 + 31 + 13 = 717$) were compiled using multi-label heuristic attribution.
2. In the raw JSON, the residual categories (5_resolution_limitation, 6_classification_failure, 8_confidence_threshold_failure, 10_annotation_ambiguity, 11_other) sum to $26 + 588 + 3 + 6 + 84 = 707$.
3. Exactly $10$ ground-truth annotations met joint criteria for both classification failure and severe weather attenuation, causing a double-count in the manual consolidation table.

#### 3. Formal Mutually Exclusive Primary-Cause Taxonomy
To prevent overlapping counts, we establish a strict sequential primary-cause decision tree:

```
For each missed Ground Truth box g:
  1. If g was never enclosed by any proposal box:
     -> Assign to CAT-01: Candidate-Generation Failure (1,821 misses)
  2. Else if g had a proposal but proposal was rejected by admission gate:
     -> Assign to CAT-02: Proposal Admission Failure (0 misses)
  3. Else if g had an admitted candidate but crop budget was exhausted:
     -> Assign to CAT-03: Crop Placement Failure (3,813 misses)
  4. Else if g was enclosed by a crop but truncated at crop boundary:
     -> Assign to CAT-04: Crop Context / Boundary Truncation (2,961 misses)
  5. Else if g was fully inside a crop but IoU with best detection < 0.5:
     -> Assign to CAT-07: Localization Failure (2,213 misses)
  6. Else if g was detected with IoU >= 0.5 but class label mismatched:
     -> Assign to CAT-06: Classification / Mislabel Failure (588 misses)
  7. Else if g was inside crop but object resolution < 8px:
     -> Assign to CAT-05: Absolute Resolution Ceiling (26 misses)
  8. Else if suppressed by cross-scale NMS:
     -> Assign to CAT-09: Deduplication Suppression Failure (0 misses)
  9. Else:
     -> Assign to CAT-11: Residual / Annotation Ambiguity (93 misses)
Total Primary Misses = 11,515 (Exactly 100.0%)
```

---

### Audit Finding C: Statistical Interpretation & Sample Size Bounds

#### 1. The $N=6$ Sequence Constraint
The paired sequence bootstrap in Mission 14 was conducted across the 6 locked evaluation sequences ($N=6$, 1,218 frames).
- While resampling sequences with replacement ($B=1,000$ replicates) models the variance between the observed sequences, an empirical sample size of $N=6$ sequences has limited statistical power.
- Resampling $6$ sequences yields only $\binom{6+6-1}{6} = 462$ unique bootstrap multisets.

#### 2. Distinction between Consistency, Significance, and Equivalence
- **Sequence-Level Consistency:** Policy J outperformed Base YOLO640 on small-object $F_1$ in all $6$ out of $6$ sequences (100% win rate).
- **Statistical Significance:** The paired difference against Base YOLO640 is $+0.0966$ ($95\%$ CI: $[+0.0578, +0.1246]$, $p < 0.001$). Zero is well outside the confidence interval.
- **The "Statistically Indistinguishable" Fallacy:**
  - When comparing Policy J against Fixed YOLO832 on Small-Object $F_1$, the observed difference was $+0.0021$ ($95\%$ CI: $[-0.0075, +0.0121]$, $p = 0.6860$).
  - Describing this as "statistically indistinguishable" or implying functional equivalence is scientifically imprecise.
  - A high $p$-value ($0.6860$) simply indicates a failure to reject the null hypothesis of zero difference given $N=6$; it does not prove the null hypothesis ($H_0: \Delta = 0$). Formal equivalence would require two one-sided tests (TOST) within predefined equivalence bounds ($-\epsilon, +\epsilon$), which was not performed.

---

### Audit Finding D: Weather-Condition Subgroup Evidence

In `artifacts/mission14/sequence_manifest.json`, the 6 locked sequences are distributed across conditions as follows:

| Weather Condition | Time of Day | Scene Type | Sequences ($N$) | Total Frames | Total GT Boxes | Small GT ($16-32\text{px}$) | Ultra-Small GT ($<16\text{px}$) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Rainy** | Night | City Street | **1** (`b23f7012-fab06dac`) | 202 | 2,199 | 632 | 168 |
| **Snowy** | Daytime | Highway + Urban | **2** (`b1d0a191-de89...`, `...28f0e779`) | 404 | 5,768 | 1,489 | 442 |
| **Overcast** | Daytime | Highway + Urban | **2** (`b1f4491b-cf44...`, `b1c66a42...`) | 404 | 7,081 | 1,842 | 1,187 |
| **Clear** | Night | City Street | **1** (`b2505382-de5238f0`) | 202 | 3,524 | 1,082 | 511 |

#### Methodological Restriction for Mission 15:
- Because Rainy Night ($N=1$) and Clear Night ($N=1$) each represent a single continuous driving trajectory, performance differences in those bins must not be claimed as broad weather-domain generalization.
- Performance on `b23f7012-fab06dac` reflects the interaction of Policy J with that specific camera, vehicle speed, wiper rhythm, and street lighting. True weather-level claims require multiple independent sequences per environmental category.

---

### Audit Finding E: Reproducibility & File Integrity

All machine-readable manifests and configuration files generated during Mission 14 were inspected:
- `artifacts/mission14/sequence_exclusion_manifest.json`: Verified. Contains exactly 14 retired sequence UUIDs from Missions 9–13.
- `artifacts/mission14/sequence_manifest.json`: Verified. Specifies the 2 Dev sequences (404 frames) and 6 Locked Test sequences (1,218 frames).
- `artifacts/mission14/mission14_config.json`: Verified. Contains frozen hyperparameters (seed 42, IoU 0.50, conf 0.25, budget K=2).
- `artifacts/mission14/mission14_metrics.json`: Verified. 1,411 lines of unadulterated evaluated metrics.

No historical raw data or benchmark metrics were modified during this audit.

---

### Phase 0 Sign-Off Summary

1. **FLIR Correction:** Clarified that Fixed YOLO832 holds the top Small $F_1$ ($0.1254$) on FLIR thermal, while Policy J achieves the top overall $F_1$ ($0.3185$) and top precision ($0.3370$).
2. **Taxonomy Reconciliation:** Established a strict mutually exclusive primary-cause hierarchy resolving the 10-count overlap in narrative reporting and verifying the exact 11,515 miss total.
3. **Statistical Boundaries:** Documented the limitations of $N=6$ sequences, distinguishing between empirical win rates, bootstrap intervals, and unverified equivalence claims.
4. **Subgroup Sample Sizes:** Formally restricted weather-specific conclusions to sequence-level findings where $N < 3$.
