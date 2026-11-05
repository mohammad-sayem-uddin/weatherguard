# MISSION 22 — PHASE 3: Statistical Re-Analysis & Hypothesis Testing

**Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** `https://github.com/mohammad-sayem-uddin/weatherguard`  
**Branch:** `master`  
**Artifact:** `artifacts/mission22/statistical_reanalysis.json`  
**Date:** November 5, 2026  

---

## 1. Executive Summary

Phase 3 conducted an independent statistical re-analysis of the 6 multi-modal decision paradigms across the 450 paired simulation trials. All methodological defects identified in Phase 0 (invalid Cohen's $d$, uncorrected p-values, cluster-count overconfidence) were replaced with mathematically defensible statistical procedures:
1. **McNemar's Exact Paired Binomial Test:** Applied to matched trial-by-trial binary classification outcomes.
2. **Holm-Bonferroni Correction:** Applied across all 5 pairwise comparisons to control the family-wise error rate at $\alpha = 0.05$.
3. **5,000-Iteration Bootstrap:** Evaluated at both the sequence-cluster level ($K=4$) and sample-stratified level ($N=450$).
4. **Standard Effect Sizes:** Reported as Odds Ratios (OR), Cohen's $g$, and paired empirical $\Delta F_1$ distributions.

---

## 2. Definitive Comparison Ledger across 6 Paradigms

| Decision Paradigm | Accuracy | Precision | Recall | F1 Score | Miss Rate | False Alarm Rate (FAR) | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Vision-Only (YOLOv8n FT)** | 53.11% | **97.27%** | 33.97% | 0.5035 | 66.03% | **2.22%** | $\text{TN}=132, \text{FP}=3, \text{FN}=208, \text{TP}=107$ |
| **2. Audio-Only (Standard CNN)** | 75.33% | 95.13% | 68.25% | 0.7948 | 31.75% | 8.15% | $\text{TN}=124, \text{FP}=11, \text{FN}=100, \text{TP}=215$ |
| **3. Simple Heuristic AND** | 44.22% | **100.0%** | 20.32% | 0.3377 | 79.68% | **0.00%** | $\text{TN}=135, \text{FP}=0, \text{FN}=251, \text{TP}=64$ |
| **4. Confidence-Weighted Avg** | 76.89% | 96.89% | 69.21% | 0.8074 | 30.79% | 5.19% | $\text{TN}=128, \text{FP}=7, \text{FN}=97, \text{TP}=218$ |
| **5. Simple Heuristic OR** | **84.22%** | 94.85% | **81.90%** | **0.8790** | **18.10%** | 10.37% | $\text{TN}=121, \text{FP}=14, \text{FN}=57, \text{TP}=258$ |
| **6. Adaptive Fusion (Engine)** | 83.11% | 95.44% | 79.68% | 0.8685 | 20.32% | 8.89% | $\text{TN}=123, \text{FP}=12, \text{FN}=64, \text{TP}=251$ |

---

## 3. Paired Hypothesis Testing (McNemar's Exact Test & Holm-Bonferroni)

Contingency tables evaluate trial-by-trial accuracy matching:
- $a$: Both models correct
- $b$: Adaptive Fusion correct, Baseline incorrect
- $c$: Baseline correct, Adaptive Fusion incorrect
- $d$: Both models incorrect

| Comparison | Discordant Pairs $(b, c)$ | Total Discordant $(b + c)$ | Raw McNemar $p$-value | Holm-Bonferroni Adjusted $p$-value | Odds Ratio $(\frac{b}{c})$ | Cohen's $g$ | Statistically Significant $(\alpha=0.05)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adaptive vs. Vision-Only** | $(153, 18)$ | 171 | $7.30 \times 10^{-28}$ | **$< 1.0 \times 10^{-15}$** | **8.50** | $+0.3947$ | **CONFIRMED SIGNIFICANT** |
| **Adaptive vs. Audio-Only** | $(36, 1)$ | 37 | $5.53 \times 10^{-10}$ | **$< 1.0 \times 10^{-8}$** | **36.00** | $+0.4730$ | **CONFIRMED SIGNIFICANT** |
| **Adaptive vs. Simple AND** | $(187, 12)$ | 199 | $1.53 \times 10^{-41}$ | **$< 1.0 \times 10^{-15}$** | **15.58** | $+0.4397$ | **CONFIRMED SIGNIFICANT** |
| **Adaptive vs. Conf-Weighted** | $(33, 5)$ | 38 | $4.26 \times 10^{-6}$ | **$9.0 \times 10^{-6}$** | **6.60** | $+0.3684$ | **CONFIRMED SIGNIFICANT** |
| **Adaptive vs. Simple OR** | $(2, 7)$ | 9 | $0.1797$ | **$0.1797$** | **0.29** | $-0.2778$ | **NOT SIGNIFICANT ($p > 0.05$)** |

---

## 4. Bootstrap Confidence Intervals (5,000 Iterations)

### 4.1 Per-Paradigm F1 Confidence Intervals
- **Vision-Only:** Mean = 0.4800, 95% Cluster CI $[0.1146, 0.6402]$
- **Audio-Only:** Mean = 0.7909, 95% Cluster CI $[0.7075, 0.8571]$
- **Simple AND:** Mean = 0.3249, 95% Cluster CI $[0.1097, 0.4407]$
- **Confidence-Weighted:** Mean = 0.7995, 95% Cluster CI $[0.7034, 0.8992]$
- **Simple OR:** Mean = 0.8723, 95% Cluster CI $[0.7860, 0.9603]$
- **Adaptive Fusion:** Mean = 0.8616, 95% Cluster CI $[0.7906, 0.9527]$

### 4.2 Pairwise $\Delta F_1$ Bootstrap Distributions

| Comparison | Cluster Bootstrap Mean $\Delta F_1$ | 95% Cluster Bootstrap CI ($K=4$) | Sample Bootstrap Mean $\Delta F_1$ | 95% Sample Bootstrap CI ($N=450$) |
| :--- | :---: | :---: | :---: | :---: |
| **Adaptive vs. Vision-Only** | $+0.3727$ | $[+0.2163, +0.6596]$ | $+0.3652$ | $[+0.3066, +0.4246]$ |
| **Adaptive vs. Audio-Only** | $+0.0728$ | $[+0.0065, +0.1101]$ | $+0.0739$ | $[+0.0507, +0.0985]$ |
| **Adaptive vs. Simple AND** | $+0.5320$ | $[+0.4355, +0.6750]$ | $+0.5309$ | $[+0.4707, +0.5932]$ |
| **Adaptive vs. Conf-Weighted** | $+0.0636$ | $[+0.0343, +0.1114]$ | $+0.0612$ | $[+0.0382, +0.0865]$ |
| **Adaptive vs. Simple OR** | **$-0.0112$** | **$[-0.0304, +0.0043]$** | **$-0.0104$** | **$[-0.0222, -0.0004]$** |

---

## 5. Critical Research Reassessment: Is Adaptive Fusion Better than Simple OR and Audio-Only?

### 5.1 Reassessment vs. Audio-Only: **CONFIRMED SUPERIOR**
- **Empirical Gain:** Adaptive Fusion increases accuracy by **$+7.78\text{ pp}$** ($75.33\% \to 83.11\%$) and F1 score by **$+0.0737$** ($0.7948 \to 0.8685$).
- **Statistical Significance:** McNemar's exact test demonstrates overwhelming significance ($p = 5.53 \times 10^{-10}$, Holm-Bonferroni $p < 10^{-8}$). Adaptive Fusion corrected 36 audio errors while causing only 1 regression (Odds Ratio = 36.0).
- **Cluster Robustness:** The 95% cluster bootstrap CI strictly excludes zero ($[+0.0065, +0.1101]$).

### 5.2 Reassessment vs. Simple Heuristic OR: **NO F1 SUPERIORITY**
- **The F1 Finding:** Simple Heuristic OR achieves a **higher F1 score (0.8790)** and higher recall (81.90%) than Adaptive Fusion (F1 = 0.8685, Recall = 79.68%).
- **Statistical Test:** McNemar's test yields $p = 0.1797$ ($p > 0.05$). There is **no statistically significant difference** in overall classification accuracy between Simple OR and Adaptive Fusion.
- **The True Differentiating Trade-Off:**
  - Simple OR maximizes recall at the expense of **higher false alarms** (FAR = $10.37\%$, 14 false alarms).
  - Adaptive Fusion gates low-margin sensor disagreements using uncertainty abstention, reducing false alarms to **$8.89\%$** (12 false alarms, a $14.3\%$ relative reduction in false deployments) and raising precision from $94.85\%$ to $95.44\%$.
  - In life-critical search and rescue, dispatching first responders to false acoustic detections wastes critical time. Adaptive Fusion trades a small amount of marginal recall ($2.22\text{ pp}$) for precision and false alarm gating. **However, claims that Adaptive Fusion achieved statistically superior F1 over Simple OR are refuted by the data.**

### 5.3 Small-Cluster Limitation ($K=4$)
- Because all 450 trials were constructed around only 4 underlying visual sequence roots, sequence-level cluster bootstrap resampling exhibits wide variance. True cluster-robust inference requires testing across at least 30 independent physical drone flight sequences.

All raw statistical outputs are archived in `artifacts/mission22/statistical_reanalysis.json`.
Phase 3 is complete.
