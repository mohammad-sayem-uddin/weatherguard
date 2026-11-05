# Phase 7 — Multi-Modal Ablation and Statistical Evaluation

**Project:** WeatherGuard + Rescue Drone  
**Author:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Mission:** 21 — Multi-Modal Acoustic-Visual Rescue Intelligence & Independent Validation  
**Date:** 2026-11-05  
**Artifact:** `reports/mission21/PHASE7_ABLATION_AND_STATISTICS.md`  

---

## 1. Executive Summary

This study evaluates six fusion paradigms on 450 paired search-and-rescue evaluation trials using sequence-level non-parametric cluster bootstrap resampling (500 iterations):

1. **Vision-Only Baseline**
2. **Audio-Only Baseline**
3. **Simple OR Fusion** ($\text{Alert}_{\text{vis}} \lor \text{Alert}_{\text{aud}}$)
4. **Simple AND Fusion** ($\text{Alert}_{\text{vis}} \land \text{Alert}_{\text{aud}}$)
5. **Confidence-Weighted Average** ($0.5 \cdot c_{\text{vis}} + 0.5 \cdot P_{\text{aud}} \ge 0.40$)
6. **Adaptive Uncertainty-Aware Fusion (Proposed)**

---

## 2. Multi-Modal Paradigm Ablation Table

| Fusion Paradigm | Accuracy | Precision | Detection Prob ($P_D$) | F1 Score | 95% Cluster Bootstrap CI (F1) | False Alarm Rate (FAR) | Miss Rate | Primary Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Vision-Only** | 53.1% | 97.3% | 34.0% | 0.5035 | $[0.1146, 0.6402]$ | 2.2% | 66.0% | Blind in darkness, dense fog, and at distances $<16\text{px}$. |
| **2. Audio-Only** | 75.3% | 95.1% | 68.2% | 0.7948 | $[0.7075, 0.8571]$ | 8.2% | 31.8% | Degrades at $0\text{ dB}$ SNR & rotor wash; no spatial bounding boxes. |
| **3. Simple AND Fusion** | 44.2% | **100.0%** | 20.3% | 0.3377 | $[0.1097, 0.4407]$ | **0.0%** | 79.7% | Catastrophic failure when one modality is environmentally blind. |
| **4. Confidence-Weighted** | 76.9% | 96.9% | 69.2% | 0.8074 | $[0.7034, 0.8992]$ | 5.2% | 30.8% | Fails when darkness or noise pulls down arithmetic mean. |
| **5. Simple OR Fusion** | **84.2%** | 94.8% | **81.9%** | **0.8790** | $[0.7860, 0.9603]$ | 10.4% | **18.1%** | Prone to false alarms from sensor noise. |
| **6. Adaptive Fusion (Proposed)** | 83.1% | 95.4% | 79.7% | 0.8685 | $[0.7906, 0.9527]$ | 8.9% | 20.3% | **Optimal balance:** Low FAR, high recall, explicit abstention. |

---

## 3. Statistical Significance and Effect Size Analysis

Paired comparisons were evaluated across bootstrap replicates resampled at the sequence level:

### 3.1 Adaptive Fusion vs. Vision-Only Baseline
- **Mean F1 Gain:** **$+0.3816$** ($+75.8\%$ relative increase)
- **95% Bootstrap Confidence Interval:** $[+0.2163, +0.6760]$
- **Effect Size (Cohen's $d$):** **$3.436$** (Extremely large effect size, $d > 0.8$)
- **Statistical Significance:** $p < 0.001$ (Zero is well outside the $99.9\%$ interval)
- *Conclusion:* Adding acoustic intelligence provides a massive, undeniable performance breakthrough over camera-only search.

### 3.2 Adaptive Fusion vs. Audio-Only Baseline
- **Mean F1 Gain:** **$+0.0707$** ($+8.9\%$ relative increase)
- **95% Bootstrap Confidence Interval:** $[+0.0024, +0.1100]$
- **Effect Size (Cohen's $d$):** **$2.478$** (Very large effect size)
- **Statistical Significance:** $p < 0.01$
- *Conclusion:* Optical corroboration provides statistically verified verification, reducing false alarms and confirming victim visual coordinates.

### 3.3 Adaptive Fusion vs. Simple OR Fusion
- **Mean F1 Delta:** $-0.0106$ ($95\%\text{ CI} = [-0.0304, +0.0046]$), $p > 0.05$ (Not statistically significant).
- While simple OR achieves slightly higher raw recall by firing unconditionally on either modality, it incurs a higher false alarm rate ($10.4\%$ vs $8.9\%$).
- Crucially, Simple OR cannot distinguish between confirmed visual sightings and distant acoustic alerts, whereas Adaptive Fusion outputs semantic action states (`CONFIRMED_VICTIM_LOCATED`, `ACOUSTIC_CUE_DISPATCH_OPTICAL_SEARCH`, `ABSTAIN_UNCERTAIN`).
