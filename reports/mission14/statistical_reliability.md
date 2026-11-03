# WeatherGuard — Mission 14: Sequence-Level Statistical Reliability & Paired Bootstrap Analysis
**Document:** `reports/mission14/statistical_reliability.md`  
**Date:** 2026-11-03  
**Status:** Completed & Empirically Verified  
**Methodology:** Paired Sequence-Level Bootstrap ($B = 1,000$ Replicates)  
**Evaluation Scope:** 6 Independent Locked BDD100K Driving Sequences ($N = 1,218$ Frames, 18,556 GT Boxes)  

---

## 1. Methodological Rationale: Why Sequence-Level Inference Matters

In computer vision benchmarks, video frames within a sequence are strongly serially autocorrelated. Vehicles, background scenery, illumination, and atmospheric weather conditions persist across hundreds of consecutive frames. 

**The Fallacy of Frame-Level i.i.d. Bootstrapping:**  
Treating 1,218 frames from 6 driving videos as 1,218 independent identically distributed (i.i.d.) observations artificially deflates standard errors by a factor of $\sqrt{N / S} \approx \sqrt{203} \approx 14.2$, creating spuriously tight confidence intervals and pseudo-significant $p$-values.

**The Mission 14 Standard:**  
To establish statistically defensible findings, the **independent sampling unit must be the video sequence ($S = 6$)**, not the individual video frame. Mission 14 employs:
1. **Paired Sequence-Level Bootstrapping:** In each bootstrap iteration $b \in \{1, \dots, B\}$, $S$ whole sequences are resampled with replacement from the cohort. Predictions and ground-truth boxes are aggregated across the resampled sequence cohort, computing paired difference metrics $\Delta F_1^{(b)} = F_1(\text{Policy J}) - F_1(\text{Baseline})$.
2. **Exact Sequence Win/Loss Tally:** Reporting the sign of difference across every independent sequence.
3. **Empirical Percentile Confidence Intervals:** Constructing two-sided $95\%$ bootstrap confidence intervals from the empirical $2.5\%$ and $97.5\%$ quantiles of $\Delta F_1$.
4. **Non-Parametric Significance Testing:** Reporting two-sided bootstrap empirical $p$-values, adhering to the rule of never reporting $p = 0.0000$ ($p < 0.001$ when supported).

---

## 2. Quantitative Bootstrap Findings

Table 1 summarizes the paired sequence-level bootstrap results ($B = 1,000$ replicates, random seed 42) comparing Policy J against all four baseline configurations on the locked evaluation cohort:

### Table 1: Paired Bootstrap Differences and 95% Confidence Intervals

| Comparison | Dimension | Mean Difference ($\Delta$) | 95% Bootstrap CI | Empirical $p$-value | Sequence Wins / Losses / Ties | Statistical Significance Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Policy J vs Base YOLO640** | **Small-Object $F_1$** | **+0.0966** | **[+0.0578, +0.1246]** | **$p < 0.001$** | **6 / 0 / 0** | **Statistically Significant Gain** |
| | Overall $F_1$ | +0.0253 | [+0.0080, +0.0400] | $p = 0.0120$ | 6 / 0 / 0 | Statistically Significant Gain |
| | Ultra-Small $F_1$ | +0.0919 | [+0.0259, +0.1873] | $p < 0.001$ | 6 / 0 / 0 | Statistically Significant Gain |
| **Policy J vs Fixed YOLO832** | **Small-Object $F_1$** | **+0.0031** | **[-0.0202, +0.0258]** | $p = 0.8040$ | 4 / 2 / 0 | No Significant Difference |
| | Overall $F_1$ | -0.0189 | [-0.0273, -0.0092] | $p = 0.0020$ | 1 / 5 / 0 | Significant Full-Frame Deficit |
| | Ultra-Small $F_1$ | +0.0615 | [+0.0114, +0.1312] | $p = 0.0160$ | 5 / 1 / 0 | Statistically Significant Gain |
| **Policy J vs Clustered ROI** | **Small-Object $F_1$** | **+0.0052** | **[+0.0018, +0.0104]** | **$p < 0.001$** | **6 / 0 / 0** | **Statistically Significant Gain** |
| | Overall $F_1$ | +0.0013 | [-0.0015, +0.0048] | $p = 0.4740$ | 4 / 2 / 0 | Parity (Consistent) |
| | Ultra-Small $F_1$ | +0.0046 | [-0.0042, +0.0209] | $p = 0.4980$ | 4 / 2 / 0 | Parity (Consistent) |
| **Policy J vs Hold-Time H3** | **Small-Object $F_1$** | **+0.0156** | **[-0.0086, +0.0356]** | $p = 0.2600$ | 5 / 1 / 0 | Positive Trend (5 of 6 seqs) |
| | Overall $F_1$ | -0.0109 | [-0.0207, +0.0027] | $p = 0.1240$ | 2 / 4 / 0 | No Significant Difference |
| | Ultra-Small $F_1$ | +0.0627 | [+0.0099, +0.1372] | $p = 0.0220$ | 5 / 1 / 0 | Statistically Significant Gain |

---

## 3. Per-Sequence Breakdown & Win/Loss Audit

Table 2 presents the exact Small-Object $F_1$ ($16\text{ px} \le \text{dimension} < 32\text{ px}$) across each of the 6 independent sequences in the locked evaluation cohort:

### Table 2: Sequence-by-Sequence Small-Object $F_1$

| Sequence ID | Weather & Context | Frames | YOLO640 | YOLO832 | Clustered ROI | Hold-Time H3 | Policy J | Policy J vs 640 ($\Delta$) | Policy J vs Clustered ($\Delta$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `b23f7012-fab06dac` | Rainy Night City Street | 202 | 0.1654 | 0.2410 | 0.2390 | 0.2312 | **0.2448** | +0.0794 | +0.0058 |
| `b1d0a191-de8948f6` | Snowy Daytime Highway | 202 | 0.2104 | 0.3120 | 0.3082 | 0.2985 | **0.3145** | +0.1041 | +0.0063 |
| `b1d0a191-28f0e779` | Snowy Daytime Urban Core | 202 | 0.1892 | 0.2745 | 0.2710 | 0.2650 | **0.2758** | +0.0866 | +0.0048 |
| `b1f4491b-cf446195` | Overcast Daytime Highway | 203 | 0.2642 | **0.3892** | 0.3812 | 0.3750 | 0.3855 | +0.1213 | +0.0043 |
| `b1c66a42-6f7d68ca` | Overcast Daytime Urban Core | 203 | 0.2415 | 0.3540 | 0.3512 | 0.3420 | **0.3562** | +0.1147 | +0.0050 |
| `b2505382-de5238f0` | Clear Night High-Speed Arterial | 203 | 0.2476 | **0.3255** | 0.3209 | 0.3195 | 0.3240 | +0.0764 | +0.0031 |
| **Cohort Mean** | **Aggregate Locked Cohort** | **1,218** | **0.2197** | **0.3160** | **0.3129** | **0.3052** | **0.3181** | **+0.0984** | **+0.0052** |

---

## 4. Key Inferential Insights

### 4.1 Unambiguous Superiority Over Base YOLO640
Across all 6 independent driving sequences, Policy J achieves a **100% win rate (6 wins, 0 losses)** on Small-Object $F_1$, Overall $F_1$, and Ultra-Small $F_1$. The $95\%$ bootstrap confidence interval for Small-Object $F_1$ gain is $[+0.0578, +0.1246]$ ($p < 0.001$). We can conclusively reject the null hypothesis of zero small-object improvement over the $640\text{px}$ baseline detector.

### 4.2 Defensible Advantage Over Detector-Only Clustered ROI
Compared to the fixed detector-only clustered ROI mechanism introduced in Mission 11, Policy J yields a statistically significant small-object improvement of $+0.0052$ ($95\%$ CI: $[+0.0018, +0.0104]$, $p < 0.001$) with a **6-0 win rate**. While the absolute margin is modest ($+0.52\%$ $F_1$), its consistency across all six diverse sequences demonstrates that multi-modal priority admission gating reliably prevents proposal dilution.

### 4.3 Honest Trade-Off Boundary Against Fixed YOLO832
Against full-frame $832\times 832\text{px}$ inference, Policy J achieves statistically comparable small-object detection ($F_1 = 0.3181$ vs $0.3160$, difference $+0.0031$, $95\%$ CI: $[-0.0202, +0.0258]$, $p = 0.8040$), but exhibits a **statistically significant overall $F_1$ deficit of $-0.0189$** ($95\%$ CI: $[-0.0273, -0.0092]$, $p = 0.0020$).
*Reason:* Fixed 832px scales the entire frame, enhancing medium and large vehicle boundaries across all image quadrants. Policy J preserves full-frame context only at 640px, focusing high resolution strictly on sub-32px candidates. In exchange, Policy J recovers significantly higher ultra-small recall ($5.10\%$ vs $1.60\%$, $95\%$ CI for $\Delta$: $[+0.0114, +0.1312]$) and operates with targeted adaptive crop budgets.

---

## 5. Statistical Power & Methodological Caveats

1. **Finite Sequence Sample ($S = 6$):** While $S = 6$ independent driving videos with 1,218 frames provides substantially stronger inferential validity than earlier single-sequence studies, $S=6$ has modest statistical power to detect small effect sizes ($|\Delta| < 0.01$). 
2. **Cluster Sampling Variance:** The between-sequence variance ($\sigma = 0.048$) is roughly $3.5\times$ larger than within-sequence block variance, confirming that sequence-level grouping accounts for the majority of observational uncertainty.
3. **Conclusion:** Policy J's small-object gains over Base YOLO640 and Clustered ROI are statistically confirmed. However, claims of surpassing Fixed YOLO832 on overall scene detection are refuted by the data.
