# WeatherGuard Mission 16 — Statistical Analysis Plan
**Document:** `reports/mission16_preparation/STATISTICAL_ANALYSIS_PLAN.md`  
**Date:** 2026-11-04  
**Author:** Mohammad Sayem Uddin (<sayemwork41@gmail.com>)  
**Mission:** Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

---

## 1. Formal Statistical Hypotheses

### Primary Hypothesis (Synthetic-to-Real Small-Object Gain)
- **Null Hypothesis ($H_{0,1}$):** Fine-tuning YOLOv8n on synthetic WeatherGuard-D1 produces no improvement or degradations in small-object $F_1$ ($16-32\text{px}$) on held-out real-world BDD100K driving video:
  $$H_{0,1}: \Delta F_{1,\text{small}} = F_{1,\text{small}}(\text{E2}) - F_{1,\text{small}}(\text{E1}) \le 0$$
- **Alternative Hypothesis ($H_{1,1}$):** Fine-tuning on D1 produces a statistically significant positive gain:
  $$H_{1,1}: \Delta F_{1,\text{small}} > 0 \quad (\alpha = 0.05)$$

### Secondary Hypothesis (Overall Acuity Preservation)
- **Two One-Sided Tests (TOST) for Non-Inferiority ($H_{0,2}$):** Fine-tuning on synthetic data does not impair large/medium object detection by more than an equivalence bound $\epsilon = 0.02$:
  $$H_{0,2}: \Delta F_{1,\text{overall}} \le -0.02 \quad \text{vs} \quad H_{1,2}: \Delta F_{1,\text{overall}} > -0.02$$

---

## 2. Resampling & Bootstrapping Protocols

### 2.1 Paired Sequence-Level Bootstrap (Real-World BDD100K Video)
Because consecutive video frames within a sequence exhibit strong temporal autocorrelation, simple i.i.d. frame sampling produces severely over-optimistic p-values.
- **Resampling Unit:** Complete driving video sequences ($N=6$ locked sequences, $1,214$ frames).
- **Replicate Count:** $B = 1,000$ bootstrap replicates with replacement.
- **Paired Evaluation:** In each bootstrap replicate $b$, the exact same sequence sample is evaluated on Model A and Model B:
  $$\Delta_b = \theta_A^{(b)} - \theta_B^{(b)}$$
- **Empirical 95% Confidence Interval:**
  $$\text{CI}_{95\%} = \left[ Q_{0.025}(\Delta), Q_{0.975}(\Delta) \right]$$
- **Two-Sided Bootstrap p-Value:**
  $$p = 2 \times \min\left( \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta_b \le 0), \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta_b \ge 0) \right)$$

### 2.2 Stratified Scenario-Level Block Bootstrap (Synthetic D1 Test Set)
Images in WeatherGuard-D1 sharing the same prompt scenario family ($F_k$) or near-duplicate dHash cluster are resampled as unified blocks:
- **Resampling Unit:** Scenario family cluster $F_k$, stratified by weather condition (Day, Night, Rain, Fog).
- **Replicate Count:** $B = 1,000$ block replicates.

---

## 3. Multiple Testing Corrections

When testing across multiple object classes (6 classes) and 4 adverse-weather conditions:
- **False Discovery Rate (FDR):** Apply the Benjamini-Hochberg procedure at level $q = 0.05$ to control type-I family-wise error rates.
- **Reporting Rule:** Uncorrected raw p-values and FDR-adjusted q-values must both be documented.

---

## 4. Effect Size & Sequence Dominance Metrics

1. **Point Estimate Delta:** $\Delta = \bar{\theta}_A - \bar{\theta}_B$.
2. **Standardized Effect Size (Cohen's $d$):**
   $$d = \frac{\Delta}{\text{SD}_{\Delta}}$$
3. **Sequence Win Rate:** The integer proportion of independent driving sequences where Model A outperforms Model B (e.g. $6/6$ wins, $0/6$ losses).
