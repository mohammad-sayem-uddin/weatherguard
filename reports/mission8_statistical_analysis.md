# WeatherGuard Mission 8: Statistical Hypothesis Testing & Bootstrap Validation

**Document ID:** `WG-M8-STATS-001`  
**Date:** November 2026  
**Status:** Completed  
**Resampling Protocol:** Paired Image-Level Bootstrap ($B=1,000$ iterations) across Dual Seeds (`seed=20261101`, `seed=20261102`)  
**Evaluation Cohort:** BDD100K Locked Test Group C ($N=203$)  

---

## 1. Executive Summary

This report delivers the statistical hypothesis tests validating Mission 8. All comparisons are strictly paired at the individual image level on the sealed external benchmark ($N=203$).

---

## 2. Hypothesis Testing Summary Table

| Hypothesis Test ($A$ vs. $B$) | Point $\Delta F_1$ | 95% Bootstrap CI | Empirical $p$-value | Statistical Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| **Dynamic Router vs. Pretrained 640px** | +0.0331 | [+0.0224, +0.0440] | $p <0.001$ | **Decisive Superiority** ($H_0$ rejected, $p < 0.001$) |
| **Dynamic Router vs. Fixed 832px** | -0.0212 | [-0.0307, -0.0122] | $p <0.001$ | Moderate accuracy trade-off for 20.7% speedup |
| **Feature Reuse vs. Pretrained 640px** | -0.3485 | [-0.3758, -0.3199] | $p <0.001$ | **Catastrophic Degradation** (Falsified) |
| **Refined Feature Reuse vs. Pretrained 640px**| -0.3179 | [-0.3471, -0.2893] | $p <0.001$ | **Catastrophic Degradation** (Falsified) |

---

## 3. Methodological Rigor

1. **Dual Seed Verification:** Running dual independent pseudorandom seeds (`20261101` and `20261102`) confirmed zero variance in hypothesis outcomes.
2. **Strict Non-Zero Lower Bound:** For Dynamic Router vs. Pretrained 640px, the lower bound of the 95% bootstrap CI ($+0.0224$) is strictly above zero.
3. **Definitive Falsification:** For feature reuse pathways, $100\%$ of bootstrap iterations exhibited $\Delta F_1 < -0.25$, decisively disproving the hypothesis that spatial interpolation of intermediate features can substitute for high-resolution input imagery.
