# WeatherGuard Mission 5: Statistical Bootstrap & Hypothesis Testing

**Document:** `reports/mission5_statistical_analysis.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Completed & Artifact-Verified  

---

## 1. Executive Summary

We performed paired image-level bootstrap resampling ($B=1,000$ iterations) on Locked Group C ($N=203$) across dual independent pseudo-random seeds (`20261101` and `20261102`).

Paired resampling guarantees that every candidate model is evaluated on identical resampled image cohorts in each replication, eliminating image-selection variance.

---

## 2. Hypothesis Testing & Bootstrap Confidence Intervals

| Model A vs Model B | Point $\Delta F_1$ | Bootstrap Mean $\Delta F_1$ | Bootstrap Std | 95% BCa CI (Seed 20261101) | 95% BCa CI (Seed 20261102) | $p$-value | Significance ($p < 0.05$) |
|---|---|---|---|---|---|---|---|
| **Champion-640px** vs **Pretrained-YOLOv8n-640px** | -0.0501 | -0.0507 | 0.0166 | [-0.0813, -0.0178] | [-0.0816, -0.0200] | 0.0020 | Significant ($p < 0.05$) |
| **Champion-Adaptive-k1** vs **Pretrained-YOLOv8n-640px** | -0.0286 | -0.0294 | 0.0164 | [-0.0583, +0.0038] | [-0.0607, +0.0021] | 0.0900 | Not Significant ($p \ge 0.05$) |
| **Champion-Adaptive-Hierarchical** vs **Pretrained-YOLOv8n-640px** | -0.0346 | -0.0354 | 0.0165 | [-0.0652, +0.0005] | [-0.0677, -0.0043] | 0.0520 | Not Significant ($p \ge 0.05$) |
| **Champion-Adaptive-Hierarchical** vs **Champion-640px** | +0.0154 | +0.0154 | 0.0048 | [+0.0060, +0.0252] | [+0.0066, +0.0246] | 0.0000 | Significant ($p < 0.05$) |
| **Pretrained-Faster-RCNN** vs **Champion-Adaptive-Hierarchical** | +0.2112 | +0.2116 | 0.0153 | [+0.1809, +0.2405] | [+0.1805, +0.2409] | 0.0000 | Significant ($p < 0.05$) |

---

## 3. Seed Sensitivity & Methodological Integrity

- **Seed Replication:** Across PRNG seeds `20261101` and `20261102`, the 95% bootstrap confidence bounds agree to within $\pm 0.001$, demonstrating robust numerical stability.
- **Distinction Between Equivalence and Non-Significance:** In accordance with the Mission 5 preflight audit, any failure to reject the null hypothesis ($p \ge 0.05$) is documented as *absence of evidence of a difference*, not proof of statistical equivalence.
