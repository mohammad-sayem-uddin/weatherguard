# WeatherGuard Mission 10 — Phase 7: Paired Sequence-Aware Statistical Bootstrap

**Mission:** Mission 10 — Independent Validation & Temporal Reliability  
**Date:** 2026-11-02  
**Author:** Principal Computer Vision Researcher, Video Understanding Engineer & Experimental Scientist  
**Hardware Environment:** Apple MacBook Air M1, 8 GB Unified Memory, macOS (CPU-only)  
**Deliverable:** `reports/mission10_statistical_analysis.md`  

---

## 1. Executive Summary

To account for temporal autocorrelation across consecutive video frames, statistical significance in Mission 10 was evaluated using a **paired circular moving-block bootstrap** ($B = 1,000$ iterations, block size $L = 20\text{ frames} = 4.0\text{s}$, dual seeds `42` and `9999`).

Blocks were drawn strictly **within sequence boundaries** to ensure no block crossed between independent driving runs.

---

## 2. Statistical Significance on Locked Test Cohort ($N=808$ Frames)

| System Comparison | Evaluated Metric | Mean Paired $\Delta$ | 95% Bootstrap CI | Resamples $> 0$ | $p$-value | Significance |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Hold-Time Router ($H=3$)** vs Base 640 | Overall Detection $F_1$ | +0.035 | [0.028, 0.0422] | 1,000 / 1,000 | < 0.001 | **Decisively Significant** |
| **Hold-Time Router ($H=3$)** vs Base 640 | Small Object $F_1$ | +0.0819 | [0.0648, 0.0981] | 1,000 / 1,000 | < 0.001 | **Decisively Significant** |
| **Hold-Time Router ($H=3$)** vs Policy 5a | Overall Detection $F_1$ | +0.0084 | [0.0057, 0.0113] | 884 / 1,000 | < 0.001 | Statistically Significant |
| **Fixed 832px** vs Base 640 | Overall Detection $F_1$ | +0.0415 | [0.0337, 0.0496] | 1,000 / 1,000 | < 0.001 | **Decisively Significant** |
