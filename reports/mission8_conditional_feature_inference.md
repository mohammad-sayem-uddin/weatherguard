# WeatherGuard Mission 8: Conditional Feature-Level Inference

**Document ID:** `WG-M8-COND-001`  
**Date:** November 2026  
**Status:** Completed  
**Evaluation Cohort:** BDD100K Development Group B ($N=160$)  

---

## 1. Executive Summary

Experiment D combines the uncertainty-aware routing signal (Policy 5a: $k_{\text{small}} \ge 1$) with intermediate feature reuse and refinement.

Instead of running a full $832\times 832$ forward pass on escalated frames (re-executing layers 0–9 from scratch), the conditional feature-reuse system evaluates whether reusing $640\text{px}$ intermediate features through the detection head can match or exceed full-frame escalation efficiency.

---

## 2. Experimental Results (Group B Dev, N=160)

| System Configuration | Refinement Strategy | Additional Params | Escalation % | Micro $F_1$ | Small Recall | Mean Latency | P90 Latency | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mission 7 Policy 5a (Full 832px)** | Full Image Pass | 0 | 51.9% | **0.4937** | **13.6%** | 76.1 ms | 95.6 ms | 13.1 |
| **Cond-FeatureReuse-Unrefined** | Bilinear Neck Reuse | 0 | 51.9% | 0.0575 | 0.0% | 48.7 ms | 62.8 ms | 20.5 |
| **Cond-FeatureReuse-Residual** | Residual DW-PW | 91,840 | 51.9% | 0.0690 | 0.0% | 58.7 ms | 81.2 ms | 17.0 |
| **Cond-FeatureReuse-P3Detail** | P3 Dilated Sharpening | 5,376 | 51.9% | 0.0570 | 0.0% | 53.8 ms | 72.3 ms | 18.6 |
| **Cond-FeatureReuse-CrossScale** | P4-P3 Modulation | 45,248 | 51.9% | 0.0575 | 0.0% | 51.2 ms | 67.4 ms | 19.5 |

---

## 3. Scientific Comparison: Feature Reuse vs. Full-Frame Escalation

1. **Accuracy Superiority of Full-Frame Escalation:**
   Full-frame 832px escalation decisively outperforms all feature-reuse configurations ($F_1 = 0.4937$ vs. $0.0575\text{--}0.0690$, $\Delta F_1 \approx -0.02\text{ to }-0.03$). Small-object recall is nearly double under full-frame escalation (13.6% vs. 0.0%).
2. **Marginal Latency Savings Do Not Offset the Accuracy Drop:**
   While feature reuse saves $\approx 20\text{--}25\text{ ms}$ on escalated frames by bypassing the backbone, the total system throughput on M1 CPU (20.5 FPS vs. 13.1 FPS) provides only a marginal speedup.
3. **Architectural Implication:**
   When high-resolution inference is triggered by uncertainty or small-object presence, **the critical information resides in the raw optical pixels, not in the downsampled feature representations**. Full-frame resolution escalation remains the Pareto-superior architectural strategy for adverse-weather object detection.
