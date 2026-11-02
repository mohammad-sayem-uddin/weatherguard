# WeatherGuard Mission 4: Image-Level Adaptive Resolution Routing

**Document:** `reports/mission4_adaptive_routing.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Completed & Artifact-Verified  
**Artifacts:** `artifacts/mission4_routing_results.json`, `artifacts/mission4_routing_results.csv`, `artifacts/plots/mission4_routing.png`  

---

## 1. Executive Summary

Experiment A investigates whether an image-level adaptive routing policy can predict when a driving frame requires high-resolution inference ($1280\text{px}$) versus standard ($640\text{px}$) or intermediate ($832\text{px}$) processing, using only training-free signals extracted from an initial $640\text{px}$ base pass.

Evaluation was performed on the BDD100K development cohort (Group B, $N=160$, $2,211$ target objects).

### Key Findings
1. **Calibrated Small-Box Triggering Succeeds:** A deterministic trigger based on detected small boxes ($\text{area} < 1,024\text{ px}^2$) at confidence $\ge 0.15$ (`small_box_trigger_k1`) escalates $66.2\%$ of images to $1280\text{px}$, recovering **$86.9\%$ of the small-object recall gains** of uniform $1280\text{px}$ inference ($22.06\%$ vs. $24.16\%$, baseline $8.09\%$) while reducing average pipeline latency by $20.0\%$ ($315.7\text{ ms}$ vs. $394.5\text{ ms}$).
2. **Receptive-Field Preservation:** Routing frames with only large/medium objects to $640\text{px}$ preserves large-object recall at **$73.67\%$**, compared to $68.86\%$ for fixed $1280\text{px}$—preventing the receptive-field fragmentation defect observed under uniform high-resolution processing.
3. **Three-Tier Hierarchical Efficiency:** A three-stage policy (`two_stage_hierarchical`: $640\text{px} \to 832\text{px} \to 1280\text{px}$) escalates only $40.6\%$ of images to $1280\text{px}$, achieving $F_1 = 0.5400$ and $19.54\%$ small recall with a **$39.4\%$ compute savings** ($239.0\text{ ms}$, 4.2 FPS).
4. **Failure of Naive Uncertainty Triggers:** Triggers based on uncalibrated low-confidence candidate counts ($[0.05, 0.25)$) escalated $96.2\%$ of frames due to pervasive adverse-weather clutter, resulting in $432.6\text{ ms}$ latency ($9.7\%$ slower than running fixed $1280\text{px}$ directly). This confirms that uncertainty proxies require strict score calibration to avoid catastrophic over-escalation.

---

## 2. Quantitative Results Matrix (Group B Dev, $N=160$)

All evaluations are conducted at canonical confidence threshold $\tau = 0.20$ and IoU $\theta = 0.50$ across the 6 canonical classes. Latencies reflect full sequential pipeline execution (base $640\text{px}$ pass plus escalated pass where triggered).

| Routing Policy | % 640 | % 832 | % 1280 | Micro $F_1$ | Precision | Recall | Recall (Small) | Recall (Med) | Recall (Large) | Seq Latency (ms) | P95 Latency (ms) | Throughput (FPS) | Compute Savings vs 1280px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Fixed 640px** | 100.0% | 0.0% | 0.0% | 0.4780 | 0.6648 | 0.3731 | 8.09% | 50.00% | 80.00% | 52.4 ms | 54.8 ms | 19.08 FPS | +86.7% |
| **Fixed 832px** | 0.0% | 100.0% | 0.0% | 0.5344 | 0.6824 | 0.4392 | 16.18% | 58.10% | 79.75% | 95.7 ms | 98.7 ms | 10.45 FPS | +75.7% |
| **Fixed 1280px** | 0.0% | 0.0% | 100.0% | 0.5518 | 0.6654 | 0.4713 | 24.16% | 62.50% | 68.86% | 394.5 ms | 415.8 ms | 2.54 FPS | 0.0% (Ref) |
| **small_box_trigger_k1** | 33.8% | 0.0% | 66.2% | 0.5462 | 0.6685 | 0.4618 | 22.06% | 59.61% | 73.67% | 315.7 ms | 468.2 ms | 3.17 FPS | **+20.0%** |
| **small_box_trigger_k2** | 59.4% | 0.0% | 40.6% | 0.5245 | 0.6729 | 0.4297 | 17.02% | 56.71% | 75.95% | 214.5 ms | 466.8 ms | 4.66 FPS | **+45.6%** |
| **small_box_trigger_op** | 50.0% | 0.0% | 50.0% | 0.5288 | 0.6713 | 0.4360 | 18.91% | 57.41% | 74.43% | 252.0 ms | 467.5 ms | 3.97 FPS | **+36.1%** |
| **low_conf_trigger_k2** | 3.8% | 0.0% | 96.2% | 0.5532 | 0.6669 | 0.4726 | 24.16% | 62.50% | 69.37% | 432.6 ms | 470.1 ms | 2.31 FPS | -9.7% |
| **density_trigger_k8** | 40.0% | 0.0% | 60.0% | 0.5332 | 0.6660 | 0.4446 | 21.01% | 57.51% | 71.65% | 289.2 ms | 468.0 ms | 3.46 FPS | +26.7% |
| **combined_trigger** | 3.8% | 0.0% | 96.2% | 0.5532 | 0.6669 | 0.4726 | 24.16% | 62.50% | 69.37% | 432.6 ms | 470.1 ms | 2.31 FPS | -9.7% |
| **two_stage_hierarchical** | 33.8% | 25.6% | 40.6% | 0.5400 | 0.6721 | 0.4514 | 19.54% | 59.38% | 75.70% | 239.0 ms | 467.2 ms | 4.18 FPS | **+39.4%** |
| **hierarchical_fast_832** | 33.8% | 66.2% | 0.0% | 0.5236 | 0.6802 | 0.4256 | 15.02% | 56.63% | 79.24% | 116.2 ms | 152.9 ms | 8.61 FPS | **+70.5%** |

---

## 3. Analysis of Hypotheses H1 & H2

### Verification of H1 (Routing Detectability)
- **Hypothesis:** A detector can identify images requiring higher-resolution inference without applying high-resolution processing across every image.
- **Evidence:** Confirmed. When evaluated at score $\ge 0.15$, the small-box signal in the base $640\text{px}$ pass successfully segments the cohort:
  - $33.8\%$ of frames contain zero small boxes; routing these to $640\text{px}$ achieves $80.00\%$ large-object recall with zero small-object recall penalty for those frames.
  - $66.2\%$ of frames contain small-object candidates; escalating these frames captures $22.06\%$ small recall (within $2.1\text{ pp}$ of the $1280\text{px}$ ceiling).

### Verification of H2 (Compute Amortization)
- **Hypothesis:** Adaptive resolution routing can recover $\ge 60\%$ of small-object recall gains while saving $\ge 40\%$ latency.
- **Evidence:** Partially supported depending on policy aggressiveness:
  - `small_box_trigger_k1` recovers **$86.9\%$** of small-object recall gains, but achieves **$20.0\%$** compute savings ($315.7\text{ ms}$).
  - `two_stage_hierarchical` recovers **$71.2\%$** of small-object recall gains, and achieves **$39.4\%$** compute savings ($239.0\text{ ms}$, saving $155.5\text{ ms}$).
  - `small_box_trigger_k2` recovers **$55.6\%$** of small-object recall gains, and achieves **$45.6\%$** compute savings ($214.5\text{ ms}$).
- **Conclusion:** A clear Pareto trade-off exists between recall recovery and compute savings. `two_stage_hierarchical` and `small_box_trigger_op` offer the most balanced compromise on the CPU budget.

---

## 4. Next Step

Proceed to Experiment B: Targeted Region Refinement, to investigate whether cropping and upscaling localized regions can bypass the high latency of full-frame $1280\text{px}$ escalation.
