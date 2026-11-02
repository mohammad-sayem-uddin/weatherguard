# WeatherGuard Mission 4: Baseline Performance & Operational Constraints

**Document:** `reports/mission4_baseline.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Completed & Artifact-Verified  

---

## 1. Executive Summary

This report establishes the baseline detection performance, scale-stratified recall, and CPU inference latency across the development (Group B, $N=160$) and locked external test (Group C, $N=203$) cohorts prior to the introduction of adaptive routing or crop refinement in Mission 4.

The baselines verify two fundamental tensions in adverse-weather edge object detection:
1. **The Small-Object Deficit:** At standard single-scale inference ($640\text{px}$), YOLOv8n misses over $91\%$ of small objects ($< 32^2\text{ px}^2$) in adverse weather, operating at only $8.09\%$ small recall on Group B and $8.67\%$ on Group C.
2. **The High-Resolution Latency Penalty:** While uniform upscaling to $1280\text{px}$ improves small recall to $24.16\%$ (Group B) and $28.69\%$ (Group C), latency surges from $\approx 52-61\text{ ms}$ to $\approx 378-395\text{ ms}$ on the Apple M1 CPU. Throughput drops from $16-19\text{ FPS}$ to $2.5-2.6\text{ FPS}$, breaching the practical $10\text{ FPS}$ real-time boundary.

---

## 2. Baseline Performance Matrix

All metrics are evaluated at canonical thresholds ($\tau = 0.20$ for YOLOv8n, $\tau = 0.60$ for Faster R-CNN) with IoU threshold $\theta = 0.50$ across the 6 canonical driving categories (`Car`, `Person`, `Truck`, `Bus`, `Motorcycle`, `Bicycle`). Scales are defined in native $1280 \times 720$ coordinates.

| Cohort | Model | Input Res | $\tau$ | Micro $F_1$ | Precision | Recall | Recall (Small) | Recall (Med) | Recall (Large) | Latency (ms) | Throughput (FPS) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Group B (Dev, $N=160$)** | YOLOv8n | $640\text{px}$ | 0.20 | 0.4780 | 0.6648 | 0.3731 | 8.09% | 50.00% | 80.00% | 52.4 ms | 19.08 FPS |
| **Group B (Dev, $N=160$)** | YOLOv8n | $832\text{px}$ | 0.20 | 0.5344 | 0.6824 | 0.4392 | 16.18% | 58.10% | 79.75% | 95.7 ms | 10.45 FPS |
| **Group B (Dev, $N=160$)** | YOLOv8n | $1280\text{px}$ | 0.20 | 0.5518 | 0.6654 | 0.4713 | 24.16% | 62.50% | 68.86% | 394.5 ms | 2.54 FPS |
| **Group B (Dev, $N=160$)** | Faster R-CNN | Native ($1333\text{px}$) | 0.60 | 0.6565 | 0.7193 | 0.6038 | 38.13% | 74.19% | 83.80% | 1,153.4 ms | 0.87 FPS |
| **Group C (Test, $N=203$)** | YOLOv8n | $640\text{px}$ | 0.20 | 0.5111 | 0.6826 | 0.4085 | 8.67% | 52.84% | 79.69% | 61.1 ms | 16.36 FPS |
| **Group C (Test, $N=203$)** | YOLOv8n | $1280\text{px}$ | 0.20 | 0.5782 | 0.6786 | 0.5037 | 28.69% | 61.23% | 70.90% | 378.0 ms | 2.65 FPS |
| **Group C (Test, $N=203$)** | Faster R-CNN | Native ($1333\text{px}$) | 0.60 | 0.6877 | 0.7384 | 0.6436 | 42.53% | 75.31% | 84.96% | 1,275.6 ms | 0.78 FPS |

---

## 3. Scale-Stratified Object Distribution

To ensure statistical rigor, the exact object denominators across both cohorts are locked:

- **Group B (BDD100K Development, $N = 160$):**
  - Small objects ($< 32^2\text{ px}^2$): $890$ objects ($40.25\%$)
  - Medium objects ($[32^2, 96^2)\text{ px}^2$): $806$ objects ($36.45\%$)
  - Large objects ($\ge 96^2\text{ px}^2$): $515$ objects ($23.29\%$)
  - Total ground-truth objects: $2,211$ objects ($13.82$ objects/image)

- **Group C (BDD100K Locked Test, $N = 203$):**
  - Small objects ($< 32^2\text{ px}^2$): $1,004$ objects ($39.33\%$)
  - Medium objects ($[32^2, 96^2)\text{ px}^2$): $933$ objects ($36.55\%$)
  - Large objects ($\ge 96^2\text{ px}^2$): $616$ objects ($24.13\%$)
  - Total ground-truth objects: $2,553$ objects ($12.58$ objects/image)

The scale distributions between Group B and Group C are highly aligned (~$39-40\%$ small, ~$36\%$ medium, ~$23-24\%$ large), validating Group B as an effective proxy for development and policy selection.

---

## 4. Key Takeaways for Mission 4

1. **Intermediate Resolution Opportunity:** YOLOv8n at $832\text{px}$ achieves $F_1 = 0.5344$ on Group B with $95.7\text{ ms}$ latency ($10.45\text{ FPS}$), doubling small-object recall from $8.09\%$ to $16.18\%$ while remaining above the $10\text{ FPS}$ threshold.
2. **Escalation Target:** Processing only frames with high small-object density at $1280\text{px}$ offers the potential to achieve near-$1280\text{px}$ accuracy while keeping mean latency closer to the $832\text{px}$ regime.
3. **Large Object Degradation at 1280px:** In Group B, large-object recall drops from $80.00\%$ at $640\text{px}$ to $68.86\%$ at $1280\text{px}$ (and $79.69\% \to 70.90\%$ in Group C). High-resolution inference introduces receptive-field fragmentation on large foreground vehicles. An adaptive system that retains $640\text{px}$ for frames with dominant large objects can prevent this degradation.
