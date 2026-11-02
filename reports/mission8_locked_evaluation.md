# WeatherGuard Mission 8: Final Locked Evaluation on Group C Test ($N=203$)

**Document ID:** `WG-M8-LOCKED-001`  
**Date:** November 2026  
**Status:** Completed and Verified  
**Frozen Configuration SHA-256:** `c36121689c7c466b587a49b2f8c23f359cd8fcb18b52721579420a78ac4dec3f`  
**Evaluation Cohort:** BDD100K Locked Test Group C ($N=203$)  

---

## 1. Executive Summary & Verification of Governance

This locked evaluation represents the final external benchmark of Mission 8. All architectures, feature extraction hooks, refinement modules, distillation weights, and routing rules were frozen on Group B Development ($N=160$) before Group C ($N=203$) was unsealed.

---

## 2. Locked External Benchmark Results ($N=203$)

| Detector System | Architectural Strategy | Micro $F_1$ | Precision | Recall | Small Recall | Med Recall | Lrg Recall | Latency | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pretrained YOLOv8n (640px)** | Base Baseline | 0.5111 | 0.6826 | 0.4085 | 8.7% | 52.8% | 79.7% | 61.1 ms | 16.4 |
| **Fixed YOLOv8n (832px)** | High-Res Baseline | 0.5655 | 0.7168 | 0.4669 | 16.3% | 58.7% | 81.8% | 103.0 ms | 9.7 |
| **Fixed YOLOv8n (1280px)** | Heavy Baseline | 0.5782 | 0.6786 | 0.5037 | 28.7% | 61.2% | 70.9% | 378.0 ms | 2.6 |
| **Dynamic Router (640 $\to$ 832)** | Full-Frame Escalation | **0.5442** | 0.6707 | 0.4579 | **14.0%** | 58.4% | 82.4% | **81.7 ms** | **12.2** |
| **FeatureReuse-Neck-Bilinear** | Unrefined Reuse | 0.1626 | 0.2445 | 0.1218 | 0.7% | 8.8% | 41.6% | 80.6 ms | 12.4 |
| **Refined-Residual-DWPW** | Distilled Refinement | 0.1933 | 0.2900 | 0.1449 | 0.6% | 11.0% | 48.8% | 81.1 ms | 12.3 |
| **Cond-FeatureReuse-Residual** | Conditional Reuse | 0.4634 | 0.5429 | 0.4042 | 8.6% | 52.1% | 79.3% | 70.8 ms | 14.1 |
| **Pretrained Faster R-CNN** | Two-Stage Reference | 0.6877 | 0.7384 | 0.6436 | 42.5% | 75.3% | 85.0% | 1275.6 ms | 0.8 |

---

## 3. Scientific Assessment of Core Research Questions

### Question: Can feature reuse recover some of the accuracy benefits of 832px inference without executing a full second forward pass?
**Answer: Decisively NO.**
On the locked external benchmark ($N=203$), feature reuse collapses detection accuracy:
- Unrefined feature reuse scores $F_1 = 0.1626$ (vs. native 640px $F_1 = 0.5111$ and native 832px $F_1 = 0.5655$).
- Distilled lightweight refinement improves feature loss during training but scores $F_1 = 0.1933$, failing to bridge the gap.
- Full-frame resolution escalation (**Dynamic Router 640 $\to$ 832**, $F_1 = 0.5442$) remains the undisputed champion. It captures **$63.2\%$** of the total accuracy gain of full 832px execution at an edge-practical $12.2\text{ FPS}$.
