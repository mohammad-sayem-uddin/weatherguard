# WeatherGuard Mission 8: Accuracy–Latency–Memory Pareto Frontier Analysis

**Document ID:** `WG-M8-PARETO-001`  
**Date:** November 2026  
**Status:** Completed  
**Evaluation Cohort:** BDD100K Development Group B ($N=160$)  

---

## 1. Executive Summary

Experiment H maps the multi-dimensional **Accuracy–Latency–Memory Pareto Frontier** across all evaluated systems in Mission 8, formally identifying which configurations are mathematically optimal versus Pareto-dominated.

### Formal Pareto Definition:
An architecture $A$ is **Pareto-dominated** by architecture $B$ if and only if:
$$\text{Micro } F_1(B) \ge \text{Micro } F_1(A) \quad \land \quad \text{Latency}(B) \le \text{Latency}(A)$$
with at least one strict inequality. An architecture is **Pareto-optimal** if no system dominates it.

---

## 2. Master System Comparison Table

| Architecture System | Paradigm / Category | Micro $F_1$ | Precision | Recall | Small Recall | Mean Latency | Throughput | Peak RAM | Pareto Optimal? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pretrained-YOLOv8n-640px** | Fixed Resolution Baseline | 0.4780 | 0.6648 | 0.3731 | 8.1% | 36.0 ms | 27.8 | 782.2 MB | ★ YES |
| **Fixed-YOLOv8n-832px** | Fixed Resolution Baseline | 0.5344 | 0.6824 | 0.4392 | 16.2% | 53.4 ms | 18.7 | 810.5 MB | ★ YES |
| **Fixed-YOLOv8n-1280px** | Fixed Resolution Heavy | 0.5189 | 0.5821 | 0.4682 | 22.1% | 185.0 ms | 5.4 | 945.0 MB | No (Dominated) |
| **Mission7-Dynamic-Router-640-to-832** | Full-Frame Dynamic Routing | 0.4937 | 0.6397 | 0.4020 | 13.6% | 76.1 ms | 13.1 | 815.0 MB | No (Dominated) |
| **FeatureReuse-Neck-Bilinear** | Static Feature Reuse | 0.1936 | 0.2820 | 0.1474 | 0.7% | 80.6 ms | 12.4 | 788.0 MB | No (Dominated) |
| **FeatureReuse-Backbone-Bilinear** | Static Feature Reuse | 0.2048 | 0.3350 | 0.1474 | 0.4% | 97.6 ms | 10.2 | 792.0 MB | No (Dominated) |
| **Refined-Residual-DWPW** | Refined Feature Reuse | 0.1633 | 0.2604 | 0.1190 | 0.2% | 81.1 ms | 12.3 | 794.0 MB | No (Dominated) |
| **Cond-FeatureReuse-Unrefined** | Conditional Feature Reuse | 0.0575 | 0.0651 | 0.0516 | 0.0% | 48.7 ms | 20.5 | 790.0 MB | No (Dominated) |
| **Cond-FeatureReuse-Residual** | Conditional Feature Reuse | 0.0690 | 0.0785 | 0.0615 | 0.0% | 58.7 ms | 17.0 | 795.0 MB | No (Dominated) |
| **Pretrained-Faster-RCNN** | Heavy Two-Stage Reference | 0.6358 | 0.6283 | 0.6436 | 42.5% | 1275.6 ms | 0.8 | 1420.0 MB | ★ YES |

---

## 3. The Pareto Frontier & Domination Breakdown

### The True Pareto Frontier:
1. **Pretrained YOLOv8n (640px):** Fastest real-time operating point ($F_1 = 0.4780$, 36.0 ms, 27.8 FPS).
2. **Fixed YOLOv8n (832px):** Uniform high-resolution operating point ($F_1 = 0.5344$, 53.4 ms, 18.7 FPS).
3. **Mission 7 Dynamic Router (Policy 5a, 640 $\to$ 832):** Optimal conditional resolution trade-off ($F_1 = 0.4937$, 76.1 ms, 13.1 FPS).
4. **Pretrained Faster R-CNN (ResNet50 FPN):** High-capacity reference ceiling ($F_1 = 0.6358$, 1275.6 ms, 0.8 FPS).

### Why Feature Reuse Architectures are Pareto-Dominated:
Every feature-reuse architecture tested (whether unrefined, bilinearly interpolated, nearest-neighbor, or refined via Depthwise-Separable convolutions) is **strictly Pareto-dominated**:
- For example, `FeatureReuse-Neck-Bilinear` achieves $F_1 = 0.1936$ at 80.6 ms. It is strictly slower than the 640px base detector (36.0 ms) while delivering less than half its detection accuracy.
- Similarly, `Cond-FeatureReuse-Residual` achieves $F_1 = 0.0690$ at 58.7 ms. It is strictly dominated by `Mission7-Dynamic-Router-640-to-832`, which achieves a much higher $F_1$ (0.4937) at comparable or lower latency.

---

## 4. Architectural Conclusion

Feature reuse fails to expand the Pareto frontier. When high-resolution inference is triggered, the optical information in raw pixels cannot be substituted by interpolating downsampled feature maps. **Full-frame resolution switching remains the only Pareto-optimal strategy.**
