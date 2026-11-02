# WeatherGuard Mission 3: Baseline Performance Report (Development Cohort)

**Document:** `reports/mission3_baseline.md`  
**Cohort:** Group B (BDD100K Development, $N = 160$ images, disjoint from locked test set)  
**Date:** 2026-11-01  
**Status:** Completed & Validated  

---

## 1. Executive Summary

This report establishes the baseline detection performance and scale sensitivity on the newly sampled **Group B BDD100K Development Cohort** ($N = 160$ images, $2,056$ target instances).

All baseline measurements reproduce the exact patterns observed on the locked test cohort:
1. **YOLOv8n Baseline ($	au = 0.20$):** Overall $F_1 = 0.4780$, mAP@0.5 = $0.1838$. Small-object recall is **8.09%** (missing 91.91% of small objects).
2. **Faster R-CNN Reference ($	au = 0.60$):** Overall $F_1 = 0.6565$, mAP@0.5 = $0.3285$. Small-object recall is **38.13%**.
3. **The Small-Object Deficit:** On Group B, Faster R-CNN maintains a **+30.04 percentage point small-object recall advantage**, confirming that YOLOv8n's small-object weakness is consistent across independent samples of BDD100K.

---

## 2. Quantitative Baseline Table (Group B Development)

| Detector | Policy | Threshold | Micro $F_1$ | Precision | Recall | mAP@0.5 | Small Recall (<32²) | Med Recall (32²-96²) | Large Recall (≥96²) | CPU Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **YOLOv8n (640x640)** | **DAWN-Selected** | **0.20** | **0.4780** | 0.6648 | 0.3731 | 0.1838 | **8.09%** | 50.00% | 80.00% | **51.7** |
| YOLOv8n (640x640) | Default | 0.50 | 0.3657 | 0.9165 | 0.2284 | 0.1182 | 1.05% | 25.23% | 70.13% | 51.7 |
| **Faster R-CNN** | **DAWN-Selected** | **0.60** | **0.6565** | 0.7193 | 0.6038 | 0.3285 | **38.13%** | 74.19% | 83.80% | **1153.4** |
| Faster R-CNN | Default | 0.50 | 0.6499 | 0.6591 | 0.6409 | 0.3459 | 42.75% | 77.31% | 86.58% | 1153.4 |

---

## 3. Scale Distribution & Ground-Truth Characteristics

In the 160 development images:
- **Small objects ($< 32^2$):** 952 instances
- **Medium objects ($32^2 \le \text{Area} < 96^2$):** 864 instances
- **Large objects ($\ge 96^2$):** 395 instances
- Total target objects: 2211

---

## 4. Key Takeaways for Mission 3 Optimization

1. **Replicated Failure Mode:** Baseline YOLOv8n ($640 \times 640$) misses **91.91%** of small objects on this development cohort, virtually identical to the $91.33\%$ miss rate on the locked test cohort.
2. **Speed Advantage Preserved:** YOLOv8n executes in $\approx 51.7\text{ ms/image}$, maintaining a $\approx 22.3\times$ throughput advantage over Faster R-CNN ($1153.4\text{ ms/image}$).
3. **Target for Experimentation:** The optimization objective in Experiments A, B, and C is to raise small-object recall from $\approx 8.1\%$ toward Faster R-CNN's $\approx 38.1\%$ while keeping CPU latency under $250\text{ ms/image}$.
