# WeatherGuard Mission 5: Pre-Training Baselines & Reference Anchors

**Document:** `reports/mission5_baseline.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Completed & Artifact-Verified  

---

## 1. Executive Summary

This report establishes the pre-training reference baselines across all experimental cohorts prior to model fine-tuning in Mission 5.

The evaluation benchmarks the off-the-shelf COCO-pretrained YOLOv8n (`yolov8n.pt`) and reference Faster R-CNN across:
1. **In-Domain Adverse Weather:** Group A DAWN Validation ($N = 206$, $1,438$ target objects).
2. **Out-of-Domain Development:** Group B BDD100K Development ($N = 160$, $2,211$ target objects).
3. **Locked External Test:** Group C BDD100K Locked Test ($N = 203$, $2,553$ target objects).

---

## 2. Baseline Performance Matrix Across Cohorts

Evaluated at canonical operational thresholds ($\tau = 0.20$ for YOLOv8n, $\tau = 0.60$ for Faster R-CNN) with IoU $\theta = 0.50$ across the 6 canonical driving categories (`Car`, `Person`, `Truck`, `Bus`, `Motorcycle`, `Bicycle`).

| Cohort | Model / Input Res | $\tau$ | Micro $F_1$ | Precision | Recall | Recall (Small) | Recall (Large) | Latency (ms) | Throughput (FPS) |
|---|---|---|---|---|---|---|---|---|---|
| **Group A Val (DAWN, $N=206$)** | Pretrained YOLOv8n ($640\text{px}$) | 0.20 | 0.5763 | 0.7136 | 0.4833 | 14.29% | 72.84% | 35.5 ms | 28.17 FPS |
| **Group A Val (DAWN, $N=206$)** | Pretrained Faster R-CNN | 0.60 | 0.6604 | 0.6622 | 0.6586 | 32.14% | 84.57% | 436.0 ms | 2.29 FPS |
| **Group B Dev (BDD, $N=160$)** | Pretrained YOLOv8n ($640\text{px}$) | 0.20 | 0.4780 | 0.6648 | 0.3731 | 8.09% | 80.00% | 52.4 ms | 19.08 FPS |
| **Group B Dev (BDD, $N=160$)** | Pretrained YOLOv8n ($832\text{px}$) | 0.20 | 0.5344 | 0.6824 | 0.4392 | 16.18% | 79.75% | 95.7 ms | 10.45 FPS |
| **Group B Dev (BDD, $N=160$)** | Pretrained YOLOv8n ($1280\text{px}$) | 0.20 | 0.5518 | 0.6654 | 0.4713 | 24.16% | 68.86% | 394.5 ms | 2.54 FPS |
| **Group B Dev (BDD, $N=160$)** | Mission 4 Adaptive-k1 | 0.20 | 0.5462 | 0.6685 | 0.4618 | 22.06% | 73.67% | 315.7 ms | 3.17 FPS |
| **Group B Dev (BDD, $N=160$)** | Pretrained Faster R-CNN | 0.60 | 0.6565 | 0.7193 | 0.6038 | 38.13% | 83.80% | 1,153.4 ms | 0.87 FPS |
| **Group C Test (BDD, $N=203$)** | Pretrained YOLOv8n ($640\text{px}$) | 0.20 | 0.5111 | 0.6826 | 0.4085 | 8.67% | 79.69% | 61.1 ms | 16.36 FPS |
| **Group C Test (BDD, $N=203$)** | Pretrained YOLOv8n ($832\text{px}$) | 0.20 | 0.5655 | 0.7011 | 0.4740 | 16.33% | 81.84% | 103.0 ms | 9.71 FPS |
| **Group C Test (BDD, $N=203$)** | Pretrained YOLOv8n ($1280\text{px}$) | 0.20 | 0.5782 | 0.6786 | 0.5037 | 28.69% | 70.90% | 378.0 ms | 2.65 FPS |
| **Group C Test (BDD, $N=203$)** | Mission 4 Adaptive-k1 | 0.20 | 0.5645 | 0.6791 | 0.4828 | 24.80% | 75.39% | 265.8 ms | 3.76 FPS |
| **Group C Test (BDD, $N=203$)** | Mission 4 Adaptive-Hier | 0.20 | 0.5671 | 0.6865 | 0.4830 | 22.71% | 77.93% | 208.8 ms | 4.79 FPS |
| **Group C Test (BDD, $N=203$)** | Pretrained Faster R-CNN | 0.60 | 0.6877 | 0.7384 | 0.6436 | 42.53% | 84.96% | 1,275.6 ms | 0.78 FPS |

---

## 3. Evaluation Objectives for Fine-Tuned Models

For any fine-tuned model trained on DAWN in Mission 5 to be deemed an empirical success, it must satisfy:
1. **In-Domain Fidelity:** Outperform pretrained YOLOv8n on Group A DAWN Val ($F_1 > 0.5763$, Precision $> 0.7136$).
2. **Cross-Dataset Generalization:** Improve zero-shot transfer on Group B BDD100K Dev at $640\text{px}$ ($F_1 > 0.4780$, Small Recall $> 8.09\%$).
3. **No Large-Object Collapse:** Preserve large-object recall ($\ge 75\%$) without inducing false-positive explosions.
4. **Efficiency Preservation:** Maintain the lightweight $35-55\text{ ms}$ inference speed of the original $640\text{px}$ architecture.
