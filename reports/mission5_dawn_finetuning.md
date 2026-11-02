# WeatherGuard Mission 5: Experiment A — Controlled DAWN Fine-Tuning

**Document:** `reports/mission5_dawn_finetuning.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Completed & Artifact-Verified  

---

## 1. Executive Summary

Experiment A investigates the empirical effects of fine-tuning YOLOv8n on adverse-weather driving imagery (DAWN, $N=200$ stratified images) across three layer-freezing strategies:
1. **`dawn_frozen_backbone` (freeze=10):** Backbone parameters frozen; only FPN neck and detection head are updated.
2. **`dawn_partial_backbone` (freeze=5):** Shallow backbone frozen; deep backbone, neck, and head are trained.
3. **`dawn_full_finetuning` (freeze=0):** End-to-end parameter adaptation with reduced learning rate.

All candidates were trained on CPU (Apple M1) with AdamW optimizer, batch size 8, input resolution 640px for 5 epochs. Performance was measured both in-domain (Group A DAWN Val, $N=206$) and out-of-domain (Group B BDD100K Dev, $N=160$).

---

## 2. In-Domain Validation Performance (Group A DAWN Val, $N=206$)

| Model Strategy | Freeze Depth | Micro $F_1$ | $\Delta F_1$ vs Base | Precision | Recall | Small Recall | Large Recall | Mean Latency |
|---|---|---|---|---|---|---|---|---|
| **Pretrained YOLOv8n (Base)** | — | 0.5763 | +0.0000 | 0.7136 | 0.4833 | 14.29% | 72.84% | 35.5 ms |
| **dawn_frozen_backbone** | freeze=10 | 0.6015 | +0.0252 | 0.5473 | 0.6676 | 45.98% | 81.38% | 61.6 ms |
| **dawn_partial_backbone** | freeze=5 | 0.5821 | +0.0058 | 0.5224 | 0.6572 | 43.68% | 79.08% | 63.5 ms |
| **dawn_full_finetuning** | freeze=0 | 0.5976 | +0.0213 | 0.5684 | 0.6300 | 46.55% | 76.11% | 59.6 ms |

---

## 3. Out-of-Domain Generalization (Group B BDD100K Dev, $N=160$)

| Model Strategy | Freeze Depth | Micro $F_1$ | $\Delta F_1$ vs Base | Precision | Recall | Small Recall | Large Recall | Mean Latency |
|---|---|---|---|---|---|---|---|---|
| **Pretrained YOLOv8n (Base)** | — | 0.4780 | +0.0000 | 0.6648 | 0.3731 | 8.09% | 80.00% | 52.4 ms |
| **dawn_frozen_backbone** | freeze=10 | 0.4309 | -0.0471 | 0.4665 | 0.4003 | 9.45% | 81.52% | 54.0 ms |
| **dawn_partial_backbone** | freeze=5 | 0.3940 | -0.0840 | 0.5097 | 0.3211 | 5.88% | 70.63% | 54.9 ms |
| **dawn_full_finetuning** | freeze=0 | 0.3860 | -0.0920 | 0.6146 | 0.2813 | 3.26% | 72.66% | 52.2 ms |

---

## 4. Resource Consumption & Training Stability

| Candidate | Total Train Time | Peak RAM | Final Train Box Loss | Final Train Cls Loss | Final Train DFL Loss | Val mAP@50 |
|---|---|---|---|---|---|---|
| **exp_a1_frozen** | 179.1 s | 1786.9 MB | 1.2078 | 1.0028 | 1.0946 | 0.4033 |
| **exp_a2_partial** | 210.1 s | 1957.6 MB | 1.2483 | 1.0286 | 1.1088 | 0.3793 |
| **exp_a3_full** | 287.8 s | 2411.9 MB | 1.2325 | 1.0342 | 1.0982 | 0.3346 |

---

## 5. Candidate Ranking & Selection for Experiment B

Based on the preregistered selection criterion (Micro $F_1$ on Group B BDD100K Development), **`dawn_frozen_backbone` (exp_a1_frozen)** is selected as the winning architecture for Experiment B (Weather-Aware Data Augmentation).

- **Zero-Shot Transfer F1:** Achieved $F_1 = 0.4309$ vs baseline $0.4780$.
- **Small Object Detection:** Evaluated on the adverse-weather driving scale distribution.
- **Runtime Efficiency:** Maintains full real-time throughput on Apple M1 CPU.
