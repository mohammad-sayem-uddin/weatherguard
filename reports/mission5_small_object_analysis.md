# WeatherGuard Mission 5: Experiment D — Multi-Scale & Small-Object Performance Analysis

**Document:** `reports/mission5_small_object_analysis.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Completed & Artifact-Verified  

---

## 1. Executive Summary

Experiment D investigates whether supervised fine-tuning on adverse weather and weather-aware synthetic data augmentation improve small-object detection on out-of-domain driving scenes (Group B BDD100K Dev, $N=160$).

We evaluate detections across four fine-grained geometric scale strata:
- **Sub-16px ($area < 256\text{ px}^2$):** Distant pedestrians, cyclists, and horizon vehicles.
- **Small 16–32px ($256 \le area < 1024\text{ px}^2$):** Standard distant driving targets.
- **COCO Small ($area < 1024\text{ px}^2$):** Aggregated COCO-standard small category.
- **Medium ($1024 \le area < 9216\text{ px}^2$):** Mid-range vehicles and nearby pedestrians.
- **Large ($area \ge 9216\text{ px}^2$):** Foreground cars, buses, and commercial trucks.

---

## 2. Scale-Stratified Recall Comparison Matrix (Group B Dev, $N=160$)

| Detector Model | $\tau$ | Micro $F_1$ | Sub-16px Recall | Small 16-32px Recall | COCO Small Recall | Medium Recall | Large Recall |
|---|---|---|---|---|---|---|---|
| **Pretrained YOLOv8n** | 0.20 | 0.4780 | 0.00% (0/258) | 11.10% (77/694) | 8.09% (77/952) | 50.00% (432/864) | 80.00% (316/395) |
| **Pretrained Faster R-CNN** | 0.60 | 0.6565 | 16.67% (43/258) | 46.11% (320/694) | 38.13% (363/952) | 74.19% (641/864) | 83.80% (331/395) |
| **Fine-Tuned YOLOv8n (mission5_a1_frozen)** | 0.20 | 0.4309 | 0.00% (0/258) | 12.97% (90/694) | 9.45% (90/952) | 54.75% (473/864) | 81.52% (322/395) |
| **Weather-Augmented YOLOv8n (mission5_b1_photo)** | 0.20 | 0.4316 | 0.00% (0/258) | 4.76% (33/694) | 3.47% (33/952) | 47.34% (409/864) | 83.04% (328/395) |

---

## 3. Empirical Insights & Hypothesis Testing

### 3.1 Hypothesis H4 Evaluation
- **Hypothesis H4:** *Fine-tuning improves small-object recall without degrading large-object detection or inducing uncalibrated false-alarm explosions.*
- The scale breakdown provides direct evidence:
  - Does fine-tuning recover sub-16px and 16-32px misses under degraded contrast?
  - Does large-object recall remain high (>75%), confirming that adapting to adverse weather does not cause catastrophic forgetting of foreground semantic features?

### 3.2 Comparison with Faster R-CNN Anchor
- Pretrained Faster R-CNN achieves higher recall on small objects due to its two-stage Region Proposal Network (RPN) and higher native feature pyramid resolution.
- We assess how much of the small-object recall gap is bridged by fine-tuning on adverse weather compared to inference-time multi-scale routing.
