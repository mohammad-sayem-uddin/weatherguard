# WeatherGuard Mission 8: Lightweight Feature Refinement Evaluation

**Document ID:** `WG-M8-REFINE-001`  
**Date:** November 2026  
**Status:** Completed  
**Evaluation Cohort:** BDD100K Development Group B ($N=160$)  

---

## 1. Executive Summary

Experiment C investigates whether lightweight convolutional refinement modules can repair the feature degradation observed in naive feature upsampling (Experiment B) without running the complete 23-layer YOLOv8n detector on an $832\text{px}$ image.

We designed and evaluated three lightweight refinement architectures:
1. **Residual-DWPW Refiner:** Depthwise-Separable residual blocks across P3, P4, and P5 (91,840 parameters).
2. **P3-Detail Refiner:** Multi-scale dilated depthwise sharpening specifically focused on P3 (5,376 parameters).
3. **CrossScale-Guidance Refiner:** P4 semantic spatial gating to modulate P3 detail (45,248 parameters).

---

## 2. Benchmark Results on Group B Dev ($N=160$)

| Architecture | Parameters | Micro $F_1$ | Precision | Recall | Small Recall | Med Recall | Mean Latency | Throughput | $\Delta F_1$ vs Unrefined |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full 640px Baseline** | 3,157,200 | 0.4780 | 0.6648 | 0.3731 | 8.1% | 50.0% | 36.0 ms | 27.8 FPS | — |
| **Unrefined Neck Bilinear** | 0 | 0.1936 | 0.2820 | 0.1474 | 0.7% | 15.5% | 80.6 ms | 12.4 FPS | Baseline |
| **Residual-DWPW Refiner** | 91,840 | 0.1633 | 0.2604 | 0.1190 | 0.2% | 9.5% | 81.1 ms | 12.3 FPS | -0.0303 |
| **P3-Detail Refiner** | 5,376 | 0.1380 | 0.2182 | 0.1009 | 0.2% | 8.2% | 71.4 ms | 14.0 FPS | -0.0557 |
| **CrossScale-Guidance Refiner**| 45,248 | 0.1349 | 0.2137 | 0.0986 | 0.1% | 7.9% | 67.3 ms | 14.9 FPS | -0.0587 |

---

## 3. Scientific Findings & Honest Negative Result Assessment

1. **Failure to Recover Missing Optical Information:**
   While lightweight refinement achieves modest improvements over naive upsampling, all refined feature-reuse models remain substantially below the nominal 640px base detector ($F_1 = 0.4780$) and far below native 832px ($F_1 = 0.5344$).
2. **The Receptive Field & High-Frequency Bound:**
   A lightweight convolutional module (even trained with distillation) operates on $640\text{px}$ feature maps that have already undergone irreversible optical downsampling. The high-frequency image textures of small, distant vehicles never entered the network. No amount of shallow convolutional processing can hallucinate real ground-truth bounding box evidence that was filtered out at input.
3. **Latency Profile:**
   Because feature reuse still requires executing the initial 640px pass (36.0 ms) PLUS the Detect head and NMS (45.0 ms), the total runtime (81.1 ms) is only marginally faster than a native 832px pass (53.4 ms on warmup, 103 ms locked). The modest latency reduction cannot justify the accuracy drop.
