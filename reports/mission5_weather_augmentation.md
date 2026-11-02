# WeatherGuard Mission 5: Experiment B — Weather-Aware Data Augmentation

**Document:** `reports/mission5_weather_augmentation.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Completed & Artifact-Verified  

---

## 1. Executive Summary

Experiment B evaluates whether physics-inspired synthetic weather augmentations during fine-tuning improve out-of-domain generalization and small-object detection on driving scenes beyond standard photometric jitter.

Using the winning layer-freezing strategy from Experiment A (**`dawn_frozen_backbone`**, freeze=10, lr0=0.001), we evaluated three augmentation policies on the 200 stratified DAWN training images:
1. **`dawn_photo_aug`:** Standard photometric jitter (brightness $\pm 20\%$, contrast $\pm 20\%$, HSV color jitter).
2. **`dawn_weather_aug`:** Physics-based weather transforms (Koschmieder atmospheric fog scattering with vertical road-depth gradient, directional rain streak synthesis, defocus blur).
3. **`dawn_combined_aug`:** Joint photometric and adverse-weather transformations.

---

## 2. Cross-Dataset Performance Comparison (Group B BDD100K Dev, $N=160$)

| Augmentation Policy | Description | Micro $F_1$ | $\Delta F_1$ vs Unaugmented | Precision | Recall | Small Recall | Large Recall | Mean Latency |
|---|---|---|---|---|---|---|---|---|
| **Unaugmented Baseline (exp_a1_frozen)** | dawn_frozen_backbone (from Exp A) | 0.4309 | +0.0000 | — | — | — | — | 52.4 ms |
| **dawn_photo_aug** | photo | 0.4316 | +0.0007 | 0.5674 | 0.3483 | 3.47% | 83.04% | 54.1 ms |
| **dawn_weather_aug** | weather | 0.4241 | -0.0067 | 0.4703 | 0.3863 | 9.45% | 80.00% | 57.5 ms |
| **dawn_combined_aug** | combined | 0.4162 | -0.0147 | 0.4536 | 0.3844 | 9.14% | 81.77% | 54.1 ms |

---

## 3. In-Domain Validation Performance (Group A DAWN Val, $N=206$)

| Augmentation Policy | Micro $F_1$ | $\Delta F_1$ vs Unaugmented | Precision | Recall | Small Recall | Large Recall | Mean Latency |
|---|---|---|---|---|---|---|
| **Unaugmented Baseline (exp_a1_frozen)** | 0.6015 | +0.0000 | — | — | — | — | 35.5 ms |
| **dawn_photo_aug** | 0.6152 | +0.0137 | 0.6043 | 0.6266 | 34.48% | 80.89% | 62.9 ms |
| **dawn_weather_aug** | 0.5849 | -0.0166 | 0.5349 | 0.6453 | 41.38% | 81.05% | 64.8 ms |
| **dawn_combined_aug** | 0.5696 | -0.0319 | 0.5164 | 0.6349 | 40.23% | 79.24% | 65.9 ms |

---

## 4. Policy Ranking & Champion Selection

The empirical champion on the validation development cohort (Group B) is **`dawn_photo_aug` (exp_b1_photo)** with $F_1 = 0.4316$.

This model will be carried forward to Experiment D (Scale Stratification), Experiment E (Adaptive Inference Integration), and the final locked Group C test evaluation.
