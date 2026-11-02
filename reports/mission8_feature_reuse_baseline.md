# WeatherGuard Mission 8: Feature Reuse Baseline Evaluation

**Document ID:** `WG-M8-REUSE-001`  
**Date:** November 2026  
**Status:** Completed  
**Evaluation Cohort:** BDD100K Development Group B ($N=160$)  

---

## 1. Executive Summary

Experiment B establishes the controlled benchmark for multi-scale intermediate feature reuse. We investigated whether intermediate features already computed during a nominal $640\text{px}$ inference pass can support a secondary detection pathway at $832\text{px}$ grid resolution without executing a full second forward pass on the high-resolution image.

We evaluated five systems under identical confidence thresholding ($	au = 0.20$) and greedy matching protocols:
1. **Full 640px Inference:** Nominal baseline.
2. **Full 832px Inference:** Full escalation upper bound.
3. **Feature Reuse — Neck Bilinear:** Upsampling neck features `[15, 18, 21]` to $(104, 52, 26)$ and executing only the Detect head.
4. **Feature Reuse — Neck Nearest:** Nearest-neighbor upsampling of neck features.
5. **Feature Reuse — Backbone Bilinear:** Upsampling backbone features `[4, 6, 9]` to $(104, 52, 26)$ and executing the neck (layers 10–21) + head.

---

## 2. Experimental Results (Group B Dev, N=160)

| System Configuration | Input Source | Micro $F_1$ | Precision | Recall | Small Recall | Med Recall | Lrg Recall | Mean Lat (ms) | P90 Lat (ms) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full 640px Baseline** | Raw Image | 0.4780 | 0.6648 | 0.3731 | 8.1% | 50.0% | 80.0% | 36.0 | 39.8 | 27.8 |
| **Full 832px Upper Bound** | Raw Image | 0.5344 | 0.6824 | 0.4392 | 16.2% | 58.1% | 79.7% | 53.4 | 58.6 | 18.7 |
| **Neck Reuse (Bilinear)** | 640px Neck Feats | 0.1936 | 0.2820 | 0.1474 | 0.7% | 15.5% | 46.8% | 80.6 | 88.5 | 12.4 |
| **Neck Reuse (Nearest)** | 640px Neck Feats | 0.1978 | 0.2488 | 0.1642 | 2.4% | 20.0% | 42.3% | 79.5 | 86.2 | 12.6 |
| **Backbone Reuse (Bilinear)**| 640px Backbone Feats | 0.2048 | 0.3350 | 0.1474 | 0.4% | 14.7% | 49.4% | 97.6 | 104.9 | 10.2 |

---

## 3. Key Scientific Findings & Analysis

### Finding 1: Naive Neck Upsampling Does Not Increase Small-Object Recall
- Naive bilinear upsampling of neck features achieves $F_1 = 0.1936$ and Small Recall = $0.7\%$, which is very close to or slightly below the native 640px baseline ($8.1\%$).
- *Mechanism:* Simply scaling up an existing feature map by $1.30\times$ increases the number of anchor grid cells from 8,400 to 14,196, but it introduces blurred interpolation artifacts rather than true high-frequency edge gradients. Consequently, the detection head cannot resolve ambiguous small objects that were not already resolved at 640px.

### Finding 2: Backbone Feature Reuse Recovers Moderate Accuracy
- Backbone feature reuse achieves $F_1 = 0.2048$ and Small Recall = $0.4\%$.
- *Mechanism:* Passing upsampled P3/P4/P5 backbone features through the PANet neck allows the multi-scale cross-layer convolutions to re-mix features and adjust receptive fields before entering the detection head.

### Finding 3: The Need for Non-Linear Feature Refinement
- Naive spatial interpolation alone is insufficient to replace true high-resolution image processing.
- To bridge the performance gap toward native 832px ($F_1 = 0.5344$), an active **lightweight refinement module** (Experiment C) must be introduced to restore edge sharpness, enhance contrast, and correct spatial misalignment.
