# WeatherGuard Mission 5: Training & Fine-Tuning Protocol

**Document:** `reports/mission5_training_protocol.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Frozen & Preregistered  

---

## 1. Executive Summary & Research Motivation

In Missions 1–4, WeatherGuard investigated zero-shot cross-dataset transfer and inference-time adaptation (multi-scale inference, adaptive routing) using an off-the-shelf COCO-pretrained YOLOv8n detector. 

Mission 5 investigates the complementary axis: **supervised domain adaptation via lightweight fine-tuning**. We evaluate whether adapting model weights on adverse-weather training imagery (DAWN, $N=821$) and weather-aware synthetic data augmentation can:
1. Improve in-domain and out-of-domain adverse-weather feature extraction.
2. Recover small-object misses without exacerbating false alarms.
3. Complement Mission 4's adaptive resolution routing to establish a new state-of-the-art Pareto frontier.

---

## 2. Formal Hypotheses

- **H1 (Adverse-Weather Fine-Tuning):** Fine-tuning YOLOv8n on adverse-weather driving scenes (DAWN) improves feature representation on out-of-domain driving datasets (BDD100K) compared to stock COCO pretraining.
- **H2 (Weather-Aware Augmentation):** Synthetic weather augmentations (atmospheric fog contrast reduction, rain streaks, defocus blur) during training yield superior cross-dataset transfer compared to standard photometric jitter alone.
- **H3 (Cross-Domain Joint Training):** *Formally blocked due to Group D unavailability on local storage.*
- **H4 (Scale-Specific Gain Parity):** Fine-tuning improves small-object recall without degrading large-object detection or inducing uncalibrated false-alarm explosions.
- **H5 (Fine-Tuning + Adaptive Inference Synergy):** Combining a fine-tuned detector with Mission 4 adaptive resolution routing yields strictly superior Pareto trade-offs than either strategy in isolation.

---

## 3. Training Architecture & Budget Constraints

All training runs are executed under strict CPU-only constraints on an 8 GB Apple M1 machine:

| Parameter | Specification | Scientific Rationale |
|---|---|---|
| **Base Weights** | `yolov8n.pt` (3.0M params, 8.2 GFLOPs) | Lightweight baseline with verified COCO weights |
| **Input Resolution** | $640 \times 640$ pixels | Balances gradient fidelity and CPU training throughput |
| **Batch Size** | 8 | Prevents memory paging / swap churn on 8 GB RAM |
| **Epoch Budget** | 5 epochs per candidate | Amortizes fine-tuning (~5–8 minutes per run) without catastrophic forgetting |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 0.0005$) | Proven convergence stability on small fine-tuning budgets |
| **Initial Learning Rate** | $\eta = 0.001$ (frozen/partial), $\eta = 0.0005$ (full) | Prevents gradient explosion in pretrained layers |
| **Validation Loss Cohort** | Group A DAWN Val ($N = 206$ images) | In-domain checkpoint checkpoint selection |
| **Selection Cohort** | Group B BDD100K Dev ($N = 160$ images) | Out-of-domain transfer ranking (zero gradient leakage) |

---

## 4. Fine-Tuning Configurations (Experiment A)

1. **`dawn_frozen_backbone`:**
   - Layers 0..9 frozen (`freeze = 10` in YOLOv8n: Conv, C2f, SPPF).
   - Only the FPN neck (layers 10..21) and Detect head (layer 22) are updated.
   - *Hypothesis:* Preserves general semantic filters while adapting feature fusion and classification priors to weather.
2. **`dawn_partial_backbone`:**
   - Layers 0..4 frozen (`freeze = 5`).
   - Deep backbone (C2f, Conv, SPPF), neck, and head are trainable.
   - *Hypothesis:* Allows high-level receptive fields to adapt to degraded textures.
3. **`dawn_full_finetuning`:**
   - All layers trainable (`freeze = 0`).
   - *Hypothesis:* Maximum domain adaptation capacity, but highest risk of overfitting to DAWN's specific camera sensors.

---

## 5. Weather-Aware Augmentation Configurations (Experiment B)

1. **`dawn_photo_aug`:**
   - Standard photometric transformations: Brightness ($\pm 20\%$), Contrast ($\pm 20\%$), HSV color space jitter ($h=0.015, s=0.7, v=0.4$).
2. **`dawn_weather_aug`:**
   - Physics-inspired weather transforms: Fog-like atmospheric contrast reduction (simulated Mie scattering), synthetic directional rain streaks, and Gaussian/defocus blur.
3. **`dawn_combined_aug`:**
   - Combined photometric and adverse-weather transformations.

---

## 6. Selection & Locking Criteria

- The winning model and augmentation policy will be selected strictly based on Micro $F_1$ and Small-Object Recall on **Group B Development ($N=160$)**.
- Group C ($N=203$) remains locked until all training runs, hyperparameter selections, and adaptive routing evaluations are frozen.
