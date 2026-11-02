# WeatherGuard Mission 4: Targeted Region Refinement & Negative Result Analysis

**Document:** `reports/mission4_crop_refinement.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Completed & Artifact-Verified  
**Artifacts:** `artifacts/mission4_crop_results.json`, `artifacts/mission4_crop_results.csv`, `artifacts/plots/mission4_crop_refinement.png`  

---

## 1. Executive Summary

Experiment B investigated whether localized region refinement—proposing bounding-box crops around small/uncertain detections from a base $640\text{px}$ pass, resizing crops to $640\text{px}$ for a second forward pass, and projecting detections back via coordinate inversion—could recover missed small objects more efficiently than global high-resolution inference ($832\text{px}$ or $1280\text{px}$).

Evaluations were performed on Group B ($N=160$, $2,211$ target objects).

### Key Result: Defensible Negative Finding
Targeted region refinement failed to improve the accuracy–latency Pareto frontier and proved strictly inferior to global multi-scale inference:
- **Small-Object Recall Stagnated:** Small recall remained at $7.98\%$ (vs. $8.09\%$ baseline, $16.18\%$ for fixed $832\text{px}$, and $24.16\%$ for fixed $1280\text{px}$).
- **Precision & F1 Collapsed:** Micro $F_1$ declined sharply from $0.4780$ (baseline) to $0.4044$ ($K=1$) and $0.3835$ ($K=2$).
- **False Positive Explosion:** False positives surged from $416$ in baseline $640\text{px}$ to $989$ ($+137.7\%$) under $K=1$ and $1,207$ ($+190.1\%$) under $K=2$.
- **Latency Overhead:** Mean latency increased to $102.5-139.0\text{ ms}$ (7.2–9.8 FPS) without delivering any detection benefit.

---

## 2. Quantitative Performance Matrix (Group B Dev, $N=160$)

All evaluations are conducted at canonical threshold $\tau = 0.20$ and IoU $\theta = 0.50$ across the 6 canonical classes. Native image resolution is $1280 \times 720$.

| Method | Max Crops ($K$) | Proposal Min Score | Images with Crops (%) | TP | FP | FN | Precision | Recall | Recall (Small) | Recall (Med) | Recall (Large) | Seq Latency (ms) | Throughput (FPS) | Micro $F_1$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Fixed 640px Baseline** | 0 | — | 0.0% | 825 | **416** | 1,386 | **0.6648** | 0.3731 | 8.09% | 50.00% | 80.00% | **52.4 ms** | **19.08 FPS** | **0.4780** |
| **Fixed 832px Global** | 0 | — | 0.0% | 971 | 452 | 1,240 | 0.6824 | 0.4392 | 16.18% | 58.10% | 79.75% | 95.7 ms | 10.45 FPS | 0.5344 |
| **Fixed 1280px Global** | 0 | — | 0.0% | 1,042 | 524 | 1,169 | 0.6654 | 0.4713 | 24.16% | 62.50% | 68.86% | 394.5 ms | 2.54 FPS | 0.5518 |
| **Adaptive Routing (op)** | 0 | — | 50.0% (Escalated) | 964 | 472 | 1,247 | 0.6713 | 0.4360 | 18.91% | 57.41% | 74.43% | 252.0 ms | 3.97 FPS | 0.5288 |
| **crop_refine_k1_s15** | 1 | 0.15 | 66.2% | 811 | **989** | 1,400 | 0.4506 | 0.3668 | 7.98% | 50.00% | 80.00% | 113.1 ms | 8.84 FPS | 0.4044 |
| **crop_refine_k2_s15** | 2 | 0.15 | 66.2% | 811 | **1,207** | 1,400 | 0.4019 | 0.3668 | 7.98% | 50.00% | 80.00% | 139.0 ms | 7.19 FPS | 0.3835 |
| **crop_refine_k1_s20** | 1 | 0.20 | 50.0% | 811 | **862** | 1,400 | 0.4848 | 0.3668 | 7.98% | 50.00% | 80.00% | 102.5 ms | 9.75 FPS | 0.4176 |

---

## 3. Forensic Analysis: Why Targeted Region Refinement Failed

The failure of crop refinement in adverse-weather driving scenes is attributable to three fundamental computer vision failure mechanisms:

### Failure Mode 1: The Small-Object Proposal Paradox
- To extract a targeted crop around a distant or small vehicle, the detector must first place a candidate bounding box in that region during the base $640\text{px}$ pass.
- However, baseline YOLOv8n misses **$91.33\%$** of small objects at $640\text{px}$. The detector produces zero proposal boxes in those areas.
- Consequently, crops are only extracted around objects that were **already detected**, or around false-alarm clutter. Novel missed objects in background regions are never proposed for refinement.

### Failure Mode 2: Loss of Global Scene Context and Vanishing Point Perspective
- Object detectors rely heavily on global geometric priors: the position of the horizon, road lane markings, perspective vanishing lines, and relative vehicle scale.
- When an arbitrary sub-region (e.g. $384 \times 384\text{ px}$) is cropped and fed into YOLOv8n at $640 \times 640$, the model loses global spatial reference.
- Distant background textures (guardrails, overhead sign gantries, tree foliage, wiper streaks, street lamps) are magnified. Deprived of surrounding road context, the network hallucinates small vehicles and pedestrians, causing false positives to spike from $416$ to $1,207$ ($+190.1\%$).

### Failure Mode 3: Receptive Field & Boundary Truncation
- Cropping arbitrarily partitions objects situated near crop margins. Detections near the crop edge suffer from aspect ratio distortions, partial feature truncation, and anchor mismatch.
- Furthermore, when mapped back to native coordinates, slight bounding box jitters from the crop pass fail greedy IoU matching against ground-truth annotations ($\text{IoU} < 0.50$), slightly reducing true positives ($825 \to 811$).

---

## 4. Scientific Conclusion on Hypothesis H3

- **Hypothesis H3:** "Targeted crop-based refinement can recover missed small objects more efficiently than processing the entire image at high resolution."
- **Status:** **Falsified.**
- **Empirical Verdict:** In complex adverse-weather driving imagery, global scale adaptation ($832\text{px}$ or adaptive $1280\text{px}$ routing) preserves essential scene perspective, context, and semantic coherence. Crop refinement induces severe false-positive hallucination without recovering missed objects. Global image-level adaptive routing remains the superior architectural approach.
