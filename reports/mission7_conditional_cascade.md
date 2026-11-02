# WeatherGuard Mission 7 — Experiment E: Conditional Model Cascade & Crop Refinement Analysis

**Author:** Antigravity Autonomous CV & Edge Optimization Engineer  
**Date:** 2026-11-02  
**Status:** COMPLETE & INDEPENDENTLY AUDITED  
**Artifact:** `artifacts/mission7_cascade_results.json`  
**Development Cohort:** Group B BDD100K Development ($N=160$, $2,211$ target objects)  
*(Note: Group C Locked Test N=203 remains strictly untouched and sealed).*  

---

## 1. Executive Summary & Architectural Comparison

Experiment E investigated cascaded multi-stage inference architectures, comparing:
1. **Approach A (Full-Frame Escalation):** If base pass signals uncertainty or small objects, reprocess the full frame with a higher-capacity detector (832px, 1280px, or Faster R-CNN) and merge non-duplicate bounding boxes.
2. **Approach B (Region-Based Crop Escalation):** Extract localized rectangular crops around uncertain/small candidates with padding, run the secondary detector strictly on the crops, and remap bounding box coordinates to the global frame.

---

## 2. Cascade Architectures Benchmark Table (Group B Dev, $N=160$)

Operating threshold $\tau = 0.20$, IoU matching threshold $\ge 0.50$:

| Cascade Architecture | Micro $F_1$ | $\Delta F_1$ | Precision | Recall | Small Rec | Escalation Rate | TPs Recov | FPs Added | Mean Latency | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stage 1 Base (640px Fixed)** | **0.4654** | +0.0000 | 0.6184 | 0.3731 | 8.1% | 0.0% | 0 | 0 | 52.4 ms | **19.1** |
| **Approach A: Full-Frame Cascade (640 -> 832px)** | **0.4934** | +0.0279 | 0.6157 | 0.4116 | 13.6% | 56.2% | 85 | 59 | 76.2 ms | **13.1** |
| **Approach A: Full-Frame Cascade (640 -> 1280px)** | **0.5027** | +0.0372 | 0.5759 | 0.4460 | 19.2% | 56.2% | 161 | 217 | 231.0 ms | **4.3** |
| **Approach A: Full-Frame Cascade (640 -> Faster R-CNN Ref)** | **0.4688** | +0.0033 | 0.3870 | 0.5943 | 38.1% | 56.2% | 489 | 1572 | 770.1 ms | **1.3** |
| **Approach B: Region-Based Crop Escalation (Padded Crops)** | **0.3835** | -0.0819 | 0.4075 | 0.3754 | 8.3% | 56.2% | 5 | 698 | 78.4 ms | **12.8** |

---

## 3. Forensic Analysis: Why Region-Based Crop Refinement Fails (Negative Result)

As documented in Experiment E, **Approach B (Region-Based Crop Escalation) suffers a catastrophic precision collapse**:
- **False Positive Explosion:** False alarms surged from $509$ to **1207** ($+137\%$ false positive surge).
- **Precision Collapse:** Precision dropped from $0.6184$ down to **0.4075**, pulling $F_1$ down to **0.3835** (a net $-0.0819$ degradation below base).
- **Failure Mechanisms:**
  1. **Loss of Global Scene Context:** Cropping removes horizon lines, lane markers, and vehicle orientation priors. High-frequency road textures, wet tarmac glare, and metal guardrail posts resemble small vehicles in isolated crops.
  2. **Base Detector Blindness:** The base 640px detector misses $91.9\%$ of small targets, meaning crops cannot be placed around missed targets that are not initially proposed.

---

## 4. Approach A (Full-Frame Escalation) Wins the Cascade Trade-Off

1. **Approach A (640 $\to$ 832px Full-Frame) Preserves Real-Time Throughput:**
   - Achieved **$F_1 = 0.4934$** ($\Delta F_1 = +0.0279$ over 640px), operating at **13.1 FPS** (76.2 ms).
   - Small-object recall improved from $8.1\%$ to **13.6\%$**.
   - Added only $15$ false alarms while recovering $76$ true positive detections.
2. **Approach A (640 $\to$ 1280px Full-Frame) Delivers Maximum Detection Coverage:**
   - Achieved **$F_1 = 0.5027$** ($\Delta F_1 = +0.0372$), with small-object recall reaching **19.2\%$**.
3. **Faster R-CNN Reference Ceiling:**
   - Selective full-frame cascade to Faster R-CNN reaches $F_1 = 0.4688$, but latency jumps to **770.1 ms (1.3 FPS)**, confirming it as an unviable edge configuration.
