# WeatherGuard Mission 10 — Phase 2: Experimental Protocol & Sequence Partitioning

**Mission:** Mission 10 — Independent Validation & Temporal Reliability  
**Date:** 2026-11-02  
**Author:** Principal Computer Vision Researcher, Video Understanding Engineer & Experimental Scientist  
**Hardware Environment:** Apple MacBook Air M1, 8 GB Unified Memory, macOS (CPU-only)  
**Deliverable:** `reports/mission10_experimental_protocol.md`  

---

## 1. Protocol Executive Summary

Mission 10 establishes a rigorous, preregistered experimental protocol to evaluate WeatherGuard on continuous driving video streams. To prevent statistical data snooping and confirmation bias:
1. **Strict Sequence-Level Partitioning:** Complete driving video sequences are assigned to partitions. No sequence appears in more than one partition, and no frames from the same video are split across partitions.
2. **Zero Historical Leakage:** All 1,213 frames across the 6 sequences were audited against BDD100K Group B ($N=160$), Group C ($N=203$), and Mission 9 ($N=1,621$). **Zero overlapping frames exist.**
3. **Strict Separation of Three Evaluation Streams (Phase 4):**
   - **Stream 1: Object Detection:** Measures raw detector precision, recall, and $F_1$. **Zero tracking or extrapolated boxes are counted as detector outputs.**
   - **Stream 2: Multi-Object Tracking:** Measures trajectory continuity, ID switches, and fragmentation on ground-truth track IDs.
   - **Stream 3: Short-Gap Recovery:** Measures the precision, duration, and false continuation rate of extrapolated states.

---

## 2. Partition Manifest & Dataset Composition

### Cohort Overview
- **Independent Development Cohort:** 2 sequences, **405 frames**, **6,411 ground-truth boxes**.
- **Independent Locked Test Cohort:** 4 sequences, **808 frames**, **14,848 ground-truth boxes**.
- **Total Volume:** 6 continuous sequences, **1,213 frames**, **21,259 ground-truth boxes**.

### Sequence Partition Specifications

| Partition | Sequence ID | Weather | Lighting | Scene | Frames | GT Boxes | Unique Tracks | Dominant Visual Challenge |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Development** | `b24380ab-6dbeb908` | Rainy | Night | Highway | 202 | 2,786 | 98 | Road spray, headlight glare, wet asphalt reflections |
| **Development** | `b255cd6c-2f889586` | Snowy | Daytime | City Street | 203 | 3,625 | 184 | Active snowfall, roadside snow accumulation, pedestrian slush |
| **Locked Test** | `b1e1a7b8-65ec7612` | Rainy | Daytime | Highway | 202 | 2,485 | 112 | Wiper blade sweep occlusion, water droplet blur |
| **Locked Test** | `b1d0a191-06deb55d` | Snowy | Daytime | Dense Urban | 202 | 3,610 | 179 | Low contrast, snowbank vehicle occlusions |
| **Locked Test** | `b24d283f-33783d1b` | Overcast | Daytime | City Street | 202 | 4,323 | 245 | Extreme multi-class vehicle density (4,323 objects) |
| **Locked Test** | `b23c9e00-b425de1b` | Clear | Night | Highway | 202 | 4,430 | 148 | High dynamic range, artificial illumination falloff |

---

## 3. Preregistered Evaluation Parameters

- **Confidence Threshold:** $\tau = 0.20$ (locked from DAWN dev tuning and cross-dataset validation).
- **IoU Matching Threshold:** $\text{IoU} = 0.50$ (standard PASCAL/COCO threshold).
- **Target Evaluation Classes (6 Canonical Automotive Classes):**
  `Car`, `Person` (pedestrian/rider), `Truck`, `Bus`, `Bicycle`, `Motorcycle`.
- **Dynamic Router Settings:**
  - Base Resolution: $640\times 640$
  - Escalated Resolution: $832\times 832$
  - Small-Object Escalation Trigger: $k_{\text{small}} \ge 1$ where $\text{area} < 1024\text{ px}^2$ ($<32\times 32$).
  - Hold Duration: $H = 3$ frames ($0.60\text{s}$ at $5.0\text{ FPS}$).
- **Tracking Settings (AdverseWeatherByteTracker):**
  - High Detection Threshold: $\tau_{\text{high}} = 0.20$
  - Low Association Threshold: $\tau_{\text{low}} = 0.10$
  - Track Confirmation Frames: $1$
  - Maximum Track Age (Coast): $6$ frames ($1.20\text{s}$)
- **Extrapolation Recovery Settings:**
  - Maximum Gap: $G = 1$ frame ($0.20\text{s}$)
  - Confidence Decay Factor: $\gamma = 0.85$
  - Decay Step: $s_{\text{extrap}} = \gamma \cdot s_{\text{prior}}$

---

## 4. Integrity and Leakage Verification

```json
{
  "group_b_overlap_frames": 0,
  "group_c_overlap_frames": 0,
  "mission9_overlap_sequences": 0,
  "manifest_sha256": "ef63bba0e6df90eb535ed53f84698ae2b7a34af4b42c07f8123a1ab3c7034bc9"
}
```
All checks passed. The independent evaluation cohort is officially frozen.
