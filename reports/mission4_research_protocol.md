# WeatherGuard Mission 4: Research Protocol & Preregistration

**Document:** `reports/mission4_research_protocol.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Frozen & Preregistered  
**Artifact:** `artifacts/mission4_protocol.json`  

---

## 1. Executive Summary & Scientific Motivation

Mission 3 demonstrated that uniform scale expansion (evaluating all frames at $1280\text{px}$) significantly improves small-object detection on the locked BDD100K adverse-weather test set ($N=203$), elevating small-object recall from $8.67\%$ to $28.69\%$ ($+20.02\text{ pp}$) and overall micro $F_1$ from $0.5111$ to $0.5782$ ($+6.71\text{ pp}$, $p < 0.001$). However, this improvement incurs a substantial computational penalty: CPU latency escalates from $61.1\text{ ms}$ (16.4 FPS) to $378.0\text{ ms}$ (2.65 FPS), representing a $6.2\times$ slowdown that drops throughput below interactive and real-time operational thresholds ($\ge 10\text{ FPS}$).

Furthermore, approximately $71.3\%$ of small objects remain undetected even at $1280\text{px}$. 

The core mission of Mission 4 is to investigate whether an **adaptive inference system**—utilizing training-free resolution routing or targeted region refinement—can recover small-object misses under adverse-weather domain shift while preserving computational efficiency, advancing the accuracy–latency Pareto frontier, and dissecting the visual and environmental causes of remaining detection failures.

---

## 2. Research Questions & Formal Hypotheses

### Primary Research Question
**Can an efficient adaptive inference strategy improve small-object detection under adverse-weather domain shift while maintaining a practical accuracy–latency trade-off?**

### Preregistered Hypotheses
- **H1 (Routing Detectability):** A detector can identify images requiring higher-resolution inference using lightweight inference signals derived from a $640\text{px}$ base pass, without applying high-resolution processing across every frame.
- **H2 (Compute Amortization):** Adaptive resolution routing can recover a significant proportion ($\ge 60\%$) of the small-object recall gains achieved by uniform $1280\text{px}$ inference while saving $\ge 40\%$ of average inference latency relative to fixed $1280\text{px}$.
- **H3 (Localized Region Refinement):** Targeted crop-based refinement of spatially clustered candidate regions can recover small-object misses more efficiently than upscaling the entire full-frame image.
- **H4 (Multi-Factor Error Dominance):** The remaining small-object false negatives (~$71\%$) are governed by visual and environmental mechanisms beyond resolution, specifically sub-pixel scale ($< 16^2\text{ px}^2$), optical extinction (fog/rain opacity), low contrast/glare, and boundary occlusion.
- **H5 (Pareto Dominance):** Adaptive inference establishes a superior accuracy–latency Pareto frontier compared with fixed single-resolution YOLOv8n operating at standard operating points.

---

## 3. Cohort Management & Leakage Prevention Protocol

The research maintains the strict three-cohort architecture established in Mission 3:

| Cohort | Source Dataset | Size | Role & Constraints |
|---|---|---|---|
| **Group A** | DAWN | $N = 821$ | Historical threshold-transfer development cohort |
| **Group B** | BDD100K | $N = 160$ | **Development & Selection Cohort:** Used exclusively for routing policy formulation, hyperparameter search, crop tuning, and threshold selection |
| **Group C** | BDD100K | $N = 203$ | **Locked External Test Cohort:** Evaluated strictly once after all policies and parameters are frozen. Zero parameter tuning or policy selection |

**Leakage Controls:**
1. Zero overlapping image IDs between Group B and Group C ($\text{Group B} \cap \text{Group C} = \emptyset$).
2. Pixel-level and hash-level deduplication confirmed.
3. No ground-truth bounding box counts, coordinates, or labels may be accessed at inference time during routing decisions or crop proposals.

---

## 4. Experimental Plan

### Experiment A: Image-Level Adaptive Resolution Routing
Evaluate deterministic, training-free routing policies that dynamically decide image resolution:
1. **Fixed Baselines:** Fixed $640\text{px}$, Fixed $832\text{px}$, Fixed $1280\text{px}$.
2. **Small-Box Trigger:** Escalate $640\text{px} \to 1280\text{px}$ if the base $640\text{px}$ pass detects $\ge 1$ small box ($\text{area} < 1,024\text{ px}^2$).
3. **Low-Confidence Trigger:** Escalate $640\text{px} \to 1280\text{px}$ if the base pass contains $\ge k$ low-confidence detections ($0.05 \le \text{score} < 0.25$).
4. **Hierarchical Two-Stage:** $640\text{px} \to 832\text{px}$ on moderate uncertainty; escalate to $1280\text{px}$ only on high small-box density.
5. **Detection-Density Trigger:** Escalate based on total bounding box density and spatial proximity.
6. **Combined Confidence-Scale Trigger:** Escalate if small boxes are detected OR low-confidence detections exceed threshold $k \ge 2$.

### Experiment B: Targeted Region Refinement
1. **Candidate Extraction:** Extract candidate RoIs from base $640\text{px}$ predictions using low-confidence or small detections.
2. **Spatial Clustering:** Merge overlapping or nearby candidate RoIs into a maximum of $K = 2$ crops.
3. **Context Expansion:** Expand crop boundaries by margin $\alpha = 0.20$ (20% padding) with coordinate boundary clipping ($[0, W]$ and $[0, H]$).
4. **Crop Inference:** Resize crop to $640 \times 640$, run YOLOv8n forward pass.
5. **Inverse Mapping:** Project detected boxes back to native $1280 \times 720$ image coordinates.
6. **Merging & Suppression:** Combine base detections with crop detections using greedy IoU-based NMS at threshold $\theta = 0.50$.

### Experiment C: Failure-Cause Analysis
Deconstruct remaining small-object false negatives into structured visual failure modes:
1. Sub-pixel scale ($< 16^2\text{ px}^2$)
2. Optical extinction & weather opacity (fog / heavy rain)
3. Low contrast & glare
4. Occlusion & spatial crowding (inter-object IoU $> 0.2$)
5. Boundary truncation ($\le 10\text{ px}$ from image edge)
6. False positive categorization (roadside furniture, wiper streaks, reflections, class confusion).

### Experiment D: Weather × Scale × Inference Interaction
Stratify performance across Clear, Overcast, Rainy, Snowy, and Foggy conditions, analyzing escalation rates, small-object recall gains, and failure modes across weather strata.

### Experiment E: Threshold & Policy Optimization
Search for optimal path-conditioned confidence thresholds $(\tau_{640}, \tau_{832}, \tau_{1280})$ on Group B development data across grid $\tau \in \{0.15, 0.20, 0.25\}$.

### Experiment F: Final External Test Evaluation & Bootstrap Validation
Evaluate winning frozen adaptive configurations on Locked Group C ($N=203$). Compute paired image-level bootstrap statistics ($B=1,000$ resamples) with dual independent random seeds (`20261101` and `20261102`).

---

## 5. Preregistration Sign-Off

The experimental protocol and constraints are frozen. All parameter searches, policy comparisons, and threshold calibrations must be completed on Group B prior to unfreezing Locked Group C.
