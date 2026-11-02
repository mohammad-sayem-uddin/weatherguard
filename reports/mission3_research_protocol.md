# WeatherGuard Mission 3 Pre-Registered Research Protocol
## Scale-Aware Object Detection, Multi-Scale Inference & Adaptive Confidence Calibration

**Document:** `reports/mission3_research_protocol.md`  
**Date:** 2026-11-01  
**Status:** FROZEN BEFORE EVALUATION  
**Author:** WeatherGuard Autonomous Research Agent  

---

## 1. Research Objectives & Questions

Mission 2 revealed that while YOLOv8n offers a $20.9\times$ steady-state CPU inference speed advantage over Faster R-CNN ($61.1\text{ ms}$ vs. $1,275.6\text{ ms}$), it suffers severe small-object blindness on driving video ($8.67\%$ recall on objects $< 32^2\text{ px}^2$, missing $91.33\%$ of small targets). 

Mission 3 investigates whether lightweight, zero-training inference enhancements—specifically **multi-scale resolution scaling**, **sliced/tiled inference**, and **scale-adaptive confidence calibration**—can bridge this small-object gap while preserving practical computational efficiency.

### Primary Research Question
> **Can scale-aware inference and object-scale-dependent confidence calibration improve small-object detection under adverse weather and cross-dataset domain shift without sacrificing computational efficiency?**

### Sub-Questions
* **RQ1 (Multi-Scale Resolution Scaling):** Does increasing YOLOv8n's inference resolution ($640 \to 832 \to 960 \to 1280$) recover distant small traffic objects, and what is the latency cost?
* **RQ2 (Sliced / Tiled Inference):** Does slicing images into overlapping local tiles improve small-object feature resolution, and what is the trade-off in duplicate detections or false alarms?
* **RQ3 (Scale-Adaptive Calibration):** Can partitioning confidence thresholds by predicted bounding-box scale ($\tau_{\text{small}}, \tau_{\text{med}}, \tau_{\text{large}}$) outperform a single global confidence threshold?
* **RQ4 (Pareto Frontier):** Which method or combination yields the optimal accuracy-versus-latency trade-off on CPU hardware?
* **RQ5 (Weather Invariance):** Are scale-aware detection gains consistent across rainy, snowy, overcast, and clear driving conditions?
* **RQ6 (Gap Reduction vs. Reference):** Can an optimized scale-aware YOLOv8n reduce its $+0.176$ $F_1$ deficit against the much heavier Faster R-CNN ResNet-50 FPN?
* **RQ7 (Taxonomic & Precision Costs):** Do small-object gains degrade large-object accuracy, inflate false positives, or compromise precision?

---

## 2. Experimental Data Strategy (Disjoint Splits)

To eliminate methodological leakage and guarantee scientific validity, the data is partitioned into three strictly disjoint groups:

```
Group A: DAWN Development (821 images)
         ├── Historical training split used to calibrate baseline operating points (0.60 FRCNN, 0.20 YOLO)
         └── Pointers: train-00000-of-00001.parquet

Group B: BDD100K Development Cohort (160 images) — NEW
         ├── Sampled from BDD100K validation split (seed 101, daytime adverse + clear)
         ├── Used exclusively for selecting multi-scale resolutions, tiling parameters, and adaptive thresholds
         └── Disjointness: Zero overlap with Group C (verified)

Group C: BDD100K Locked Test Cohort (203 images) — LOCKED
         ├── The exact 203 images evaluated in Mission 2 (seed 42)
         ├── Frozen; NO parameter tuning or threshold selection permitted on this cohort
         └── Used strictly for unbiased final testing of frozen configurations
```

---

## 3. Predefined Metrics & Scale Definitions

### 3.1 Object Scale Conventions
Object area is defined in the **original un-resized image coordinate system** ($1280 \times 720$):
$$\text{Area} = \text{width} \times \text{height} = (x_2 - x_1) \times (y_2 - y_1)$$
- **Small Objects:** $\text{Area} < 32^2 = 1,024\text{ px}^2$.
- **Medium Objects:** $1,024 \le \text{Area} < 96^2 = 9,216\text{ px}^2$.
- **Large Objects:** $\text{Area} \ge 96^2 = 9,216\text{ px}^2$.

### 3.2 Evaluation Metrics
* **Micro $F_1$:** Harmonic mean of micro-precision and micro-recall across all evaluated instances at the specified threshold.
* **Macro mAP@0.5:** VOC 101-point interpolated AP, averaged across present target classes.
* **Scale-Specific Recall ($R_{\text{small}}, R_{\text{med}}, R_{\text{large}}$):** Proportion of ground-truth objects in each scale bin matched by a valid prediction ($\text{IoU} \ge 0.50$).
* **Steady-State Latency ($T_{\text{infer}}$):** Mean wall-clock inference time per image on CPU (ms), excluding one-time model initialization.
* **Relative Latency Factor ($L_{\text{rel}}$):** $\frac{T_{\text{infer}}}{T_{\text{baseline}}}$.
* **F1 per CPU-Second:** Descriptive efficiency index $\frac{F_1}{T_{\text{infer}} \text{ (seconds)}}$.

---

## 4. Experimental Designs

### Experiment A: Multi-Scale Inference
* **Hypothesis:** Ultralytics YOLOv8n downsamples $1280 \times 720$ inputs to $640 \times 640$ by default, collapsing small objects into sub-feature resolution. Increasing inference resolution will directly increase small-object recall.
* **Candidate Resolutions:**
  - $640 \times 640$ (Baseline, downscaled $2.0\times$)
  - $832 \times 832$ (Moderate scaling, $1.3\times$ scaling)
  - $960 \times 960$ (High resolution, $1.5\times$ scaling)
  - $1280 \times 1280$ (Native resolution, preserving full $1280\text{ px}$ width)
* **Protocol:** Evaluate all 4 candidate resolutions on Group B (Development). Record small/medium/large recall, mAP@0.5, and latency.

### Experiment B: Sliced / Tiled Inference
* **Hypothesis:** Slicing high-resolution $1280 \times 720$ images into overlapping $2 \times 2$ tiles allows YOLOv8n to process sub-regions at native pixel scale without running full-frame high-resolution convolutions.
* **Tiling Protocol:**
  - Grid: $2 \times 2$ tiles (4 tiles per image).
  - Tile dimensions: $720 \times 450$ px with $\approx 20\%$ spatial overlap.
  - Coordinate re-projection: Each tile prediction $[x_{\text{tile}}, y_{\text{tile}}]$ is translated back to full-frame coordinates:
    $$x_{\text{global}} = x_{\text{tile}} + x_{\text{offset}}, \quad y_{\text{global}} = y_{\text{tile}} + y_{\text{offset}}$$
  - Cross-Tile NMS: Standard greedy NMS ($\text{IoU} = 0.50$) applied across merged predictions to eliminate duplicate edge detections.

### Experiment C: Scale-Adaptive Confidence Calibration
* **Hypothesis:** High confidence thresholds filter out faint, distant small objects, while low thresholds introduce large phantom false alarms. Tuning separate thresholds conditioned on *predicted box area* will optimize global $F_1$.
* **Critical Methodological Guardrail:**
  > **Thresholding must be conditioned strictly on PREDICTED box area, NEVER on ground-truth box area.** Ground-truth scale is unknown at deployment.
* **Formulation:**
  $$\tau(\hat{B}) = \begin{cases} 
  \tau_{\text{small}} & \text{if } \text{Area}(\hat{B}) < 1,024\text{ px}^2 \\
  \tau_{\text{med}} & \text{if } 1,024 \le \text{Area}(\hat{B}) < 9,216\text{ px}^2 \\
  \tau_{\text{large}} & \text{if } \text{Area}(\hat{B}) \ge 9,216\text{ px}^2 
  \end{cases}$$
* **Optimization Objective:** Select $(\tau_{\text{small}}^*, \tau_{\text{med}}^*, \tau_{\text{large}}^*)$ via grid search in $[0.05, 0.70]$ (step $0.05$) to maximize micro $F_1$ on Group B (Development).

### Experiment D: Combined Scale-Aware Configurations
* Formulate up to 2 promising combined configurations based on Group B findings (e.g., Best Resolution + Scale-Adaptive Threshold).

### Experiment E: Computational Efficiency & Pareto Analysis
* Construct the empirical accuracy-latency Pareto frontier across all configurations on CPU.

---

## 5. Protocol Freezing & Test Evaluation Gate

All architectural decisions, resolution choices, tiling parameters, and adaptive threshold values must be **selected on Group B and locked** before executing any inference on Group C (Locked Test Cohort).

Statistical significance will be verified on Group C using paired image-level bootstrap resampling ($B = 1,000$, seed 20261101, secondary sensitivity seed 20261102).
