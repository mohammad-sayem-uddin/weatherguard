# WeatherGuard Pre-Registered Protocol: Cross-Dataset Generalization & External Validation

**Document:** `reports/cross_dataset_protocol.md`  
**Status:** FROZEN BEFORE EXTERNAL EVALUATION  
**Date:** 2026-11-01  
**Authors:** WeatherGuard Research Team  

---

## 1. Research Objectives & Questions

The primary objective of Mission 2 is to evaluate whether object detection performance and confidence thresholds calibrated on the DAWN adverse-weather benchmark generalize to an independent, real-world adverse-weather driving dataset (**BDD100K**).

### Primary Research Question
> **How well do detection performance and confidence thresholds selected using DAWN transfer to an independent adverse-weather driving dataset?**

### Sub-Questions
* **RQ1 (Detector Performance Shift):** How much does detector performance (mAP@0.5 and micro F1) change when evaluated on an independent driving dataset under adverse weather?
* **RQ2 (Threshold Transferability):** Do the confidence thresholds selected on the DAWN development set (0.60 for Faster R-CNN, 0.20 for YOLOv8n) retain their effectiveness on the external dataset, or do they suffer performance degradation compared to the default (0.50) or external oracle operating points?
* **RQ3 (Condition-Specific Degradation):** Which adverse weather conditions (rain, snow, fog, overcast) exhibit the greatest cross-dataset performance drops?
* **RQ4 (Architectural Resilience):** Do Faster R-CNN (two-stage, ResNet-50 FPN) and YOLOv8n (single-stage, anchor-free CNN) exhibit different degradation patterns under cross-dataset domain shift?
* **RQ5 (Detector Ranking Consistency):** Does the detector with higher performance on DAWN (Faster R-CNN) retain its superiority on the external dataset?
* **RQ6 (Class-Specific Sensitivity):** Which object classes (e.g., small objects like pedestrians and bicycles vs. large objects like trucks and buses) suffer the most severe domain-shift penalties?

---

## 2. Evaluation Protocol & Metrics

To guarantee scientific comparability with the prior DAWN study, all metrics, matching algorithms, and class mappings are frozen according to the definitions in `src/metrics.py` and `configs/default.yaml`.

### 2.1 Metrics
* **Primary Metric:** Micro-averaged $F_1$ score across all evaluated instances at the specified confidence threshold:
  $$F_1 = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
* **Secondary Metric:** Mean Average Precision at IoU 0.50 ($\text{mAP@0.5}$), calculated using the VOC 101-point interpolated precision-recall curve, macro-averaged across classes with at least one ground-truth box.
* **Operational Metrics:** Precision, Recall, True Positives (TP), False Positives (FP), and False Negatives (FN).
* **Computational Metrics:** Steady-state per-image CPU inference latency (ms), cold initialization time (s), and model memory footprint.

### 2.2 Matching Algorithm
* **Greedy IoU Matching:** Predictions are sorted in descending order of confidence score. Each prediction is matched to the highest-IoU ground-truth box sharing the same target class, provided $\text{IoU} \ge 0.50$.
* Each ground-truth box can be matched at most once.
* Unmatched predictions above the confidence threshold are counted as False Positives (FP).
* Unmatched ground-truth boxes are counted as False Negatives (FN).
* Predictions with predicted classes outside the target 6-class vocabulary are ignored (not penalized as FP), preserving parity with the DAWN evaluation protocol.

### 2.3 Operating Points Under Evaluation
1. **DAWN-Selected Dev Threshold (Frozen):**
   - Faster R-CNN ResNet-50 FPN: **0.60**
   - YOLOv8n: **0.20**
   *Strictly locked from the DAWN development study; no external tuning is permitted.*
2. **Default Standard Baseline:**
   - Confidence threshold: **0.50** for both detectors.
3. **External Diagnostic Oracle:**
   - The threshold in $[0.05, 0.95]$ (sweep step 0.05) that achieves the highest micro $F_1$ on the external dataset.
   *Used exclusively for diagnostic evaluation of the threshold transfer gap; never presented as a deployable operational policy.*

---

## 3. Class Mapping & Taxonomy

We restrict evaluation strictly to the six core traffic categories present in both DAWN and COCO:

| BDD100K Native Label | Canonical Internal Label | Pretrained COCO Detector Class | Notes |
|---|---|---|---|
| `pedestrian` | `Person` | `person` | Mapped directly. |
| `car` | `Car` | `car` | Includes sedans, coupes, SUVs, hatchbacks. |
| `truck` | `Truck` | `truck` | Commercial freight trucks and haulers. |
| `bus` | `Bus` | `bus` | Public transit and coach buses. |
| `motorcycle` | `Motorcycle` | `motorcycle` | Two-wheeled motorized vehicles. |
| `bicycle` | `Bicycle` | `bicycle` | Pedal bicycles. |

### Explicit Exclusions (Taxonomic Safeguards)
* **`rider`:** In BDD100K, people riding bicycles or motorcycles are labeled `rider`. In COCO, riders are annotated as `person`. To prevent silent taxonomic distortion and maintain semantic purity, `rider` instances are **excluded** from the ground-truth target set and not merged into `Person`.
* **Other BDD100K categories:** `traffic light`, `traffic sign`, `train`, `other vehicle`, `trailer`, and `other person` are excluded from the target evaluation set. Detector predictions predicting these classes are ignored.

---

## 4. Sampling Strategy & Dataset Stratification

To ensure a balanced, reproducible, and computationally tractable benchmark on our Apple M1 hardware, we sample a weather-stratified subset of $N = 203$ images from the official BDD100K validation split:

| Strata (Weather Condition) | Time of Day | Target Sample Size | Justification |
|---|---|---:|---|
| **Rainy** | Daytime | 60 | Primary adverse weather condition with water droplets, wiper streaks, and road reflections. |
| **Snowy** | Daytime | 60 | Primary adverse weather condition with snow accumulation, diffuse scattering, and low contrast. |
| **Foggy** | All available | 13 | Exhaustive census of all 13 foggy scenes in the BDD100K validation split (5 daytime, 7 night, 1 dusk). |
| **Overcast** | Daytime | 35 | Natural adverse lighting condition exhibiting severe contrast reduction without precipitation. |
| **Clear** | Daytime | 35 | **Control group** to decouple sensor/viewpoint domain shift from adverse-weather degradation. |
| **Total** | | **203** | Perfectly mirrors the scale of DAWN heldout ($N = 202$). |

### Sampling Procedure
- Sampling is executed deterministically using Python's `random.Random(seed=42)` over sorted image identifiers.
- Each sampled image must satisfy all quality-control criteria detailed below.

---

## 5. Quality Control & Exclusion Criteria

Images and annotations must pass the following 10 validation checks before inclusion:
1. **Image Existence & Integrity:** JPEG file must exist, load cleanly via PIL without truncation errors, and have dimensions exactly $1280 \times 720$.
2. **Coordinate Bounds:** Bounding boxes must have valid coordinates ($0 \le x_{\min} < x_{\max} \le 1280$ and $0 \le y_{\min} < y_{\max} \le 720$). Boxes with zero or negative area are rejected.
3. **Duplicate Detection:** SHA-256 byte hashing across all downloaded images to verify that no duplicate frames exist within the sample.
4. **Empty Image Handling:** Images with zero target annotations are preserved to faithfully evaluate false-alarm rates.
5. **Cross-Split Contamination:** All sampled images originate strictly from the official BDD100K `val` split; no training frames are utilized.

---

## 6. Statistical Hypothesis Testing & Bootstrap Framework

1. **Paired Image-Level Bootstrap (Model vs. Model on External Data):**
   - Resampling unit: Image ($N = 203$).
   - Number of bootstrap iterations: $B = 1,000$.
   - Seed: `20261101`.
   - Metric: Paired difference in micro $F_1$ ($\Delta F_1 = F_{1, \text{FRCNN}} - F_{1, \text{YOLO}}$).
   - 95% Confidence Intervals computed via the percentile method.

2. **Paired Image-Level Bootstrap (Threshold Comparison):**
   - Paired difference in micro $F_1$ between DAWN-selected threshold and default 0.50 threshold on external images ($\Delta F_1 = F_{1, \text{DAWN-thresh}} - F_{1, \text{default}}$).

3. **Independent Two-Sample Bootstrap (Cross-Dataset Comparison):**
   - Because DAWN ($N = 202$) and BDD100K ($N = 203$) comprise distinct images, a paired bootstrap is methodologically invalid.
   - We perform independent two-sample bootstrap resampling ($B = 1,000$) to estimate the cross-dataset performance gap:
     $$\Delta F_{1, \text{cross}} = F_{1, \text{BDD100K}} - F_{1, \text{DAWN heldout}}$$
     with 95% confidence intervals.
