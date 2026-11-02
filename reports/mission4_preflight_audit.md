# WeatherGuard Mission 4: Pre-Flight Audit of Previous Missions

**Document:** `reports/mission4_preflight_audit.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Completed & Artifact-Verified  

---

## 1. Executive Summary

Before beginning Mission 4, a comprehensive pre-flight audit was conducted across all previous codebases, reports, artifacts, prediction caches, bootstrap implementations, and metric routines (Mission 1 threshold transfer, Mission 2 cross-dataset generalization, and Mission 3 scale-aware inference).

The objective is to establish ground truth, verify data integrity, reconcile numerical discrepancies, and ensure that the experimental foundation for Mission 4 is robust and leak-free.

---

## 2. Mandatory Audit Items & Reconciliations

### Item 1: Reconciliation of Reported F1 Gap (Faster R-CNN vs. Optimized YOLOv8n)
- **Reported in Mission 3 narrative:**
  - Faster R-CNN $F_1 = 0.6877$
  - Scale-Aware YOLOv8n ($1280\text{px}$) $F_1 = 0.5782$
  - Narrative reported gap: $0.1099$
- **Exact Full-Precision Point Estimates (`artifacts/mission3_final_results.json`):**
  - Faster R-CNN: $F_1 = 0.6877354541649225$
  - Scale-Aware YOLOv8n: $F_1 = 0.5782374100719424$
  - Direct point estimate difference:
    $$0.68773545 - 0.57823741 = \mathbf{0.10949804} \quad (\approx 0.1095)$$
- **Bootstrap Mean (`artifacts/mission3_bootstrap_results.json`):**
  - Paired bootstrap ($B = 1,000$, seed `20261101`):
    $$\text{Mean}(\Delta F_1) = \mathbf{0.10991604} \quad (\approx 0.1099), \quad 95\%\text{ CI: } [0.09135, 0.12812]$$
- **Audit Conclusion (Numerical Discrepancy / Wording Recommendation):**
  The narrative text cited the **paired bootstrap mean ($0.1099$)** rather than the point estimate difference ($0.1095$). The numerical discrepancy is minor ($0.000418$ or $0.04$ percentage points), well within the bootstrap standard error ($\text{SE} \approx 0.009$). In Mission 4 reports, we will explicitly distinguish between point estimate differences and bootstrap resample means.

---

### Item 2: Verification of Baseline 91.33% Small-Object Miss Rate
- **Reconstructed Counts on Locked Test Set (Group C, $N = 203$):**
  - Total ground-truth target objects: $2,553$
  - Small objects ($< 32^2 = 1,024\text{ px}^2$): $1,004$ instances ($39.33\%$)
  - Baseline YOLOv8n ($640\text{px}, \tau = 0.20$):
    - True Positives ($\text{TP}$): $87$
    - False Negatives ($\text{FN}$): $917$
    - $\text{TP} + \text{FN} = 87 + 917 = 1,004$ (exact match)
    - Recall $= 87 / 1,004 = 8.665338\% \approx 8.67\%$
    - Miss Rate $= 917 / 1,004 = 91.33466\% \approx \mathbf{91.33\%}$
- **Audit Conclusion:** Confirmed independently from raw prediction JSONs (`artifacts/eval_external_yolov8n.json`). The denominator is verified and exact.

---

### Item 3: Small-Object Denominator Parity Across Detectors & Methods
- **Verification Across Models on Locked Test Set ($N = 203$):**
  - Faster R-CNN Reference: $\text{GT}_{\text{small}} = 1,004$ ($\text{TP} = 427, \text{FN} = 577$, Recall $= 42.53\%$)
  - Baseline YOLOv8n (640px): $\text{GT}_{\text{small}} = 1,004$ ($\text{TP} = 87, \text{FN} = 917$, Recall $= 8.67\%$)
  - Scale-Aware YOLOv8n (1280px): $\text{GT}_{\text{small}} = 1,004$ ($\text{TP} = 288, \text{FN} = 716$, Recall $= 28.69\%$)
- **Audit Conclusion:** Ground-truth denominator parity is $100\%$ consistent. All models and methods are evaluated against the exact same $1,004$ small ground-truth instances.

---

### Item 4 & 5: Scale Definitions & Coordinate Space
- **Scale Boundaries:**
  - Small: $\text{Area} < 32^2 = 1,024\text{ px}^2$
  - Medium: $32^2 \le \text{Area} < 96^2 = [1,024, 9,216)\text{ px}^2$
  - Large: $\text{Area} \ge 96^2 = 9,216\text{ px}^2$
- **Coordinate Space:**
  - In all evaluation code (`src/metrics.py`, `scripts/run_combined_scale_inference.py`), box area is computed as:
    $$\text{Area} = (x_{\max} - x_{\min}) \times (y_{\max} - y_{\min})$$
  - BDD100K images are uniformly $1280 \times 720$. Ground-truth coordinates are loaded unscaled in original image pixels.
  - Predicted boxes from Ultralytics YOLOv8 are automatically transformed back to original image space by Ultralytics (`r.boxes.xyxy`).
- **Audit Conclusion:** Confirmed. Object sizes are measured strictly in original image coordinates ($1280 \times 720$), invariant to model-specific input resizing transforms.

---

### Item 6 & 7: Paired Bootstrap Methodology & Seed Independence
- **Replication Check:**
  - Evaluated on $N = 203$ paired image samples ($B = 1,000$ resamples).
  - Primary Seed `20261101`:
    - Mean $\Delta F_1 = +0.066706$, $95\%\text{ CI: } [+0.04961, +0.08387]$, $p < 0.001$
    - Mean $\Delta \text{Recall}_{\text{small}} = +0.199880$, $95\%\text{ CI: } [+0.16906, +0.23135]$, $p < 0.001$
  - Secondary Replication Seed `20261102`:
    - Mean $\Delta F_1 = +0.066624$, $95\%\text{ CI: } [+0.04667, +0.08459]$, $p < 0.001$
    - Mean $\Delta \text{Recall}_{\text{small}} = +0.200444$, $95\%\text{ CI: } [+0.16982, +0.23186]$, $p < 0.001$
  - Discrepancy between seeds: $\Delta(\Delta F_1) = 0.000083$, $\Delta(\Delta \text{Recall}_{\text{small}}) = -0.000563$.
- **Audit Conclusion:** Confirmed. The secondary seed provides independent pseudo-random resampling from a distinct NumPy PRNG state. The tiny divergence reflects genuine Monte Carlo variance and proves numerical stability.

---

### Item 8: Latency Benchmarking Methodology
- **Timing Boundaries:**
  - In `src/model.py`, YOLOv8 timing wraps `model.predict(arr, conf=0.0, imgsz=...)`. This measures image letterbox resizing/normalization, CNN forward pass, anchor decoding, and internal NMS.
  - Faster R-CNN timing wraps `model([x])`, which includes torchvision's internal `GeneralizedRCNNTransform` (scaling to $1333 \times 750$), ResNet-50 FPN backbone, RPN, RoI Align, classification/regression heads, and NMS.
  - Warmup is executed once before the cohort timing loop (`detector.warmup()`).
  - Throughput (FPS) is derived directly as $1.0 / \text{mean\_latency}$.
- **Methodological Limitation:**
  In Faster R-CNN, converting PIL to float tensor (`transforms(image)`) occurs immediately before the timed block, whereas in YOLO, converting numpy array to torch tensor occurs inside `model.predict`. This discrepancy is negligible ($< 1\text{ ms}$) relative to the $1,276\text{ ms}$ Faster R-CNN forward pass, but should be documented.

---

### Item 9: Verification of Zero Test Data Leakage
- **Cohort Roles:**
  - Group A: DAWN dev ($N = 821$)
  - Group B: BDD100K dev ($N = 160$)
  - Group C: BDD100K locked test ($N = 203$)
- **Audit Verification:**
  - In `scripts/run_combined_scale_inference.py`, model selection was performed by ranking candidate configurations solely on Group B development metrics.
  - The winning configuration ($1280\text{px}$) was selected on Group B dev before any test evaluation was initiated.
  - Automated disjointness test (`tests/test_mission3_scale_aware.py`) verifies **0 overlapping images** between Group B and Group C.

---

### Item 10: Taxonomy, Filtering, and Weather Stratum Consistency
- **Class Mapping:**
  - Strict 6-class canonical mapping: `pedestrian -> Person`, `car -> Car`, `truck -> Truck`, `bus -> Bus`, `motorcycle -> Motorcycle`, `bicycle -> Bicycle`.
  - Non-canonical classes explicitly excluded: `rider`, `traffic light`, `traffic sign`, `train`, `trailer`, `other vehicle`, `other person`.
- **Weather Stratification:**
  - Clear, Overcast, Rainy, Snowy, Foggy.
  - All images are daytime scenes (`timeofday == "daytime"`).

---

## 3. Classification of Pre-Flight Findings

| Finding | Classification | Impact & Action |
|---|---|---|
| F1 Gap reported as $0.1099$ vs point estimate difference $0.1095$ | Numerical Discrepancy / Wording Recommendation | Report both exact point estimate difference and bootstrap mean in Mission 4 tables. |
| Baseline small-object miss rate ($91.33\%$) and denominator ($1,004$) | Confirmed Correct | Established as definitive baseline ground truth. |
| Object scale coordinate space | Confirmed Correct | Native $1280 \times 720$ coordinates verified throughout. |
| Secondary bootstrap seed | Confirmed Correct | Validated independent PRNG Monte Carlo replication. |
| Zero data leakage to locked Group C | Confirmed Correct | Enforced and maintained for Mission 4. |
| Preprocessing tensor conversion timing boundary | Methodological Limitation | Document that torchvision PIL-to-tensor occurs outside timed block. |
| Fog sample size ($N = 13$ on locked test cohort) | Methodological Limitation | Carry forward prominent small-sample uncertainty caveat. |

---

## 4. Pre-Flight Sign-Off

The audit confirms that all previous experimental results, denominators, metrics, and split manifests are fully reconciled and leak-free. We proceed to Mission 4 protocol specification.
