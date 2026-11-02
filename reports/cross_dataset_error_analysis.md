# WeatherGuard Cross-Dataset Error Analysis: BDD100K External Evaluation

**Date:** 2026-11-01  
**Authors:** WeatherGuard Research Team  
**Artifacts Referenced:** `artifacts/cross_dataset_error_taxonomy.json`, `artifacts/qualitative/external/`  
**Status:** Completed  

---

## 1. Executive Summary

This report analyzes the failure modes observed when evaluating pretrained object detectors on the 203-image external BDD100K evaluation cohort under their DAWN-calibrated operating points:
- **Faster R-CNN ResNet-50 FPN** at threshold $\tau = 0.60$:
  - Total Ground Truth: 2,553
  - True Positives: 1,643 | False Positives: 582 | False Negatives: 910
  - Micro $F_1 = 0.6877$ (Precision: $0.7384$, Recall: $0.6436$)
- **YOLOv8n** at threshold $\tau = 0.20$:
  - Total Ground Truth: 2,553
  - True Positives: 1,043 | False Positives: 485 | False Negatives: 1,510
  - Micro $F_1 = 0.5111$ (Precision: $0.6826$, Recall: $0.4085$)

We examine four error dimensions: (1) Scale-dependent missed detections, (2) Weather-specific error profiles, (3) Class confusion and taxonomic boundaries, and (4) Threshold dynamics and false alarm trade-offs.

---

## 2. Scale-Dependent Missed Detections (The Primary Failure Mode)

As established in the domain-shift analysis, small objects ($< 32^2\text{ px}^2$) constitute $39.3\%$ of all external annotations ($1,004$ instances). The empirical breakdown across object scales demonstrates stark architectural divergence:

| Scale Bin (Area in $\text{px}^2$) | Ground Truth Count | Faster R-CNN TP | Faster R-CNN FN | FRCNN Recall | YOLOv8n TP | YOLOv8n FN | YOLOv8n Recall | Recall Gap ($\Delta$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Small ($< 32^2 = 1,024$)** | 1,004 | 427 | 577 | **42.53%** | 87 | 917 | **8.67%** | **$+33.86\%$** |
| **Medium ($32^2 \le \text{Area} < 96^2$)** | 1,037 | 781 | 256 | **75.31%** | 548 | 489 | **52.84%** | **$+22.47\%$** |
| **Large ($\ge 96^2 = 9,216$)** | 512 | 435 | 77 | **84.96%** | 408 | 104 | **79.69%** | **$+5.27\%$** |

### Key Diagnostic Observations:
1. **YOLOv8n Small-Object Blindness:** YOLOv8n missed **917 out of 1,004 small objects** ($91.33\%$ failure rate). In wide-angle dashcam footage, oncoming vehicles and pedestrians further than $\approx 35$ meters appear smaller than $32 \times 32$ pixels. YOLOv8n's feature downsampling discards fine spatial cues necessary for anchor-free regression at this scale.
2. **Convergence on Large Objects:** For objects larger than $96 \times 96$ pixels (e.g., vehicles directly in the host car's lane within 15 meters), YOLOv8n reaches **$79.69\%$ recall**, closely trailing Faster R-CNN's $84.96\%$. The single-stage detector is competent at immediate collision-hazard ranges, but severely degrades in situational horizon perception.
3. **Faster R-CNN Pyramid Efficacy:** Faster R-CNN detected nearly $5\times$ more small objects (427 vs. 87) thanks to its $P_2$ lateral feature maps and two-stage Region Proposal Network (RPN).

---

## 3. Weather-Specific Error Profiles

We analyzed False Positives (FP) and False Negatives (FN) across all 5 evaluation strata:

| Weather Condition | Sample Size ($N$) | Ground Truth | FRCNN TP | FRCNN FP | FRCNN FN | FRCNN $F_1$ | YOLO TP | YOLO FP | YOLO FN | YOLO $F_1$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Clear (Control)** | 35 | 465 | 323 | 101 | 142 | **0.7267** | 200 | 78 | 265 | **0.5384** |
| **Overcast** | 35 | 452 | 311 | 104 | 141 | **0.7174** | 198 | 91 | 254 | **0.5344** |
| **Snowy** | 60 | 778 | 490 | 162 | 288 | **0.6853** | 303 | 128 | 475 | **0.5012** |
| **Rainy** | 60 | 711 | 446 | 171 | 265 | **0.6717** | 298 | 154 | 413 | **0.5125** |
| **Foggy** | 13 | 147 | 73 | 44 | 74 | **0.5530** | 44 | 34 | 103 | **0.3911** |

### Environmental Observations:
1. **Fog is the Most Destructive Condition:**
   - Both detectors suffered their lowest performance in **Foggy** conditions ($F_1 = 0.5530$ for FRCNN, $F_1 = 0.3911$ for YOLOv8n).
   - Atmospheric scattering suppresses high spatial frequencies, blurring vehicle silhouettes and reducing contrast between car bodies and background mist.
2. **Rain-Induced False Positives:**
   - Rainy conditions produced the highest rate of false positives per image for both models (FRCNN: $2.85\text{ FP/img}$; YOLO: $2.57\text{ FP/img}$).
   - Visual inspection of failure examples reveals that water droplets on the windshield and specular reflections on wet asphalt create phantom bounding boxes falsely classified as `Car` or `Person`.
3. **Snowy Contrast Degradation:**
   - In snowy scenes, snowbanks along curbs and white vehicle roofs against overcast skies caused false negative vehicle clusters (FRCNN: 288 FN, YOLO: 475 FN).

---

## 4. Class-Level Error Dynamics & Confusion

| Target Class | Ground Truth Instances | FRCNN Detections | FRCNN TP | FRCNN FP | FRCNN FN | FRCNN $F_1$ | YOLO TP | YOLO FP | YOLO FN | YOLO $F_1$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Car** | 2,107 | 1,806 | 1,409 | 397 | 698 | **0.7202** | 900 | 313 | 1,207 | **0.5422** |
| **Person** | 262 | 250 | 165 | 85 | 97 | **0.6445** | 98 | 51 | 164 | **0.4769** |
| **Truck** | 116 | 129 | 49 | 80 | 67 | **0.4000** | 28 | 92 | 88 | **0.2373** |
| **Motorcycle** | 7 | 2 | 2 | 0 | 5 | **0.4444** | 2 | 1 | 5 | **0.4000** |
| **Bus** | 45 | 29 | 14 | 15 | 31 | **0.3784** | 11 | 26 | 34 | **0.2683** |
| **Bicycle** | 16 | 9 | 4 | 5 | 12 | **0.3200** | 4 | 2 | 12 | **0.3636** |

### Confusion Patterns:
1. **Commercial Vehicle Confusion (`Truck` vs. `Car` vs. `Bus`):**
   - Truck precision was low for both models ($38.0\%$ for FRCNN, $23.3\%$ for YOLO). Large pickup trucks and delivery vans in BDD100K are frequently predicted as `Car` by the detector or vice versa. In addition, city transit buses are occasionally misclassified as trucks.
2. **The `Rider` vs. `Person` Semantic Tension:**
   - Per protocol, BDD100K `rider` annotations were excluded. In 11 instances where bicyclists/motorcyclists were present, detectors predicted `Person` at the rider's position. Because `rider` was excluded, these valid human detections were formally recorded as false positives (accounting for $\approx 10\%$ of pedestrian false alarms). This confirms the critical necessity of documented taxonomic auditing.

---

## 5. Threshold Operating Dynamics: Default (0.50) vs. Dev-Calibrated

Comparing the default threshold ($\tau = 0.50$) to the DAWN-calibrated thresholds reveals why threshold transfer matters in real deployments:

- **For YOLOv8n:**
  - At Default $\tau = 0.50$: Precision is very high ($0.9150$), but Recall collapses to **$0.2573$** (657 TP vs. **1,896 missed objects!**). F1 is **$0.4017$**.
  - At DAWN-Calibrated $\tau = 0.20$: Recall rises to **$0.4085$** (1,043 TP, salvaging **386 additional objects**). Precision moderates to $0.6826$. F1 surges to **$0.5111$** ($+0.1094$ absolute gain).
  - *Finding:* Using the default 0.50 threshold on YOLOv8n in adverse weather driving is dangerously conservative; it misses nearly 3 out of every 4 objects. The lower 0.20 threshold calibrated on DAWN transfers successfully and rescues operational utility.

- **For Faster R-CNN:**
  - At Default $\tau = 0.50$: F1 is $0.6714$ (1,715 TP, 841 FP, 838 FN).
  - At DAWN-Calibrated $\tau = 0.60$: FP drops by **259 alarms** (from 841 to 582, a $-30.8\%$ reduction), while TP only drops by 72. F1 increases to **$0.6877$**.
  - *Finding:* A slightly more conservative threshold ($0.60$) filters out transient weather artifacts and phantom boxes, improving overall F1 on external data.

---

## 6. Qualitative Failure Gallery

High-resolution side-by-side failure visualizations (Ground Truth vs. Model Detections) were rendered locally to `artifacts/qualitative/external/`:
1. `failure_foggy_b7dc9921-fcfe3332.png`: Demonstrates severe contrast loss in dense fog where distant oncoming cars are completely undetected.
2. `failure_snowy_ba4bcdc5-06e22f87.png`: Illustrates snow accumulation obscuring vehicle boundaries and creating false alarms along snowbanks.
3. `failure_rainy_b4b7133e-94c56651.png`: Shows windshield water streaks and pavement reflections triggering false pedestrian/vehicle boxes.
4. `failure_overcast_b1f4491b-cf446195.png`: Illustrates scale-limited detection where multiple sub-$32^2$ distant passenger cars are missed in highway traffic.

*(In compliance with dataset licensing, raw original images are kept outside git and summaries are reported numerically above).*
