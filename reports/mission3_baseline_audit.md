# WeatherGuard Mission 3: Baseline Audit & Methodological Verification

**Document:** `reports/mission3_baseline_audit.md`  
**Date:** 2026-11-01  
**Status:** Completed & Verified  
**Auditor:** WeatherGuard Autonomous Research Agent  

---

## 1. Executive Summary

Prior to developing scale-aware inference and adaptive calibration methods in Mission 3, we performed a comprehensive audit of the Mission 1 (DAWN) and Mission 2 (BDD100K external) codebases, cached prediction records, ground-truth annotations, and statistical pipelines. 

Every key empirical metric from Mission 2 was independently re-evaluated and confirmed. We verified the ground-truth integrity across models, the coordinate system definitions, the exact timing boundaries, and the reproducibility of the bootstrap confidence intervals. We also resolved a linguistic ambiguity regarding the "8.7% small-object recall" phrasing in the previous draft report.

---

## 2. Detailed Audit of the 10 Pre-Flight Items

### Item 1: Small-Object Definition
* **Audit Finding:** The previous study defined scale bins following standard COCO conventions:
  - **Small:** $\text{Area} < 32^2 = 1,024\text{ px}^2$.
  - **Medium:** $32^2 \le \text{Area} < 96^2 = 9,216\text{ px}^2$.
  - **Large:** $\text{Area} \ge 96^2 = 9,216\text{ px}^2$.
* **Assessment:** Scientifically standard and sound. In continuous distributions, small objects in the 203-image test cohort range from $42\text{ px}^2$ ($6 \times 7$) to $1,023\text{ px}^2$ ($31 \times 33$), representing distant horizon targets.

### Item 2: Coordinate Space for Object Scale Measurement (Before vs. After Resizing)
* **Audit Finding:** Object sizes are measured in the **original un-resized image coordinate space** ($1280 \times 720$).
  - Ground truth boxes are mapped directly from normalized relative coordinates $[x_{\text{rel}}, y_{\text{rel}}, w_{\text{rel}}, h_{\text{rel}}]$ multiplied by width ($1280$) and height ($720$).
  - Predicted boxes from both Faster R-CNN and YOLOv8n are mapped back to this identical $1280 \times 720$ coordinate system before matching and metric evaluation.
* **Assessment:** Valid. Measuring bounding boxes in original sensor coordinates ensures that scale binning reflects physical sensor resolution rather than model-specific letterboxing.

### Item 3: Independent Verification of the 91.33% YOLO Small-Object Miss Rate
* **Audit Finding:** Re-running the greedy IoU matching protocol independently on `artifacts/eval_external_yolov8n.json` at $\tau = 0.20$ yielded:
  - Total small ground-truth objects ($< 32^2$): **$1,004$**
  - True Positives (TP): **$87$**
  - False Negatives (FN / Missed): **$917$**
  - **Small-Object Recall:** $\frac{87}{1,004} = \mathbf{8.6653\%}$ ($\approx 8.67\%$).
  - **Small-Object Miss Rate:** $\frac{917}{1,004} = \mathbf{91.3347\%}$ ($\approx 91.33\%$).
* **Assessment:** Confirmed to exact decimal precision.

### Item 4: Clarification of the "8.7% Recall Gap" Phrasing
* **Audit Finding:** The previous draft occasionally referred to "an 8.7% small-object recall gap".
  - Strictly speaking, **$8.67\%$ is YOLOv8n's absolute recall** on small objects ($87 / 1,004$).
  - Faster R-CNN achieved **$42.53\%$ absolute recall** on small objects ($427 / 1,004$).
  - The actual **recall gap** between Faster R-CNN and YOLOv8n is $42.53\% - 8.67\% = \mathbf{+33.86\text{ percentage points}}$.
* **Correction:** Throughout Mission 3, we strictly distinguish *absolute recall* ($8.67\%$ for baseline YOLOv8n) from the *inter-detector recall gap* ($+33.86\text{ pp}$ in favor of Faster R-CNN).

### Item 5: Ground-Truth Parity Across Detectors
* **Audit Finding:** Checked image-by-image across `artifacts/eval_external_fasterrcnn_resnet50_fpn.json` and `artifacts/eval_external_yolov8n.json`.
  - Both files evaluate the exact same 203 images in identical order.
  - Number of ground-truth boxes matches identically on every image (total = $2,553$).
  - Box coordinates and category labels match with zero discrepancy.
* **Assessment:** Parity confirmed. Both models were evaluated against identical ground-truth targets.

### Item 6: Consistency of Class Mappings and Taxonomic Boundaries
* **Audit Finding:** Both detectors map COCO 80-class outputs to the canonical 6 traffic classes:
  - `person` $\to$ `Person`
  - `car` $\to$ `Car`
  - `truck` $\to$ `Truck`
  - `bus` $\to$ `Bus`
  - `motorcycle` $\to$ `Motorcycle`
  - `bicycle` $\to$ `Bicycle`
  - BDD100K annotations map `pedestrian` $\to$ `Person`.
  - `rider` ($11$ instances) was excluded from ground-truth evaluation, avoiding taxonomic distortion.
* **Assessment:** Class mapping is consistent, audited, and preserved.

### Item 7: Status of the 203-Image BDD100K Test Cohort
* **Audit Finding:** The 203 BDD100K validation images evaluated in Mission 2 are recorded in `artifacts/external_dataset_summary.json` under `"cohort_image_ids"`.
* **Methodological Mandate:** These 203 images are **permanently locked** as the independent test cohort for Mission 3. All parameter selection, threshold tuning, and scale-bin optimization in Mission 3 must occur strictly on a **disjoint development cohort** ($N = 160$ newly sampled images).

### Item 8: Timing Boundaries and the Reported 20.9× Speed Ratio
* **Audit Finding:**
  - Faster R-CNN latency: $1,275.6\text{ ms/image}$ (mean steady-state).
  - YOLOv8n latency: $61.1\text{ ms/image}$ (mean steady-state).
  - Ratio: $\frac{1,275.6}{61.1} = 20.88\times \approx 20.9\times$.
* **Critical Technical Distinction:** Both frameworks measure from pre-formatted numpy/tensor input through bounding box decoding. However:
  - Torchvision Faster R-CNN internally executes `GeneralizedRCNNTransform`, resizing the $1280 \times 720$ image to $\approx 1333 \times 750$.
  - Ultralytics YOLOv8n by default resizes and letterboxes the $1280 \times 720$ image to $640 \times 640$.
  - Consequently, YOLOv8n evaluates at $\approx \frac{1}{2}$ the linear resolution and $\approx \frac{1}{4}$ the pixel count of Faster R-CNN.
  - This resolution discrepancy directly explains both YOLOv8n's $20.9\times$ speed advantage and its $8.67\%$ small-object recall: downsampling a $1280 \times 720$ frame to $640 \times 640$ shrinks a $20 \times 20$ vehicle down to $10 \times 10$ pixels, which vanishes in YOLOv8n's stride-8 ($P_3$) feature map.

### Item 9: Bootstrap & Threshold Sweep Reproducibility
* **Audit Finding:**
  - Re-running the bootstrap script (`seed = 20261101`, 1,000 resamples) reproduces:
    - Faster R-CNN external $F_1$: $0.6877$ (oracle $F_1 = 0.6877$, gap $= 0.0000$).
    - YOLOv8n external $F_1$ at $\tau=0.20$: $0.5111$ (oracle $F_1 = 0.5124$, gap $= 0.0013$).
    - YOLOv8n external $F_1$ at $\tau=0.50$: $0.4017$ ($+0.1092$ gain for $\tau=0.20$, 95% CI $[+0.0925, +0.1250]$).
    - Model gap (FRCNN - YOLO): $+0.1763$, 95% CI $[+0.1596, +0.1934]$.
* **Assessment:** 100% reproducible.

### Item 10: Weather-Specific Sample Sizes & Fog Small-Sample Caveat
* **Audit Finding:**
  - `rainy` ($N = 60$, $711$ targets): statistically robust.
  - `snowy` ($N = 60$, $778$ targets): statistically robust.
  - `overcast` ($N = 35$, $452$ targets): adequate support.
  - `clear` ($N = 35$, $465$ targets): adequate support.
  - `foggy` ($N = 13$, $147$ targets): **small sample size**. All 13 foggy images in the official BDD100K validation split were sampled. Any weather conclusions regarding fog must explicitly state this small-sample limitation ($N=13$).

---

## 3. Audit Conclusion & Approval for Mission 3

The previous baseline numbers are fully verified and methodologically sound. The primary cause of YOLOv8n's small-object failure is identified: **input downscaling to $640 \times 640$ on wide-angle $1280 \times 720$ driving video**. 

We approve proceeding to Mission 3 under a strict protocol that locks the 203 test images and selects all multi-scale, tiling, and adaptive calibration parameters on a newly sampled, disjoint BDD100K development cohort.
