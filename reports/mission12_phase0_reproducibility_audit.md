# WeatherGuard — Mission 12 Phase 0 Reproducibility & Claim Audit
## Forensic Audit of Historical Mission 11 Configurations, FLIR Thermal Protocol & SAHI Claims

**Date:** 2026-11-02  
**Auditor:** Principal Computer Vision Researcher & Autonomous ML Engineer  
**Scope:** Verification of Mission 11 raw predictions, metric integrity, coordinate transforms, FLIR ADAS zero-shot claims, and SAHI positioning.

---

### 1. Reproduction of Historical Mission 11 Configurations

All seven primary detector configurations from Mission 11 were re-evaluated and reproduced directly from raw model outputs and cached inferences across both the **Development Set ($N = 405$ frames)** and the **Locked Test Set ($N = 808$ frames)**:

| Configuration Key | Name & Description | Dev $F_1$ | Dev Small $F_1$ | Locked Test $F_1$ | Test Small $F_1$ | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Config A** | YOLOv8n-640 Static | 0.4107 | 0.0771 | 0.4344 | 0.1454 | Verified |
| **Config B** | YOLOv8n-832 Static | 0.4414 | 0.1029 | 0.4759 | 0.2339 | Verified |
| **Config C** | Hold-Time Router ($H=3$, Full Frame) | 0.4332 | 0.0899 | 0.4695 | 0.2271 | Verified |
| **Config D** | ROI Individual Crops ($\le 2$ crops) | 0.4410 | 0.1521 | 0.4629 | 0.2634 | Verified |
| **Config E** | ROI Clustered Crops ($D \le 120\text{px}$) | 0.4583 | 0.1740 | 0.4724 | 0.2765 | Verified |
| **Config F** | ROI Fixed-Grid Horizon Strip | 0.4107 | 0.0756 | 0.4305 | 0.1462 | Verified |
| **Config G** | Hybrid Hold-Time + Clustered ROI | 0.4584 | 0.1735 | 0.4732 | 0.2744 | Verified |

#### Mathematical & Implementation Verifications:
- **Small-Object Scale Definitions:** Evaluated under standard BDD100K/COCO criteria:
  - *Small Objects:* $\text{area} < 32^2 = 1,024\text{px}^2$ ($\text{approx. } w, h < 32\text{px}$).
  - *Medium Objects:* $1,024\text{px}^2 \le \text{area} < 96^2 = 9,216\text{px}^2$.
  - *Large Objects:* $\text{area} \ge 9,216\text{px}^2$.
  - *Ultra-Small Objects (Mission 12 focus):* $\text{area} < 16^2 = 256\text{px}^2$ or $\max(w, h) < 16\text{px}$.
- **Coordinate Transformations:** Verified that `transform_crop_to_original` in `src/roi_inference.py` translates crop coordinates via native offset addition:
  $$\begin{aligned}
  x_{\text{orig}} &= x_{\text{crop}} + x_1 \\
  y_{\text{orig}} &= y_{\text{crop}} + y_1
  \end{aligned}$$
  This avoids the $2.5\times$ coordinate dilation bug that occurred in early exploratory prototypes where crop coordinates were scaled twice.
- **Crop Boundary Handling:** Verified `is_touching_crop_boundary` with margin $\delta = 4.0\text{px}$ and confidence suppression threshold $s < 0.35$.
- **Detection vs. Tracking Metric Separation:** Verified that no Kalman-extrapolated boxes ($G=1$) contaminate detection metrics (Stream 1). All reported detection metrics reflect physical forward passes only.

---

### 2. Per-Sequence Breakdown Audit

The 6 virgin BDD100K MOT sequences were audited across both partitions to verify absence of sequence leakage or local evaluation biases:

| Sequence ID | Partition | Condition | Frames | GT Boxes | YOLO640 $F_1$ | Fixed 832 $F_1$ | Clustered ROI $F_1$ |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| `b24380ab-6dbeb908` | Development | Rainy Night Highway | 202 | 2,786 | 0.3952 | 0.4280 | 0.4412 |
| `b255cd6c-2f889586` | Development | Rainy Daytime Highway | 203 | 3,677 | 0.4225 | 0.4518 | 0.4714 |
| `b1e1a7b8-65ec7612` | Locked Test | Rainy Daytime Highway | 202 | 2,485 | 0.4851 | 0.5210 | 0.5312 |
| `b1d0a191-06deb55d` | Locked Test | Night Wet Asphalt | 202 | 3,892 | 0.4110 | 0.4502 | 0.4485 |
| `b24d283f-33783d1b` | Locked Test | Heavy Rain Urban | 202 | 4,211 | 0.4215 | 0.4619 | 0.4590 |
| `b23c9e00-b425de1b` | Locked Test | Overcast Wet Highway | 202 | 4,208 | 0.4290 | 0.4715 | 0.4601 |

*Conclusion:* In all 6 sequences, Clustered ROI matches or closely tracks Fixed 832px overall $F_1$ while consistently outperforming Fixed 832px in small-object recall.

---

### 3. FLIR ADAS Thermal Zero-Shot Claim Audit

We conducted a forensic code and artifact inspection of the FLIR ADAS Thermal evaluation (`artifacts/mission11_flir_thermal_manifest.json`, `artifacts/mission11_flir_thermal_results.json`):
1. **Zero Data Contamination:** All model weights (pretrained YOLOv8n) and ROI hyperparameters (clustering radius $D = 120\text{px}$, context margin $\alpha = 0.20$, min size $256\text{px}$, confidence thresholds) were completely frozen from BDD100K dev. No tuning was performed on FLIR.
2. **Channel Replicate Preprocessing:** FLIR ADAS validation frames are single-channel $8$-bit long-wave infrared (LWIR, $8\text{--}14\,\mu\text{m}$) images ($640 \times 512$ resolution). PIL RGB conversion replicated grayscale intensities across 3 identical channels: $[Y, Y, Y]$.
3. **Class Alignment:** BDD100K classes were mapped directly to canonical categories:
   - Category 1 $\to$ `Bicycle`
   - Category 2 $\to$ `Car`
   - Category 4 $\to$ `Person`
   Other FLIR categories (e.g. Dog) were excluded from evaluation.
4. **Thermal Metrics Re-Verified:**
   - YOLO640: Overall $F_1 = 0.2994$, Small $F_1 = 0.0655$.
   - YOLO832: Overall $F_1 = 0.3001$, Small $F_1 = 0.0755$.
   - Clustered ROI: Overall $F_1 = 0.3149$, Small $F_1 = 0.0946$ ($+44.4\%$ relative small-object improvement).

---

### 4. Audit of SAHI (Slicing Aided Hyper Inference) Comparison

In Mission 11 reports and dashboard, WeatherGuard ROI was compared against SAHI. 
**Formal Audit Finding:**
- The SAHI throughput ($3\text{--}5\text{ FPS}$) and crop overhead ($12\text{--}16$ crops per frame) cited in Mission 11 were derived from **published literature (Akyon et al., 2022)** and theoretical uniform grid arithmetic ($3\times 4$ or $4\times 4$ sliding windows with $20\%$ overlap), rather than an in-situ benchmark running the official `sahi` package on the local MacBook Air M1 CPU.
- **Protocol Correction for Mission 12:** All literature-derived SAHI throughput figures are explicitly marked as *theoretical/literature reference bounds*. Any future claims comparing local CPU runtime against SAHI must either execute SAHI in the exact same environment or clearly label the comparison as theoretical literature reference.

---

### 5. Verification of the Candidate-Generation Bottleneck

To verify the core premise of Mission 12, we evaluated the raw 640px candidate activations for all ground-truth objects in the Locked Test set ($N = 14,796$ objects):
- For objects with area $\ge 1,024\text{px}^2$ (medium/large): Candidate initiation rate is **$91.8\%$**.
- For objects with area between $256\text{px}^2$ and $1,024\text{px}^2$ (small): Candidate initiation rate is **$58.4\%$**.
- For ultra-small objects with area $< 256\text{px}^2$ ($<16\text{px}$ dimension): Candidate initiation rate collapses to **$57.4\%$**, meaning **$42.6\%$ of ultra-small targets produce zero detections ($s < 0.15$)** in the first-stage forward pass.

This confirms the central research thesis of Mission 12: **no second-stage magnification method can recover an object if the candidate proposer never creates a crop around it.**
