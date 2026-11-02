# WeatherGuard Cross-Dataset Domain-Shift Analysis: DAWN vs. BDD100K

**Date:** 2026-11-01  
**Authors:** WeatherGuard Research Team  
**Artifacts Referenced:** `artifacts/cross_dataset_domain_shift.json`, `artifacts/plots/cross_dataset_domain_shift.png`  
**Status:** Completed  

---

## 1. Executive Summary

A central finding of Mission 2 is the distinct cross-dataset behavior between two prominent architectures under identical pretrained weights:
- **Faster R-CNN ResNet-50 FPN:** Held-out DAWN $F_1 = 0.6738 \to$ External BDD100K $F_1 = 0.6877$ ($\Delta F_1 = +0.0139$, 95% CI $[-0.0199, +0.0454]$). The DAWN-calibrated threshold ($\tau = 0.60$) transferred without degradation and matched the external empirical oracle.
- **YOLOv8n:** Held-out DAWN $F_1 = 0.5781 \to$ External BDD100K $F_1 = 0.5111$ ($\Delta F_1 = -0.0670$, 95% CI $[-0.1020, -0.0250]$).

To understand why detector ranking widened from a $+0.097$ gap on DAWN to a $+0.176$ gap on BDD100K, we performed a systematic domain-shift analysis across low-level photometrics, camera geometry, object scale distributions, and traffic scene density.

---

## 2. Low-Level Photometric & Environmental Distributions

We extracted per-image pixel statistics across the 202 clean held-out DAWN images and the 203 weather-stratified BDD100K external evaluation images:

| Photometric Metric | DAWN Held-out ($N=202$) | BDD100K External ($N=203$) | Absolute Shift | Interpretation / Hypotheses |
|---|---:|---:|---:|---|
| **Mean Luminance (Brightness)** | $148.86 \pm 29.92$ | $103.83 \pm 18.59$ | **$-45.02$** | DAWN images (scraped from web search engines) contain high-key photographic compositions, intense backscattering fog, and overexposed snow fields. BDD100K dashboard cameras are auto-exposed for high dynamic range road pavement, producing darker average pixel values. |
| **RMS Contrast** | $41.68 \pm 14.32$ | $61.27 \pm 8.16$ | **$+19.59$** | DAWN scenes frequently exhibit severe low-contrast atmospheric wash (heavy fog, sandstorms, haze). BDD100K images maintain sharper local contrast between asphalt, lane markings, and vehicle chassis. |
| **Color Saturation (HSV S-channel)** | $43.34 \pm 48.31$ | $53.65 \pm 26.63$ | **$+10.31$** | DAWN includes monochromatic sandstorms and dense mist where color saturation collapses. BDD100K urban scenes exhibit higher saturation from traffic signals, brake lights, and road signage. |
| **Sensor Resolution** | Variable (median $800 \times 600$) | Uniform $1280 \times 720$ (HD) | Standardized | DAWN contains non-uniform web images with varied aspect ratios; BDD100K uses standard 16:9 dashcam aspect ratio. |

*Critical Caveat:* While photometric shifts between DAWN and BDD100K are statistically pronounced, correlation does not establish causation. A darker image with higher local contrast does not by itself degrade single-stage detection; geometric factors play an even larger role.

---

## 3. Geometric & Object Scale Shift (The Dominant Mechanism)

Evaluating bounding-box pixel distributions revealed the single most impactful domain difference between the two benchmarks: **object scale and scene clutter**.

We categorized all ground-truth instances into standard COCO scale bins:
- **Small objects:** Area $< 32^2 = 1,024\text{ px}^2$.
- **Medium objects:** $1,024 \le \text{Area} < 9,216\text{ px}^2$ ($96^2$).
- **Large objects:** Area $\ge 96^2 = 9,216\text{ px}^2$.

| Geometric Dimension | DAWN Held-out ($N=202$) | BDD100K External ($N=203$) | Divergence ($\Delta$) |
|---|---:|---:|---:|
| **Total Annotated Target Boxes** | 1,406 | 2,553 | $+1,147$ |
| **Target Instances per Image (Mean)** | $6.96 \pm 5.1$ | **$12.58 \pm 7.4$** | **$+5.62$** ($+80.7\%$) |
| **Target Instances per Image (Median)** | 5.0 | **13.0** | **$+8.0$** ($+160\%$) |
| **Small Objects ($< 32^2$)** | 166 (**$11.81\%$**) | 1,004 (**$39.33\%$**) | **$+27.52\text{ percentage points}$** |
| **Medium Objects ($32^2 \le \text{Area} < 96^2$)** | 645 (**$45.87\%$**) | 1,037 (**$40.62\%$**) | $-5.26\text{ percentage points}$ |
| **Large Objects ($\ge 96^2$)** | 595 (**$42.32\%$**) | 512 (**$20.05\%$**) | **$-22.26\text{ percentage points}$** |
| **Bounding Box Aspect Ratio ($W/H$, Median)** | 1.18 | 1.22 | $+0.04$ |

### Why Object Scale Diverged
1. **Camera Mounting & FOV:** BDD100K is captured by wide-angle dashboard cameras mounted on vehicle windshields looking down urban roads and multi-lane expressways. Distant cars and pedestrians occupy minuscule pixel footprints ($10 \times 15$ to $25 \times 25$ pixels).
2. **Web-Crawled Framing Bias in DAWN:** DAWN images were largely captured by pedestrians or photographers targeting specific vehicles in adverse conditions, resulting in centered, medium-to-large subject framing where objects occupy a substantial fraction of the field of view.

---

## 4. Architectural Resilience Under Scale-Dominant Domain Shift

The scale divergence directly explains the divergence between Faster R-CNN and YOLOv8n:

1. **Faster R-CNN ResNet-50 FPN:**
   - Incorporates a 5-level Feature Pyramid Network ($P_2, P_3, P_4, P_5, P_6$) with top-down lateral connections.
   - The high-resolution $P_2$ feature stride ($4\times$ downsampling) preserves spatial localization for sub-$32^2$ objects.
   - Achieved **$42.5\%$ small-object recall** (427 / 1,004 detected).
   - This architectural capacity allowed Faster R-CNN to absorb the $27.5\%$ surge in small objects on BDD100K without collapsing.

2. **YOLOv8n (Nano Backbone):**
   - Built with an ultra-lightweight backbone designed for edge efficiency ($3.2\text{M}$ parameters).
   - Uses an anchor-free decoupled head downsampled by $8\times, 16\times, 32\times$ ($P_3, P_4, P_5$). Sub-$32^2$ objects are compressed into single spatial bins or lost in feature downsampling.
   - Achieved only **$8.7\%$ small-object recall** (87 / 1,004 detected, 917 missed!).
   - On large objects ($\ge 96^2$), YOLOv8n achieved **$79.7\%$ recall** (close to Faster R-CNN's $85.0\%$).
   - Therefore, YOLOv8n's cross-dataset degradation was not primarily caused by adverse weather photometrics, but by **geometric scale shift** inherent to wide-angle operational driving cameras.

---

## 5. Traffic Class & Categorical Representation Shift

| Canonical Class | DAWN Held-out Share | BDD100K External Share | Nature of Shift |
|---|---:|---:|---|
| **Car** | 57.5% | **82.5%** | Severe class imbalance toward passenger cars in US metropolitan driving. |
| **Person** | 20.3% | 10.3% | Lower pedestrian density on multi-lane highways; pedestrians are heavily distant/small. |
| **Truck** | 12.1% | 4.5% | DAWN over-indexed on commercial trucks in snow/sand storms; BDD100K contains fewer freight haulers. |
| **Bus** | 4.8% | 1.8% | Low frequency in highway traffic. |
| **Bicycle** | 3.4% | 0.6% | Very rare in adverse weather US dashcam footage (16 instances). |
| **Motorcycle** | 1.9% | 0.3% | Virtually absent during adverse precipitation (7 instances). |

*Taxonomic Insight:* US dashcam driving footage in adverse weather is heavily dominated by passenger vehicles ($82.5\%$). Rare vulnerable road user classes (bicycles, motorcycles) have insufficient sample volume ($n < 20$) for statistically robust AP estimation, which explains why macro mAP@0.5 dropped across both models.

---

## 6. Synthesis: What Explains the Domain Shift?

In order of empirical impact:
1. **Object Scale & Resolution (Primary Driver):** Small-object prevalence increased from $11.8\%$ to $39.3\%$, penalizing detectors lacking high-resolution feature pyramids.
2. **Scene Clutter & Density:** Target density doubled ($6.96 \to 12.58$ objects/image), increasing occlusion and multi-object competition in greedy NMS.
3. **Severe Class Imbalance:** Extreme concentration in the `Car` category ($82.5\%$), with near-zero occurrences of two-wheelers in rainy/snowy conditions.
4. **Photometric / Environmental Shift:** Darker pavement, sharper local contrast, and artificial night/dusk illumination compared to high-exposure internet imagery.

*Conclusion:* Cross-dataset generalization in autonomous driving is fundamentally governed by sensor geometry and object scale distributions, which exert a stronger influence on detector operating points than adverse weather precipitation alone.
