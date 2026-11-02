# WeatherGuard — Mission 12 Cross-Domain Validation on FLIR ADAS Thermal
## Zero-Shot Infrared Evaluation, Static Salience Behavior & Thermal Blooming Diagnostics

**Date:** 2026-11-02  
**Dataset:** FLIR ADAS Thermal Object Detection (LWIR, $8\text{--}14\,\mu\text{m}$, 150 stratified validation frames, 1,071 ground-truth annotations)  
**Deliverable File:** `reports/mission12_cross_domain_analysis.md`  
**Artifact:** `artifacts/mission12_flir_thermal_results.json`

---

### 1. Research Motivation & Domain Gap

Visible RGB cameras suffer severe degradation under night driving, dense fog, and glare because they depend on ambient reflected photons. Long-Wave Infrared (LWIR) thermal sensors capture emitted blackbody radiation, operating independently of illumination and penetrating haze.

However, cross-modality zero-shot transfer from RGB to LWIR introduces severe domain shifts:
1. **Contrast Inversion:** Cold metal surfaces appear dark, while hot tires and engine hoods appear bright white, inverting standard RGB edge gradients.
2. **Thermal Blooming:** Radiative heat from exhaust tailpipes blooms into saturated white halos that obscure vehicle body outlines.
3. **Absence of Color & Texture:** Grayscale thermal imagery lacks chromatic cues (taillight red, turn indicator amber) used by RGB detectors.

Mission 12 investigated whether **detector-independent static visual salience** and **spatial priors** transfer zero-shot to thermal imagery without requiring thermal fine-tuning or annotations.

---

### 2. Methodological Protocol & Safeguards

1. **Non-Consecutive Image Dataset Integrity:** The public FLIR ADAS validation split in our Parquet repository (`validation.parquet`) consists of individual driving keyframes sampled non-consecutively across thermal driving sessions.
   - *Protocol Safeguard:* Temporal motion differencing (`FrameDifferenceProposal`) requires consecutive frames ($\Delta t \le 0.2\text{s}$). Because consecutive frames are unavailable on the static FLIR split, the controller's causal safety check **automatically bypassed temporal differencing and engaged static visual salience and base detector proposals**.
   - No synthetic temporal interpolations were fabricated.
2. **Grayscale Channel Conversion:** Single-channel $8$-bit LWIR intensities were replicated across three identical channels: $[Y, Y, Y]$.
3. **Frozen Parameters:** No thresholds or weights were tuned on the FLIR validation set. All parameters were held identical to the BDD100K development configuration.

---

### 3. Quantitative Cross-Domain Results

| Model Configuration | Modality | Overall Precision | Overall Recall | Overall $F_1$ | Small $F_1$ | Ultra-Small $F_1$ | Latency (ms) | FPS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLOv8n-640 Static** | Thermal (LWIR) | 0.7123 | 0.1895 | 0.2994 | 0.0655 | 0.0000 | 49.91 | 20.03 |
| **YOLOv8n-832 Static** | Thermal (LWIR) | 0.7199 | 0.1895 | 0.3001 | 0.0755 | 0.0000 | 84.62 | 11.82 |
| **Mission 11 Clustered ROI** | Thermal (LWIR) | 0.6844 | 0.2045 | 0.3149 | 0.0946 | 0.0215 | 72.99 | 13.70 |
| **Mission 12 Fused Proposals ROI** | Thermal (LWIR) | 0.6720 | **0.2185** | **0.3298** | **0.1120** | **0.0385** | **68.45** | **14.61** |

---

### 4. Key Scientific Discoveries

1. **Static Visual Salience Bridges the Thermal Contrast Gap:**
   - Thermal radiation creates sharp thermal boundary gradients between warm vehicle bodies ($30^\circ\text{C}\text{--}80^\circ\text{C}$) and cold asphalt roadbeds ($5^\circ\text{C}\text{--}15^\circ\text{C}$).
   - The perspective-weighted Sobel operator in `StaticSalienceProposal` detected these temperature boundary transitions, proposing candidate regions around distant vehicles even when the base 640px RGB-trained detector produced zero initial activations.
   - Small-object $F_1$ improved from $0.0655 \to \mathbf{0.1120}$ (**$+71.0\%$ relative improvement** over Base 640px).
2. **Thermal Blooming False Proposal Diagnosis (FM-07):**
   - High exhaust temperatures ($>150^\circ\text{C}$) create localized intensity saturation peaks ($I = 255$).
   - On 14 images, exhaust blooming triggered small false salience clusters.
   - *Mitigation:* The conservative NMS merging policy (`new_crop_conf_threshold = 0.35`) successfully rejected $85.7\%$ of these false blooming crops, preserving precision ($0.6720$).
3. **Zero-Shot Transferability:**
   - Region-adaptive high-resolution magnification operates on raw spatial geometry. Because a pixel sub-window physically expands thermal vehicle silhouettes by $2.5\times$, second-stage inference recovers distant targets without requiring thermal domain adaptation or fine-tuning.
