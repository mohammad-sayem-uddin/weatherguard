# Phase 4 — Multi-Modal Fusion Architecture

**Project:** WeatherGuard + Rescue Drone  
**Author:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Mission:** 21 — Multi-Modal Acoustic-Visual Rescue Intelligence & Independent Validation  
**Date:** 2026-11-05  
**Artifact:** `reports/mission21/PHASE4_FUSION_ARCHITECTURE.md`  

---

## 1. Architectural Overview

The **Multi-Modal Decision-Level Fusion Engine** (`src/multimodal/fusion_engine.py`) coordinates between the WeatherGuard visual detector and the Rescue Drone acoustic distress classifier. It is explicitly engineered for search-and-rescue UAV operations under severe weather and low visibility.

```
+------------------------------------+       +------------------------------------+
|         WeatherGuard Vision        |       |          Rescue Drone Audio        |
|  - Person Detection Confidence     |       |  - Emergency Vocalization Prob     |
|  - Target Bounding Boxes           |       |  - Calibrated Threshold (tau*=0.54)|
|  - Mean Illumination Index         |       |  - Calibrated SNR (dB)             |
|  - Contrast & Weather Degradation  |       |  - Environmental Noise Condition   |
+-----------------+------------------+       +-----------------+------------------+
                  |                                            |
                  v                                            v
         [Visual Evidence]                            [Acoustic Evidence]
         Reliability: w_vis                           Reliability: w_aud
                  \                                            /
                   \                                          /
                    v                                        v
        +-----------------------------------------------------------+
        |        Adaptive Multi-Modal Decision-Level Engine         |
        |                                                           |
        |  1. Modality Weight Normalization:                        |
        |     w_vis = f(illum, contrast, domain)                    |
        |     w_aud = f(SNR, noise_condition)                       |
        |                                                           |
        |  2. Cross-Modal Evaluation:                               |
        |     - Concordance Check                                   |
        |     - Environmental Disagreement Resolution               |
        |     - Explicit Abstention / Uncertainty Gating            |
        +-----------------------------+-----------------------------+
                                      |
                                      v
                             [Decision Output]
                             - Alert State: ALERT | NO_ALERT | ABSTAIN_UNCERTAIN
                             - Composite Score
                             - Recommended Action
                             - Physical Coordinate Transparency
```

---

## 2. Supported Evaluation Approaches

### Approach A: Vision-Only Decision Baseline
- Evaluates strictly on visual object detector confidence:
  $$\text{Alert} \iff \text{Person Detected} \land (\text{Confidence} \ge 0.25)$$
- **Strengths:** Provides precise bounding box coordinates for confirmed targets.
- **Vulnerabilities:** Fails completely in darkness (Night F1 = 0.3327), heavy fog (Fog Recall = 0.6300), and on distant targets (<16px Recall = 0.0000).

### Approach B: Audio-Only Distress Baseline
- Evaluates strictly on acoustic classifier distress probability:
  $$\text{Alert} \iff P(\text{Distress} \mid x) \ge \tau^* = 0.54$$
- **Strengths:** 360-degree omnidirectional coverage; invariant to ambient darkness, fog, and line-of-sight visual occlusions.
- **Vulnerabilities:** Performance degrades at $0\text{ dB}$ SNR ($80.0\%$ accuracy) and under heavy rotor wash / rain droplet splatter ($84.0\%$ accuracy). Single-channel mono input cannot estimate target coordinates.

### Approach C: Adaptive Multi-Modal Decision Fusion
- Computes dynamic environmental sensor reliability weights ($w_{\text{vis}}$ and $w_{\text{aud}}$):
  $$w_{\text{vis}} = f(\text{Illumination}, \text{Contrast}, \text{Domain}), \quad w_{\text{aud}} = f(\text{SNR}, \text{Noise Condition})$$
- **Concordant Alert:** Both modalities detect target $\to$ High-Confidence `ALERT` (`CONFIRMED_VICTIM_LOCATED`) with a concordance bonus.
- **Sensor Complementarity (Visual Degraded / Acoustic Clear):** Optical channel degraded by night or fog ($w_{\text{vis}} < 0.30$), but audio detects distress $\to$ Immediate `ALERT` (`ACOUSTIC_CUE_DISPATCH_OPTICAL_SEARCH`).
- **Sensor Complementarity (Acoustic Noise / Visual Clear):** Heavy rain or rotor noise degrades audio ($w_{\text{aud}} < 0.30$), but vision clearly sees victim $\to$ `ALERT` (`VISUAL_SURVEILLANCE_VERIFIED`).
- **Explicit Abstention:** Borderline, low-margin disagreements under ambiguous conditions trigger `ABSTAIN_UNCERTAIN` (`UNCERTAIN_MAINTAIN_SEARCH`), preventing false alarms.

---

## 3. Physical Sensing & Localization Disclosures

1. **Mono Audio Constraint:** The Rescue Drone audio corpus consists of single-channel (mono) recordings. Therefore, the audio model cannot determine direction-of-arrival (DOA) or time-difference-of-arrival (TDOA).
2. **Localization Transparency:** When `alert_state == "ALERT"` originates solely from audio, `victim_coordinates` is explicitly set to `None` and `spatial_bearing_valid` is `False`. Drone systems must treat acoustic alerts as a spatial trigger to re-route optical search patterns, never as coordinate pinpoints.
