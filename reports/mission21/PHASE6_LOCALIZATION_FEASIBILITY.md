# Phase 6 — Victim Localization Feasibility Study

**Project:** Rescue Drone  
**Author:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Mission:** 21 — Multi-Modal Acoustic-Visual Rescue Intelligence & Independent Validation  
**Date:** 2026-11-05  
**Artifact:** `reports/mission21/PHASE6_LOCALIZATION_FEASIBILITY.md`  

---

## 1. Executive Summary & Hardware Grounding

A strict audit of all 1,500 audio samples in the Rescue Drone corpus (`artifacts/mission21/localization_feasibility_analysis.json`) confirms:
- **Mono-Channel Files:** Exactly 1,500 / 1,500 ($100.0\%$) are single-channel mono audio captures sampled at 16,000 Hz.
- **Multi-Channel Array Files:** 0 / 1,500 ($0.0\%$).

**Core Scientific Conclusion:** In accordance with Mission Principles (*"Do not claim real-world drone localization without real localization measurements; implement localization only if the required microphone information is available; otherwise document localization as a future hardware-dependent capability"*), **acoustic victim localization cannot be claimed or simulated as an active feature.**

Single-channel scalar audio provides omnidirectional signal power $p(t)$ without inter-microphone phase or delay differences, making spatial direction-of-arrival (DOA) mathematically underdetermined.

---

## 2. Mathematical Requirements for Acoustic Localization

To achieve spatial bearing and range estimation in future drone iterations, the system requires a multi-microphone array executing Time-Difference-of-Arrival (TDOA) estimation:

### 2.1 Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
For a pair of microphones $i$ and $j$ receiving source signal $s(t)$ corrupted by noise:
$$R_{ij}^{\text{PHAT}}(\tau) = \int_{-\infty}^{\infty} \frac{X_i(f) X_j^*(f)}{|X_i(f) X_j^*(f)|} e^{j 2 \pi f \tau} df$$

The phase transform normalizes the cross-spectrum magnitude, producing sharp Dirac-like delta peaks even under moderate reverberation:
$$\hat{\tau}_{ij} = \arg\max_\tau R_{ij}^{\text{PHAT}}(\tau)$$

### 2.2 Direction of Arrival (DOA) Geometry
For a planar microphone array with separation vector $\mathbf{d}_{ij}$ and speed of sound $c \approx 343\text{ m/s}$:
$$\hat{\tau}_{ij} = \frac{\mathbf{d}_{ij} \cdot \mathbf{u}}{c} = \frac{\|\mathbf{d}_{ij}\|}{c} \cos(\theta_{ij})$$

- **Azimuth Only (2D bearing):** Minimum 3 non-collinear microphones.
- **Azimuth + Elevation (3D bearing):** Minimum 4 non-coplanar microphones (e.g., regular tetrahedron or planar quad with central elevation offset).

---

## 3. Physical Interference & Acoustic Challenges on Edge Drones

### 3.1 Drone Rotor Wash & Blade Pass Frequency (BPF)
Small search-and-rescue quadcopters operate with 2-blade or 3-blade propellers spinning between $4,000$ and $8,000\text{ RPM}$:
$$f_{\text{BPF}} = \frac{N_{\text{blades}} \times \text{RPM}}{60} \approx 133\text{ Hz} - 400\text{ Hz}$$
- Rotor wash produces a dense acoustic harmonic series ($2 f_{\text{BPF}}, 3 f_{\text{BPF}}, \dots, 2.5\text{ kHz}$) that directly overlaps the critical first and second formants ($F_1, F_2$) of Bengali vocal cries ($300-2,200\text{ Hz}$).
- In our Phase 3 benchmarks, drone rotor noise reduced single-channel distress classification accuracy to $88.00\%$. For localization, rotor harmonics cause severe false correlation peaks unless canceled via multi-channel adaptive beamforming (Minimum Variance Distortionless Response, MVDR).

### 3.2 Atmospheric Attenuation and Environmental Multipath
- **Dense Fog:** Suspended water droplets induce excess acoustic attenuation at high frequencies ($>4\text{ kHz}$), reducing the bandwidth available for sharp GCC-PHAT peak resolution.
- **Heavy Rain:** Kinetic impact of water droplets on microphone diaphragms generates non-stationary, wideband acoustic spikes.
- **Multipath Reverberation in Disaster Zones:** Collapsed buildings, water surfaces, and dense tree canopies reflect acoustic wavefronts, producing reverberation times ($RT_{60} > 0.4\text{s}$) that smear TDOA delay estimates.

---

## 4. Current Operational Role: Omnidirectional Cueing

While spatial localization is hardware-dependent, the current single-channel model fulfills a vital operational role in search-and-rescue:

$$\text{Omnidirectional Trigger} \longrightarrow \text{Flight Pattern Switch} \longrightarrow \text{Localized Optical Acquisition}$$

1. **Wide-Area Transit:** The drone flies high-altitude grid lines ($h \approx 30-50\text{ m}$) where the camera field-of-view is narrow and distant victims (<16px) cannot be visually resolved.
2. **Omnidirectional Acoustic Cue:** A Bengali distress cry ("বাঁচাও") is detected omnidirectionally with **$92.67\%$ accuracy** and **$0.9706$ ROC-AUC**, invariant to whether the victim is behind trees, inside a structure, or enveloped in dense fog.
3. **Autonomous Search Focus:** The acoustic trigger instructs the flight controller to suspend transit and initiate an expanding low-altitude spiral or lawnmower search pattern, bringing high-resolution camera optics directly over the victim.

---

## 5. Future Hardware Roadmap for Multi-Channel Localization

```
+-------------------------------------------------------------+
| Stage 1: Microphone Array Hardware Selection                |
| - 4x Knowles SPH0645LM4H-B digital I2S MEMS microphones    |
| - Vibration-isolated acoustic enclosure below fuselage      |
| - Circular array geometry with 18 cm diameter baseline      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Stage 2: Embedded DSP Architecture                          |
| - Dual-core ARM Cortex-M7 / ESP32-S3 or FPGA coprocessor   |
| - Real-time 4-channel synchronized DMA audio ingestion      |
| - Fixed-point GCC-PHAT calculation (10 ms sliding frames)    |
| - Rotor noise notch filtering on BPF harmonics              |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Stage 3: Flight Controller Integration                      |
| - MAVLink protocol message: ACOUSTIC_BEARING(yaw_angle, conf)|
| - Drone auto-rotates gimbal directly along victim azimuth    |
| - WeatherGuard camera zooms in to confirm victim identity   |
+-------------------------------------------------------------+
```
