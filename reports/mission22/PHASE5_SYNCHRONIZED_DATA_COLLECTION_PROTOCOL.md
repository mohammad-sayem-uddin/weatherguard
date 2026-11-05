# MISSION 22 — PHASE 5: Genuine Synchronized Audiovisual Data Collection Protocol & Annotation Guide

**Principal Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** `https://github.com/mohammad-sayem-uddin/weatherguard`  
**Branch:** `master`  
**Schema Definition:** `configs/multimodal_synchronized_schema.yaml`  
**Date:** November 5, 2026  

---

## 1. Executive Directive

> **Mandatory Research Disclaimer:**  
> This document specifies the engineering design, hardware synchronization mechanics, metadata schema, and annotation protocols for future synchronized data collection.  
> **No synchronized physical UAV audiovisual dataset currently exists in this repository.**  
> Previous Mission 21 scenario pairings were asynchronous experimental simulations. Real-world simultaneity cannot and must not be claimed until data is collected under this protocol.

---

## 2. Hardware Synchronization Architecture

To eliminate the "phantom pairing" artifacts that affected Mission 21, the physical data acquisition system must achieve microsecond-grade timestamp correlation across independent sensor modalities.

```
       [ GPS / Precision Time Reference (PTP IEEE 1588) ]
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
      [ Hardware Trigger / PPS ]    [ Audio Sample Clock Sync ]
                 |                             |
                 v                             v
      [ Global Shutter Camera ]    [ Digital I2S Microphone ]
        Exposure Center: T_vis       Buffer Start: T_aud
                 |                             |
                 +--------------+--------------+
                                |
                   Delta T = |T_vis - T_aud| <= 5 ms
                                v
               [ Common Synchronized Stream (.h5 / .mcap) ]
```

### 2.1 Hardware Specifications
1. **Clock Master:** GPS-synchronized Pulse-Per-Second (PPS) or onboard microcontroller running Precision Time Protocol (PTP IEEE 1588) with sub-millisecond precision.
2. **Video Sensor:**
   - Global shutter RGB sensor (e.g., Sony IMX477 or Basler dart).
   - Capture resolution: $1920\times 1080$ at $30.0\text{ FPS}$.
   - Exact hardware exposure midpoint logged in nanoseconds (`frame_timestamp_utc_ns`).
3. **Audio Transducer:**
   - Digital $I^2S$ MEMS omnidirectional microphone with vibration-damped mounting.
   - Sampling rate: $16,000\text{ Hz}$, 16-bit linear PCM mono.
   - Buffer alignment: Audio buffered in $1.0\text{ s}$ overlapping windows ($50\%$ overlap, $500\text{ ms}$ hop).
4. **Permissible Drift:**
   $$\Delta T = |t_{\text{frame\_mid}} - t_{\text{audio\_window\_mid}}| \le 5.0\text{ ms}$$
   Any record with drift $|\Delta T| > 5.0\text{ ms}$ must be flagged as `SYNC_DRIFT_EXCEEDED` and excluded from benchmark evaluation.

---

## 3. Environmental Scenario Matrix (Controlled Staged & Field)

The collection protocol mandates balanced coverage across a $3\times 3$ environmental matrix:

| Optical Condition | Clean Acoustic Speech | Drone Rotor Noise (Hovering @ 10m) | Adverse Acoustic Weather (Rain / Wind / Flood) |
| :--- | :--- | :--- | :--- |
| **Daylight Clear** ($> 500\text{ lux}$) | Baseline calibration | Motor masking evaluation | Rain / wind turbulence |
| **Adverse Weather** (Fog / Rain / Snow) | Optical attenuation study | Multi-modal degradation | Severe dual-channel stress |
| **Nighttime / Low Light** ($< 5\text{ lux}$) | Acoustic-dominant trigger | Rotor interference in darkness | Disaster zone search test |

---

## 4. Cross-Modal Annotation Protocol

### 4.1 Frame-Level Visual Annotations
For every video frame ($30\text{ FPS}$):
- Format: Standard YOLO format `[class_id cx cy w h]`.
- Classes:
  - `0: person_standing`
  - `1: person_fallen_unconscious`
  - `2: vehicle`
  - `3: boat_rescue_craft`
- Occlusion Tag: `visible (>75%)`, `partially_occluded (25-75%)`, `severely_occluded (<25%)`.

### 4.2 Segment-Level Acoustic Annotations
For every paired $1.0\text{ s}$ audio buffer:
- Label: `distress_cry` (*"বাঁচাও"*, screams, groans) vs. `normal_speech` vs. `environmental_noise_only`.
- Language & Dialect: Bengali (Standard, Sylheti, Chittagonian, Noakhali).
- Estimated Signal-to-Noise Ratio (SNR) in dB.

### 4.3 The Critical Cross-Modal Association Link
To prevent the phantom-person flaw of Mission 21, every synchronized sample must declare the exact cross-modal identity relationship:
- `same_individual`: The visible person is actively vocalizing the distress cry.
- `different_individual`: A person is visible, but the distress cry originates from a different occluded/unseen person.
- `visual_person_silent`: A person is visible, but no distress vocalization occurs (unconscious victim or calm bystander).
- `audio_person_occluded`: Distress cry is heard, but the victim is completely occluded by darkness, fog, or debris (zero bounding box).
- `no_persons`: Neither visible person nor vocalization exists (control sample).

---

## 5. Human-Subject Ethics, Safety & Consent Protocol

1. **Informed Consent:** Every human participant vocalizing distress phrases or appearing in video frames must sign an Institutional Ethics Review Board (IRB) approved consent form allowing anonymized research dissemination.
2. **Privacy Protection:** Identifiable facial features in non-distress bystander categories must be blurred prior to public benchmark release.
3. **Acoustic Safety:** Staged recordings simulating distress must be conducted in controlled safety corridors with advance notification to local emergency services to avoid false municipal dispatches.

---

## 6. Data Tier Segregation Rules

All future corpora must strictly segregate data across four tiers in filenames and manifests:
1. `TIER_1_FIELD_SYNCHRONIZED`: Authentic drone flight recordings with hardware-synchronized video and audio.
2. `TIER_2_STAGED_SYNCHRONIZED`: Controlled field experiments with actors and common hardware clock.
3. `TIER_3_BENCHMARK_AUGMENTED`: Physical audio/video augmented with calibrated synthetic noise.
4. `TIER_4_SYNTHETIC_SIMULATION`: Asynchronous pairings used solely for architectural proof-of-concept.

**Non-Negotiable Rule:** TIER 4 data must never be reported as TIER 1 or TIER 2 physical evidence.

Phase 5 is complete.
