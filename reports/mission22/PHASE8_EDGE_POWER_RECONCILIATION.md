# MISSION 22 — PHASE 8: Edge Deployment Feasibility, Latency Benchmarks & Flight Power Reconciliation

**Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** `https://github.com/mohammad-sayem-uddin/weatherguard`  
**Branch:** `master`  
**Artifact:** `artifacts/mission22/edge_power_reconciliation.json`  
**Script:** `scripts/mission22/reconcile_edge_power.py`  
**Date:** November 5, 2026  

---

## 1. Executive Summary

Phase 8 audited the $3.8\times$ flight endurance extension claimed in Mission 21, measured real on-device model latencies on local hardware, and reconciled the underlying electrical and aerodynamic physics.

### Core Findings & Scientific Disambiguation:
1. **Mathematical Origin of the $3.8\times$ Multiplier:**  
   The $3.8\times$ ratio was reproduced exactly from the compute power equation:
   $$P_{\text{avg}} = P_{\text{standby}} \cdot (1 - d) + P_{\text{active}} \cdot d$$
   where $P_{\text{active}} = 18.5\text{ W}$, $P_{\text{standby}} = 2.8\text{ W}$, and active tracking fraction $d = 0.13$.  
   This yields $P_{\text{avg}} = 4.84\text{ W}$, representing a **$3.82\times$ reduction in compute power** ($18.5\text{ W} / 4.84\text{ W} = 3.82$).
2. **The Propulsion Physics Reality Check:**  
   - The $3.8\times$ extension applies **ONLY to the compute subsystem** (or a dedicated auxiliary compute battery).
   - For a standard $1.5\text{ kg}$ quadrotor UAV, **motor hover propulsion consumes $\approx 180-220\text{ W}$**, dwarfing onboard compute.
   - On a shared flight battery ($59.2\text{ Wh}$ usable energy), total UAV power drops from $203.5\text{ W} \to 189.8\text{ W}$, yielding an overall flight endurance gain of **$+6.86\%$ (+1.2 minutes: $17.5\text{ min} \to 18.7\text{ min}$)**.
   - Representing duty-cycling as extending physical drone flight time from 18 to 68 minutes was an ungrounded extrapolation that failed to account for multirotor aerodynamics.
3. **Hardware Latency Benchmarked on Apple M1 CPU:**  
   - Vision YOLOv8n: **$51.46\text{ ms}$** ($19.4\text{ FPS}$) at $640\times 640$ FP32.
   - Audio Standard CNN: **$1.39\text{ ms}$** per audio spectrogram window.
   - Audio Lightweight Separable CNN: **$0.98\text{ ms}$** per audio window.
4. **Hardware Status Labeling:**  
   Raspberry Pi 4B and Raspberry Pi 5 figures are explicitly classified as **UNMEASURED ENGINEERING PROJECTIONS**, preserving scientific integrity.

---

## 2. Real Hardware Latency Measurements (Apple M1 CPU)

Evaluated on Apple MacBook Air M1 (8-core ARM, 3.2 GHz, 8 GB Unified Memory):

| Model Architecture | Task | Input Dimension | Parameter Count | Disk Size | Measured Latency (Median) | Throughput | Measurement Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n (Fine-Tuned)** | Object Detection | $1\times 3\times 640\times 640$ | 3,139,218 | 6.19 MB | **51.46 ms** | 19.4 FPS | **DIRECTLY BENCHMARKED** |
| **Audio Standard CNN** | Distress Spotting | $1\times 1\times 64\times 126$ | 1,076,417 | 4.12 MB | **1.39 ms** | 719.4 windows/s | **DIRECTLY BENCHMARKED** |
| **Audio Lightweight CNN** | Distress Spotting | $1\times 1\times 64\times 126$ | 6,705 | 0.04 MB | **0.98 ms** | 1,020.4 windows/s | **DIRECTLY BENCHMARKED** |

---

## 3. Engineering Projections vs. Hardware Measurements

| Platform Target | Processor Architecture | Vision YOLOv8n | Audio Standard CNN | Audio Lightweight CNN | Provenance & Validation Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Apple M1 CPU** | 8-core ARM (3.2 GHz) | 51.46 ms | 1.39 ms | 0.98 ms | **BENCHMARKED (CONFIRMED)** |
| **NVIDIA Tesla T4** | Turing Cloud (16 GB VRAM) | 3.80 ms | 0.15 ms | 0.08 ms | **CLOUD BENCHMARKED (CONFIRMED)** |
| **Raspberry Pi 5** | Quad Cortex-A76 (2.4 GHz) | 78.0 ms (INT8 proj) | 7.82 ms (proj) | 4.61 ms (proj) | **ENGINEERING PROJECTION (UNMEASURED)** |
| **Raspberry Pi 4B** | Quad Cortex-A72 (1.5 GHz) | 185.0 ms (INT8 proj) | 18.2 ms (proj) | 10.64 ms (proj) | **ENGINEERING PROJECTION (UNMEASURED)** |

---

## 4. Rigorous Power & Battery Endurance Reconciliation

### 4.1 Case A: Dedicated Compute Payload Battery (Subsystem Level)
Assumes a separate $2\text{S}$ auxiliary LiPo battery ($7.4\text{ V}, 1,000\text{ mAh}$, $5.92\text{ Wh}$ usable):
- **Continuous Mode:** $P = 18.5\text{ W} \implies T = \frac{5.92}{18.5} \times 60 = \mathbf{19.2\text{ minutes}}$.
- **Duty-Cycled Mode:** $P_{\text{avg}} = 2.8 \times 0.87 + 18.5 \times 0.13 = 4.84\text{ W} \implies T = \frac{5.92}{4.84} \times 60 = \mathbf{73.4\text{ minutes}}$.
- **Endurance Multiplier:**
  $$\text{Compute Battery Gain} = \frac{73.4}{19.2} = \mathbf{3.82\times}$$
- **Verdict:** Validated strictly as a **compute payload subsystem metric**.

### 4.2 Case B: Shared Multirotor Flight Battery (Full System Level)
Assumes a standard $1.5\text{ kg}$ SAR quadrotor powered by a $4\text{S}$ LiPo battery ($14.8\text{ V}, 5,000\text{ mAh}$, $59.2\text{ Wh}$ usable):
- Motor Hover Propulsion: $180.0\text{ W}$
- Flight Controller & Avionics: $5.0\text{ W}$
- **Continuous Mode Power:** $180 + 5 + 18.5 = 203.5\text{ W} \implies T = \frac{59.2}{203.5} \times 60 = \mathbf{17.5\text{ minutes}}$.
- **Duty-Cycled Mode Power:** $180 + 5 + 4.84 = 189.84\text{ W} \implies T = \frac{59.2}{189.84} \times 60 = \mathbf{18.7\text{ minutes}}$.
- **Flight Time Gain:**
  $$\text{Real Flight Extension} = \frac{18.7 - 17.5}{17.5} \times 100\% = \mathbf{+6.86\%} \quad (+1.2\text{ minutes})$$

### 4.3 Scientific Guidance for Future Publications
In academic papers, the $3.8\times$ factor must be described accurately as:
> *"Acoustic-triggered visual duty-cycling reduces companion computer power drain by $73.8\%$ ($18.5\text{ W} \to 4.84\text{ W}$, a $3.8\times$ compute endurance extension), yielding a $+6.9\%$ flight endurance extension on a shared $1.5\text{ kg}$ multirotor flight battery."*

Never represent compute power savings as physical multirotor flight duration without propulsion modeling.

Phase 8 is complete.
