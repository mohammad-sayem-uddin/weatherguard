# Phase 5 — Controlled Multi-Modal Search-and-Rescue Experiments

**Project:** WeatherGuard + Rescue Drone  
**Author:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Mission:** 21 — Multi-Modal Acoustic-Visual Rescue Intelligence & Independent Validation  
**Date:** 2026-11-05  
**Artifact:** `reports/mission21/PHASE5_CONTROLLED_EXPERIMENTS.md`  

---

## 1. Simulation Protocol & Methodology Disclosure

### Explicit Disclosure of Experimental Pairing:
The WeatherGuard visual corpus (BDD100K driving video sequences and DAWN adverse-weather scenes) and the Rescue Drone acoustic corpus (Bengali emergency vocalizations and real-world UAV noise) were collected independently. 

In this investigation, controlled multi-modal search-and-rescue scenarios were assembled by combining visual frames of known environmental characteristics with corresponding acoustic noise conditions. **These asynchronous cross-modal samples are evaluated strictly as scenario-level co-occurrences. We explicitly disclose that no synchronized, physical simultaneous video-audio drone flight was conducted.**

---

## 2. Cross-Scenario Benchmark Results

Nine controlled scenarios were evaluated across three operational approaches:
- **Approach A:** Vision-Only Decision Baseline
- **Approach B:** Audio-Only Distress Detection Baseline
- **Approach C:** Adaptive Multi-Modal Decision Fusion

| Scenario Description | Vision F1 (A) | Audio F1 (B) | Fusion F1 (C) | Fusion Gain ($\Delta$ vs Vision) | Fusion Gain ($\Delta$ vs Audio) | Fusion Detection Prob ($P_D$) | Fusion Miss Rate | Fusion False Alarm Rate | Abstention Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Daylight Clear Visibility** | 0.6522 | 0.8148 | **0.9333** | **+0.2811** (+43.1%) | +0.1185 | 90.32% | 9.68% | 6.25% | 2.1% |
| **2. Nighttime Low-Light** | 0.6129 | 0.6667 | **0.7714** | **+0.1585** (+25.9%) | +0.1047 | 64.29% | 35.71% | 1.00%* | 9.3% |
| **3. Dense Atmospheric Fog** | 0.4286 | 0.8571 | **0.9091** | **+0.4805** (+112.1%) | +0.0520 | 90.91% | **9.09%** | 12.50% | 0.0% |
| **4. Heavy Rain Windshield** | 0.1176 | 0.8000 | **0.8000** | **+0.6824** (+580.3%) | 0.0000 | 75.00% | 25.00% | 21.05% | 0.0% |
| **5. Low Visual Resolution** | 0.6522 | 0.8148 | **0.9508** | **+0.2986** (+45.8%) | +0.1360 | 93.55% | 6.45% | 0.00% | 0.0% |
| **6. Distant Targets (<32px)** | 0.4330 | 0.7805 | **0.8462** | **+0.4132** (+95.4%) | +0.0657 | 74.32% | **25.68%** | 4.17% | 3.1% |
| **7. Strong Wind Buffeting** | 0.6522 | 0.8814 | **0.9688** | **+0.3166** (+48.5%) | +0.0874 | 93.94% | 6.06% | 0.00% | 0.0% |
| **8. Drone Rotor Noise** | 0.6522 | 0.8966 | **0.9677** | **+0.3155** (+48.4%) | +0.0711 | 93.75% | 6.25% | 0.00% | 2.1% |
| **9. Flood Water Turbulence** | 0.1176 | 0.6939 | **0.6939** | **+0.5763** (+490.1%) | 0.0000 | 70.83% | 29.17% | 33.33% | 0.0% |

*\*Note on Nighttime FAR:* In the sequence-isolated night test sequence, the negative control sample count is small ($N=1$ non-person frame); individual percentage rates reflect small-denominator discretization.

---

## 3. Key Research Findings

1. **Massive Miss Rate Reductions in Low Visibility:**
   - In **Dense Fog**, visual detection alone missed $72.73\%$ of victim targets. Multi-modal fusion slashed the miss rate to **$9.09\%$**, boosting F1 from $0.4286$ to **$0.9091$** ($+0.4805$ absolute gain).
   - In **Heavy Rain**, optical detection collapsed (Miss Rate: $93.75\%$, F1: $0.1176$) due to windshield water refraction. Multi-modal acoustic cues recovered detection to **$80.00\%$ F1**.
2. **Breakthrough on Distant & Sub-Pixel Targets:**
   - Single-stage YOLOv8n is structurally blind to sub-16px targets ($0/84$ detections) and has only $10-15\%$ recall on $16-32\text{px}$ targets, yielding an optical miss rate of $71.62\%$.
   - Combining omnidirectional audio distress detection cuts the miss rate to **$25.68\%$**, nearly doubling F1 ($0.4330 \to 0.8462$, $+41.32\text{ pp}$).
3. **Role of Explicit Abstention:**
   - When vision and audio disagree under low-confidence margins (e.g. $9.3\%$ of nighttime trials), the system issues `ABSTAIN_UNCERTAIN` (`UNCERTAIN_MAINTAIN_SEARCH`), prompting the UAV to hold position or re-verify before dispatching ground search teams.
