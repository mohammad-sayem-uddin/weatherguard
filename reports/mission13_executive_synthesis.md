# WeatherGuard — Mission 13 Executive Research Synthesis
## Priority-Gated Candidate Allocation, Crop-Budget Optimization & Reliable Edge Inference

**Project:** WeatherGuard  
**Mission:** Mission 13  
**Execution Environment:** Apple MacBook Air M1, 8-core CPU, 8 GB unified memory, macOS, CPU-only  
**Author:** Principal Computer Vision Researcher & Edge-AI Optimization Engineer  
**Status:** Completed & Validated  

---

## 1. Executive Summary

Mission 13 resolves the central scientific dilemma identified at the conclusion of Mission 12: **the proposal dilution problem**. 

While Mission 12 demonstrated that inexpensive visual and motion cues can identify distant objects completely invisible to a low-resolution base detector, indiscriminate proposal fusion caused high-novelty independent proposals to displace critical detector-generated candidates under fixed crop budgets ($K=2$).

In Mission 13, we designed, implemented, and validated the **Priority-Gated Candidate Allocation Architecture (`src/priority_allocation.py`)**. Evaluating 10 factorial configurations across 1,213 adverse-weather video frames (405 dev, 808 locked reference) and 150 FLIR ADAS Thermal images establishes the following definitive conclusions:

1. **Elimination of Proposal Dilution:** Under hierarchical detector reservation (**Policy J**), detector candidate displacement drops from **6,695 instances** in unconstrained scoring to **strictly 0 (0.0% displacement)**.
2. **Superior Small-Object Acuity:** Policy J achieves **Small-Object $F_1 = 0.3289$** on the locked test cohort, surpassing Base YOLO640 ($0.1737$) by **+89.3% relative (+0.1552 absolute)**, outperforming full-frame Fixed 832px ($0.2809$) by **+17.1% relative (+0.0480 absolute)**, and exceeding detector-only clustered ROI ($0.3177$) by **+3.5% relative (+0.0112 absolute)**.
3. **Ultra-Small Recovery Preserved:** Policy J achieves **Ultra-Small $F_1 = 0.0259$** (recovering distant horizon vehicles where Base YOLO640 and Fixed 832px score $0.0000$).
4. **Statistically Confirmed Superiority:** Paired circular block bootstrapping ($B=1,000, L=20$) confirms that small-object gains over Base 640px ($\Delta = +0.1554, p < 0.001$) and Fixed 832px ($\Delta = +0.0494, p < 0.001$) are statistically significant and robust across block lengths $L \in \{10, 20, 40\}$.
5. **Cross-Modality Transfer:** Zero-shot transfer to FLIR ADAS Thermal (LWIR) yields **Overall $F_1 = 0.3229$** and **Small $F_1 = 0.1023$** (+56.2% relative gain over Base 640px) and **Ultra-Small $F_1 = 0.0870$**.
6. **Edge Hardware Viability:** On Apple M1 CPU within 502 MB RAM, standardized non-overlapping execution demonstrates that adaptive crop budgeting and exclusion of costly optical flow delivers sustained, robust inference.

---

## 2. Answers to the 10 Final Research Questions

### RQ1: Can priority-gated candidate allocation prevent the proposal dilution problem observed in Mission 12?
**Yes, completely.**  
In Mission 12, tri-modal proposal fusion allowed unconstrained independent proposals to compete directly with detector proposals for crop slots, displacing valuable detector seeds and dropping small-object F1 from $0.3117 \to 0.2223$.  
Under Mission 13's **PriorityAllocationController**, a two-tier priority queue with guaranteed detector reservation ($K_{\text{det}} = \min(K, N_{\text{det\_clusters}})$) ensures that detector proposals are never displaced. In the locked reference audit of 808 frames, detector displacement was **exactly 0** ($0.0\%$ displacement rate), completely eliminating proposal dilution.

### RQ2: What allocation policy achieves the best trade-off between small-object recall, ultra-small recovery, and edge inference speed?
**Policy J (Best Priority-Gated Combination).**  
Policy J synergistically combines:
1. **Tier 1 Detector Reservation:** Prevents detector displacement.
2. **Tier 2 Four-Gate Admission Filter:** Rejects 98.7% of nuisance independent proposals via horizon prior, aspect ratio, spatial novelty ($\text{IoU} < 0.25$), and evidence hurdle ($s \ge 0.15$).
3. **Adaptive Crop Budgeting ($K \in \{0, 1, 2\}$):** Dynamically skips crops on barren frames and single-crops isolated clusters.
4. **Decoupled Stream 2 ByteTrack Association:** Preserves high detection precision without trajectory extrapolation corruption.  
Policy J delivers the highest Small $F_1$ ($0.3289$), preserves ultra-small recovery ($0.0259$), and maintains edge CPU execution.

### RQ3: How much do independent proposals contribute to final detection performance when detector-generated candidates are prioritized?
**Independent proposals provide a focused, high-value marginal recovery of distant vehicles near the vanishing line without diluting core recall.**  
Because detector candidates are prioritized, independent proposals only receive crops when budget remains. Across 808 test frames, 358 independent proposals were admitted; 41.6% of these resolved into true-positive detections of distant vehicles that were completely undetectable by the base 640px pass. This yielded an incremental $+0.0113$ Small $F_1$ ($p = 0.0130$) and $+0.0259$ Ultra-Small $F_1$ ($p = 0.0010$) over detector-only inference.

### RQ4: Does adaptive crop budgeting improve average inference speed without hurting worst-case detection recall?
**Yes.**  
By evaluating crop worthiness dynamically:
- Frames with no candidate clusters require 0 crops ($43.19\text{ ms}, 23.15\text{ FPS}$).
- Frames with a single isolated cluster execute 1 crop ($70.99\text{ ms}, 14.09\text{ FPS}$).
- Frames with complex traffic execute 2 crops ($98.79\text{ ms}, 10.12\text{ FPS}$).  
Because crop execution is only bypassed when no candidate meets the spatial evidence hurdle, worst-case recall is identical to fixed $K=2$ execution, while average sustained throughput increases by **+23.9%**.

### RQ5: How does priority-gated candidate allocation perform under severe weather conditions compared to full-frame high-resolution inference?
**Priority-gated allocation systematically outperforms full-frame 832px inference on small and ultra-small objects under severe weather.**  
On locked test driving videos featuring heavy rain, dusk, and pavement glare:
- Full-frame 832px achieves Small $F_1 = 0.2809$ and Ultra-Small $F_1 = 0.0000$.
- Policy J achieves Small $F_1 = \mathbf{0.3289}$ (+17.1% relative gain) and Ultra-Small $F_1 = \mathbf{0.0259}$.  
By extracting localized $448\times 448\text{px}$ crops, the effective spatial resolution on distant vehicles is $2.0\times$ higher than downscaling the entire 1280px frame to 832px, enabling the detector head to resolve headlights and wipers through dense rain streaks.

### RQ6: Does priority-gated candidate allocation generalize across different sensing modalities (e.g., FLIR ADAS Thermal)?
**Yes, robustly.**  
On 150 FLIR ADAS Thermal (LWIR) validation images:
- Base YOLO640: Overall $F_1 = 0.2994$, Small $F_1 = 0.0655$, Ultra-Small $F_1 = 0.0000$.
- Policy J: Overall $F_1 = \mathbf{0.3229}$, Small $F_1 = \mathbf{0.1023}$ (+56.2% relative gain), Ultra-Small $F_1 = \mathbf{0.0870}$.  
Perspective-weighted static visual salience effectively captures high-contrast thermal gradients from vehicle radiators and pedestrians, transferring seamlessly to infrared data without retraining.

### RQ7: What are the computational bottlenecks of priority-gated candidate allocation on edge hardware, and how can they be mitigated?
The standardized M1 micro-benchmark identifies three distinct bottlenecks:
1. **Serial Crop Inference ($27.8\text{ ms}$ per 448px crop):** Serial execution of 2 crops consumes $55.6\text{ ms}$. *Mitigation:* Adaptive crop budgeting reduces average crop count from $2.0 \to 1.38$, and batched tensor inference reduces forward pass overhead.
2. **Sparse Optical Flow ($17.75\text{ ms}$):** Tracking Shi-Tomasi corners imposes heavy CPU latency with negligible incremental recall yield ($0.3549$ yield/ms). *Mitigation:* Optical flow is excluded from edge execution; lightweight frame differencing ($1.25\text{ ms}$) provides $15.8\times$ higher recall efficiency.
3. **Coordinate Restoration & Slicing ($1.71\text{ ms}$):** Negligible CPU cost (<2 ms combined).

### RQ8: How does priority-gated candidate allocation interact with multi-object tracking and temporal video detection?
**Decoupled two-stage integration provides optimal perception stability.**  
Following the causal principles established in Mission 10, priority-gated crop detections are merged into Stream 1 using conservative NMS ($\tau_{\text{merge}} \ge 0.35, \text{IoU} \ge 0.45$). This clean detection stream is then fed into `AdverseWeatherByteTracker` (Stream 2), achieving an **IDF1 of 0.4180** and an ID switch rate of only **0.209 switches/frame** with zero track fragmentation caused by boundary artifacts. Trajectory extrapolation is strictly excluded from detection outputs to prevent precision collapse.

### RQ9: Is the candidate-generation bottleneck fundamentally solved by priority-gated allocation, or does an irreducible ceiling remain?
**The algorithmic bottleneck of proposal dilution is solved, but a physical optical ceiling remains.**  
- **Solved Bottleneck:** The system can now generate, filter, prioritize, and verify proposals without displacing detector candidates or suffering false alarm floods.
- **Irreducible Ceiling:** For objects below $6\times 6\text{px}$ ($<36\text{px}^2$) under heavy fog or distant spray, optical information is destroyed by atmospheric scattering before reaching the lens. No high-resolution crop can recover features when optical contrast is below sensor noise floors.

### RQ10: What is the definitive architecture for edge-efficient adverse-weather perception in WeatherGuard?
The definitive WeatherGuard perception architecture comprises:
```
                                   ┌───────────────────────────────────┐
                                   │  Incoming Driving Video Frame     │
                                   │        (720p @ 30 FPS)            │
                                   └─────────────────┬─────────────────┘
                                                     │
                                   ┌─────────────────▼─────────────────┐
                                   │  Base YOLOv8n Pass @ 640x640px    │ (38.0 ms)
                                   └───────┬───────────────────┬───────┘
                                           │                   │
                        Detections >= 0.35 │                   │ Low-Conf & Candidate Cues
                                           │                   ▼
                                           │       ┌───────────────────────────────┐
                                           │       │ PriorityAllocationController  │
                                           │       │ 1. Detector Reservation (T1)  │ (1.73 ms)
                                           │       │ 2. Salience / FrameDiff (T2)  │
                                           │       │ 3. 4-Gate Admission Filter    │
                                           │       │ 4. Adaptive Budget K in {0,1,2}
                                           │       └───────────────┬───────────────┘
                                           │                       │
                                           │                       ▼ Allocated Crops (K <= 2)
                                           │       ┌───────────────────────────────┐
                                           │       │ Crop Forward Pass @ 448x448px │ (27.8 - 55.6 ms)
                                           │       └───────────────┬───────────────┘
                                           │                       │
                                           │                       ▼
                                           │       ┌───────────────────────────────┐
                                           │       │ Coordinate Restoration &      │
                                           │       │ 4px Boundary Edge Suppression │ (0.51 ms)
                                           │       └───────────────┬───────────────┘
                                           │                       │
                                           ▼                       ▼
                                   ┌───────────────────────────────────────────┐
                                   │ Two-Tier Conservative NMS Merge           │
                                   │ (tau_merge >= 0.35, IoU >= 0.45)          │
                                   └─────────────────────┬─────────────────────┘
                                                         │ Stream 1 (Pure Detections)
                                                         ▼
                                   ┌───────────────────────────────────────────┐
                                   │ AdverseWeatherByteTracker (Stream 2)      │
                                   │ Low-confidence association (0.20 / 0.10)  │ (0.35 ms)
                                   └─────────────────────┬─────────────────────┘
                                                         │
                                                         ▼ Final Output
                                              Tracks & Validated Objects
```
This unified architecture achieves **0.4662 Overall $F_1$**, **0.3289 Small-Object $F_1$**, **0.0259 Ultra-Small $F_1$**, **0.4180 IDF1**, and runs reliably within 502 MB RAM on edge hardware.
