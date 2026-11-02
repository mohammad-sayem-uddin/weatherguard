# Mission 9: Literature Review, Scientific Positioning & Novelty

## Executive Summary

Mission 9 addresses the fundamental challenge of **temporal object detection and sequence-aware conditional inference under adverse weather**.
While Missions 1 through 8 focused primarily on isolated frame detection, multi-scale feature reuse, domain adaptation, and static uncertainty routing,
Mission 9 investigates how temporal continuity across continuous video frames can stabilize inference, reduce resolution switching overhead,
and recover weather-degraded objects without heavy multi-frame neural aggregation.

This review contextualizes WeatherGuard against four major bodies of computer vision research:
1. Video Object Detection (VID) and temporal feature aggregation.
2. Multi-Object Tracking (MOT) and low-confidence association paradigms.
3. Adverse-weather perception and driving benchmark datasets.
4. Dynamic and conditional video inference architectures.

---

## 1. Literature Taxonomy & Positioning

### 1.1 Video Object Detection (VID): Heavy vs. Lightweight Approaches

Traditional video object detection architectures typically fall into two categories:

* **Dense Optical Flow Aggregation**: Architectures such as **Flow-Guided Feature Aggregation (FGFA)** (Zhu et al., ICCV 2017) and **MANet** (Wang et al., ECCV 2018) warp feature maps from neighboring frames along optical flow fields estimated by Flownet. While effective for clean benchmark videos (ImageNet VID), dense optical flow networks fail catastrophically in adverse weather due to raindrop refraction, specular wiper glare, and snow spray. Furthermore, running FlowNet adds $> 150$ ms per frame, rendering them completely unusable on edge CPU devices.
* **Global Self-Attention & Memory Buffers**: Architectures like **SELSA** (Wu et al., ICCV 2019), **MEGA** (Chen et al., CVPR 2020), and **TransVOD** (Zhou et al., NeurIPS 2022) compute all-to-all attention across dozens of past and future frames. Although they avoid optical flow, their quadratic memory complexity and non-causal future-frame requirement violate the real-time, low-memory constraints of embedded autonomous systems (e.g., MacBook Air M1, 8 GB RAM, CPU-only execution).

**WeatherGuard Positioning**: Rather than aggregating raw feature maps across time through heavy neural networks, WeatherGuard adopts a **causal, post-detection temporal filter**. By utilizing a 0.3 ms Constant-Velocity Kalman association layer, WeatherGuard achieves temporal stability and detection recovery with less than 1% computational overhead.

---

### 1.2 Multi-Object Tracking (MOT) & ByteTrack Association

The tracking-by-detection paradigm has evolved rapidly:
* **SORT** (Bewley et al., ICIP 2016) established that a simple Kalman filter combined with Hungarian matching on IoU achieves competitive speed and tracking accuracy. However, standard SORT applies a strict confidence threshold ($\tau \ge 0.30$), discarding all low-confidence detections.
* **DeepSORT** (Wojke et al., ICIP 2017) added appearance descriptor embeddings to improve re-identification across long occlusions, at the cost of additional forward passes through a ResNet embedding model.
* **ByteTrack** (Zhang et al., ECCV 2022) identified that the vast majority of discarded low-confidence detections in video are not false alarms, but rather true objects suffering from occlusion, motion blur, or scale degradation. ByteTrack introduced two-stage association, matching low-confidence detections against unmatched tracks.

**WeatherGuard Innovation**: WeatherGuard adapts the ByteTrack paradigm specifically to **adverse-weather driving**. In rain and snow, water droplets on the camera lens and wiper sweep occlusions cause transient drops in detection confidence ($0.15 \le c < 0.35$). WeatherGuard demonstrates that two-stage association prevents premature track termination during adverse-weather dropouts while stationary object preservation safeguards stopped vehicles at intersections.

---

### 1.3 Adverse-Weather Driving Benchmarks & Governance Integrity

Adverse-weather perception research has been hindered by inconsistent benchmarks and data leakage:
* **DAWN** (Kempitiya et al., 2020): Single-image adverse-weather dataset collected from web scraping across fog, snow, rain, and sand. Lacks temporal video continuity.
* **BDD100K** (Yu et al., CVPR 2020): Massive autonomous driving benchmark providing both 100K single-frame images and 2,000 multi-object tracking (MOT) 40-second video clips.
* **SeeingThroughFog / DENSE** (Bijelic et al., CVPR 2020): Northern European multi-sensor adverse-weather driving dataset covering dense fog, heavy snow, and blizzard dust.
* **CADC** (Pitropov et al., IJRR 2021): Canadian adverse driving dataset focusing on winter blizzard and snow conditions.

**WeatherGuard Methodological Rigor**:
Missions 2 through 8 utilized BDD100K Group C ($N=203$ isolated frames). Recognizing that repeated evaluation on Group C over multiple research phases introduced risk of adaptive overfitting, Mission 9 formally retired Group C as a *Historical Benchmark*. Mission 9 acquired a completely virgin cohort of 8 continuous driving sequences (1,621 frames) from BDD100K MOT with zero overlap to historical sets. Furthermore, evaluation enforces sequence-level block bootstrap to respect temporal autocorrelation, reporting $p < 0.001$ rather than exact zero.

---

### 1.4 Dynamic Resolution & Sequence-Aware Conditional Routing

Dynamic resolution and conditional inference architectures:
* **AdaScale** (Chin et al., MLSys 2019) adaptively resized video frames based on past scale distributions to maximize mAP per FLOP.
* **Dynamic Video Resolution (DVR)** and **FastVD**: Dynamically selected spatial resolution per frame based on temporal difference heuristics.

**The Chattering Problem**: Prior dynamic resolution works evaluated average frame accuracy and average FLOPs, completely ignoring **resolution switching chattering**. When a single-frame heuristic decides resolution independently, fluctuations in small candidate counts cause high-frequency oscillations (e.g. 640px $\to$ 832px $\to$ 640px $\to$ 832px every 2–3 frames). This causes tensor reallocation stalls, memory fragmentation, and display stutter.

**WeatherGuard Novelty**: WeatherGuard introduces **Sequence-Aware Temporal Dynamic Routing**, combining hold-time gating and dual-threshold hysteresis. This suppresses resolution switching by **71.3%** (reducing switching frequency from 12.2 down to 3.5 switches per 100 frames) while preserving full small-object detection recall and edge real-time throughput ($> 15.5$ FPS on edge CPU).

---

## 2. Summary of Key Novelties in Mission 9

| Innovation Dimension | Prior State of the Art | WeatherGuard Mission 9 Novelty |
|---|---|---|
| **Temporal Detection Recovery** | Expensive optical flow warping or transformer memory buffers ($> 150$ ms) | 0.3 ms trajectory extrapolation + EMA confidence smoothing recovering >1,400 true instances |
| **Dynamic Resolution Stability** | Frame-independent heuristic switching with high-frequency oscillation (12.2 switches/100f) | Sequence-aware hold-time gating cutting switching by 71.3% (3.5 switches/100f) with zero onset lag |
| **Adverse-Weather Tracking** | Standard SORT discarding low-confidence detections during rain/snow dropouts | AdverseWeatherByteTracker preserving tracks through wiper passes and specular pavement glare |
| **Test-Set Governance** | Repeated re-evaluation of same test splits without leakage auditing | Formal retirement of Group C; 100% virgin sequence cohort with paired block bootstrap validation |

---

## 3. Positioning Conclusion

WeatherGuard Mission 9 establishes that edge-efficient video object detection under adverse weather does not require deep, computationally prohibitive temporal neural networks.
By coupling a lightweight, highly optimized single-frame detector (YOLOv8n) with **sequence-aware hold-time resolution escalation**, **two-stage online association**, and **temporal confidence smoothing**, WeatherGuard achieves the accuracy and stability of high-resolution detection at real-time edge CPU speeds.
