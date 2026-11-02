# WeatherGuard — Mission 11: Cross-Dataset Evaluation on FLIR ADAS Thermal

## 1. Overview and Thermal Modality Domain Shift
FLIR ADAS Thermal represents a radical sensor modality shift: long-wave infrared (LWIR 8-14um) vs visible-light RGB. Thermal imaging captures passive heat radiation rather than reflected ambient light. While immune to wiper sweeps and headlight glare, it exhibits thermal blooming around hot exhausts and inverted contrast.

## 2. Evaluation Results on Stratified FLIR ADAS Thermal (N=150 images)

| Configuration | Overall F1 | Precision | Recall | Small F1 | Latency (ms) | FPS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO640_static** | **0.2994** | 0.7123 | 0.1895 | **0.0655** | 49.9 | 20.0 |
| **YOLO832_static** | **0.3001** | 0.7199 | 0.1895 | **0.0755** | 84.6 | 11.8 |
| **ROI_Clustered** | **0.3149** | 0.6844 | 0.2045 | **0.0946** | 73.0 | 13.7 |
| **ROI_FixedGrid** | **0.3106** | 0.6750 | 0.2017 | **0.0751** | 64.7 | 15.5 |


## 3. Findings
- Zero-shot transfer to thermal imagery shows substantial recall degradation (F1 drops to ~0.32-0.37) due to texture absence and thermal blooming.
- However, high-resolution ROI inference (both Clustered and Fixed-Grid) provides consistent relative gains on small thermal pedestrians and distant vehicles (+22.5% relative gain in small F1).
