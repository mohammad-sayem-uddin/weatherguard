# WeatherGuard — Mission 11: Edge CPU Benchmark on Apple M1

**Platform:** Apple MacBook Air M1, 8-core CPU, 8 GB unified memory | **Device:** cpu | **Threads:** 7

## 1. Component Latency Breakdown (ms)
- `base_640_inference`: **38.00 ms**
- `candidate_generation_and_clustering`: **0.35 ms**
- `crop_extraction_448px`: **1.20 ms**
- `crop_inference_448px`: **27.73 ms**
- `coordinate_restoration`: **0.06 ms**
- `duplicate_safe_nms_merge`: **0.42 ms**
- `bytetrack_association`: **0.35 ms**

## 2. Pipeline End-to-End Throughput

| Pipeline Architecture | Mean Latency (ms) | Throughput (FPS) | Peak Memory (MB) |
|---|:---:|:---:|:---:|
| **yolov8n_640_static** | 43.19 | **23.15** | 425 |
| **yolov8n_832_static** | 57.19 | **17.48** | 512 |
| **hold_time_h3_fullframe** | 46.85 | **21.34** | 518 |
| **roi_individual_crops** | 55.40 | **18.05** | 490 |
| **roi_clustered_crops** | 48.95 | **20.43** | 495 |
| **roi_fixed_grid_horizon** | 50.15 | **19.94** | 492 |
| **weatherguard_roi_decoupled** | 49.38 | **20.25** | 502 |

**Verdict:** Clustered ROI runs at 20.43 FPS, operating +16.9% faster than fixed 832px (17.48 FPS).
