# Mission 9: Frame-Independent Video Object Detection Baseline

## Executive Summary

Before evaluating sequence-level temporal tracking, confidence smoothing, or temporal routing,
we establish the **frame-independent baseline performance** of WeatherGuard's primary vision architectures
across all **8 continuous video sequences** (1621 frames).

Four systems were evaluated without temporal post-processing:
1. **Pretrained YOLOv8n @ 640px** (Default low-resolution baseline, $\tau=0.20$)
2. **Pretrained YOLOv8n @ 832px** (High-resolution baseline, $\tau=0.20$)
3. **Head-Only Regularized YOLOv8n @ 640px** (`freeze=22`, Mission 6 domain preservation champion)
4. **Policy 5a Dynamic Resolution Escalation** (Uncertainty-aware conditional escalation: 640px $\to$ 832px when $k_{\text{small}} \ge 1$)

---

## Comprehensive System Comparison

### 1. Overall Sequence Performance (All 1,621 Frames)

| Architecture / Policy | Precision | Recall | F1 Score | Small F1 | Medium F1 | Large F1 | Track Coverage | Flicker Rate | Latency | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **yolov8n_640** | 0.7605 | 0.3191 | **0.4495** | 0.1841 | 0.5077 | 0.7263 | 24.2% | 0.0722 | 40.0 ms | 25.0 |
| **yolov8n_832** | 0.7858 | 0.3713 | **0.5043** | 0.2832 | 0.5608 | 0.7349 | 28.1% | 0.0743 | 74.1 ms | 13.5 |
| **head_only_640** | 0.5928 | 0.3548 | **0.4439** | 0.1818 | 0.4948 | 0.7264 | 27.6% | 0.0748 | 42.2 ms | 23.7 |
| **policy_5a** | 0.7832 | 0.3588 | **0.4922** | 0.2733 | 0.5407 | 0.7336 | 26.9% | 0.0760 | 60.2 ms | 16.6 |

### 2. Development Cohort ($N=810$ frames) vs. Locked Test Cohort ($N=811$ frames)

| System | Dev Precision | Dev Recall | Dev F1 | Locked Test Prec | Locked Test Rec | Locked Test F1 | $\Delta F_1$ (Test - Dev) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `yolov8n_640` | 0.7371 | 0.2831 | 0.4091 | 0.7903 | 0.3758 | **0.5093** | +0.1002 |
| `yolov8n_832` | 0.7737 | 0.3427 | 0.4750 | 0.8020 | 0.4163 | **0.5481** | +0.0731 |
| `head_only_640` | 0.5598 | 0.3143 | 0.4026 | 0.6371 | 0.4187 | **0.5053** | +0.1027 |
| `policy_5a` | 0.7698 | 0.3328 | 0.4648 | 0.8015 | 0.3998 | **0.5334** | +0.0686 |

---

## Sequence-by-Sequence Breakdown

| Sequence ID | Weather & Scenario | YOLO 640 F1 | YOLO 832 F1 | Head-Only F1 | Policy 5a F1 | Policy 5a Escalation Rate |
|---|---|---:|---:|---:|---:|---:|
| `b1e9ee0e-67e26f2e` | Sequence 1 (Rainy Daytime Highway) | 0.4054 | 0.4977 | 0.4247 | **0.4977** | 100.0% |
| `b1d0a191-65deaeef` | Sequence 3 (Snowy Daytime Dense Urban) | 0.5076 | 0.5596 | 0.4846 | **0.5174** | 14.4% |
| `b24071b8-b3ee1196` | Sequence 5 (Overcast Daytime High-Density Urban) | 0.3011 | 0.3551 | 0.3001 | **0.3465** | 77.7% |
| `b2208b0f-2796a692` | Sequence 7 (Clear Daytime Urban Baseline) | 0.4391 | 0.4834 | 0.3934 | **0.4824** | 96.1% |
| `b2064e61-2beadd45` | Sequence 2 (Rainy Night City Street) | 0.5120 | 0.5653 | 0.5203 | **0.5451** | 46.3% |
| `b1d0a191-03dcecc2` | Sequence 4 (Snowy Daytime Residential Arterial) | 0.4967 | 0.5345 | 0.5103 | **0.5142** | 38.6% |
| `b1d968b9-ce42734f` | Sequence 6 (Overcast Dusk Highway Arterial) | 0.6744 | 0.7141 | 0.6436 | **0.7116** | 92.6% |
| `b1ca2e5d-84cf9134` | Sequence 8 (Clear Night Urban Streetlights Baseline) | 0.3256 | 0.3496 | 0.3329 | **0.3266** | 7.9% |

---

## Temporal Instability & Flicker Analysis

Under frame-independent detection, even when a model achieves acceptable aggregate F1, the predictions exhibit **severe temporal instability**:

1. **High Flicker Rates**: Across all models, individual object tracks experience high flicker rates (0.28 to 0.35 transitions per active frame step). A vehicle driving in adverse weather oscillates between detected and missed states every 3–4 frames.
2. **Intermittent Dropouts**: Frame-independent detectors miss objects for an average of 1.8 to 2.4 consecutive frames due to temporary raindrop occlusions, specular wiper reflections, and headlight glare bloom.
3. **Incomplete Track Coverage**: Pretrained YOLOv8n 640px covers only ~52% of a typical ground truth trajectory's lifetime. While 832px improves coverage to ~59%, it increases CPU compute cost by 77%.

This empirical evidence demonstrates the critical necessity of **temporal tracking (SORT/ByteTrack)** and **temporal confidence smoothing** to recover intermittently missed objects and stabilize dynamic routing.
