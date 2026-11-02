# WeatherGuard Mission 10 — Phase 1: Independent Video Dataset Discovery & Review

**Mission:** Mission 10 — Independent Validation & Temporal Reliability  
**Author:** Principal Computer Vision Researcher, Video Understanding Engineer & Experimental Scientist  
**Date:** 2026-11-02  
**Hardware Environment:** Apple MacBook Air M1, 8 GB Unified Memory, macOS (CPU-only)  
**Deliverable:** `reports/mission10_dataset_review.md`  

---

## 1. Executive Summary

Mission 10 expands WeatherGuard's evaluation framework to rigorously test whether the temporal perception gains observed in Mission 9 generalize beyond the initial 8 sequences. Following the scientific protocol established in Phase 1, we conducted an exhaustive investigation into candidate public adverse-weather driving datasets to identify continuous video sequences with valid object tracking ground truth.

Five prominent candidate datasets were evaluated against strict criteria:
1. **DAWN** (Detection in Adverse Weather Nature)
2. **ACDC** (Adverse Conditions Dataset with Correspondences)
3. **CADC** (Canadian Adverse Driving Conditions Dataset)
4. **KuAvLab/EMT** (Emirates Multi-Task Dataset)
5. **BDD100K MOT** (Berkeley DeepDrive Multiple Object Tracking 2020)

### Primary Decision:
- **DAWN is formally disqualified as a video benchmark:** Our forensic inspection confirmed that DAWN consists strictly of 1,023 static, web-crawled still images with zero temporal continuity, zero frame timestamps, and zero track IDs. Concatenating still images to simulate video violates scientific ethics and is rejected.
- **ACDC is disqualified for dense video tracking:** While recorded from vehicle-mounted video, ACDC is publicly distributed as semantic/panoptic segmentation image pairs and keyframes (32+ GB archive). It lacks continuous dense temporal tracking IDs.
- **BDD100K MOT Virgin Sequences Selected for Primary Independent Validation:** BDD100K MOT 2020 contains 1,600 continuous driving sequences (1,400 train, 200 val) at 5.0 FPS with full 2D bounding box track IDs. Exactly 8 sequences were accessed in Mission 9. All remaining 192 validation sequences and 1,400 training sequences are completely virgin and uninspected.
- We selected 6 completely independent virgin sequences ($N=1,215$ continuous frames, 21,259 ground-truth annotations) with **zero frame or sequence overlap** with any previous WeatherGuard cohort (Group B, Group C, or Mission 9).

---

## 2. Comparative Review of Candidate Datasets

| Evaluation Criterion | DAWN (Rashid et al. 2020) | ACDC (Sakaridis et al. 2021) | CADC (Pitropov et al. 2020) | KuAvLab/EMT (2025) | BDD100K MOT (Yu et al. 2020) |
|---|---|---|---|---|---|
| **Official Source** | HuggingFace / IEEE ICCVW | ETH Zurich / ICCV 2021 | Univ. of Waterloo / IJRR | HuggingFace / arXiv | Berkeley DeepDrive / CVPR |
| **License / Terms** | CC BY-NC 3.0 (Research) | Non-commercial Academic | CC BY-NC-SA 4.0 | CC BY-NC-SA 4.0 | Berkeley DeepDrive License |
| **Continuous Video?** | **NO** (Static web images) | Partial (Keyframe pairs) | **YES** (75 video runs) | **YES** (57 mins video) | **YES** (1,600 video sequences) |
| **Temporal Track IDs?** | **NO** (None) | **NO** (Segmentation masks) | **YES** (3D Cuboid tracks) | **YES** (2D BBox tracks) | **YES** (2D BBox tracks) |
| **Annotation Density** | Isolated images | Keyframes only | Dense LiDAR/Cam sweeps | Dense (10 FPS) | Dense (5.0 FPS, all frames) |
| **Weather Conditions** | Rain, Snow, Fog, Sand | Rain, Snow, Fog, Night | Extreme Snow / Blizzard | Rain, Night reflections | Rain, Snow, Overcast, Night |
| **Automotive Classes** | Car, Bus, Truck, Person, Bike, Moto | 19 Cityscapes classes | 10 3D Classes | 8 Agent categories | 8 Standard driving classes |
| **Resolution & FPS** | Variable static images | 1080p (30 Hz source) | Multi-cam (10 Hz) | 1080p (10 FPS) | 1280x720 (5.0 FPS) |
| **Download Size** | ~100 MB | >32 GB (Tarball) | >50 GB (ROS bags) | 8.3 GB (Monolithic tar) | **Targeted Range: ~150 MB** |
| **M1 CPU Feasibility** | High | Infeasible (Disk & Memory) | Infeasible (Disk & 3D) | Borderline (8.3 GB tar) | **Optimal (Range-Slice HTTP)** |
| **Final Recommendation** | **DISQUALIFIED (Not video)** | **DISQUALIFIED (Keyframes)** | **REJECTED (Resource limit)** | **SECONDARY CANDIDATE** | **SELECTED BENCHMARK** |

---

## 3. Detailed Forensic Findings on Disqualified Datasets

### 3.1 DAWN (Detection in Adverse Weather Nature)
- **Investigation:** A full query of DAWN repository metadata and official documentation confirms that DAWN was constructed by scraping search engines (Google, Bing, Flickr) for adverse-weather keywords.
- **Structural Flaw:** It is an unordered collection of individual image files (`Fog_1.jpg`, `Rain_2.jpg`, etc.). It contains no temporal metadata, camera sensor telemetry, or sequential tracking annotations.
- **Scientific Conclusion:** Treating DAWN as video would require synthetic sequencing or motion interpolation. This introduces artificial temporal artifacts and is categorically disqualified.

### 3.2 ACDC (Adverse Conditions Dataset with Correspondences)
- **Investigation:** ACDC captures vehicle-mounted footage across 4 challenging conditions (Fog, Night, Rain, Snow) paired with clear-day correspondences.
- **Structural Flaw:** Public distribution focuses on semantic and panoptic segmentation benchmarks. Bounding-box annotations are provided in COCO detection format (`instancesonly_*_gt_detection.json`), but are generated per keyframe without multi-frame temporal tracklet links (`track_id`).
- **Resource Flaw:** The official download archive exceeds 32 GB, requiring manual authentication through ETH Zurich's portal. This violates our resource constraints (~17 GB free disk on M1).

### 3.3 KuAvLab/EMT (Emirates Multi-Task Dataset)
- **Investigation:** Hosted on HuggingFace (`KuAvLab/EMT`), this 2025 dataset provides 57 minutes of 10 FPS driving video with explicit `track_id` multi-object tracking annotations under rain, night, and glare.
- **Limitation:** The test images are packaged as a single monolithic 8.31 GB archive (`test_images.tar.gz`), preventing selective sequence slicing without downloading the entire multi-gigabyte tarball. It is cataloged as a strong future benchmark for high-capacity storage environments.

---

## 4. The Selected Virgin BDD100K MOT Video Cohort

BDD100K Multiple Object Tracking 2020 contains 200 validation sequences recorded via vehicle dashboard cameras under diverse environmental conditions at a calibrated 5.0 FPS (1280x720 resolution). Every frame is densely annotated with 2D bounding boxes and persistent tracking identifiers (`id`).

### 4.1 Zero-Leakage Verification
To ensure 100% scientific integrity, candidate sequences were audited against all previous WeatherGuard cohorts:
- **Historical Group B Dev ($N=160$ keyframes):** Checked against all frame filenames. **0 overlap.**
- **Historical Group C Test ($N=203$ keyframes):** Checked against all frame filenames. **0 overlap.**
- **Mission 9 Sequences (8 sequences, 1,621 frames):** Checked against sequence IDs. **0 overlap.**
- **Total available virgin sequences in validation split:** 192 / 192 completely virgin sequences.

### 4.2 Selected 6-Sequence Independent Cohort

We selected 6 diverse, high-density sequences capturing authentic weather corruptions, low contrast, road spray, windshield wiper motion, and night illumination falloff:

| Sequence ID | Weather Condition | Lighting / Time | Scene Type | Total Frames | Ground Truth Boxes | Dominant Visual Challenge |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `b24380ab-6dbeb908` | **Rainy** | Night | Highway | 202 | 2,786 | Heavy road spray, headlight glare, wet asphalt reflections |
| `b255cd6c-2f889586` | **Snowy** | Daytime | City Street | 203 | 3,625 | Active snowfall, roadside snow accumulation, pedestrian slush |
| `b1e1a7b8-65ec7612` | **Rainy** | Daytime | Highway | 202 | 2,485 | Windshield wiper blade occlusion, water droplet blur |
| `b1d0a191-06deb55d` | **Snowy** | Daytime | Dense Urban | 202 | 3,610 | Low contrast overcast overcast glare, snowbank vehicle occlusions |
| `b24d283f-33783d1b` | **Overcast** | Daytime | City Street | 202 | 4,323 | Extreme traffic density (4,323 objects), small vehicle clutter |
| `b23c9e00-b425de1b` | **Clear** | Night | Highway | 202 | 4,430 | High dynamic range, artificial illumination falloff baseline |
| **TOTALS** | **4 Conditions** | **3 Time Regimes** | **2 Scene Types** | **1,215** | **21,259** | **Complete authentic adverse driving spectrum** |

---

## 5. Sequence-Level Partitioning Strategy (Phase 2 Preview)

To prevent hyperparameter over-fitting, the 6 sequences are strictly divided at sequence boundaries:
- **Independent Development Cohort ($N=2$ sequences, 405 frames, 6,411 objects):**
  - `b24380ab-6dbeb908` (Rainy Night Highway)
  - `b255cd6c-2f889586` (Snowy Daytime City)
  - Used for parameter validation, failure taxonomy, and causal ablation analysis.
- **Independent Locked Test Cohort ($N=4$ sequences, 810 frames, 14,848 objects):**
  - `b1e1a7b8-65ec7612` (Rainy Daytime Highway)
  - `b1d0a191-06deb55d` (Snowy Daytime Urban)
  - `b24d283f-33783d1b` (Overcast City Street)
  - `b23c9e00-b425de1b` (Clear Night Highway)
  - Strictly locked and evaluated in a single blind pass.
