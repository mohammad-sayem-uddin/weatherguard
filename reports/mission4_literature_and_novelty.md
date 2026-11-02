# WeatherGuard Mission 4: Literature Review & Novelty Assessment

**Document:** `reports/mission4_literature_and_novelty.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Completed & Artifact-Verified  

---

## 1. Executive Summary

This report establishes the bibliographic foundation, related works, and technical boundaries of WeatherGuard Mission 4. It places our empirical findings on adaptive resolution routing, crop refinement, and adverse-weather domain shift into context with modern computer vision literature.

We rigorously distinguish between established paradigms in the literature and WeatherGuard's specific contributions, avoiding exaggerated claims of novelty.

---

## 2. Review of Related Work

### 2.1 Multi-Scale Object Detection & Scale-Specific Invariance
- **Feature Pyramid Networks (FPN):** Lin et al. (*Feature Pyramid Networks for Object Detection*, CVPR 2017) introduced top-down pathways and lateral connections to build high-level semantic feature maps at all scales. While FPN improves multi-scale detection, the shallowest feature map (stride 8, P3) remains inadequate for sub-16px objects in $640\text{px}$ inputs.
- **SNIP & SNIPER:** Singh and Davis (*An Analysis of Scale Invariance in Object Detection - SNIP*, CVPR 2018; *SNIPER: Efficient Multi-Scale Training*, NeurIPS 2018) demonstrated that training and testing detectors on extreme scale variations causes gradient and feature degradation. They showed that selectively processing scale-normalized image chips or resolutions mitigates receptive-field mismatch. WeatherGuard's finding that uniform $1280\text{px}$ inference degrades large-object recall by $8.8-11.1\text{ pp}$ directly corroborates the core insight of SNIP.

### 2.2 Sliced Inference & Region-Based Upscaling (SAHI)
- **Slicing Aided Hyper Inference (SAHI):** Akyon, Altinuc, and Temizel (*Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection*, IEEE ICIP 2022) proposed uniform window slicing for aerial and drone imagery, extracting regular overlapping tiles to detect small objects with off-the-shelf detectors.
- **Why Targeted Crop Refinement Diverges from SAHI:** While SAHI uniformly tiles the entire image grid (which Mission 3 evaluated and found increased CPU latency to $628\text{ ms}$), WeatherGuard Experiment B attempted heuristic *targeted* crop proposal based on initial 640px detections. Our empirical finding that targeted crop refinement collapses due to (1) the proposal paradox (undetected small objects are never cropped) and (2) false positive explosion from context loss provides a critical negative result delineating when localized cropping fails in forward-facing driving cameras.

### 2.3 Dynamic Neural Networks & Adaptive Inference
- **Dynamic Neural Networks Survey:** Han et al. (*Dynamic Neural Networks: A Survey*, IEEE TPAMI 2021) cataloged input-dependent dynamic architectures, including dynamic routing, early exiting, and adaptive resolution networks.
- **Resolution Routing:** Yang et al. (*Resolution Adaptive Networks for Efficient Inference*, CVPR 2020) and Wang et al. (*Adaptive Image Transformation for Object Detection*, ICCV 2021) demonstrated that easy samples can be classified at lower resolutions while difficult samples escalate to higher resolutions. WeatherGuard adopts this principle for edge object detection, demonstrating that a **deterministic, training-free small-box trigger** derived from base predictions achieves $80.6\%$ of full-resolution recall gains while saving $29.7-44.8\%$ of CPU latency without requiring policy network retraining.

### 2.4 Adverse Weather & Cross-Dataset Domain Shift
- **Adverse Weather Benchmarks:** Kenk and Hassaballah (*DAWN: Vehicle Detection in Adverse Weather Nature*, IEEE ITSC 2020) and Yu et al. (*BDD100K: A Diverse Driving Video Database*, CVPR 2020) created public benchmarks for evaluating autonomous driving models under adverse environmental conditions.
- **Domain Adaptation Limitations:** Michaelis et al. (*Benchmarking Robustness in Object Detection*, NeurIPS Workshop 2019) demonstrated that state-of-the-art detectors suffer severe out-of-distribution performance drops under corruption and weather shifts. WeatherGuard's findings extend this by quantifying how atmospheric extinction (fog/rain opacity) sets an upper physical bound on detection recall that cannot be resolved through spatial magnification alone.

---

## 3. Honest Classification of Technical Contribution

To maintain scientific integrity, WeatherGuard's contributions are explicitly classified as:

| Contribution Dimension | Claimed Level | Justification |
|---|---|---|
| **Novel Architectural Design** | *Engineering Integration (No Claim of Fundamental Novelty)* | The combination of multi-scale YOLO, heuristic routing triggers, and NMS merging utilizes established primitives. We do not claim invention of a new network architecture. |
| **Empirical Scientific Contribution** | **High / Defensible Finding** | Systematic quantification of: (1) the sub-16px optical floor, (2) the failure mechanics of targeted crop refinement in road scenes, and (3) the exact Pareto frontier trade-offs on edge CPU hardware under adverse-weather domain shift. |
| **Negative Result Contribution** | **High / Defensible Finding** | Thorough falsification of Hypothesis H3: proving that localized crop refinement causes false-positive explosion (+190%) and cannot overcome the initial miss rate of single-stage detectors. |
| **Reproducible Benchmark Study** | **High / Artifact-Verified** | Open, reproducible three-cohort evaluation (Group A, Group B, Group C) with 100% locked test verification and dual-seed bootstrap statistical validation. |

---

## 4. Key Distinctions Against Prior Work

1. **Edge CPU Reality:** Much prior literature benchmarks on high-end NVIDIA GPUs (A100, V100, RTX 3090) where FLOPs and batch parallelism dominate. WeatherGuard measures wall-clock latency on an edge Apple M1 CPU (8 GB RAM), where sequential forward passes impose strict penalties, demonstrating that naive multi-stage pipelines can easily run slower than uniform execution unless escalations are strictly calibrated.
2. **True Out-of-Domain Adverse Weather:** While many multi-scale studies evaluate on standard COCO val, WeatherGuard evaluates cross-dataset transfer from DAWN to locked BDD100K rainy, snowy, overcast, and foggy driving scenes, capturing genuine real-world optical extinction and windshield distortion.
