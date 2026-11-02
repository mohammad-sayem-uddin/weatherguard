# WeatherGuard Mission 8: Pre-Flight Audit & Baseline Verification

**Document ID:** `WG-M8-AUDIT-001`  
**Date:** November 2026  
**Status:** Validated and Reconciled  
**Frozen Configuration SHA-256 (from Mission 7):** `f4e04789c35144175db6b95d23b9da996ada32cd22a8f718630a5baa9710b7ff`  

---

## 1. Executive Summary

Mission 8 investigates **Multi-Scale Feature Reuse, Lightweight Refinement, and Temporal Stability** for efficient adverse-weather object detection.

Before designing feature-reuse mechanisms or refinement modules, this pre-flight audit:
1. Re-verifies canonical baseline metrics against locked Mission 7 evaluation artifacts.
2. Formally audits the 4-cohort data isolation guarantees between Group B ($N=160$) and Group C ($N=203$).
3. Audits the physical availability of temporal sequential data in the BDD100K evaluation split.
4. Confirms checkpoint availability, weights integrity, and hardware execution readiness.

---

## 2. Canonical Baseline Reconciliation

Every baseline metric reported below is directly validated against `artifacts/mission7_locked_results.json`:

| Detector System | Input Resolution | Micro $F_1$ | Small Recall ($a < 32^2$) | Mean Latency (ms) | Throughput (FPS) | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pretrained YOLOv8n (Base)** | $640\times 640$ | 0.4974 | 8.7% | 61.1 ms | 16.4 FPS | **VERIFIED** |
| **Policy 5a Dynamic ($640 \to 832$)** | $640 \to 832$ | 0.5270 | 13.1% | 81.7 ms | 12.2 FPS | **VERIFIED** |
| **Fixed YOLOv8n 832px** | $832\times 832$ | 0.5442 | 16.3% | 103.0 ms | 9.7 FPS | **VERIFIED** |
| **Fixed YOLOv8n 1280px** | $1280\times 1280$ | 0.5449 | 28.7% | 378.0 ms | 2.6 FPS | **VERIFIED** |

- **Reconciliation Note:** All metrics match reported values within scalar rounding tolerances ($|\Delta F_1| < 0.0001$, $|\Delta\text{lat}| < 0.1\text{ ms}$).
- **Core Motivation for Mission 8:** While Policy 5a dynamic routing achieves $F_1 = 0.5270$ at $12.2\text{ FPS}$ (saving $20.7\%$ latency vs full 832px), each escalated frame re-executes the entire 23-layer network on an $832\times 832$ image. The primary objective is to investigate whether intermediate feature maps from the initial 640px pass can be reused to capture high-resolution benefits with substantially lower marginal computation.

---

## 3. Dataset Governance & Strict Cohort Isolation

- **Group B Development ($N=160$):** Strictly reserved for feature-reuse architecture exploration, refinement module design, distillation loss tuning, and threshold selection.
- **Group C Locked Test ($N=203$):** Permanently sealed. Zero test annotations or test predictions may be used to guide architecture development. Group C will be evaluated strictly once after all configurations are frozen.
- **Cohort Overlap:** $|\text{Group B} \cap \text{Group C}| = 0$ (Zero leakage).

---

## 4. Temporal Data Availability Audit

A thorough inspection of image filenames, split manifests, and parquet metadata in the WeatherGuard repository reveals:
- **Group B:** 160 images contain 148 unique 8-character video prefix IDs ($92.5\%$ unique).
- **Group C:** 203 images contain 197 unique 8-character video prefix IDs ($97.0\%$ unique).
- **Finding:** The BDD100K 100K bounding box detection benchmark consists of **single independent 10-second keyframes** sampled across tens of thousands of separate driving trips. Continuous 30 FPS video sequences are part of the separate BDD100K MOT / Tracking dataset and are not present in this detection benchmark.
- **Protocol Directive:** Per Section 7 and Section 8 of the research protocol, temporal routing and temporal consistency algorithms will be implemented and unit-tested on controlled sequential data, but temporal evaluation on Group B and Group C will be reported honestly as **blocked / limited by data availability**, avoiding any fabricated video sequences.

---

## 5. Checkpoint & Hardware Readiness

| Checkpoint Identifier | File Path | File Size | Readiness |
| :--- | :--- | :---: | :---: |
| **Pretrained YOLOv8n** | `yolov8n.pt` | 6.25 MB | ✅ Ready |
| **Mission 5 Champion** | `models/mission5_b1_photo.pt` | 6.20 MB | ✅ Ready |
| **Mission 6 Head-Only** | `models/mission6_e2_head_only.pt` | 6.20 MB | ✅ Ready |
| **Mission 6 Frozen-Backbone** | `models/mission5_a1_frozen.pt` | 6.20 MB | ✅ Ready |

Execution is configured for Apple MacBook Air M1 (ARM64 CPU, 8 GB RAM) with PyTorch 2.14.1 and Ultralytics 8.4.176.
