# WeatherGuard — Mission 20: Phase 1 Dataset Verification Report

**Date:** 2026-11-05  
**Author / Principal Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Corpus Under Audit:** WeatherGuard V2 Road Perception Corpus ($N=1,000$ frames)  
**Machine-Readable Acceptance Report:** [`artifacts/mission20/weatherguard_v2_acceptance_report.json`](file:///Users/sayemuddin/Desktop/WeatherGuard/artifacts/mission20/weatherguard_v2_acceptance_report.json)  
**Acceptance Status:** **ACCEPTED**

---

## 1. Executive Summary

Phase 1 of Mission 20 has performed an exhaustive, byte-level verification of the curated 1,000-frame WeatherGuard V2 road perception corpus and its 13,838 object annotations. Every image file, label file, coordinate tuple, perceptual hash, scale categorization, and manual review record was audited against the published V2 manifest (`weatherguard_v2_manifest.json`).

The verification confirmed zero missing files, zero dimension mismatches, zero normalized coordinate violations, zero invalid class IDs, and zero SHA-256 digest discrepancies.

---

## 2. Verification Checklist & Empirical Results

| Verification Dimension | Expected / Target Standard | Verified Result | Compliance Status |
| :--- | :--- | :--- | :---: |
| **Total Frames** | Exactly 1,000 JPEG images | 1,000 images found & decoded | **PASS** |
| **Total Labels** | Exactly 1,000 YOLO format text files | 1,000 files found & parsed | **PASS** |
| **Image Resolution** | $1280 \times 720$ (dashcam 16:9) | 1,000 / 1,000 match metadata | **PASS** |
| **Coordinate Bounds** | Normalized $[x_c, y_c, w, h] \in [0.0, 1.0]$ | 13,838 / 13,838 valid | **PASS** |
| **Canonical Classes** | 6 classes: `truck`, `person`, `bicycle`, `car`, `motorcycle`, `bus` | Zero invalid class IDs | **PASS** |
| **Per-Image SHA-256** | 1,000 digests matching manifest | 1,000 / 1,000 exact match | **PASS** |
| **Manual Review Queue** | Exactly 352 candidate boxes | 352 boxes audited (all $<12\text{px}$) | **PASS** |
| **Augmentation Lineage**| 300 augmented frames traceable to clean parent | 100% parent reference match | **PASS** |
| **Dataset YAML Config** | `configs/weatherguard_v2.yaml` | Class order 0–5 matches runner | **PASS** |

---

## 3. Class & Scale Distribution (13,838 Annotated Objects)

### 3.1. Canonical Class Breakdown
- **Car (`class_id: 3`):** 11,226 objects (81.12%)
- **Person (`class_id: 1`):** 1,462 objects (10.56%)
- **Truck (`class_id: 0`):** 722 objects (5.22%)
- **Bus (`class_id: 5`):** 269 objects (1.94%)
- **Bicycle (`class_id: 2`):** 108 objects (0.78%)
- **Motorcycle (`class_id: 4`):** 51 objects (0.37%)

### 3.2. Scale Tiers ($\max(w_{\text{px}}, h_{\text{px}})$)
- **Ultra-Small ($< 16\text{ px}$):** 466 objects (3.37%)
- **Small ($16\text{ to } 32\text{ px}$):** 3,311 objects (23.93%)
- **Medium ($32\text{ to } 96\text{ px}$):** 6,639 objects (47.98%)
- **Large ($\ge 96\text{ px}$):** 3,422 objects (24.73%)

---

## 4. Perceptual Hash & Near-Duplicate Clustering

Using 64-bit difference hashing (dHash) with a Hamming distance threshold $\le 4$:
- **Exact dHash Pairs ($d=0$):** 88 pairs (principally clear vs. foggy versions of identical source dashcam scenes).
- **Near-Duplicate Pairs ($1 \le d \le 4$):** 412 pairs across temporally adjacent video sequence frames.
- **Methodological Directive for Phase 2:** Near-duplicate frames and augmented pairs must never cross train, validation, and test partitions. Transitive component clustering (DSU) will enforce strict isolation.

---

## 5. Audit of the 352 Manual Review Queue Entries

All 352 entries in `data/weatherguard_corpus/metadata/manual_review_queue.csv` were inspected:
- **Flagging Reason:** `ultra_small_sub_12px_verification` (100% of entries).
- **Spatial Characteristics:** Distant horizon targets or traffic participants at extreme vanishing points where bounding box dimensions range from $3\text{px}$ to $11.8\text{px}$.
- **Decision:** Retained as canonical ground-truth for evaluating extreme-distance candidate generation (Policy J / Policy K) under strict IoU matching.

---

## 6. Dataset YAML Alignment

The configuration [`configs/weatherguard_v2.yaml`](file:///Users/sayemuddin/Desktop/WeatherGuard/configs/weatherguard_v2.yaml) was created and verified:
```yaml
path: data/weatherguard_corpus
train: images/train
val: images/val
test: images/test

names:
  0: truck
  1: person
  2: bicycle
  3: car
  4: motorcycle
  5: bus
```
The numerical order strictly aligns with the class-ID schema generated during corpus expansion.
