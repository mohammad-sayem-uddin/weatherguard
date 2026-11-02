# WeatherGuard — Mission 12 Candidate-Generation Oracle Analysis
## Theoretical Upper-Bound Evaluation of First-Stage Discovery vs. Second-Stage Resolution

**Date:** 2026-11-02  
**Dataset:** BDD100K MOT Sequences (Development $N = 405$ frames, Locked Test $N = 808$ frames)  
**Total Evaluated Ground-Truth Targets:** 21,207 bounding boxes  
**Artifact:** `artifacts/mission12_candidate_oracle.json`

---

### 1. Executive Summary: The First-Stage Discovery Bottleneck

Mission 11 proved that region-adaptive ROI magnification ($448\times 448\text{px}$) outperforms full-frame 832px inference in small-object detection ($0.2765$ vs $0.2339$). However, Mission 11 relied exclusively on the base 640px detector to propose candidate regions.

This Oracle Analysis investigated the theoretical performance ceiling:
**If ROI crops were placed perfectly around ground-truth targets, how many missed objects could be recovered?**

```
+---------------------------------------------------------------------------------------------------+
| FIRST-STAGE CANDIDATE DISCOVERY VS. ORACLE RECOVERABILITY (LOCKED TEST: N = 14,796 GT OBJECTS)     |
+-------------------+-----------+-------------+------------------+-------------+--------------------+
| Object Scale Tier | Total GT  | YOLO640 Hit | YOLO640 Uncertain| YOLO640 Miss| Oracle Recoverable |
|                   | Instances | (conf >=0.45)| (0.15<=s<0.45)   | (s < 0.15)  | (% of Missed GT)   |
+-------------------+-----------+-------------+------------------+-------------+--------------------+
| Ultra-Small (<16px)| 1,282     | 0.0% (0)    | 0.0% (0)         | 100.0%(1282)| 92.7% (1,188)      |
| Small (16 to 32px)| 5,006     | 2.7% (135)  | 10.7% (538)      | 86.6% (4333)| 99.9% (4,329)      |
| Medium (32 to 96px| 6,140     | 21.5%(1320) | 17.1% (1053)     | 61.4% (3767)| 100.0% (3,767)     |
| Large (> 96px)    | 2,368     | 70.1%(1660) | 10.7% (252)      | 19.3% (456) | 100.0% (456)       |
+-------------------+-----------+-------------+------------------+-------------+--------------------+
| OVERALL           | 14,796    | 21.0%(3115) | 12.5% (1843)     | 66.5% (9838)| 98.9% (9,740)      |
+-------------------+-----------+-------------+------------------+-------------+--------------------+
```

---

### 2. Forensic Breakdown by Scale Tier

#### Tier 1: Ultra-Small Objects ($<16\text{px}$, Area $<256\text{px}^2$)
- **Empirical Reality:** The standard YOLOv8n detector at 640px produces **exactly 0 activations ($s \ge 0.15$)** across all 1,282 ultra-small targets in the locked test set (100% miss rate).
- **Physical Reason:** An 8–14px vehicle silhouette downsampled to $640\times 360$ occupies merely $4\times 7$ pixels. In the P3 feature stride ($8\times$ downsampling), this target projects to less than a single feature cell ($0.5\times 0.9$ grid units), completely vanishing before anchor-free regression heads can operate.
- **Oracle Recoverability:** When an oracle crop ($448\times 448\text{px}$) is centered on these targets, **$92.7\%$ ($1,188$ targets)** have sufficient raw pixel resolution ($\ge 8\text{px}$) to be resolved by the high-resolution forward pass.
- **Unrecoverable Remainder ($7.3\%$):** Only $94$ objects are sub-6px or severely degraded by wiper streaks and heavy spray where even a $2.5\times$ crop cannot recover the structural silhouette.

#### Tier 2: Small Objects ($16\text{px} \le \text{Dimension} < 32\text{px}$)
- In the base 640px forward pass, only $2.7\%$ are confident hits and $10.7\%$ are uncertain candidates ($0.15 \le s < 0.45$).
- **$86.6\%$ ($4,333$ targets) are completely missed.**
- Under oracle placement, **$99.9\%$ ($4,329$ targets)** are fully recoverable.

---

### 3. Core Scientific Finding: The Proposal Gap

The theoretical ceiling of candidate generation is extraordinarily high:
1. **The detector itself is not the primary resolution limitation.** A $448\times 448$ crop forward pass has sufficient capacity to resolve $92.7\%$ of ultra-small vehicles if given the correct spatial crop.
2. **The failure is exclusively spatial proposal generation.** Because low-resolution inference yields zero activations for distant vanishing-point targets, detector-dependent ROI inference is blind to their existence.
3. **Requirement for Mission 12:** To break through the performance ceiling, WeatherGuard must incorporate **detector-independent candidate proposal mechanisms** (frame differencing, sparse optical flow, static visual salience, and perspective horizon priors) that propose crops even when the low-resolution detector is completely silent.
