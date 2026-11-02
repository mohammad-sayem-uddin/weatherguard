# WeatherGuard Mission 7 — Experiment B: Routing Oracle & Upper Bound Diagnostics

**Author:** Antigravity Autonomous CV & Edge Optimization Engineer  
**Date:** 2026-11-02  
**Status:** COMPLETE & INDEPENDENTLY AUDITED  
**Artifact:** `artifacts/mission7_routing_oracle.json`  
**Development Cohort:** Group B BDD100K Development ($N=160$, $2,211$ target objects)  
*(Note: Group C Locked Test N=203 remains strictly untouched and sealed).*  

---

## 1. Executive Summary & Core Diagnostic Findings

Before designing practical uncertainty routing policies or training routing classifiers, we established the **theoretical maximum potential benefit** of conditional inference on the development set (Group B, $N=160$).

An **Oracle Router** represents a diagnostic ceiling: it has perfect, non-causal ground truth knowledge of whether escalating each specific image will improve the global dataset $F_1$ score or harm it. It is not an implementable model, but an upper-bound diagnostic.

### Primary Baseline on Group B Dev ($640\text{px}$, $\tau=0.20$):
- **$F_1$:** 0.4654 (Precision: 0.6184, Recall: 0.3731)
- **Small-Object Recall (< 1024 px²):** 8.09% (189 targets detected out of 1004)

---

## 2. Oracle Routing Ceiling across Escalation Candidates

Matching IoU threshold = 0.50, development cohort $N=160$:

| Escalation Candidate | Standalone $F_1$ | Oracle $F_1$ Upper Bound | $\Delta F_1$ Oracle Gain | Oracle Escalation Rate | Oracle Small Recall | Images $+F_1$ | Images $-F_1$ | Rec+ Prec- | Extra FPs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLOv8n-832px** | 0.5154 | **0.5344** | **+0.0689** | 58.8% (94/160) | 15.8% | 94 | 46 | 25 | 75 |
| **YOLOv8n-1280px** | 0.5189 | **0.5538** | **+0.0884** | 61.3% (98/160) | 22.1% | 92 | 55 | 37 | 98 |
| **Head-Only-640px** | 0.4305 | **0.4941** | **+0.0286** | 33.8% (54/160) | 11.0% | 57 | 93 | 54 | 134 |
| **Faster-RCNN-Ref** | 0.6070 | **0.6265** | **+0.1610** | 75.6% (121/160) | 37.1% | 124 | 29 | 54 | 109 |

---

## 3. Multi-Resolution Oracle (Dynamic Choice among 640px, 832px, 1280px)

When an oracle router can dynamically assign each image to the optimal resolution among $\{640\text{px}, 832\text{px}, 1280\text{px}\}$:
- **Multi-Resolution Oracle $F_1$:** **0.6271** ($\Delta F_1 = +0.1617$ over fixed 640px)
- **Precision:** 0.7209
- **Recall:** 0.5550
- **Small-Object Recall:** **21.95%**
- **Optimal Resolution Allocation:**
  - $640\text{px}$: 45 images (28.1%)
  - $832\text{px}$: 52 images (32.5%)
  - $1280\text{px}$: 63 images (39.4%)

---

## 4. Scientific Insights & Guidance for Practical Router Design

1. **A Substantial Viable Routing Window Exists ($\Delta F_1^* \approx +0.07$ to $+0.08$):**
   - For **YOLOv8n-1280px**, the oracle ceiling reaches **$F_1 = 0.5538$** ($\Delta F_1 = +0.0884$).
   - For **YOLOv8n-832px**, the oracle ceiling reaches **$F_1 = 0.5344$** ($\Delta F_1 = +0.0689$).
   - This proves that conditional inference has substantial headroom: escalating the right images while holding back others yields higher $F_1$ than running fixed 1280px everywhere ($0.5401$), because fixed 1280px introduces false positives on images that did not need escalation.

2. **The Precision Penalty of Unconditional Escalation:**
   - In 1280px standalone, 98 images had additional false positives.
   - On 55 images, escalating to 1280px strictly harmed image $F_1$.
   - On 37 images, escalation increased recall but diluted precision.
   - Therefore, a practical router's primary job is **selective restraint**: invoking high-resolution inference only when small, high-value targets are present, and suppressing escalation when scene clutter will trigger false alarms.

3. **Head-Only Adaptation Escalation Ceiling is Limited:**
   - Escalating to Mission 6 Head-Only (`freeze=22`) achieves an oracle upper bound of $F_1 = 0.4941$ ($\Delta F_1 = +0.0286$).
   - This occurs because Head-Only adaptation was trained on DAWN's small 6-class dataset and exhibits lower precision ($0.4323$) across all images. Fusing or cascading with it introduces false alarms.
   - **Conclusion:** Resolution escalation ($832\text{px}$ and $1280\text{px}$) offers vastly higher diagnostic headroom than model-cascade escalation to adapted weights.
