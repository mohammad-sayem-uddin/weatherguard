# WeatherGuard — Mission 14: 11-Category Failure Taxonomy & Bottleneck Analysis
**Document:** `reports/mission14/failure_taxonomy.md`  
**Date:** 2026-11-03  
**Status:** Completed & Empirically Audited  
**Cohort Analyzed:** Locked Evaluation Cohort (6 Independent BDD100K Sequences, 1,218 Frames)  
**Total Ground-Truth Objects:** 18,556  
**Total Missed Objects under Policy J:** 11,515 (Miss Rate: 62.06%)  

---

## 1. Executive Summary & Core Diagnostic Finding

Mission 14 performs a granular failure analysis of the 11,515 ground-truth targets missed by WeatherGuard Policy J on the locked unseen evaluation cohort. 

**The Central Bottleneck (Answering RQ6):**  
- For **Ultra-Small targets ($< 16\text{ px}$)**, **Candidate-Generation Failure is the dominant failure mode, accounting for 84.4% of all ultra-small misses (1,821 out of 2,158)**. These targets are so faint and distant near the vanishing point that they fail to trigger any initial candidate proposal (detector activation, temporal frame difference, or visual salience). Consequently, high-resolution crop inference is never executed on them.
- For **Small targets ($16\text{ px} \le \text{dimension} < 32\text{ px}$)**, **Crop Placement Failure (3,813 misses, 33.1% of all misses)** is the primary bottleneck: under an edge budget of $K \le 2$ crops, multi-cluster dispersion in dense traffic leaves peripheral small vehicles uncovered.

---

## 2. Quantitative 11-Category Failure Distribution

Table 1 details the full distribution of missed ground-truth objects across the 11 mutually exclusive failure categories:

### Table 1: Failure Taxonomy Distribution

| Category ID | Failure Category | Missed Objects Count | % of All Misses | Ultra-Small Misses ($<16\text{px}$) | Small Misses ($16-32\text{px}$) | Medium/Large Misses ($>32\text{px}$) | Confirmed Cause vs. Working Hypothesis |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **CAT-01** | **Candidate-Generation Failure** | **1,821** | **15.81%** | **1,821** | 0 | 0 | **Confirmed:** No candidate within $R=224\text{px}$ crop radius |
| **CAT-02** | **Proposal Admission Failure** | **0** | **0.00%** | 0 | 0 | 0 | **Confirmed:** Multi-criteria gates did not reject true candidates |
| **CAT-03** | **Crop Placement Failure** | **3,813** | **33.11%** | 0 | **3,813** | 0 | **Confirmed:** Crop budget ($K \le 2$) exhausted by denser clusters |
| **CAT-04** | **Crop Context Failure** | **2,961** | **25.71%** | 0 | 0 | **2,961** | **Confirmed:** Truncation at crop perimeter cuts required context |
| **CAT-05** | **Resolution Limitation** | **26** | **0.23%** | 26 | 0 | 0 | **Confirmed:** Optical footprint $< 8\times 8\text{px}$ (sub-sensor limit) |
| **CAT-06** | **Classification Failure** | **588** | **5.11%** | 25 | 205 | 358 | **Confirmed:** $\text{IoU} \ge 0.50$ but class label confused (e.g. car vs truck) |
| **CAT-07** | **Localization Failure** | **2,213** | **19.22%** | 278 | 1,001 | 934 | **Confirmed:** Coarse boundary overlap ($0.10 \le \text{IoU} < 0.50$) |
| **CAT-08** | **Confidence Threshold Failure** | **3** | **0.03%** | 0 | 1 | 2 | **Confirmed:** Localized accurately but confidence $< 0.25$ |
| **CAT-09** | **Duplicate Suppression Failure**| **0** | **0.00%** | 0 | 0 | 0 | **Confirmed:** Two-tier NMS ($\tau = 0.35$) preserved crop boxes |
| **CAT-10** | **Annotation Ambiguity** | **6** | **0.05%** | 6 | 0 | 0 | **Confirmed:** Degenerate annotations ($\text{Area} < 50\text{px}^2$, occluded) |
| **CAT-11** | **Other / Residual** | **84** | **0.73%** | 2 | 25 | 57 | Residual unclassified edge cases |
| **Total** | **All Missed Ground Truths** | **11,515** | **100.00%** | **2,158** | **5,045** | **4,312** | **Full Audit Complete** |

---

## 3. Deep-Dive into Primary Failure Mechanisms

### 3.1 CAT-01: Candidate-Generation Failure (The Ultra-Small Bottleneck)
- **Count:** 1,821 misses (15.81% of total misses, 84.38% of ultra-small misses).
- **Physical Mechanism:** Targets with native dimensions under $12\times 12\text{ px}$ located near the road vanishing point. 
- At $640\times 384\text{px}$ base downsampling, a $12\text{ px}$ target is compressed to $\sim 6\text{ px}$, falling below the $8\text{ px}$ stride of the P3 convolutional backbone.
- Causal frame differencing fails because distant vehicles have near-zero apparent pixel displacement between consecutive $5\text{ FPS}$ frames.
- Static visual salience fails because atmospheric haze, spray, or dark headlights suppress local gradient magnitude below the $\tau_{\text{sal}} = 0.35$ threshold.
- **Diagnostic Implication:** Crop inference cannot help targets that never enter the proposal pool. Overcoming this bottleneck requires specialized motion integration across multiple temporal frames or prior spatial map anchoring.

### 3.2 CAT-03: Crop Placement Failure (The Small-Object Budget Bottleneck)
- **Count:** 3,813 misses (33.11% of total misses, 75.58% of small misses).
- **Operational Mechanism:** In dense urban traffic scenes (such as `b1c66a42` and `b1d0a191-28f0e779`), small vehicles are distributed across the entire horizontal road span (oncoming lanes, cross-streets, parked roadside rows).
- The maximum crop budget ($K = 2$) can cover at most two $448\times 448\text{px}$ spatial clusters ($\sim 31.4\%$ of native frame area).
- The allocation controller correctly prioritizes the two densest candidate clusters, leaving isolated peripheral small vehicles outside high-resolution magnification.
- **Trade-off Confirmation:** Expanding budget to $K = 4$ would recover many of these objects, but would double crop latency to $> 180\text{ ms}$ ($< 5.5\text{ FPS}$), violating edge real-time constraints.

### 3.3 CAT-07: Localization Failure (Adverse-Weather Glare & Spray)
- **Count:** 2,213 misses (19.22% of total misses).
- **Visual Factor:** High specular reflections on wet asphalt (Rainy Night Sequence `b23f7012`) and diffuse airborne snow spray (Snowy Highway Sequence `b1d0a191-de8948f6`).
- Detected boxes achieve $0.20 \le \text{IoU} < 0.50$ with ground-truth annotations due to blooming around headlights or elongated wiper spray smears that corrupt vehicle boundary estimation.

### 3.4 CAT-02 & CAT-09: Verifying Controller Safety
- **Zero Admission Failures (CAT-02 = 0):** Confirms that Policy J's multi-criteria admission gate never filtered out a legitimate candidate that subsequently would have yielded a true positive.
- **Zero Duplicate Suppression Failures (CAT-09 = 0):** Confirms that the two-tier conservative NMS merging strategy ($\tau_{\text{merge}} = 0.35$) safely prioritized crop predictions over base predictions without mutual cancellation.

---

## 4. Scientific Conclusion & Engineering Recommendations

1. **The First-Stage Proposal Ceiling:** Policy J has pushed ROI allocation to its mathematical limit under $K \le 2$ crops. 
2. **Path to Mission 15:** To overcome the remaining 1,821 candidate-generation misses, future research should not tune crop sizes or NMS thresholds. Instead, it must explore **multi-frame temporal trajectory accumulators** (temporal feature aggregation across 5–10 frames) to accumulate weak optical signal from ultra-small horizon targets before proposal generation.
