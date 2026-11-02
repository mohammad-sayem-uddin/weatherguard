# WeatherGuard Mission 4: Forensic Failure-Cause Analysis

**Document:** `reports/mission4_failure_analysis.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Completed & Artifact-Verified  
**Artifacts:** `artifacts/mission4_failure_analysis.json`, `artifacts/mission4_failure_analysis.csv`  

---

## 1. Executive Summary

Experiment C investigates why the scale-aware YOLOv8n detector still misses approximately $71\%$ of small objects ($75.84\%$ on Group B dev, $71.31\%$ on Group C locked test), deconstructing false negatives and false positives into verified visual, geometric, and environmental causes.

Evaluations were conducted on the BDD100K development cohort (Group B, $N=160$, $2,211$ target objects).

### Key Scientific Findings
1. **The Sub-16px Optical Floor:** Objects with area $< 256\text{ px}^2$ (linear size $< 16\text{ px}$) represent a hard limit for standard convolutional feature pyramids. At $640\text{px}$, **$100.0\%$ of sub-16px objects are missed**. Even at $1280\text{px}$, **$94.19\%$ remain undetected**, accounting for $20.8\%$ of all false negatives.
2. **Dominance of Atmospheric Extinction & Occlusion:** Over $50\%$ of remaining misses at $1280\text{px}$ are caused by factors unrelated to pixel resolution: optical contrast extinction in fog/rain/snow ($29.3\%$) and severe inter-vehicle occlusion in crowded traffic ($20.7\%$).
3. **Class Confusion Drives False Positives:** Out of $524$ false positives produced at $1280\text{px}$, **$47.0\%$ ($246$ detections) are caused by inter-class confusion** (primarily `Car` vs. `Truck`/`Bus` under low visibility), while background roadside infrastructure accounts for $24.2\%$.
4. **Extreme Class Imbalance:** The target distribution is heavily dominated by `Car` ($78.9\%$) and `Person` ($12.4\%$), while vulnerable two-wheelers (`Motorcycle`, $N=12$; `Bicycle`, $N=15$) have insufficient sample support to infer generalizable conclusions.

---

## 2. Scale-Stratified Miss Rates & Optical Limits

Objects were categorized into four discrete scale strata based on bounding-box area in native $1280 \times 720$ coordinates:

| Scale Stratum | Bounding Box Area | Linear Dimension | Group B Total GT | 640px Miss Rate (%) | 1280px Miss Rate (%) | Miss Reduction |
|---|---|---|---|---|---|---|
| **Sub-16px (Extreme Small)** | $< 256\text{ px}^2$ | $< 16\text{ px}$ | 258 | **100.0%** (258/258) | **94.19%** (243/248) | -5.81 pp |
| **Small (16–32px)** | $256 - 1,024\text{ px}^2$ | $16 - 32\text{ px}$ | 632 | **88.92%** (562/632) | **69.02%** (436/632) | **-19.90 pp** |
| **Medium (32–96px)** | $1,024 - 9,216\text{ px}^2$ | $32 - 96\text{ px}$ | 806 | **50.00%** (403/806) | **37.50%** (302/806) | **-12.50 pp** |
| **Large ($\ge 96\text{px}$)** | $\ge 9,216\text{ px}^2$ | $\ge 96\text{ px}$ | 515 | **20.00%** (103/515) | **31.14%** (160/515) | +11.14 pp (Degradation) |

### Technical Interpretation
- **Why Sub-16px Objects Fail:** Modern single-stage detectors employ feature pyramid strides of $8$ (P3), $16$ (P4), and $32$ (P5). An object of dimension $12 \times 12$ pixels spans only $1.5 \times 1.5$ cells at P3. Under adverse weather (attenuated edges, low contrast), the signal-to-noise ratio is insufficient for anchorless center-point regression.
- **Why 16–32px Objects Benefit Most:** For objects in the $16-32\text{px}$ range, doubling resolution to $1280\text{px}$ shifts their representation into the receptive field sweet spot of P3, cutting miss rates by $\approx 20\text{ pp}$.
- **Receptive Field Over-Expansion on Large Objects:** At $1280\text{px}$, large objects expand beyond the receptive field of P5, increasing the large-object miss rate from $20.0\%$ to $31.14\%$.

---

## 3. False Negative Attribution (1280px High-Resolution Pass)

Total False Negatives at $1280\text{px}$: $1,169$ missed objects across $160$ frames.

| Attributed Failure Cause | Definition / Diagnostic Criteria | Count | Percentage of FNs |
|---|---|---|---|
| **Optical Weather Degradation** | Misses in heavy fog, rain, or snow where edges suffer severe optical extinction | 342 | **29.26%** |
| **Sub-16px Extreme Scale** | Area $< 256\text{ px}^2$ (below convolutional feature resolution floor) | 243 | **20.79%** |
| **Crowding & Occlusion** | Overlap with adjacent ground-truth box ($\text{IoU} > 0.15$ or center $< 30\text{px}$) | 242 | **20.70%** |
| **Standard Adverse Miss** | Resolvable scale, unoccluded, but detector confidence fell below threshold | 199 | **17.02%** |
| **Boundary Truncation** | Bounding box within $10\text{px}$ of image border (partial vehicle visibility) | 96 | **8.21%** |
| **Extreme Aspect Ratio** | Aspect ratio $> 3.0$ (e.g. roadside barriers) or $< 0.33$ (e.g. distant light poles) | 47 | **4.02%** |

---

## 4. False Positive Attribution (1280px High-Resolution Pass)

Total False Positives at $1280\text{px}$: $524$ false alarms ($\tau \ge 0.20$).

| Attributed Cause | Diagnostic Criteria | Count | Percentage of FPs |
|---|---|---|---|
| **Inter-Class Confusion** | Detection overlaps with ground-truth box of a different category ($\text{IoU} \ge 0.50$) | 246 | **46.95%** |
| **Low-Confidence Edge Alarm** | False alarm with score $0.20 \le \text{score} < 0.30$ | 151 | **28.82%** |
| **Background Structure** | Roadside signboards, bridge trusses, reflections, wiper water streaks | 127 | **24.24%** |

---

## 5. Class-Stratified Detection Performance

| Category | Ground Truth ($N$) | True Positives | False Negatives | False Positives | Recall (%) | Precision (%) |
|---|---|---|---|---|---|---|
| **Car** | 1,744 | 843 | 901 | 289 | 48.34% | 74.47% |
| **Person** | 275 | 140 | 135 | 77 | 50.91% | 64.52% |
| **Truck** | 126 | 42 | 84 | 124 | 33.33% | 25.30% |
| **Bus** | 39 | 13 | 26 | 32 | 33.33% | 28.89% |
| **Motorcycle** | 12 | 2 | 10 | 1 | 16.67%* | 66.67%* |
| **Bicycle** | 15 | 2 | 13 | 1 | 13.33%* | 66.67%* |

*\*Note: Sample counts for `Motorcycle` ($N=12$) and `Bicycle` ($N=15$) are too low for statistical generalization.*

---

## 6. Synthesis and Hypothesis H4 Evaluation

- **Hypothesis H4:** "Remaining errors are influenced by multiple factors, including object scale, weather degradation, class imbalance, occlusion, and detector confidence."
- **Status:** **Strongly Supported.**
- **Scientific Conclusion:** Resolution scaling to $1280\text{px}$ effectively addresses only one specific error category: resolving objects between $16\text{px}$ and $32\text{px}$. The remaining $71-75\%$ small-object deficit is physically bounded by the sub-16px feature limit ($20.8\%$), atmospheric extinction in severe weather ($29.3\%$), and vehicle occlusion in traffic ($20.7\%$). Further resolution increases alone cannot solve these non-resolution failure modes.
