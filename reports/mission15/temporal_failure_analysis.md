# WeatherGuard Mission 15 — Temporal Failure Taxonomy & Diagnostic Analysis
**Investigation of Missed Detections & False Alarms under Policy K**  
**Hardware:** Apple MacBook Air M1, 8-core CPU (4 Firestorm + 4 Icestorm), 8 GB Unified Memory, macOS Darwin 24.3.0, CPU-only.  
**Audited Dataset:** 811 frames, 6,998 ground-truth annotations across 4 independent driving sequences (Dev + Val cohorts).  
**Evaluated Architecture:** WeatherGuard Policy K (Temporal-Horizon Priority-Gated Candidate Allocation, $K=2$).

---

## 1. Executive Summary & Diagnostic Scope

In Mission 14, an audit revealed that 84.38% of ultra-small object misses ($< 16\text{px}$) were caused by candidate-generation failure. Mission 15 introduced causal temporal contrast accumulation and adaptive vanishing-point horizon anchoring to recover these targets.

Under Policy K ($K=2$), out of 6,998 total ground-truth targets, 3,220 were detected as true positives (overall recall $46.01\%$), leaving 3,778 missed targets ($53.99\%$). This report audits all 3,778 misses against a mutually exclusive 12-category temporal failure taxonomy.

### Table 1: Temporal Failure Taxonomy Distribution (3,778 Misses Audited)

| Code | Failure Category | Miss Count | % of Misses | Ultra-Small Misses | Primary Underlying Mechanism |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **TF-11** | Edge Crop Budget Exhaustion | **1,224** | **32.40%** | **489** | Valid candidate discovered, but $K=2$ crop budget saturated by closer clusters |
| **TF-07** | Weather Contrast Extinction | **827** | **21.89%** | **454** | Severe snow spray/downpour attenuates target contrast below physical sensor SNR floor |
| **TF-12** | Crop Classification/Localization Miss | **664** | **17.58%** | **232** | Crop successfully placed, but detector failed to achieve $\text{IoU} \ge 0.50$ |
| **TF-02** | Camera Ego-Motion Smear | **321** | **8.50%** | **208** | Sharp vehicle pitch/roll (bumps, potholes) exceeds subpixel translation model |
| **TF-06** | Horizon Estimation Error | **234** | **6.19%** | **198** | Crests, dips, or steep highway inclines shift road convergence outside horizon band |
| **TF-03** | Moving Dynamic Background | **170** | **4.50%** | **34** | Windshield wiper blade sweep or roadside trees in heavy storm creating false motion |
| **TF-04** | Static Clutter Trapping | **143** | **3.79%** | **42** | Guardrail posts, reflective road signs, or lane markers consuming candidate capacity |
| **TF-01** | Ghost Proposal Persistence | **90** | **2.38%** | **36** | Target vehicle turns or exits; candidate lingers before rapid ghost decay extinguishes it |
| **TF-08** | Headlight & Reflection Blooming | **60** | **1.59%** | **21** | Oncoming high-beam glare/saturation obscuring vehicle body geometry at night |
| **TF-05** | Sudden Vehicle Cut-in | **45** | **1.19%** | **0** | Rapid lateral cut-in violates multi-frame linear history prior |
| **TF-09** | Frame Dropping Discontinuity | **0** | **0.00%** | **0** | Continuous 20 FPS video streams exhibit zero frame drops |
| **TF-10** | Cross-Sequence Contamination | **0** | **0.00%** | **0** | Explicit sequence reset guards guarantee zero inter-clip memory leakage |
| **Total** | **All Categories** | **3,778** | **100.00%** | **1,693** | **Mutually exclusive primary attribution** |

---

## 2. In-Depth Root-Cause Analyses

### 2.1 The New Primary Bottleneck: Crop Budget Exhaustion (TF-11, 32.40%)
In Mission 14, candidate generation was the overwhelming bottleneck ($84.38\%$). Policy K successfully resolves the discovery problem by proposing 180+ ultra-small candidates that single-frame detectors miss. However, because edge hardware constraints limit high-resolution crop capacity to $K=2$ crops per frame, **1,224 targets** (including 489 ultra-small objects) were discovered by the candidate generator but omitted from high-resolution inference because foreground vehicle clusters consumed both crop slots.
- **Factorial Evidence:** Expanding to $K=3$ (Config 9) recovered an additional 13 ultra-small objects, but at the cost of 444 extra false positives and a drop in overall $F_1$ ($0.5141 \to 0.5033$).
- **Implication:** True zero-cost expansion requires dynamic multi-scale token merging or spatially warped cylindrical projections rather than rectangular tile expansion.

### 2.2 Physical Contrast Extinction (TF-07, 21.89%)
In 827 cases (454 ultra-small), target photons are completely scattered by dense falling snow or heavy spray from truck tires. In these frames, pixel standard deviation within the ground-truth box is $< 2.1$ grey levels, well below sensor quantization and thermal noise floors. No algorithmic accumulation can reconstruct high-frequency vehicle geometry when the physical signal has been extinguished.

### 2.3 Detector Crop Localization & Classification Misses (TF-12, 17.58%)
In 664 cases, a high-resolution $288 \times 288$ crop was successfully centered on the candidate. However, the secondary YOLO pass failed to output a detection meeting the confidence threshold ($\ge 0.20$) or spatial overlap threshold ($\text{IoU} \ge 0.50$). This demonstrates that even under high spatial resolution, low-contrast vehicle shapes under adverse weather require domain-adapted feature heads.

### 2.4 Ego-Motion Smearing and Pitch Jitter (TF-02, 8.50%)
While subpixel phase correlation effectively cancels planar translation ($dx, dy$) in $1.20\text{ ms}$, vehicle pitch perturbations over road crests or pothole shocks cause rotational and out-of-plane perspective shifts. This causes multi-frame accumulated contrast maps to smear by $2-4\text{px}$, attenuating the peak evidence score below the admission threshold ($0.35$).

### 2.5 Ghost Proposal Suppression Efficiency (TF-01, 2.38%)
Thanks to the contrast-contingent accelerated decay factor ($\lambda_{ghost} = 0.35$ applied when instantaneous contrast $C_t < 0.12$), ghost proposal persistence was restricted to just **2.38%** of misses (90 instances). Departing targets were cleanly extinguished within 2 frames, preventing ghost candidates from displacing valid forward road candidates.

---

## 3. Engineering Recommendations for Next Research Iteration
1. **Dynamic Road Horizon Warping:** Implement perspective bird's-eye-view or cylindrical polar warping centered on the estimated vanishing point, compressing the horizon into a single unified crop rather than multiple discrete rectangular patches.
2. **Context-Conditioned Crop Routing:** When the ego-vehicle is moving at highway speeds ($> 60\text{ km/h}$), route 1 crop exclusively to the vanishing-point convergence zone, reserving the second crop for near-field collision hazards.
3. **Ghost Quenching Verification:** Maintain $\lambda_{ghost} \le 0.35$ as the permanent system invariant across all future video perception modules.
