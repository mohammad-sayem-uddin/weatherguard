# WeatherGuard — Mission 14: Unseen Sequence Selection & Cohort Partitioning Report
**Document:** `reports/mission14/sequence_selection_report.md`  
**Date:** 2026-11-03  
**Status:** Completed & Manifests Created  
**Principal Investigator:** Principal Computer Vision Researcher & Edge Inference Architect  

---

## 1. Objective and Isolation Principles

A critical limitation identified in prior research (Missions 9–13) was the reliance on a restricted set of 14 BDD100K MOT driving sequences (8 sequences in Mission 9, and 6 sequences across Missions 10–13). Repeated evaluation on the same video sequences risks sequence overfitting, hyperparameter memorization, and optimistic performance estimates due to temporal correlation.

Mission 14 establishes a **strictly virgin, unseen sequence benchmark** satisfying the following criteria:
1. **Zero Sequence Overlap:** Every sequence evaluated in Missions 9, 10, 11, 12, and 13 is explicitly excluded via an audited machine-readable manifest (`artifacts/mission14/sequence_exclusion_manifest.json`).
2. **Disjoint Cohort Partitioning:** Sequences are split strictly into a non-overlapping Development cohort (for implementation verification and non-test sanity checks) and a Locked Evaluation cohort (frozen before model evaluation).
3. **Environmental and Weather Stratification:** Sequences encompass diverse adverse weather conditions (rain, snow, overcast), challenging lighting (night with high glare/reflections, dawn/dusk, daytime), and diverse driving contexts (high-speed highways, congested urban arterials, pedestrian crossings).
4. **Rich Small-Object Distribution:** Selection favors sequences with substantial counts of small ($16\text{ px} \le \text{dimension} < 32\text{ px}$) and ultra-small ($< 16\text{ px}$) ground-truth annotations to guarantee statistical power for scale-specific evaluations.

---

## 2. Complete BDD100K Validation Inventory

The entire locally available BDD100K 20fps/5fps tracking validation set in `data/bdd100k/box_track_labels_trainval.zip` (200 sequence annotations) was audited by joining tracking labels with official keyframe metadata from `data/bdd100k/samples.json`.

### Inventory Summary:
- **Total Validation Sequences Inspected:** 200
- **Sequences Excluded (Historical Missions 9–13):** 14
- **Virgin / Unused Sequences Available:** 186
- **Unused Weather Distribution:**
  - Clear: 105
  - Overcast: 23
  - Undefined: 21
  - Snowy: 13
  - Rainy: 12
  - Partly Cloudy: 12
- **Unused Time-of-Day Distribution:**
  - Daytime: 99
  - Night: 67
  - Dawn/Dusk: 19
  - Undefined: 1

---

## 3. Historical Exclusion Audit

All 14 previously used sequences are formally retired and cataloged in `artifacts/mission14/sequence_exclusion_manifest.json`:

| Sequence ID | Historical Role | Original Mission | Frames | Total Boxes | Weather / Context |
|:---|:---:|:---:|:---:|:---:|:---|
| `b1e9ee0e-67e26f2e` | Dev | Mission 9 | 203 | 8,719 | Rainy Daytime Highway |
| `b1d0a191-65deaeef` | Dev | Mission 9 | 203 | 4,281 | Snowy Daytime Urban |
| `b24071b8-b3ee1196` | Dev | Mission 9 | 203 | 3,115 | Overcast Daytime Highway |
| `b2208b0f-2796a692` | Dev | Mission 9 | 203 | 2,890 | Clear Night Highway |
| `b2064e61-2beadd45` | Test | Mission 9 | 202 | 7,450 | Rainy Daytime Arterial |
| `b1d0a191-03dcecc2` | Test | Mission 9 | 202 | 3,892 | Snowy Daytime Urban |
| `b1d968b9-ce42734f` | Test | Mission 9 | 202 | 2,980 | Overcast Daytime Highway |
| `b1ca2e5d-84cf9134` | Test | Mission 9 | 202 | 3,410 | Clear Night Highway |
| `b24380ab-6dbeb908` | Dev | Missions 10–13 | 202 | 2,786 | Rainy Night Highway Spray |
| `b255cd6c-2f889586` | Dev | Missions 10–13 | 203 | 3,625 | Snowy Daytime Urban Slush |
| `b1e1a7b8-65ec7612` | Test | Missions 10–13 | 202 | 2,702 | Rainy Daytime Wiper Glare |
| `b1d0a191-06deb55d` | Test | Missions 10–13 | 202 | 4,323 | Snowy Daytime Urban Traffic |
| `b24d283f-33783d1b` | Test | Missions 10–13 | 202 | 4,323 | Overcast Daytime Arterial |
| `b23c9e00-b425de1b` | Test | Missions 10–13 | 202 | 3,500 | Clear Night High-Speed Highway |

---

## 4. Mission 14 Cohort Definitions

From the 186 eligible virgin sequences, 8 diverse sequences (1,620 total continuous frames, 23,288 ground-truth annotations) were partitioned into two disjoint cohorts:

### 4.1 Development Cohort (2 Sequences, 406 Frames, 4,740 GT Boxes)
Used strictly for code verification, pipeline sanity checks, and debugging:
1. **`b1e1a7b8-b397c445` (Rainy Daytime City Street):**
   - 203 frames, 2,173 annotations (810 small, 69 ultra-small).
   - Features wet windshield wiper sweeps, street-level water reflections, and oncoming vehicle headlight glare.
2. **`b204a5c1-064b0040` (Snowy Daytime Urban Arterial):**
   - 203 frames, 2,567 annotations (1,142 small, 39 ultra-small).
   - Features heavy roadside snowbanks, road slush, and dense pedestrian and cyclist crossings.

### 4.2 Locked Evaluation Cohort (6 Independent Sequences, 1,218 Frames, 18,548 GT Boxes)
Frozen prior to running any model comparisons. Evaluates generalization across unseen environments:
1. **`b23f7012-fab06dac` (Rainy Night City Street):**
   - 202 frames, 2,199 annotations (770 small, 168 ultra-small).
   - Low ambient illumination, high specular reflections on soaked asphalt, severe blooming around streetlights.
2. **`b1d0a191-de8948f6` (Snowy Daytime Highway):**
   - 202 frames, 3,321 annotations (845 small, 195 ultra-small).
   - High-speed driving, snow spray from lead vehicles, roadside snow glare, distant vehicles near horizon.
3. **`b1d0a191-28f0e779` (Snowy Daytime Urban Core):**
   - 202 frames, 2,447 annotations (864 small, 247 ultra-small).
   - Dense stop-and-go urban traffic, partial occlusions by snowbanks, cross-traffic pedestrian clusters.
4. **`b1f4491b-cf446195` (Overcast Daytime Highway Arterial):**
   - 203 frames, 3,840 annotations (1,567 small, 730 ultra-small).
   - Diffuse low-contrast sky, long sightlines with high density of ultra-small vehicles approaching vanishing point.
5. **`b1c66a42-6f7d68ca` (Overcast Daytime Urban Core):**
   - 203 frames, 3,241 annotations (1,339 small, 457 ultra-small).
   - Multi-lane arterial with dense queues of passenger cars, buses, and delivery trucks.
6. **`b2505382-de5238f0` (Clear Night High-Speed Arterial):**
   - 203 frames, 3,524 annotations (1,125 small, 511 ultra-small).
   - Clear weather baseline providing night contrast, headlight glare, and taillight clustering at high speeds.

---

## 5. Statistical Power Analysis

The locked evaluation cohort provides:
- **6 independent sequence sampling units** (allowing paired sequence-level bootstrapping with $2^6 = 64$ exact permutation states and $B=1,000$ paired resampling replicates).
- **1,218 continuous evaluation frames** (providing $>10\times$ the statistical observations of single-image test sets).
- **18,548 total ground-truth annotations**, including:
  - **6,510 Small Objects** ($16\text{ px} \le \text{dimension} < 32\text{ px}$)
  - **2,308 Ultra-Small Objects** ($< 16\text{ px}$)
- This large sample of sub-32px objects ensures narrow confidence intervals and statistical defensibility for scale-specific hypotheses.
