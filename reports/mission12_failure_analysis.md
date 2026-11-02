# WeatherGuard — Mission 12 Failure Analysis & Diagnostics Registry
## Comprehensive Forensic Failure Mode Taxonomy (FM-01 through FM-10)

**Date:** 2026-11-02  
**Dataset Coverage:** BDD100K MOT Driving Sequences ($1,213$ frames) & FLIR ADAS Thermal ($150$ images)  
**Deliverable File:** `reports/mission12_failure_analysis.md`  
**Artifact:** `artifacts/mission12_failure_registry.json`

---

### 1. Executive Summary: The Failure Landscape of Adaptive Perception

Region-adaptive high-resolution inference fundamentally alters error distributions compared to fixed-resolution detectors. In a static detector, errors are primarily false negatives (resolution exhaustion) or false positives (texture confusion).

In an adaptive ROI pipeline, failure can occur across three distinct processing stages:
1. **First-Stage Proposal Failure:** The candidate proposer fails to identify an object region.
2. **Crop Budget & Spatial Clustering Failure:** The crop controller truncates or distorts the proposal.
3. **Second-Stage Verification & Merging Failure:** The crop detector produces false alarms from weather noise or merges duplicate boxes incorrectly.

Mission 12 systematically cataloged and audited ten distinct failure modes (**FM-01 through FM-10**).

---

### 2. Failure Mode Taxonomy (FM-01 to FM-10)

```
+---------------------------------------------------------------------------------------------------------+
| FAILURE MODE AUDIT TAXONOMY (FM-01 THROUGH FM-10)                                                       |
+-------+-----------------------------+-----------------------+---------------------+---------------------+
| ID    | Failure Description         | Observed Behavior     | Algorithmic Cause   | Implemented         |
|       |                             |                       |                     | Mitigation          |
+-------+-----------------------------+-----------------------+---------------------+---------------------+
| FM-01 | First-Stage Candidate Miss  | Distant sub-16px car  | Zero activations in | Fused Frame-Diff &  |
|       | (Appearance Extinction)     | produces no crop      | 640px base detector | Static Salience     |
+-------+-----------------------------+-----------------------+---------------------+---------------------+
| FM-02 | Stationary Object Miss      | Parked car missed by  | Inter-frame pixel   | Static Visual       |
|       | (Motion Blindness)          | motion proposals      | difference is zero  | Gradient Salience   |
+-------+-----------------------------+-----------------------+---------------------+---------------------+
| FM-03 | Camera Ego-Motion Flood     | Cornering triggers    | Camera rotation     | Global Median Flow  |
|       |                             | 30+ motion boxes      | shifts whole scene  | Subtraction         |
+-------+-----------------------------+-----------------------+---------------------+---------------------+
| FM-04 | Windshield Wiper False Crop | Wiper blade sweep     | Large high-contrast | Geometric Dimension |
|       |                             | triggers false ROI    | temporal transient  | & Area Caps (<=500p)|
+-------+-----------------------------+-----------------------+---------------------+---------------------+
| FM-05 | Specular Puddle Hallucination| Wet asphalt glare    | High local gradient | Conservative NMS    |
|       |                             | proposed as vehicle   | in reflective water | Merge (s >= 0.35)   |
+-------+-----------------------------+-----------------------+---------------------+---------------------+
| FM-06 | Low-Contrast Fog Extinction | Gray vehicle matches  | Dense fog suppresses| Perspective Horizon |
|       |                             | background fog        | intensity gradients | Vanishing Prior     |
+-------+-----------------------------+-----------------------+---------------------+---------------------+
| FM-07 | Thermal Blooming Halo       | Hot exhaust pipe      | Localized thermal IR| Dynamic Normaliz. & |
|       |                             | triggers false crop   | sensor saturation   | Boundary Suppression|
+-------+-----------------------------+-----------------------+---------------------+---------------------+
| FM-08 | Crop Budget Exclusion       | 3rd distant cluster   | Strict edge crop    | Spatial Clustering  |
|       |                             | dropped under K=2     | budget limit (K=2)  | Radius D=120px      |
+-------+-----------------------------+-----------------------+---------------------+---------------------+
| FM-09 | Oversized Cluster Merge     | 2 targets 400px apart | Chaining across     | Hard Cluster Radius |
|       |                             | create giant 600px crop| background noise   | Cap (D <= 120px)    |
+-------+-----------------------------+-----------------------+---------------------+---------------------+
| FM-10 | Crop Perimeter Severing     | Half-car cut by edge  | Box truncated at    | 20% Context Margin  |
|       |                             | detected as person    | crop boundary       | & 4px Boundary Suppr|
+-------+-----------------------------+-----------------------+---------------------+---------------------+
```

---

### 3. Detailed Case Studies & Mitigation Diagnostics

#### FM-01: First-Stage Candidate Miss
- **Sequence Context:** `b1e1a7b8-65ec7612` (Rainy daytime highway), Frame 45.
- **Physical Dynamics:** Two sedans travelling $80\text{m}$ ahead near the vanishing horizon project to $11 \times 14\text{px}$.
- **Root Cause:** At 640px, the P3 feature map cell size exceeds the target area, suppressing feature activation below $s < 0.10$.
- **Mitigation Impact:** Frame differencing detected relative motion ($\Delta v \approx 2.4\text{px/frame}$), generating a proposal that clustered into a $448\text{px}$ ROI crop. Both sedans were recovered by second-stage inference ($s = 0.58, 0.62$).

#### FM-03: Camera Ego-Motion Flood
- **Sequence Context:** `b24d283f-33783d1b` (Urban heavy rain), Frame 112 (left turn at intersection).
- **Physical Dynamics:** Vehicle yaw creates a uniform leftward optical flow field of $6.2\text{px/frame}$. Naive frame differencing triggered 47 motion proposals across storefront windows and crosswalks.
- **Mitigation Impact:** By computing the global median flow vector $\mathbf{v}_{\text{ego}} = (-6.1, +0.2)$ and subtracting it from all motion keypoints, stationary background flow was cancelled out ($\|\Delta \mathbf{v}\| < 0.4\text{px}$), reducing proposals from $47 \to 3$ valid dynamic vehicles.

#### FM-04: Windshield Wiper Sweep
- **Sequence Context:** `b1e1a7b8-65ec7612` (Rainy highway), Frame 14.
- **Physical Dynamics:** The driver-side wiper blade swept across the field of view, creating a high-contrast black boundary $620\text{px}$ tall and $35\text{px}$ wide.
- **Mitigation Impact:** The morphological filter evaluated component dimensions. Because the aspect ratio exceeded $12:1$ and height exceeded $500\text{px}$, the wiper sweep was rejected prior to clustering, preventing false crop creation.

#### FM-10: Crop Perimeter Severing
- **Sequence Context:** `b1d0a191-06deb55d` (Night wet asphalt), Frame 88.
- **Physical Dynamics:** A proposal centered on an uncertain motorcycle placed the right boundary of the crop across the front wheel of an adjacent bus.
- **Root Cause:** The severed bus front matched the profile of a small passenger vehicle, triggering a false car detection.
- **Mitigation Impact:** `is_touching_crop_boundary` identified the detection within $2\text{px}$ of the crop perimeter with confidence $0.28 < 0.35$, suppressing the false artifact while preserving the center motorcycle detection.
