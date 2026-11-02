# WeatherGuard — Mission 11: ROI Inference Architecture
## Region-Adaptive Multi-Scale Perception for Edge Computer Vision

**Author:** Principal Autonomous Computer Vision Researcher & Edge-AI Optimization Engineer  
**Date:** November 2, 2026  
**Hardware Environment:** Apple MacBook Air M1 (8-core CPU, 8 GB Unified Memory, CPU-only execution)  
**Deliverable:** `reports/mission11_roi_architecture.md`  
**Source Code:** `src/roi_inference.py`

---

## 1. Motivation and Problem Formulation

In adverse-weather autonomous driving, small and distant objects (vehicles, pedestrians, cyclists at distances $>30\text{m}$) suffer severe feature degradation when downsampled to the standard base resolution ($640\times 640$). Under heavy rain, snow backscatter, or headlight glare, sub-$32\times 32$ pixel instances lack the gradient contrast required to trigger high-confidence activations in deep convolutional feature pyramids (e.g., YOLOv8 P3 stride-8 feature maps).

While full-frame high-resolution inference ($832\times 832$) recovers substantial recall ($\Delta\text{Small } F_1 = +0.0885$), it increases CPU inference time by **$+32.4\%$** ($43.19\text{ ms} \to 57.19\text{ ms}$, dropping frame rate from $23.15\text{ FPS}$ to $17.48\text{ FPS}$). Full-frame escalation processes millions of background pixels (empty sky, asphalt, trees) that contain zero foreground targets.

**Mission 11 investigates Region-Adaptive High-Resolution Inference:**
Can we selectively extract and infer high-resolution sub-windows (crops) only around regions containing small or uncertain candidates, achieving the detection fidelity of $832\text{px}$ inference at a fraction of the computational overhead?

---

## 2. Mathematical Architecture

```
                                WeatherGuard ROI Inference Pipeline
                                
       [ Input Frame (1280x720) ] 
                   |
                   v
       +------------------------+
       | Fast Base YOLO (640px) | ===> Generates Initial Detections {d_i}
       +------------------------+
                   |
                   v
       +------------------------+
       | Candidate Filter       | ===> Selects Small (<1024px²) or Uncertain (0.15 <= conf <= 0.40)
       +------------------------+
                   |
                   v
       +------------------------+
       | Spatial Clustering     | ===> Groups Proximity Candidates into K <= 2 Unified Boxes
       | (Connected Components) |      Applies Context Margin (+20% Padding)
       +------------------------+
                   |
                   +------------------------------+
                   | (If K = 0: Skip to Output)   | (If K > 0: Extract Sub-Crops)
                   v                              v
       [ Pass Base Detections ]        +--------------------+
                   |                   | Crop Forward Pass  | ===> YOLOv8n @ 640px on Sub-Window
                   |                   +--------------------+
                   |                              |
                   |                              v
                   |                   +--------------------+
                   |                   | Boundary Filter    | ===> Suppresses Truncated Edge Ghosts
                   |                   +--------------------+
                   |                              |
                   |                              v
                   |                   +--------------------+
                   |                   | Coord Restoration  | ===> Exact Affine Projection to Full Frame
                   |                   +--------------------+
                   |                              |
                   +------------+-----------------+
                                |
                                v
                   +------------------------+
                   | Duplicate-Safe NMS     | ===> Merges Base & Crop Detections
                   | (Priority to High-Res) |      IoU >= 0.50 Duplicate Suppression
                   +------------------------+
                                |
                                v
                   [ Final Frame Detections ]
```

### 2.1 Candidate Region Proposal
For every base detection $d_i = (c_i, s_i, x_1, y_1, x_2, y_2)$ from the $640\text{px}$ base pass:
$$\text{Area}(d_i) = (x_2 - x_1) \cdot (y_2 - y_1)$$
An ROI candidate is triggered if either condition is satisfied:
$$\mathcal{C}_{\text{small}}(d_i) = \mathbb{I}(\text{Area}(d_i) < 1024\text{ px}^2 \land s_i \ge 0.10)$$
$$\mathcal{C}_{\text{uncertain}}(d_i) = \mathbb{I}(0.15 \le s_i \le 0.40)$$

### 2.2 Spatial Candidate Clustering
Naive region refinement extracts an independent crop around every candidate box, leading to overlapping redundant inference and high latency when multiple vehicles travel in a platoon.
WeatherGuard implements spatial agglomeration:
1. Compute pairwise Euclidean distances between candidate centers $c_i, c_j$:
   $$D(i, j) = \|\mathbf{c}_i - \mathbf{c}_j\|_2$$
2. Construct an adjacency graph with edge condition $D(i, j) \le D_{\text{cluster}} = 96.0\text{ px}$.
3. Form connected components $\{C_k\}$. For each cluster, compute the bounding envelope:
   $$[\min_{i \in C_k}(x_1^i), \min_{i \in C_k}(y_1^i), \max_{i \in C_k}(x_2^i), \max_{i \in C_k}(y_2^i)]$$
4. Expand the envelope with context margin $m = 0.20$ ($20\%$ padding in each direction) and enforce a minimum crop size ($256\times 256$) to prevent extreme high-frequency magnification.

### 2.3 Coordinate Restoration Math
Let crop region in native image coordinates be $R = [X_1, Y_1, X_2, Y_2]$ with width $W_R = X_2 - X_1$ and height $H_R = Y_2 - Y_1$.
The crop is resized to input resolution $(W_c, H_c) = (640, 640)$ for the detector. For any detection $[x_1^c, y_1^c, x_2^c, y_2^c]$ generated inside the crop:
$$x_1^{\text{orig}} = X_1 + x_1^c \cdot \frac{W_R}{W_c}$$
$$y_1^{\text{orig}} = Y_1 + y_1^c \cdot \frac{H_R}{H_c}$$
$$x_2^{\text{orig}} = X_1 + x_2^c \cdot \frac{W_R}{W_c}$$
$$y_2^{\text{orig}} = Y_1 + y_2^c \cdot \frac{H_R}{H_c}$$

### 2.4 Boundary Artifact Suppression
A known hazard of crop inference is **edge truncation**: vehicles partially crossing the crop border are severed in half, causing the detector to generate false partial-box detections (e.g. truncated car roofs or bumper segments).
WeatherGuard applies a boundary filter:
$$\text{IsBoundaryBox}(\mathbf{b}) = \left(x_1^c \le \delta \lor y_1^c \le \delta \lor x_2^c \ge W_c - \delta \lor y_2^c \ge H_c - \delta\right)$$
where $\delta = 4.0\text{ px}$. If a crop detection touches the boundary AND has confidence $s < 0.35$, it is suppressed to prevent boundary false alarms.

### 2.5 Duplicate-Safe NMS Merging
Base detections and restored crop detections are merged:
- Base detections provide global context and detect large nearby vehicles.
- Crop detections provide high-resolution feature resolution for small distant objects.
- When a base detection and a crop detection overlap with $\text{IoU} \ge 0.50$, the system selects the crop detection if the object is small ($<1024\text{px}^2$) or the higher-confidence detection otherwise.

---

## 3. Comparison with Slicing Aided Hyper Inference (SAHI)

| Feature / Dimension | SAHI (Akyon et al. 2022) | Naive Crop Refinement (Mission 4) | WeatherGuard Clustered ROI (Mission 11) |
|---|---|---|---|
| **Slicing Strategy** | Uniform static sliding grid (e.g. 3x3) | 1 unclustered crop per candidate | Dynamic spatial clustering + Context padding |
| **Crops Per Frame** | Fixed $N_{\text{slices}} \ge 4-9$ crops | Unbounded ($K = N_{\text{candidates}}$) | Adaptive ($K \in [0, 2]$, median $1$) |
| **Empty Frame Cost** | Runs all slices unconditionally | 0 crops | **0 crops (Pure base latency)** |
| **Edge Boundary Handling** | Overlap slicing ($20-30\%$) | None (high boundary false alarms) | Context expansion + Boundary truncation suppression |
| **M1 CPU Latency** | $>150\text{ ms/frame}$ (infeasible for real-time) | Unstable ($40-90\text{ ms}$) | **Target: $42-50\text{ ms/frame}$** |
| **Duplicate Merging** | Full-frame NMS / WBF | Greedy NMS | Class-aware scale-prioritized NMS |

---

## 4. The Fundamental Scientific Challenge: First-Stage Candidate Bottleneck

A critical research question formulated for Mission 11 is:
> **"What if the first-stage 640px detector completely fails to detect the small object in the first place?"**

If an object is so small, dark, or occluded that the $640\text{px}$ model outputs zero boxes (even below threshold $0.10$), the candidate generator will not propose an ROI around it.
To causally quantify and address this bottleneck, Mission 11 tests five distinct strategies in Phase 2 & 3:
1. **Config A:** Full-frame 640px (Baseline).
2. **Config B:** Full-frame 832px (Upper bound).
3. **Config C:** Hold-Time $H=3$ (Mission 10 benchmark).
4. **Config D:** Naive individual candidate crops.
5. **Config E:** Clustered ROI inference.
6. **Config F:** Overlapping fixed-grid horizon strip (which guarantees high resolution across the vanishing line regardless of first-stage detections).
7. **Config G:** Hold-Time $H=3$ + Clustered ROI.
