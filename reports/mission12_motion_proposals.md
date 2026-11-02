# WeatherGuard — Mission 12 Motion-Based Candidate Proposals
## Frame Differencing, Sparse Optical Flow & Causal Temporal Motion Discovery

**Date:** 2026-11-02  
**Author:** Principal Computer Vision Researcher & Edge AI Engineer  
**Hardware Environment:** Apple MacBook Air M1, 8-core CPU, macOS, CPU-only execution  
**Module Reference:** `src/candidate_generation.py` (`FrameDifferenceProposal`, `SparseOpticalFlowProposal`)

---

### 1. Research Motivation & Theoretical Foundations

In Mission 11, the primary bottleneck in detecting ultra-small distant vehicles was the **spatial proposal failure**: when an 8–14px vehicle produces zero activations in the initial 640px forward pass, the region-adaptive detector never creates a crop around it.

However, moving vehicles in traffic possess a strong physical invariant independent of appearance: **temporal pixel motion**. Even when a vehicle's pixel contrast is washed out by heavy rain, fog, or headlight glare, its spatial translation across consecutive video frames produces detectable intensity changes and motion vectors.

Mission 12 investigated two complementary causal motion proposal mechanisms:
1. **Consecutive Frame Differencing:** Ultra-lightweight temporal difference with morphological filtering and geometric wiper rejection.
2. **Sparse Optical Flow:** Pyramidal Lucas-Kanade feature point tracking with global ego-motion compensation.

---

### 2. Method A: Causal Frame Differencing (`FrameDifferenceProposal`)

#### Algorithmic Design:
1. **Downscaled Intensity Representation:** Full-frame input ($1280 \times 720$) is converted to grayscale and downsampled to $320 \times 180$ via area interpolation, reducing pixel operations by **$16\times$**:
   $$I_{\text{small}, t}(x, y) = \text{Downsample}(I_t(x, y))$$
2. **Temporal Difference Operator:**
   $$\Delta I_t(x, y) = |I_{\text{small}, t}(x, y) - I_{\text{small}, t-1}(x, y)|$$
3. **Road Horizon Prior Mask:** Motion in the sky ($y < 0.20 H$) and immediate hood ($y > 0.82 H$) is masked to zero:
   $$\Delta I_{\text{masked}, t} = \Delta I_t \odot M_{\text{road}}$$
4. **Adaptive Thresholding & Morphological Cleanup:**
   - Threshold $\tau_{\text{diff}} = 25$ isolates meaningful motion.
   - Morphological opening ($3\times 3$ rectangular kernel) strips isolated rain streak noise.
   - Morphological closing ($3\times 3$ kernel) bridges fragmented vehicle contours (e.g. headlight and bumper reflections).
5. **Connected Component Analysis & Artifact Filtering:**
   - Evaluates connected components with 8-way connectivity.
   - **Wiper Sweep Rejection:** Any component with area $> 40,000\text{px}^2$ (scaled) or width/height $> 500\text{px}$ is discarded as an ego-wiper stroke or sudden illumination shift.
   - **Noise Rejection:** Components with area $< 40\text{px}^2$ are discarded.
6. **Proposal Scoring:**
   $$s_{\text{diff}} = \min\left(1.0, \frac{\bar{I}_{\text{diff}}}{80.0}\right)$$

---

### 3. Method B: Sparse Optical Flow with Ego-Motion Compensation (`SparseOpticalFlowProposal`)

#### Algorithmic Design:
1. **Feature Point Seeding:** Extracts up to 150 Shi-Tomasi corner keypoints within the active road horizon band:
   $$\mathbf{p}_{i, t-1} = \text{goodFeaturesToTrack}(I_{t-1}, N=150, q=0.02, d_{\min}=12)$$
2. **Pyramidal Lucas-Kanade Flow:** Tracks feature points across time using a 2-level image pyramid with an $11\times 11$ pixel search window:
   $$\mathbf{p}_{i, t} = \text{calcOpticalFlowPyrLK}(I_{t-1}, I_t, \mathbf{p}_{i, t-1})$$
3. **Ego-Motion Compensation (Median Flow Subtraction):** In forward driving, vehicle movement induces a strong radial background optical flow field. To separate moving targets from stationary guardrails and road markings, we estimate ego-motion via the spatial median displacement vector:
   $$\mathbf{v}_{\text{ego}} = \left(\text{median}_i(\Delta x_i), \text{median}_i(\Delta y_i)\right)$$
4. **Residual Motion Vector Isolation:**
   $$\Delta \mathbf{v}_i = (\mathbf{p}_{i, t} - \mathbf{p}_{i, t-1}) - \mathbf{v}_{\text{ego}}$$
   Points with residual velocity $\|\Delta \mathbf{v}_i\| \ge 1.8\text{px}$ are tagged as independent dynamic movers.
5. **Spatial Clustering:** Moving points within Euclidean radius $D \le 48\text{px}$ are merged into candidate bounding boxes with $35\%$ context padding.

---

### 4. Empirical Benchmark & Engineering Trade-Off on Apple M1 CPU

Benchmarking sustained operation over 1,213 video frames on an Apple M1 CPU (7 intra-op threads):

| Motion Proposal Method | Mean Latency (ms) | P90 Latency (ms) | Peak RAM (MB) | Proposals / Frame | Ultra-Small Recall Gain |
|---|:---:|:---:|:---:|:---:|:---:|
| **Frame Differencing** | **$1.25\text{ ms}$** | **$1.45\text{ ms}$** | **$495\text{ MB}$** | $14.2$ | $\mathbf{+18.4\%}$ |
| **Sparse Optical Flow** | $17.75\text{ ms}$ | $22.10\text{ ms}$ | $525\text{ MB}$ | $6.8$ | $+19.1\%$ |

#### Critical Engineering Verdict:
- **Sparse Optical Flow is too computationally expensive for real-time edge CPU deployment.** At $17.75\text{ ms/frame}$, sparse optical flow consumes over $36\%$ of the entire edge frame budget ($50\text{ ms} = 20\text{ FPS}$), dropping total pipeline throughput from $20.43\text{ FPS} \to 14.63\text{ FPS}$.
- **Frame Differencing achieves virtually identical candidate recall (+18.4% vs +19.1%) while running $14.2\times$ faster ($1.25\text{ ms}$ vs $17.75\text{ ms}$).**
- Therefore, the optimal motion component for WeatherGuard Mission 12 is **downscaled morphological Frame Differencing**.

---

### 5. Causal Temporal Constraints & Edge Case Diagnostics

1. **Startup Behavior ($t = 0$):** At frame 0 of a video sequence, no previous frame exists. The motion generator outputs an empty proposal set; the controller falls back cleanly to the static base detector and spatial prior without throwing exceptions or stalling execution.
2. **Stationary Vehicles at Red Lights (FM-02):** Stopped vehicles have zero relative velocity and generate zero frame differences. This proves that motion proposals *cannot operate alone* and must be fused with appearance-based static salience.
3. **Wiper Motion Rejection (FM-04):** Without geometric aspect-ratio and area bounding, windshield wipers trigger large vertical motion masks. Capping maximum bounding box dimensions at $500\text{px}$ eliminates $98.2\%$ of wiper false proposals.
