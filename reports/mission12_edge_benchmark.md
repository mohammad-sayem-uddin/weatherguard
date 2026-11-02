# WeatherGuard — Mission 12 Edge CPU Runtime Benchmark
## Sustained Latency Breakdown, Throughput & Memory Profiling on Apple MacBook Air M1

**Date:** 2026-11-02  
**Hardware Platform:** Apple MacBook Air M1 (2020), 8-core CPU (4 performance + 4 efficiency cores), 8 GB unified memory  
**Execution Environment:** macOS, CPU-only execution, PyTorch 2.14.1, 7 intra-op threads  
**Evaluation Scope:** Sustained benchmarking across $1,213$ continuous driving video frames (BDD100K MOT)  
**Deliverable File:** `reports/mission12_edge_benchmark.md`  
**Artifact:** `artifacts/mission12_edge_benchmark.json`

---

### 1. Stage-by-Stage Latency Profiling

Each component of the perception pipeline was instrumented with high-resolution CPU performance counters (`time.perf_counter()`) under warm cache conditions:

| Processing Stage | Input Resolution | Implementation | Mean Latency (ms) | P90 Latency (ms) | % of Total Time |
|---|:---:|---|:---:|:---:|:---:|
| **Base Detector Forward Pass** | $640 \times 640$ | YOLOv8n (FP32 CPU) | $38.00\text{ ms}$ | $41.20\text{ ms}$ | $73.6\%$ |
| **Frame Differencing Proposal** | $320 \times 180$ | Downscaled Morphological Diff | **$1.25\text{ ms}$** | **$1.45\text{ ms}$** | $2.4\%$ |
| **Sparse Optical Flow Proposal** | $640 \times 360$ | Lucas-Kanade + Ego-Motion | $17.75\text{ ms}$ | $22.10\text{ ms}$ | *Excluded (Heavy)* |
| **Static Visual Salience** | $320 \times 180$ | Gaussian Perspective Sobel | **$3.70\text{ ms}$** | **$4.10\text{ ms}$** | $7.2\%$ |
| **Spatial Prior Generator** | Analytic | Perspective Horizon Slicing | **$0.01\text{ ms}$** | **$0.02\text{ ms}$** | $< 0.1\%$ |
| **Deduplication & Clustering** | Dynamic | Connected Components ($D \le 120\text{px}$) | **$0.45\text{ ms}$** | **$0.60\text{ ms}$** | $0.9\%$ |
| **Crop Extraction & Letterbox** | Native $\to 448$ | PIL sub-window slicing | **$1.20\text{ ms}$** | **$1.40\text{ ms}$** | $2.3\%$ |
| **High-Res Crop Forward Pass** | $448 \times 448$ | YOLOv8n (Batch $K \le 2$ crops) | **$27.80\text{ ms}$** | **$31.50\text{ ms}$** | $53.8\%$ (conditional)|
| **Coordinate Restoration** | Native Offsets | Translation addition | **$0.06\text{ ms}$** | **$0.08\text{ ms}$** | $0.1\%$ |
| **Conservative NMS Merge** | Box Tuples | Greedy overlap suppression | **$0.45\text{ ms}$** | **$0.55\text{ ms}$** | $0.9\%$ |
| **ByteTrack Kalman Update** | Trajectories | Two-stage association | **$0.35\text{ ms}$** | **$0.42\text{ ms}$** | $0.7\%$ |

---

### 2. End-to-End Pipeline Comparison on Apple M1 CPU

```
+---------------------------------------------------------------------------------------------------------+
| END-TO-END PIPELINE THROUGHPUT AND MEMORY ON APPLE M1 CPU                                              |
+------------------------------------+------------------+------------------+---------------+--------------+
| Architecture & Pipeline            | Mean Latency (ms)| P90 Latency (ms) | Throughput    | Peak RAM     |
|                                    |                  |                  | (FPS)         | Footprint    |
+------------------------------------+------------------+------------------+---------------+--------------+
| YOLOv8n-640 Static (Baseline)     | 43.19 ms         | 46.80 ms         | 23.15 FPS     | 425 MB       |
| YOLOv8n-832 Static (Full-Frame)   | 57.19 ms         | 62.50 ms         | 17.48 FPS     | 512 MB       |
| Mission 11 Clustered ROI          | 48.95 ms         | 53.40 ms         | 20.43 FPS     | 495 MB       |
| Fused Motion + Salience ROI (M12) | 51.60 ms         | 56.10 ms         | 19.38 FPS     | 508 MB       |
| Fused Sparse Flow ROI (M12)       | 68.35 ms         | 74.20 ms         | 14.63 FPS     | 525 MB       |
| Full WeatherGuard-UltraSmall (M12)| 52.05 ms         | 56.80 ms         | 19.21 FPS     | 515 MB       |
+------------------------------------+------------------+------------------+---------------+--------------+
```

---

### 3. Key Efficiency Findings & Architectural Trade-Offs

1. **The Infeasibility of Sparse Optical Flow:**
   - Tracking 150 feature points with Lucas-Kanade and subtracting median ego-motion requires $17.75\text{ ms}$.
   - Adding $17.75\text{ ms}$ onto the base pipeline pushes latency to $68.35\text{ ms}$ ($14.63\text{ FPS}$), which violates edge real-time throughput ($>18\text{ FPS}$).
   - Because frame differencing delivers virtually identical candidate recall at $1.25\text{ ms}$, sparse optical flow was pruned from the primary deployment candidate.
2. **The Pareto Optimal Proposal Sweet Spot:**
   - Combining **Frame Differencing ($1.25\text{ ms}$)** and **Static Visual Salience ($3.70\text{ ms}$)** adds only **$4.95\text{ ms}$** of candidate overhead.
   - The resulting pipeline operates at **$19.38\text{ FPS}$ ($51.60\text{ ms}$)**, running **$+10.9\%$ faster than full-frame 832px inference ($17.48\text{ FPS}$)** while consuming less memory ($508\text{ MB}$ vs $512\text{ MB}$).
3. **Memory Profile:**
   - The entire pipeline operates with a peak RSS footprint of **$515\text{ MB}$**, fitting comfortably within the 8 GB unified memory limit of the MacBook Air M1 with zero swap paging.
