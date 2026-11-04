# WeatherGuard — Research Paper Draft Outline
## Photorealistic Synthetic Data Fine-Tuning and Adaptive Edge Inference for Adverse-Weather Object Detection

**Document:** `reports/mission16_preparation/PAPER_DRAFT_OUTLINE.md`  
**Authors:** Mohammad Sayem Uddin et al.  
**Working Title:** *Bridging the Synthetic-to-Real Gap in Adverse-Weather Edge Perception: Lessons from Photorealistic Dashcam Synthesis and Horizon-Anchored Candidate Allocation*  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / IEEE Robotics and Automation Letters (RA-L)

---

### Abstract
Adverse weather conditions—heavy precipitation, fog, and nighttime illumination glare—severely degrade object detection reliability in autonomous driving. While modern detectors achieve high accuracy on near-field vehicles, small and distant targets ($< 32\text{px}$) near the vanishing point suffer catastrophic recall failure due to low physical contrast and scarce adverse-weather training representations. In this work, we investigate whether photorealistic synthetic adverse-weather dashcam data can overcome this bottleneck and transfer to real-world edge perception. We introduce **WeatherGuard-D1**, a curated corpus of 2,000 synthetic scenes (Day, Night, Rain, Fog) paired with independently validated multi-tier annotations. We formulate a leakage-safe prompt-family grouping protocol and investigate small-object-aware fine-tuning of lightweight detectors (YOLOv8n) on Apple M1 edge CPU hardware. Furthermore, we evaluate the interaction between synthetic feature fine-tuning and adaptive candidate allocation (Policy K). On held-out real-world driving video (BDD100K locked cohort), our framework achieves `[PENDING_MISSION16_E2_E6_RESULTS]`, demonstrating that targeted synthetic domain adaptation can recover distant road users without incurring prohibitive latency penalties.

---

### 1. Introduction
- The imperative for small-object perception at stopping distances ($> 50\text{ m}$).
- The failure of single-frame detectors in adverse weather (CAT-01 candidate failure).
- The opportunity and perils of generative diffusion models for autonomous perception.
- Contributions of WeatherGuard Mission 16:
  1. A rigorous ingestion, validation, and leakage-safe grouping methodology for synthetic diffusion data.
  2. A controlled factorial study of synthetic fine-tuning (E1–E12) isolating small-object oversampling, adverse-weather augmentation, and layer freezing.
  3. Empirical quantification of the synthetic-to-real domain gap on real-world driving video.

---

### 2. Related Work & Research Gap
- **Adverse-Weather Perception:** DAWN, BDD100K, ACDC, Foggy Cityscapes.
- **Synthetic Data for Autonomous Driving:** SYNTHIA, CARLA, diffusion-based scenario generation.
- **Small-Object Detection & Slicing:** SAHI, ClusteredDet, scale-aware routing.
- **The Research Gap:** Prior synthetic driving studies either focus on daytime driving or evaluate only on synthetic test sets, failing to isolate small-object transfer on CPU-constrained edge hardware.

---

### 3. WeatherGuard-D1: Construction & Quality Control
- **Generative Process:** Google Flow prompt templates and condition quotas (Day: 600, Night: 500, Rain: 500, Fog: 400).
- **Independent Validation Pipeline:** Transforming Gemini Flash $[0..1000]$ coordinates, recalculating object scales on $\max(\text{pixel width}, \text{pixel height})$, and pruning duplicates.
- **Three-Tier Uncertainty Triage:** Accepted Structural, Needs Visual Review, Rejected.
- **Leakage Prevention Invariant:** Scenario family and dHash cluster isolation across Train ($60\%$), Validation ($20\%$), and Test ($20\%$).

---

### 4. Methodology
- **Small-Object-Aware Sampling:** Counteracting the natural heavy-tailed scale distribution.
- **Targeted Photometric Augmentation:** Simulating fog optical thickness and wet asphalt reflectance.
- **Adaptive Edge Inference:** Marrying fine-tuned detection heads with Policy K causal temporal evidence accumulation and horizon-anchored candidate generation.

---

### 5. Experimental Setup
- **Hardware Platform:** Apple MacBook Air M1, 8-core CPU, 8 GB Unified Memory, macOS Darwin 24.3.0, CPU-only FP32.
- **Baseline Models:** Pretrained YOLOv8n (640px, 832px).
- **Evaluation Splits:** D1 Internal Test (400 images), BDD100K Locked Video (6 sequences, 1,214 frames), FLIR ADAS Thermal (200 images).
- **Statistical Significance:** Paired sequence-level bootstrap ($B=1,000$) with empirical 95% confidence intervals and TOST non-inferiority margins.

---

### 6. Empirical Results & Domain Transfer (Placeholders)
- *Table 1: Synthetic In-Domain Performance (D1 Test Set).* `[PENDING_EXECUTION]`
- *Table 2: Real-World Cross-Domain Performance (BDD100K Locked Video).* `[PENDING_EXECUTION]`
- *Table 3: Cross-Spectral Transfer (FLIR ADAS Thermal).* `[PENDING_EXECUTION]`

---

### 7. Ablation Studies (Placeholders)
- *Ablation 1:* Small-object oversampling factor (1.0x vs 2.0x vs 3.0x).
- *Ablation 2:* Layer-freezing depth (head-only vs backbone stages 0..4 vs full fine-tuning).
- *Ablation 3:* Condition-balanced mini-batch updates vs uniform random sampling.

---

### 8. Diagnostic Error Analysis
- Application of Mission 15's 12-category temporal failure taxonomy.
- Tracking resolution of TF-12 (crop classification misses) under fine-tuned heads.
- Residual bottlenecks: TF-11 (crop budget exhaustion) and TF-07 (physical contrast extinction).

---

### 9. Computational Latency & Edge Feasibility
- Micro-benchmarks of forward-pass latency on Apple M1 CPU.
- RAM RSS stability and memory leak diagnostics.
- Throughput comparison against full-frame high-resolution inference (YOLO-832).

---

### 10. Threats to Validity & Limitations
- Diffusion generation artifacts and hallucinated background objects.
- Finite number of locked real-world driving sequences ($N=6$).
- Decoupling of static still image training from causal temporal video accumulation.

---

### 11. Conclusion
- Summary of synthetic-to-real transfer findings.
- Recommendation of optimal edge perception pipeline combining synthetic fine-tuning with Policy K priority allocation.
