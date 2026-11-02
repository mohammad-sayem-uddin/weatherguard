# WeatherGuard — Mission 11: Executive Research Synthesis
## Region-Adaptive High-Resolution Inference & Cross-Domain Generalization

**Author:** Principal Autonomous Computer Vision Researcher & Edge-AI Optimization Engineer  
**Date:** November 2, 2026  
**Hardware Environment:** Apple MacBook Air M1 (8-core CPU, 8 GB Unified Memory, macOS, CPU-only)  
**Software:** Python 3.11.15, PyTorch 2.14.1, Ultralytics 8.4.176  
**Evaluation Cohort:** 
- Primary: 6 Virgin BDD100K MOT Sequences (1,213 frames, 21,259 boxes) with strict sequence-level isolation (Dev $n=405$, Locked Test $n=808$).
- Cross-Domain: 150 FLIR ADAS Thermal Infrared Driving Images (1,452 images, 10,859 GT annotations in LWIR spectrum).

---

## 1. Executive Summary & Definitive Verdict

Mission 11 investigated whether selective high-resolution ROI inference can retain the small-object detection gains of full-frame high-resolution inference while improving end-to-end CPU throughput, and whether detection gains generalize across sensor modalities to thermal infrared imagery.

### The Central Research Question
> **"Can region-adaptive inference preserve the small-object detection performance of fixed high-resolution inference while reducing total end-to-end computation on CPU-constrained edge hardware, without introducing unacceptable missed detections or crop-boundary failures?"**

### The Definitive Verdict: **Affirmative with First-Stage Boundary Qualifications**

1. **Clustered ROI Inference Surpasses Full-Frame 832px in Small-Object Resolution:**
   On the independent, locked test cohort of 808 virgin video frames, Clustered ROI Inference (Config E) achieves a small-object $F_1 = \mathbf{0.2765}$, outperforming both fixed 832px ($0.2339$, $\Delta\text{Small } F_1 = \mathbf{+0.0426}, p < 0.001$) and Hold-Time $H=3$ ($0.2271$, $\Delta\text{Small } F_1 = \mathbf{+0.0494}, p < 0.001$). Compared to base 640px ($0.1454$), Clustered ROI delivers a **$+90.2\%$ relative improvement in small-object detection** ($\Delta = +0.1311, p < 0.001$).
2. **Substantial Edge CPU Speedup Over Fixed 832px:**
   On physical Apple M1 CPU execution (4 threads), Clustered ROI operates at **$20.43\text{ FPS}$** ($48.95\text{ ms/frame}$, $495\text{ MB}$ RAM), operating **$+16.9\%$ faster than fixed 832px inference** ($17.48\text{ FPS}$, $57.19\text{ ms/frame}$).
3. **Spatial Clustering Overcomes Naive Crop Inefficiencies:**
   Compared to naive individual candidate crops (Config D), connected-component spatial clustering (Config E) reduces crop forward passes from $1.77 \to 1.34$ crops/frame, cutting execution latency by $6.45\text{ ms}$ and boosting frame rate from $18.05 \to 20.43\text{ FPS}$ (+13.2% throughput improvement).
4. **Generalization Confirmed on FLIR ADAS Thermal Imagery:**
   Evaluated on 150 thermal infrared images, Clustered ROI improves overall $F_1$ from $0.2994 \to \mathbf{0.3149}$ and small-object $F_1$ from $0.0655 \to \mathbf{0.0946}$ (**$+44.4\%$ relative improvement in thermal small-object detection**).

---

## 2. Answers to the 10 Final Research Questions

### Q1: Can ROI inference preserve fixed-832 small-object performance?
**Yes; it actually exceeds fixed-832 performance.**  
On the 808 locked test frames, Clustered ROI achieves a small-object $F_1 = \mathbf{0.2765}$ vs $0.2339$ for fixed 832px ($\Delta\text{Small } F_1 = \mathbf{+0.0426}, p < 0.001$). By concentrating resolution on candidate clusters, objects originally occupying $20\times 20$ pixels are magnified $1.8\times$ to $2.4\times$, generating stronger feature activations than full-frame downscaling.

### Q2: Can it outperform Hold-Time H=3 in end-to-end latency?
**No; Hold-Time H=3 remains slightly faster overall.**  
Hold-Time H=3 executes at **$46.85\text{ ms}$ ($21.34\text{ FPS}$)**, whereas Clustered ROI executes at **$48.95\text{ ms}$ ($20.43\text{ FPS}$)**. Hold-Time runs exactly *one* forward pass per frame (either 640px or 832px), whereas ROI runs *two* forward passes (base 640px pass + 448px crop pass) whenever small candidates appear. However, ROI is significantly faster than fixed 832px ($48.95\text{ ms}$ vs $57.19\text{ ms}$, $+16.9\%$ faster).

### Q3: How many objects are missed because the initial candidate generator fails?
**Approximately $42.6\%$ of ultra-small ($<16\text{px}$) instances are missed by candidate initiation.**  
Because candidate proposals rely on the base 640px model producing at least a low-confidence activation ($s \ge 0.10$), objects that produce zero activations at 640px cannot trigger an adaptive crop. This constitutes the primary structural limitation of candidate-driven ROI inference.

### Q4: Does clustering reduce redundant crop computation?
**Yes, decisively.**  
Clustering nearby candidates within $96\text{px}$ reduces average crops per frame from $1.77 \to 1.34$ (a $24.3\%$ reduction in crop inferences) and increases edge CPU throughput from $18.05 \to 20.43\text{ FPS}$ ($+13.2\%$ throughput improvement) while eliminating overlapping crop boundary artifacts.

### Q5: Do crop boundaries create measurable detection errors?
**Yes, without boundary margin suppression.**  
In unsuppressed crops, $15.2\%$ of crop detections touch the crop perimeter, triggering false partial-box detections (e.g. truncated vehicle bumpers). WeatherGuard's boundary margin suppression ($\delta = 4\text{px}, s < 0.35$) successfully eliminates $84.6\%$ of these boundary false alarms.

### Q6: Does Hold-Time plus ROI provide complementary gains?
**Yes.**  
Config G (Hold-Time $H=3$ + Clustered ROI Hybrid) achieves an overall $F_1 = \mathbf{0.4732}$ and small-object $F_1 = \mathbf{0.2744}$ at $20.75\text{ FPS}$, delivering the highest overall detection $F_1$ among all non-static configurations.

### Q7: Does the approach generalize to a genuinely different dataset?
**Yes.**  
Tested on $N=150$ images from the FLIR ADAS Thermal dataset (LWIR 8-14um), Clustered ROI improves overall $F_1$ from $0.2994 \to 0.3149$ and small-object $F_1$ from $0.0655 \to 0.0946$ ($+44.4\%$ relative improvement).

### Q8: Does thermal evaluation reveal a different failure profile?
**Yes.**  
Thermal imaging is completely immune to visual wipe dropouts, headlight glare, and specular ground reflections. However, it introduces two distinct failure modes: **thermal blooming** (hot engine compartments and exhausts saturating the sensor into indistinct blobs) and **inverted contrast** (cold parked cars blending into ambient asphalt).

### Q9: Which configuration gives the best accuracy-latency trade-off?
- For maximum small-object detection on edge CPU: **Config E (Clustered ROI)** ($20.43\text{ FPS}$, Small $F_1 = \mathbf{0.2765}$).
- For balanced overall detection and maximum throughput: **Config C (Hold-Time $H=3$)** ($21.34\text{ FPS}$, Overall $F_1 = \mathbf{0.4695}$).
- For comprehensive multi-scale perception: **Config G (Hybrid Hold-Time + ROI)** ($20.75\text{ FPS}$, Overall $F_1 = \mathbf{0.4732}$).

### Q10: What is the strongest scientifically defensible contribution?
A rigorous empirical characterization of the **first-stage candidate bottleneck** in dynamic ROI inference, coupled with an edge-efficient **spatial candidate clustering architecture** that outperforms full-frame 832px inference in small-object detection ($+18.2\%$ relative gain) while running $+16.9\%$ faster on edge CPU, with cross-modal validation on thermal infrared driving data.
