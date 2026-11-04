# WeatherGuard Mission 16 — Threats to Scientific Validity
**Document:** `reports/mission16_preparation/THREATS_TO_VALIDITY.md`  
**Date:** 2026-11-04  
**Author:** Mohammad Sayem Uddin (<sayemwork41@gmail.com>)  
**Mission:** Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

---

## 1. Internal Validity

1. **AI-Assisted Annotation Bias (Gemini Flash):**
   - *Threat:* Foundation vision-language models may exhibit systematic classification biases, hallucinating distant vehicles in foggy backgrounds or omitting low-contrast pedestrians.
   - *Mitigation:* The multi-tier validation engine (`scripts/validate_d1_annotations.py`) mathematically verifies bounding box coordinates, recalculates object scale $d_{max}$, prunes duplicates, and isolates uncertain cases into human review queues.
2. **Diffusion Prompt Overfitting & Leakage:**
   - *Threat:* Generative diffusion models produce visual artifacts (texture smoothing, repetitive wheel geometries) across identical prompts.
   - *Mitigation:* Split isolation groups images by scenario family and perceptual dHash clusters, ensuring near-duplicates are never distributed across train and test.
3. **Hyperparameter Selection Confounding:**
   - *Threat:* Tuning early stopping or augmentations directly on test data would invalidate generalization claims.
   - *Mitigation:* Strict cohort separation; the internal test set ($400$ images) and real-world locked video cohort ($1,214$ frames) remain locked and evaluated exactly once.

---

## 2. External Validity & Domain Transfer

1. **Simulation-to-Reality Domain Gap:**
   - *Threat:* High performance on synthetic test data might fail to transfer to real cameras due to real-world sensor noise, CMOS rolling shutter, headlight lens flare, and raindrop distortion on windshield glass.
   - *Mitigation:* Mission 16 treats the real-world BDD100K locked video test set as the primary arbiter of success. Synthetic scores are never presented as evidence of real-world deployment readiness.
2. **Geographical & Sensor Distribution Shift:**
   - *Threat:* Synthetic generation may over-represent clean modern sedans on wide multi-lane highways, under-representing complex urban street clutter.
   - *Mitigation:* BDD100K evaluation covers diverse scenes: residential roads, night city streets, high-speed highways, and rural twilight.

---

## 3. Construct Validity

1. **Object-Scale Threshold Boundaries:**
   - *Threat:* Inconsistent scale definitions across literature (COCO area $< 1024\text{ px}^2$ vs pixel max dimension $< 16\text{ px}$) can obscure where gains originate.
   - *Mitigation:* Explicit adoption of $\max(\text{pixel width}, \text{pixel height})$ across all evaluation scripts, reporting Ultra-Small ($<16\text{px}$) and Small ($16-32\text{px}$) separately.
2. **Single-Frame Still vs Temporal Video Disconnect:**
   - *Threat:* Conflating image-level fine-tuning with multi-frame temporal accumulation.
   - *Mitigation:* Strict decoupling: D1 synthetic data is evaluated purely as static still imagery; temporal policies (Policy K) are evaluated strictly on continuous video sequences.

---

## 4. Statistical Conclusion Validity

1. **Finite Sequence Sample Size ($N=6$ Locked Sequences):**
   - *Threat:* Evaluating 6 driving sequences ($1,214$ frames) cannot support unbounded claims of universal weather invariance.
   - *Mitigation:* All comparative conclusions are bounded by empirical 95% bootstrap confidence intervals and sequence win/loss ratios.
2. **Temporal Autocorrelation:**
   - *Threat:* Treating adjacent video frames as independent observations artificially shrinks standard errors.
   - *Mitigation:* Resampling units are entire driving sequences ($N=6$) rather than isolated frames.
