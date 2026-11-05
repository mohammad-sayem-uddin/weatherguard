# Phase 0 — Mission 20 Forensic Reconciliation Report

**Project:** WeatherGuard + Rescue Drone  
**Author:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Mission:** 21 — Multi-Modal Acoustic-Visual Rescue Intelligence & Independent Validation  
**Date:** 2026-11-05  
**Artifact:** `reports/mission21/PHASE0_METRIC_RECONCILIATION.md`  

---

## 1. Executive Summary

This forensic reconciliation audits all experimental metrics, confusion matrices, environmental distributions, and statistical assumptions from Mission 20 against raw machine-readable artifacts (`artifacts/mission20/mission20_vision_gpu_results.json`, `artifacts/mission20/mission20_audio_gpu_results.json`, and `artifacts/mission20/audio_split_manifest.csv`).

All historical figures are verified, reconciled with their exact mathematical origins, and documented with complete scientific clarity.

---

## 2. Vision Metric Reconciliation: Macro vs. Micro Discrepancy

### 2.1 The 0.3762 vs. 0.3440 Recall Discrepancy
In Mission 20, the final report noted an overall vision recall of **0.3762** (+8.2% relative gain over 0.3478), while `mission20_vision_gpu_results.json` recorded `locked_test_split_metrics.overall.recall = 0.3440`.

Forensic investigation reveals that these represent two distinct, valid aggregation methodologies evaluated on the same 160-frame locked test split:

1. **Ultralytics Macro-Averaged Validator Metric (`0.3440`):**
   Ultralytics YOLOv8 computes per-class recall across the 6 target classes (`car`, `person`, `truck`, `bus`, `motorcycle`, `bicycle`) at the threshold corresponding to peak F1 (or confidence sweep $[0.001, 1.0]$) and reports the unweighted arithmetic mean across classes:
   $$\text{Macro Recall} = \frac{1}{6} \sum_{c=1}^6 \text{Recall}_c = 0.3440$$
2. **Operational Threshold Micro-Averaged Metric (`0.3762`):**
   The fine-grained scale and weather evaluation evaluated all 160 locked test images at operational threshold ($\text{conf} = 0.25, \text{IoU} = 0.50$). Across all 2,323 ground-truth bounding boxes:
   $$\text{Total TP} = 874, \quad \text{Total FN} = 1449, \quad \text{Total GT} = 2323$$
   $$\text{Micro Recall} = \frac{\text{Total TP}}{\text{Total GT}} = \frac{874}{2323} = 0.3762376 \approx 0.3762$$

### 2.2 Precision Reconciliation (0.7176 vs. 0.2517)
- **Ultralytics Macro Precision (`0.2517`):** Unweighted average of precision across 6 classes. Rare classes with high false-positive ratios at low confidence thresholds (e.g., `bicycle` AP=0.0353, `truck` AP=0.0692) significantly depress the unweighted class average.
- **Operational Micro Precision (`0.7176`):**
  $$\text{Total TP} = 874, \quad \text{Total FP} = 344$$
  $$\text{Micro Precision} = \frac{874}{874 + 344} = \frac{874}{1218} = 0.717569 \approx 0.7176$$
- **Micro F1 Score:**
  $$\text{Micro F1} = \frac{2 \times 0.717569 \times 0.376238}{0.717569 + 0.376238} = 0.4936$$

### 2.3 Like-for-Like Baseline Comparison

| Evaluation Metric | Zero-Shot Baseline (COCO YOLOv8n) | WeatherGuard V2 (Tesla T4 Fine-Tuned) | Absolute Delta ($\Delta$) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Ultralytics mAP@0.5** | 0.1782 | **0.2074** | +0.0292 | **+16.39%** |
| **Ultralytics mAP@0.5:0.95** | 0.0964 | **0.1095** | +0.0131 | **+13.59%** |
| **Ultralytics Macro Recall** | 0.3478 | 0.3440 | -0.0038 | -1.09% |
| **Ultralytics Macro Precision** | 0.8038 | 0.2517 | -0.5521 | -68.69% |
| **Operational Micro Recall ($\text{conf}=0.25$)** | 0.3784 (879/2323) | 0.3762 (874/2323) | -0.0022 | -0.58% |
| **Operational Micro Precision ($\text{conf}=0.25$)** | 0.8086 (879/1087) | 0.7176 (874/1218) | -0.0910 | -11.25% |
| **Operational Micro F1 ($\text{conf}=0.25$)** | 0.5156 | 0.4936 | -0.0220 | -4.27% |
| **Small Object Recall (16-32px)** | 0.1023 (63/616) | **0.1461** (90/616) | **+0.0438** | **+42.82%** |
| **Small Object F1 (16-32px)** | 0.1829 | **0.2344** | **+0.0515** | **+28.16%** |
| **Dense Fog Recall (DAWN)** | 0.4300 (43/100) | **0.6300** (63/100) | **+0.2000** | **+46.51%** |
| **Dense Fog F1 (DAWN)** | 0.5584 | **0.7000** | **+0.1416** | **+25.36%** |

*Methodological note:* In Mission 20 reporting, comparing zero-shot macro recall (0.3478) with fine-tuned operational micro recall (0.3762) mixed two distinct definitions. When evaluated consistently under Ultralytics mAP, the fine-tuned model yields a solid **+16.39% relative gain in mAP@0.5** and **+13.59% relative gain in mAP@0.5:0.95**. Under operational thresholds, overall recall remained steady while detection shifted selectively toward low-visibility and smaller scale targets.

---

## 3. Dense Fog Improvement: Relative vs. Percentage-Point Clarification

The reported "+46.5% gain" in dense fog detection represents **relative improvement**, not raw percentage points:

- **Zero-Shot Dense Fog Recall:** $43.0\%$ ($43 / 100$ ground-truth targets detected)
- **Fine-Tuned Dense Fog Recall:** $63.0\%$ ($63 / 100$ ground-truth targets detected)
- **Percentage-Point Improvement ($\Delta$):**
  $$63.0\% - 43.0\% = +20.00\text{ percentage points (pp)}$$
- **Relative Improvement:**
  $$\frac{63.0\% - 43.0\%}{43.0\%} = \frac{20.0\%}{43.0\%} = +46.51\%$$
- **Dense Fog F1 Gain:** $0.5584 \to 0.7000$ ($+14.16\text{ pp}$, or $+25.36\%$ relative gain).

Both figures are accurate; scientific presentations must report both the $+20.0\text{ pp}$ absolute delta and the $+46.5\%$ relative gain.

---

## 4. Audio Confusion Matrix and Derived Metrics

On the 300-sample locked test partition evaluated at calibrated threshold $\tau^* = 0.54$:

$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 139 & 11 \\ 11 & 139 \end{pmatrix}$$

- **Total Ground Truth Negatives ($N$):** $\text{TN} + \text{FP} = 139 + 11 = 150$
- **Total Ground Truth Positives ($P$):** $\text{FN} + \text{TP} = 11 + 139 = 150$
- **Accuracy:** $\frac{\text{TP} + \text{TN}}{P + N} = \frac{278}{300} = 92.67\%$
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{139}{150} = 92.67\%$
- **Recall (Sensitivity / TPR):** $\frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{139}{150} = 92.67\%$
- **Specificity (TNR):** $\frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{139}{150} = 92.67\%$
- **F1 Score:** $2 \times \frac{0.9267 \times 0.9267}{0.9267 + 0.9267} = 0.9267$
- **False Positive Rate (FPR):** $\frac{\text{FP}}{\text{TN} + \text{FP}} = \frac{11}{150} = 7.33\%$
- **False Negative Rate (FNR):** $\frac{\text{FN}}{\text{TP} + \text{FN}} = \frac{11}{150} = 7.33\%$
- **ROC-AUC:** $0.9706$ (computed across continuous probability outputs)

All derived calculations are strictly verified and self-consistent.

---

## 5. Audio SNR and Noise Breakdown

### 5.1 Noise Condition Distribution in Locked Test Set ($N=300$)
- **Clean Speech:** 200 samples (66.7%), Accuracy: 95.00%, F1: 0.9438
- **Strong Wind:** 29 samples (9.7%), Accuracy: 89.66%, F1: 0.9189
- **Drone Propeller Noise:** 25 samples (8.3%), Accuracy: 88.00%, F1: 0.9032
- **Heavy Rain Splatter:** 25 samples (8.3%), Accuracy: 84.00%, F1: 0.8750
- **Flood Water:** 21 samples (7.0%), Accuracy: 90.48%, F1: 0.9091

### 5.2 Calibrated SNR Distribution ($N=300$)
- **$\infty\text{ dB}$ (Clean):** 200 samples, Accuracy: 95.00%, F1: 0.9438
- **$20.0\text{ dB}$:** 19 samples, Accuracy: 94.74%, F1: 0.9600
- **$10.0\text{ dB}$:** 22 samples, Accuracy: 95.45%, F1: 0.9630
- **$5.0\text{ dB}$:** 21 samples, Accuracy: 95.24%, F1: 0.9655
- **$0.0\text{ dB}$:** 25 samples, Accuracy: 80.00%, F1: 0.8148 (Acoustic boundary)
- **$15.0\text{ dB}$:** 13 samples, Accuracy: 69.23%, F1: 0.7143 (Small-sample anomaly: 4 errors on 13 samples)

---

## 6. Dataset Provenance Audit: Collected vs. Augmented

### 6.1 Vision Test Set Provenance ($N=160$)
- **Authentic Real Camera Frames:** 160 (100.0%)
- **Augmented Derivative Frames:** 0 (0.0%)
- *Conclusion:* Vision test evaluation is completely free of synthetic/augmented derivative bias.

### 6.2 Audio Test Set Provenance ($N=300$)
- **Owner-Collected Originals:** 38 samples (12.67%)
- **Augmented Derivatives:** 262 samples (87.33%)
- **Unique Root Recordings Represented:** 207 roots
- *Conclusion:* While cross-split root isolation ensures that neither a root recording nor its derivative appears in multiple splits, 87.33% of test samples represent augmented derivatives (noise injection, SNR scaling, acoustic distance filtering) of these 207 parent recordings.

---

## 7. Audio Root Identifiers: Recordings vs. Speaker Identities

A critical distinction must be maintained between **source recordings** and **speaker identities**:

1. **What `root_source` establishes:**
   `root_source` (e.g., `v2_pos_0001`, `v2_neg_0012`) identifies the **parent acoustic recording/utterance**. The partitioning algorithm strictly enforces:
   $$\text{Roots}(\text{Train}) \cap \text{Roots}(\text{Val}) = \emptyset, \quad \text{Roots}(\text{Train}) \cap \text{Roots}(\text{Test}) = \emptyset, \quad \text{Roots}(\text{Val}) \cap \text{Roots}(\text{Test}) = \emptyset$$
   This guarantees that acoustic noise patterns, timing artifacts, or specific sentence tokens cannot leak across splits.
2. **What `root_source` does NOT establish:**
   `root_source` does **not** establish independent biological speakers.
   In `dataset_v2_500`, 500 parent utterances were synthesized across 6 distinct neural voice profiles (`bn-BD-NabanitaNeural`, `bn-BD-PradeepNeural`, `bn-IN-TanishaaNeural`, `bn-IN-BashkarNeural`, `Piya`, `gTTS_bn_standard`).
   Therefore, multiple distinct root recordings share the same underlying neural voice profile.
3. **Mission 21 Rule:**
   We will never fabricate or assume missing human speaker identities. In Phase 1 and Phase 3, evaluations will explicitly report performance stratified by:
   - Root-isolated recordings (745 roots total, 207 in test)
   - Underlying voice identities where available
   - Collected originals vs. augmented derivatives.

---

## 8. Statistical Sampling Units for Uncertainty Intervals

In previous reports, confidence intervals or standard errors were sometimes evaluated by resampling individual images or audio samples.

**Methodological Violation:** Treating individual frames from a driving video or augmented derivatives from a parent recording as independent and identically distributed (i.i.d.) violates exchangeability:
- Adjacent video frames share scene geometry, lighting, and weather conditions.
- Augmented audio clips sharing a root utterance share fundamental pitch, timbre, and speech envelope.

**Mission 21 Corrective Policy:**
- For **Vision**: Non-parametric bootstrap resampling must be conducted at the **sequence level** (cluster bootstrap over sequences/domains: `bdd_video_b24d283f`, `dawn_snow`, `bdd_video_b255cd6c`, `dawn_dense_fog`).
- For **Audio**: Non-parametric bootstrap resampling must be conducted at the **root recording cluster level** (cluster bootstrap over the 207 independent root recordings in the test set).
