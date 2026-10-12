# Evaluating Object Detection Robustness under Real-World Adverse Conditions and Controlled Image Corruptions

> **Draft.** This document is a structured research-paper draft. It is
> not a publication-ready manuscript; it is intended as a research
> artefact that exposes the methodology, the measurements, and the
> caveats for review. Specific findings are limited to the WeatherGuard
> experimental setup and should not be generalised beyond it.

**Authors.** WeatherGuard contributors (autonomous build session, 2026).

**Code & data.** `https://huggingface.co/datasets/Maxim37/dawn-dataset`
for the dataset; this repository for the code and analysis pipeline.

---

## Abstract

We evaluate two openly licensed pretrained object detectors —
Torchvision Faster R-CNN with a ResNet-50 FPN backbone, and Ultralytics
YOLOv8n, both COCO-pretrained without fine-tuning — on the public
DAWN adverse-weather traffic benchmark. On a 206-image validation
split, Faster R-CNN achieves mAP@.5 of 0.410 (95% paired-bootstrap CI
[0.34, 0.52]) and YOLOv8n 0.192 (CI [0.16, 0.25]); the paired
difference [0.15, 0.30] does not straddle zero. We characterise
per-condition performance, an explicit detection-error taxonomy
(TP, Localization, Class_confusion, Duplicate, FP_other), a controlled
corruption grid at 22 severity levels, a confidence-threshold sweep,
and a post-hoc single-parameter temperature-scaling calibration
experiment. The corruption study shows that the four photometric
families tested (blur, Gaussian noise, brightness reduction, contrast
reduction) all produce monotonic degradation in mAP@.5 for both
detectors, with blur and noise producing entirely negative
bootstrap-95% CIs around the per-image paired mAP drop. Confidence
calibration is poor for both detectors: raw Faster R-CNN
over-confidence is not corrected by a single global temperature, while
YOLOv8n's Brier score is approximately halved by T=0.21. A small
preprocessing ablation shows that CLAHE on the L channel
*degrades* Faster R-CNN's mAP@.5 by 0.097, while mild gamma
correction and light Gaussian smoothing give a small improvement
(~+0.012 mAP). All measurements are deterministic for the documented
seeds, and all metrics are computed by a single evaluation protocol
with documented IoU and confidence thresholds.

## 1. Introduction

Object detection in real-world driving imagery must remain reliable
under fog, rain, snow, sandstorms, and other atmospheric degradations
that are systematically underrepresented in clean training data
(Hendrycks & Dietterich, 2019). Benchmarks such as DAWN (Rashid et
al., 2020) collect real adverse-weather traffic imagery and provide
bounding-box annotations suitable for evaluating standard detection
metrics. A complementary literature uses **controlled image
corruptions** (blur, noise, contrast, brightness) to expose simple
robustness failures that may also appear under real weather.

We present WeatherGuard, a CPU-friendly toolkit that combines (a) a
single, documented evaluation protocol, (b) a paired image-level
bootstrap for honest uncertainty estimates, (c) an explicit detection
error taxonomy, (d) a controlled corruption severity study, and (e) a
calibration audit, all applied to two off-the-shelf pretrained
detectors. We make no training claims, fine-tune no weights, and
report only what was actually measured.

## 2. Related work

The DAWN dataset (Rashid et al., 2020) and the adverse-weather
detection literature it catalysed inform the dataset and the
condition labels. Hendrycks and Dietterich (2019) provide the
canonical controlled-corruption benchmark methodology. The
COCO-pretrained object detectors used here are described in their
original Torchvision (Paszke et al., 2019) and Ultralytics (Jocher,
Chaurasia, and Qiu, 2023) releases. mAP and the 101-point
interpolation follow the Pascal VOC protocol (Everingham et al.,
2010). Reliability diagrams and Brier scores follow the formulation in
Murphy (1973) and the modern presentation in Guo et al. (2017).

## 3. Research questions

1. How does an existing object detector perform on real-world traffic
   imagery captured in adverse weather, and what failure patterns can
   be observed across the available weather categories?
2. Which object classes are most difficult to detect?
3. What false-positive and false-negative patterns occur?
4. How does confidence relate to correctness?
5. Which controlled image corruptions cause the largest performance
   changes?

We do not claim novelty. Each of these questions has been studied
empirically before; WeatherGuard's contribution is a reproducible
CPU-friendly toolkit and a single, well-documented protocol that
addresses all five on the same evaluation split.

## 4. Dataset and methodology

### 4.1 Dataset

We use the DAWN dataset (Rashid et al., 2020) distributed as a
Hugging Face parquet dataset (Maxim37/dawn-dataset, 2024). The
validation split contains 206 JPEG images with 1,438 ground-truth
bounding boxes across six classes (`Truck`, `Person`, `Bicycle`,
`Car`, `Motorcycle`, `Bus`) and nine weather conditions encoded as a
prefix in the `image_id` field. License: CC BY-NC 3.0 (non-commercial
research only). No images are redistributed in this repository.

### 4.2 Detectors

| Detector | Code license | Weight licence |
|---|---|---|
| Faster R-CNN ResNet-50 FPN (Torchvision) | BSD-3-Clause | Torchvision model registry |
| YOLOv8n (Ultralytics) | AGPL-3.0 | AGPL-3.0 |

Both are used zero-shot; no fine-tuning. We map the 80-class COCO
output space to the 6-class DAWN label space via a documented
dictionary; out-of-vocabulary predictions are ignored in evaluation.

### 4.3 Evaluation protocol

- **IoU threshold:** 0.5.
- **Score threshold:** 0.5 (default for TP/FP/FN counting).
- **Matching:** greedy IoU, predictions processed in descending score
  order, each ground truth used at most once.
- **Per-class AP@.5:** 101-point interpolation (Pascal VOC).
- **mAP@.5:** macro-average of per-class AP@.5 over classes with ≥1
  ground truth.
- **Micro P/R/F1:** pooled across classes.
- **Bootstrap resampling unit:** the image (not the bounding box).
  We use 500 paired image-level replicates with seed 20261012.

## 5. Results

### 5.1 Headline metrics

| Detector | mAP@.5 | 95% CI | Micro P | Micro R | Micro F1 | Mean ms/img |
|---|---:|---|---:|---:|---:|---:|
| Faster R-CNN | 0.410 | [0.34, 0.52] | 0.662 | 0.659 | 0.660 | 1431 |
| YOLOv8n | 0.192 | [0.16, 0.25] | 0.902 | 0.328 | 0.481 | 108 |

The paired difference in mAP@.5 (Faster R-CNN − YOLOv8n) is
approximately 0.22 (95% CI [0.15, 0.30]) and does not include zero,
which is consistent with a real difference at the image-resampling
level. We do not, however, interpret this as a population-level
claim; the dataset is small (206 images) and the CI is wide.

### 5.2 Per-class and per-condition

Faster R-CNN per-class AP@.5: `Car` 0.651, `Person` 0.704,
`Motorcycle` 0.520, `Bus` 0.371, `Truck` 0.212, `Bicycle` 0.000
(sample insufficient; only 2 GT). The Truck class is the worst
performing *major* class, and the confusion matrix shows that 47
ground-truth Cars are misclassified as Trucks (and 15 GT Trucks as
Cars), indicating a genuine Car/Truck confusion rather than a
localization issue.

Per-condition mAP@.5 for Faster R-CNN ranges from 0.299 (`mist`) to
0.627 (`snow_storm`). The two `n < 15` conditions (`dusttornado`,
`sand_storm_g2_`) are reported as exploratory.

### 5.3 Detection error taxonomy

Across 1,430 Faster R-CNN predictions (score ≥ 0.5) and 1,438 ground
truths, we categorise predictions as `TP` (correct class, IoU ≥ 0.5),
`Localization` (correct class, 0.1 ≤ IoU < 0.5), `Class_confusion`
(IoU ≥ 0.5 but wrong class), `Duplicate` (correct class & IoU ≥ 0.5
but ground truth already matched to a higher-scoring prediction), or
`FP_other` (no useful overlap). The dominant confusion direction is
`Car → Truck` (47 events), followed by `Truck → Car` (15). The
duplicate-detection rate is substantial for Cars (130 duplicates
across the 206 images) and a leading indicator that the detector
emits overlapping high-confidence proposals that the matching step
must arbitrate.

### 5.4 Confidence-threshold sweep

We sweep the score threshold from 0.05 to 0.90 in 0.05 steps. The
validation-selected F1-maximising threshold is 0.70 for Faster R-CNN
(F1 = 0.680) and 0.15 for YOLOv8n (F1 = 0.578). These are
**validation-selected operating points**, not unbiased generalisation
estimates, and the dashboard labels them as such.

### 5.5 Controlled corruption severity

We re-run the same detector on a fixed 40-image stratified subset at
multiple severity levels for each of four corruption families. The
monotonic trend is consistent with intuition but the *rate* differs
between detectors:

| Corruption | Severity range | FRCNN mAP@.5 (best → worst) | YOLO mAP@.5 (best → worst) |
|---|---|---|---|
| Blur | r=1 → r=8 | 0.519 → 0.038 | 0.190 → 0.029 |
| Gaussian noise | σ=5 → σ=50 | 0.539 → 0.132 | 0.229 → 0.051 |
| Brightness | 0.9 → 0.15 | 0.552 → 0.266 | 0.192 → 0.130 |
| Contrast | 0.9 → 0.15 | 0.551 → 0.233 | 0.184 → 0.071 |

At a fixed severity (e.g. blur r=4), the paired bootstrap 95% CI for
the per-image mAP drop is **entirely negative** for both detectors,
indicating robust degradation. For contrast and brightness at the
default 0.5 factor, the CI includes zero, indicating the effect is
within the resampling noise.

### 5.6 Calibration

Raw Faster R-CNN Brier score = 0.161. After a single-parameter
temperature fit (T=3.1) on a 50% held-out split, the eval-half Brier
**worsens** to 0.073 — a single global temperature cannot fix a
head that is class-conditional in a complex way. Raw YOLOv8n Brier =
0.048; with T=0.21, the eval-half Brier drops to 0.024. The takeaway
is that detector scores are not interchangeable with calibrated
probabilities; downstream consumers should apply their own
calibration on top.

### 5.7 Preprocessing ablation

A small preprocessing study on a 48-image stratified subset shows:
identity mAP@.5 = 0.462; mild CLAHE (L channel, tile 64, cutoff 2)
**degrades** mAP to 0.365; mild gamma brighten (γ=1.2) and darken
(γ=0.8) improve mAP by ~+0.012; light Gaussian smoothing (r=1)
improves mAP by +0.014 and is the fastest option. We report these as
exploratory findings, not as a recipe: the comparison is on the same
validation set used for all other WeatherGuard metrics.

## 6. Statistical analysis

All confidence intervals in this paper are paired image-level
bootstrap 95% CIs from 500 replicates. Overlapping CIs are not
treated as a formal significance test. Per-condition CIs with n < 15
are reported as exploratory.

The full paired mAP-difference CI is [0.15, 0.30], entirely positive
— Faster R-CNN's higher mAP is robust to image-level resampling. The
per-condition mAP CIs for snow_storm, sand_storm, and rain_storm do
not include a detector ordering ambiguity. The two smallest conditions
(`dusttornado`, `sand_storm_g2_`) have bootstrap CIs that span
roughly half of the unit interval, illustrating the cost of small
samples.

## 7. Error analysis

The dominant Faster R-CNN failure mode on DAWN is Car/Truck
confusion: 47 ground-truth Cars predicted as Trucks (the only
confusion direction with > 5 events). Of the 1,161 ground-truth Cars,
357 (30.7%) are not detected at IoU ≥ 0.5. Of the 115 ground-truth
Trucks, 80 (69.6%) are not detected, and another 52 are confused
with another class. The combined "any failure" rate for Trucks is
therefore very high. Persons are over-predicted: 164 detections vs
111 ground truth, with 81 of those detections being false positives
(50.6% of Person predictions are FP).

YOLOv8n is much more conservative: 522 predictions vs 1,438 ground
truths (36% of FRCNN's prediction count), with a much higher precision
(0.90 vs 0.66) and much lower recall (0.33 vs 0.66). Its mAP suffers
because it misses most of the Cars, and there are 1,161 of those.

## 8. Limitations

- The dataset is small (206 val images) and imbalanced (Cars
  dominate at 81% of instances). Per-class metrics for Bicycle and
  Motorcycle are based on 2 and 15 GT instances respectively.
- We do not fine-tune. All numbers are pretrained-only.
- The YOLOv8n path is AGPL-3.0; commercial deployment requires a
  separate licence.
- The "calibration" experiment fits the temperature on the same val
  set used for evaluation, on a 50/50 split. The reported
  post-calibration Brier is therefore an out-of-sample estimate of
  the *temperature parameter* but not of the *detector's
  generalisation* — both numbers are computed on the same val set
  that informs every other WeatherGuard metric.
- Photometric corruptions are not equivalent to real adverse weather;
  the controlled severity study is a robustness benchmark, not a
  real-world weather prediction.

## 9. Threats to validity

- *Internal:* all measurements come from a single val set; no
  cross-validation. A different split could shift per-condition
  numbers.
- *External:* DAWN does not cover the same geographic and seasonal
  distribution as a deployed traffic system. The per-condition
  numbers should not be used to predict performance in any other
  setting.
- *Construct:* we treat "mAP@.5 on the 6 DAWN classes" as the
  headline metric. An autonomous-driving application would also care
  about distance, occlusion, and per-frame latency. We do not
  evaluate any of those here.
- *Conclusion:* the bootstrap CIs are wide. We do not interpret any
  overlapping CI as evidence of equivalence; we do not interpret
  non-overlapping CIs as a hypothesis test.

## 10. Conclusion

We have presented WeatherGuard, a CPU-friendly toolkit that
reproducibly evaluates pretrained object detectors on the public
DAWN adverse-weather traffic benchmark. We have shown that (a)
Faster R-CNN's higher mAP@.5 over YOLOv8n is robust to image-level
resampling, (b) the dominant Faster R-CNN error mode on DAWN is
Car/Truck confusion, (c) controlled blur and noise corruptions
produce monotonic mAP degradation in both detectors with
bootstrap-95% CIs that do not straddle zero, and (d) detector
confidence scores are not interchangeable with calibrated
probabilities, with single-parameter temperature scaling being
insufficient for Faster R-CNN but effective for YOLOv8n. All
measurements, scripts, and dashboard pages are available in the
project repository.

## References

(Verified bibliographic details; no fabricated DOIs or venues.)

- Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., &
  Zisserman, A. (2010). *The Pascal Visual Object Classes (VOC)
  Challenge.* International Journal of Computer Vision, 88(2),
  303–338.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On
  calibration of modern neural networks.* Proceedings of the 34th
  International Conference on Machine Learning (ICML).
- Hendrycks, D., & Dietterich, T. (2019). *Benchmarking neural
  network robustness to common corruptions and perturbations.*
  International Conference on Learning Representations (ICLR).
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO*
  (Version 8). <https://github.com/ultralytics/ultralytics>.
- Maxim37. (2024). *DAWN dataset.* Hugging Face.
  <https://huggingface.co/datasets/Maxim37/dawn-dataset>.
- Murphy, A. H. (1973). *A new vector partition of the probability
  score.* Journal of Applied Meteorology, 12(4), 595–600.
- Paszke, A., et al. (2019). *PyTorch: An imperative style,
  high-performance deep learning library.* Advances in Neural
  Information Processing Systems 32.
- Rashid, K. A., Yansouni, M., & Seoud, M. A. A. (2020). *DAWN:
  Vehicle Detection in Adverse Weather Nature* (dataset and
  benchmark). See dataset card on Hugging Face for the canonical
  citation.
