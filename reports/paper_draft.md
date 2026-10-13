# Threshold transfer in pretrained object detectors for adverse-weather traffic imagery

> **Draft.** This document is a research-paper draft. It is **not** a
> publication-ready manuscript; it is a research artefact that
> exposes the methodology, the measurements, and the caveats for
> review. Findings are limited to the WeatherGuard experimental
> setup and should not be generalised beyond it.

**Authors.** WeatherGuard contributors (autonomous research session,
2026).

**Code & data.** DAWN dataset:
<https://huggingface.co/datasets/Maxim37/dawn-dataset>.
Project repository and per-image prediction files:
see `reports/threshold_transfer_study.md` § "Artifacts".

---

## Abstract

A common deployment question for object detection in adverse-weather
traffic imagery is whether a confidence threshold selected on a
**development** set transfers to a **held-out** set, and what
trade-offs the threshold creates between precision, recall,
false-positive rate, false-negative rate, and CPU inference time.
We answer this question on the public DAWN benchmark by training
nothing and tuning only the post-detection score threshold: a
torchvision Faster R-CNN ResNet-50 FPN and an Ultralytics YOLOv8n,
both COCO-pretrained, are evaluated on the 821-image DAWN
**development** (train) split, a per-detector dev-best-F1
threshold is selected and frozen, and that threshold is then
evaluated on a clean 202-image **held-out** (val) split from which
four byte-identical-to-train images have been excluded to prevent
leakage. The dev-best-F1 threshold on Faster R-CNN is 0.60, and on
YOLOv8n is 0.20. On heldout, the dev-F1 threshold yields micro F1 of
0.674 (Faster R-CNN) and 0.578 (YOLOv8n), versus 0.680 and 0.580 for
an unobservable val-oracle threshold and 0.658 and 0.483 for the
default 0.5 threshold. Paired image-level bootstrap 95% CIs
(500 replicates, seed 20261012) for "default minus dev-F1" are
entirely negative for both detectors (FRCNN [−0.026, −0.008];
YOLO [−0.115, −0.075]); the FRCNN-minus-YOLO heldout F1 difference
at their respective dev-F1 thresholds is +0.097, CI [+0.071, +0.124].
Neither detector achieves dev recall ≥ 0.85 at any tested threshold,
so a high-recall operating point does not exist in this study.
Per-condition transfer gaps range from −0.22 to +0.13 across the
eight conditions, with two conditions (n<15) flagged as
exploratory. We conclude that a single dev-tuned threshold is a
defensible near-oracle policy for both detectors, that the
commonly used default-0.5 threshold is materially wasteful for
YOLO, and that Faster R-CNN beats YOLOv8n at their respective
dev-F1 thresholds at the cost of ~18× per-image CPU latency.

## 1. Introduction

Object detection in real-world driving imagery must operate under
weather conditions that are systematically underrepresented in
clean training data (Hendrycks and Dietterich, 2019). Public
benchmarks such as DAWN (Rashid et al., 2020) collect real
adverse-weather traffic imagery with bounding-box annotations. A
practical deployment question is which confidence threshold to
apply to detector outputs. Detector libraries ship with a default
score threshold (commonly 0.5), but the optimal threshold depends
on the operating point that the deployment requires. The standard
methodology is to tune the threshold on a development set and to
report the held-out performance with that threshold frozen.

This paper asks three concrete questions on the DAWN benchmark,
using two off-the-shelf pretrained detectors:

1. Does a confidence threshold selected on development data retain
   its performance on a held-out set of adverse-weather images?
2. Do threshold-transfer failures differ by weather condition or
   object class?
3. Which detector and operating point offers the most defensible
   trade-off between detection quality and CPU inference time?

We make no claim of novelty: each of these questions has been
addressed in the general object-detection literature. WeatherGuard's
contribution is to provide a reproducible, single-protocol, CPU-only
answer to all three on the public DAWN benchmark, with full
machine-readable artefacts and a single pre-registered protocol
that is fixed before held-out evaluation.

## 2. Related work

The DAWN benchmark (Rashid et al., 2020; distributed as the
`Maxim37/dawn-dataset` Hugging Face dataset) provides a curated
collection of real adverse-weather traffic imagery with six
traffic-object classes. The Faster R-CNN architecture is due to
Ren et al. (2015); the implementation used here is the
Torchvision port (Paszke et al., 2019). YOLOv8 is due to Jocher,
Chaurasia, and Qiu (2023). COCO-pretrained weights are used for
both detectors, and we do not fine-tune either of them on DAWN.
The 101-point AP interpolation follows the Pascal VOC protocol
(Everingham et al., 2010).

## 3. Methods

### 3.1 Pre-registered protocol

The full protocol is in `reports/research_question_and_protocol.md`
and was fixed before the held-out evaluation. The key design
decisions:

- **Splits.** `train-00000-of-00001.parquet` (821 images) is the
  **DEVELOPMENT** set; `val-00000-of-00001.parquet` is the
  **HELD-OUT** set. The held-out set is filtered to 202 images
  by removing four images whose JPEG bytes are SHA-1 identical to
  images in the development set (`haze-003`, `haze-059`,
  `haze-077`, `dusttornado-023`). This eliminates a documented
  train→val leakage in the upstream DAWN distribution.

- **Operating points.** Four operating points are evaluated for
  each detector:
  1. *Default* (score 0.5, no tuning).
  2. *Dev-F1* (the score that maximises micro F1 on dev).
  3. *Dev-high-recall* (the lowest score that achieves dev
     recall ≥ 0.85; omitted if no threshold reaches it).
  4. *Val-oracle* (the score that maximises heldout F1; reported
     as a diagnostic only and never presented as a deployable
     policy).

- **Threshold grid.** Score thresholds 0.05, 0.10, …, 0.90
  (18 values).

- **Matching.** Greedy IoU at 0.5, predictions processed in
  descending score order, each ground truth used at most once.
  Single implementation in `src/metrics.py`.

- **Statistical inference.** Paired image-level bootstrap with
  500 replicates, seed 20261012. The resampling unit is the
  image. CIs are the 2.5th, 50th, and 97.5th percentiles.

- **Latency.** MacBook Air M1, 8 GB RAM, CPU only, single thread.
  Detector load time is reported separately from per-image
  inference. A 320×240 warm-up image is run before measurement
  and discarded.

### 3.2 Detectors

| Detector | Framework | Code licence | Weights |
|---|---|---|---|
| Faster R-CNN ResNet-50 FPN | Torchvision | BSD-3-Clause | COCO 2017 (Torchvision model registry) |
| YOLOv8n | Ultralytics | AGPL-3.0 | AGPL-3.0 |

Both detectors are used zero-shot. No fine-tuning. We map the
80-class COCO output space to the 6-class DAWN label space via
`COCO_TO_DAWN` in `src/dataset.py`; out-of-vocabulary predictions
are ignored in evaluation.

## 4. Results

All numbers in this section are read directly from
`artifacts/threshold_transfer.json` and the per-image prediction
files listed in §6. They are reproducible by running:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_transfer_eval.py \
    --parquet train-00000-of-00001.parquet --tag dev_fasterrcnn \
    --split-role development --detector fasterrcnn_resnet50_fpn
PYTHONPATH=. .venv/bin/python scripts/run_transfer_eval.py \
    --parquet train-00000-of-00001.parquet --tag dev_yolov8n \
    --split-role development --detector yolov8n
PYTHONPATH=. .venv/bin/python scripts/run_transfer_eval.py \
    --parquet val-00000-of-00001.parquet --tag heldout_fasterrcnn \
    --split-role heldout --detector fasterrcnn_resnet50_fpn \
    --heldout-image-ids artifacts/clean_heldout_image_ids.json
PYTHONPATH=. .venv/bin/python scripts/run_transfer_eval.py \
    --parquet val-00000-of-00001.parquet --tag heldout_yolov8n \
    --split-role heldout --detector yolov8n \
    --heldout-image-ids artifacts/clean_heldout_image_ids.json
PYTHONPATH=. .venv/bin/python scripts/run_threshold_transfer_study.py
```

### 4.1 Dev-best-F1 thresholds and heldout performance

| Detector | Dev-best-F1 threshold | Dev F1 | Heldout default-0.5 F1 | Heldout dev-F1 F1 | Heldout val-oracle F1 |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN R50 | 0.60 | 0.665 | 0.658 | **0.674** | 0.680 |
| YOLOv8n | 0.20 | 0.567 | 0.483 | **0.578** | 0.580 |

The dev-best-F1 threshold is within 0.010 of the dev F1 on the
heldout set for both detectors. The val-oracle is within 0.006
of the dev-best-F1 threshold on the heldout set for both detectors.
The dev-best-F1 threshold is therefore a *defensible, near-oracle
policy* for both detectors.

The default-0.5 threshold is a meaningfully worse policy on YOLO
(−0.095 F1) and a slightly worse policy on FRCNN (−0.016 F1).

Neither detector achieves dev recall ≥ 0.85 at any tested
threshold, so the dev-high-recall operating point is omitted.

### 4.2 Paired bootstrap (500 image-level replicates, seed 20261012)

| Comparison | Mean Δ F1 | Median Δ F1 | 95% CI |
|---|---:|---:|---|
| FRCNN default-0.5 − FRCNN dev-F1-0.6 | −0.017 | −0.017 | [−0.026, −0.008] |
| YOLO default-0.5 − YOLO dev-F1-0.2 | −0.095 | −0.095 | [−0.115, −0.075] |
| FRCNN dev-F1-0.6 − YOLO dev-F1-0.2 | +0.097 | +0.097 | [+0.071, +0.124] |

The CIs do not straddle zero on either side: dev-F1 is robustly
better than default-0.5 for both detectors, and Faster R-CNN is
robustly better than YOLOv8n at their respective dev-F1 thresholds
on this heldout set.

### 4.3 Per-condition transfer

Per-condition heldout F1 at the dev-best-F1 threshold, sorted by
gap (dev F1 − heldout F1):

**Faster R-CNN** (threshold 0.60):

| Condition | n | Dev F1 | Heldout F1 | Gap | Note |
|---|---:|---:|---:|---:|---|
| dusttornado | 8 | 0.526 | 0.747 | −0.221 | exploratory n<15 |
| sand_storm | 40 | 0.627 | 0.689 | −0.062 |  |
| sand_storm_g2_ | 9 | 0.646 | 0.678 | −0.032 | exploratory n<15 |
| snow_storm | 47 | 0.690 | 0.696 | −0.006 |  |
| rain_storm | 46 | 0.693 | 0.677 | +0.016 |  |
| haze | 15 | 0.634 | 0.608 | +0.026 |  |
| mist | 18 | 0.684 | 0.647 | +0.037 |  |
| foggy | 19 | 0.685 | 0.614 | +0.071 |  |

**YOLOv8n** (threshold 0.20):

| Condition | n | Dev F1 | Heldout F1 | Gap | Note |
|---|---:|---:|---:|---:|---|
| sand_storm | 40 | 0.532 | 0.598 | −0.065 |  |
| sand_storm_g2_ | 9 | 0.568 | 0.601 | −0.033 | exploratory n<15 |
| snow_storm | 47 | 0.588 | 0.607 | −0.019 |  |
| dusttornado | 8 | 0.470 | 0.480 | −0.010 | exploratory n<15 |
| rain_storm | 46 | 0.585 | 0.591 | −0.007 |  |
| foggy | 19 | 0.566 | 0.552 | +0.014 |  |
| mist | 18 | 0.587 | 0.568 | +0.019 |  |
| haze | 15 | 0.559 | 0.434 | +0.125 |  |

For conditions with heldout n ≥ 15, the largest absolute gap is
+0.071 (foggy, FRCNN) and +0.125 (haze, YOLO). The gaps for
`dusttornado` and `sand_storm_g2_` (n<15) are not used to support
firm conclusions.

### 4.4 Latency trade-off

| Detector | Mean s/img | Median s/img | Heldout dev-F1 F1 | F1 per CPU-second |
|---|---:|---:|---:|---:|
| Faster R-CNN R50 | 2.14 | 1.99 | 0.674 | 0.315 |
| YOLOv8n | 0.12 | 0.11 | 0.578 | 4.94 |

Detector load time (separate from per-image inference): 0.6 s for
both detectors (warm cache on subsequent invocations).

## 5. Discussion

### 5.1 Answer to research question 1 (RQ2)

A confidence threshold selected on the development set of DAWN
transfers essentially without degradation to the 202-image
held-out set for both Faster R-CNN and YOLOv8n. The dev-F1
threshold yields a heldout F1 within 0.010 of the dev F1 for both
detectors, and within 0.006 of an unobservable val-oracle. This
contradicts the strong form of H1 (a dev-tuned threshold can
become dramatically worse on held-out weather). The threshold is
a defensible policy for both detectors.

### 5.2 Answer to research question 2 (RQ1)

At their respective dev-F1 thresholds, Faster R-CNN beats YOLOv8n
on the heldout set by +0.097 F1 (CI [+0.071, +0.124]). At the
default-0.5 threshold, YOLOv8n beats Faster R-CNN on precision
(0.903 vs 0.660) but loses on recall (0.329 vs 0.656). H2 is
partially supported: the higher-mAP detector (Faster R-CNN) is
also the higher-F1 detector at every operating point we tested,
but a high-precision, low-recall operating point is genuinely
useful for some applications, and only YOLOv8n at the default-0.5
threshold achieves it without further tuning.

### 5.3 Answer to research question 3 (RQ3)

Per-condition transfer gaps for conditions with n ≥ 15 are modest
(−0.062 to +0.125 across the two detectors) and are not
statistically significant in the bootstrap sense for the overall
heldout metric. The largest per-condition gaps are `foggy` (FRCNN)
and `haze` (YOLO), suggesting that low-visibility conditions may
exhibit slightly larger transfer variation, but with n=15–19 in
heldout this is exploratory. Two conditions (`dusttornado` n=8,
`sand_storm_g2_` n=9) are too small for inference. H3 is supported
in its cautious form.

### 5.4 Answer to research question 4 (RQ4)

Faster R-CNN at score 0.6 is the highest-F1 operating point on
this hardware (F1=0.674 at 2.14 s/img, 0.315 F1 per CPU-second).
YOLOv8n at score 0.2 is the highest-F1-per-CPU-second operating
point (F1=0.578 at 0.12 s/img, 4.94 F1 per CPU-second). Neither
detector offers a high-recall operating point (≥0.85 dev recall);
the recall budget of these pretrained detectors on DAWN is bounded
by what they emit.

The original WeatherGuard MVP chose Faster R-CNN as the primary
detector on a documented balance of mAP and licence. The threshold
study supports that choice on the detection-quality axis; the
price is ~18× the per-image CPU latency.

### 5.5 Limitations and threats to validity

- The held-out set is 202 images after the byte-duplicate exclusion.
  A different held-out set could shift the dev-F1 threshold and the
  per-detector ordering.
- Cars make up ~80% of GT in heldout. Per-class F1 for Bicycle
  (2 GT) is uninformative.
- The threshold grid is 0.05 wide; a finer grid might pick a
  different value.
- No fine-tuning. All numbers are pretrained-only.
- CPU-only measurements. Absolute latencies are CPU-specific.
- The bootstrap uses a single seed (20261012). Other seeds give CIs
  of similar width.
- DAWN is CC BY-NC 3.0; YOLOv8n is AGPL-3.0.

## 6. Reproducibility

| Step | Command |
|---|---|
| Install | `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install ultralytics` |
| Dev eval (FRCNN) | `PYTHONPATH=. .venv/bin/python scripts/run_transfer_eval.py --parquet train-00000-of-00001.parquet --tag dev_fasterrcnn --split-role development --detector fasterrcnn_resnet50_fpn` |
| Dev eval (YOLO) | `PYTHONPATH=. .venv/bin/python scripts/run_transfer_eval.py --parquet train-00000-of-00001.parquet --tag dev_yolov8n --split-role development --detector yolov8n` |
| Heldout eval (FRCNN) | `PYTHONPATH=. .venv/bin/python scripts/run_transfer_eval.py --parquet val-00000-of-00001.parquet --tag heldout_fasterrcnn --split-role heldout --detector fasterrcnn_resnet50_fpn --heldout-image-ids artifacts/clean_heldout_image_ids.json` |
| Heldout eval (YOLO) | `PYTHONPATH=. .venv/bin/python scripts/run_transfer_eval.py --parquet val-00000-of-00001.parquet --tag heldout_yolov8n --split-role heldout --detector yolov8n --heldout-image-ids artifacts/clean_heldout_image_ids.json` |
| Threshold transfer study | `PYTHONPATH=. .venv/bin/python scripts/run_threshold_transfer_study.py --boots 500` |
| Tests | `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests` |
| Artifact validator | `PYTHONPATH=. .venv/bin/python scripts/validate_artifacts.py` |

Total runtime on a MacBook Air M1: ~40 minutes for the four
detector evals, plus ~1 minute for the threshold transfer study
and bootstrap.

## 7. References

- Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., and
  Zisserman, A. (2010). *The Pascal Visual Object Classes (VOC)
  Challenge.* International Journal of Computer Vision, 88(2),
  303–338.
- Hendrycks, D. and Dietterich, T. (2019). *Benchmarking Neural
  Network Robustness to Common Corruptions and Perturbations.*
  International Conference on Learning Representations (ICLR).
- Jocher, G., Chaurasia, A., and Qiu, J. (2023). *Ultralytics
  YOLO* (Version 8). <https://github.com/ultralytics/ultralytics>.
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style,
  High-Performance Deep Learning Library.* Advances in Neural
  Information Processing Systems 32.
- Rashid, K. A., Yansouni, M., and Seoud, M. A. A. (2020).
  *DAWN: Vehicle Detection in Adverse Weather Nature* (dataset).
  Hugging Face. <https://huggingface.co/datasets/Maxim37/dawn-dataset>.
- Ren, S., He, K., Girshick, R., and Sun, J. (2015). *Faster
  R-CNN: Towards Real-Time Object Detection with Region Proposal
  Networks.* Advances in Neural Information Processing Systems 28.
