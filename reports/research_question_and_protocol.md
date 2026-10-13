# Research question and protocol — WeatherGuard focused study

**Status:** This document is pre-registered. It states the research
question, hypotheses, data split strategy, metrics, and leakage
safeguards **before** the experiment is run. It is not modified after
results are known.

---

## 1. Primary research question

> **How well do confidence thresholds selected on development data
> transfer to held-out adverse-weather traffic images, and what
> trade-offs do they create between detection quality, missed objects,
> false alarms, and inference speed?**

## 2. Sub-questions

- **RQ1.** How do Faster R-CNN (Torchvision, ResNet-50 FPN, COCO
  pretrained) and YOLOv8n (Ultralytics, COCO pretrained) compare when
  evaluated under the same held-out protocol?
- **RQ2.** Does a confidence threshold selected on development data
  retain its performance on held-out images?
- **RQ3.** Do threshold-transfer failures differ by weather condition
  or object class, and how certain are those differences?
- **RQ4.** Which detector and operating point offers the most
  defensible trade-off between detection quality and CPU inference
  time?

## 3. Hypotheses (to test, not to assume)

- **H1.** A threshold optimized on development data may not remain
  optimal on held-out weather conditions. We expect the validation
  F1 of the dev-selected threshold to be *no better* than a fixed
  default (score = 0.5) and *worse* than a validation-set oracle.
- **H2.** The detector with higher mAP may not be the best option
  for every precision/recall operating requirement. Faster R-CNN has
  higher mAP on the existing val eval; YOLOv8n is faster and has
  higher precision at low thresholds. The question is which trade-off
  is more useful.
- **H3.** Threshold transfer and error rates may differ across
  weather conditions and object classes, but the small per-class and
  per-condition sample sizes (e.g. Bicycle: 2 GT; dusttornado: 9
  images) may prevent firm conclusions. We will report per-class and
  per-condition numbers with explicit sample-size caveats.

## 4. Data split strategy

### 4.1 Splits as published by DAWN

| Split | Images | Purpose in this study |
|---|---:|---|
| `train-00000-of-00001.parquet` | 821 | **Development** (DEVELOPMENT). Used for any threshold selection. Untouched. |
| `val-00000-of-00001.parquet`   | 206 | **Held-out** (HELD-OUT). Used **only** for the final evaluation of the locked dev-selected threshold. |

We treat the published DAWN `val` split as held-out because the
research question requires a clean held-out set. This is the standard
split that comes with the dataset; we are not creating a new split.

### 4.2 Leakage audit and remediation

We computed SHA-1 hashes of every image's raw bytes and found that
**4 of the 206 val images are byte-identical to images in the
training split** but carry different `image_id` strings:

| Val image_id | Train image_id with identical bytes |
|---|---|
| `haze-059` | `haze-036` |
| `haze-003` | `foggy-011` |
| `dusttornado-023` | `sand_storm-057` |
| `haze-077` | `haze-098` |

These are likely duplicate files in the upstream DAWN distribution
with mismatched metadata. To prevent train → val leakage in the
threshold-transfer experiment, we **exclude** these 4 images from the
held-out evaluation, leaving a clean held-out set of 202 images.

The 4 excluded images are reported in the results but not used to
score the locked dev threshold. This is the single modification we
make to the published val split.

### 4.3 Why train, not val, for threshold selection

Threshold selection is a method-tuning operation. We deliberately do
it on `train` (DEVELOPMENT) and freeze the result before evaluating
`val` (HELD-OUT). This makes the held-out evaluation a true
out-of-sample test of the policy.

If the dev split were too small to support threshold selection, we
would either (a) use a documented dev/val re-split with a fixed
random seed and re-budget, or (b) document the limitation and report
the dev-only results as exploratory. We do **not** substitute a
different protocol silently.

## 5. Evaluation protocol (frozen before running)

### 5.1 Matching and metrics

- **IoU threshold:** 0.5.
- **Score threshold:** 0.5 by default; the dev-selected threshold and
  the high-recall threshold are evaluated separately (see §5.4).
- **Matching:** greedy IoU, predictions processed in descending
  score order, each ground truth used at most once.
- **Per-class AP@.5:** 101-point interpolation (Pascal VOC). Classes
  with no ground truth in the evaluation set contribute NaN and are
  excluded from the mAP average.
- **mAP@.5:** macro-average of per-class AP@.5 over classes with ≥1
  ground truth.
- **Micro P/R/F1:** pooled across classes.
- **Out-of-DAWN-vocabulary predictions** (e.g. `potted plant`,
  `bench`) are **ignored** in evaluation, not counted as FP. This
  matches the WeatherGuard protocol in `src/metrics.py`.

### 5.2 Operating-point policy

For each detector, we define three operating points:

1. **Default (0.5).** Score threshold 0.5, no tuning.
2. **Dev-selected F1.** Score threshold that maximises development
   micro F1. Selected on the train split, frozen, then evaluated on
   val.
3. **High-recall (target recall ≥ 0.85 on dev).** If multiple
   thresholds achieve ≥ 0.85 recall on dev, we take the one with
   the highest precision. If no threshold reaches 0.85, we use the
   threshold that maximises recall. Reported only if dev recall
   reaches 0.85 for at least one threshold; otherwise the
   high-recall point is omitted with a documented reason.

We also report a **diagnostic val-oracle F1 threshold** that
maximises validation F1, but we label it as diagnostic only. The
oracle is **never** presented as a deployable policy.

### 5.3 Threshold grid

We sweep score thresholds from 0.05 to 0.90 in 0.05 steps (18
thresholds). The same grid is used for both detectors and for both
splits. The grid is declared here; the dev sweep is run on
`train-00000-of-00001.parquet` and the held-out sweep is run on the
202-image `val-...-clean.parquet`.

### 5.4 Sample-size disclosure

- Per-class: Bicycle has only 2 GT in val and ~24 in train; AP and
  F1 for this class are flagged as **uninformative** (sample
  insufficient for inference).
- Per-condition: `dusttornado` has 9 val images and 27 train images;
  `sand_storm_g2_` has 9 val and 44 train. We report them but
  mark any per-condition claim with n < 15 as exploratory.

### 5.5 Statistical inference

- **Paired image-level bootstrap** at the held-out stage to
  characterise uncertainty in the difference between (a) two
  operating points on the same detector and (b) the two detectors at
  the same operating point.
- **Replicates:** 500.
- **Seed:** 20261012.
- **Resampling unit:** the image (boxes within an image are not
  independent).
- We do not interpret overlapping CIs as a significance test. We
  state which CIs do and do not straddle the relevant reference
  point (zero for paired differences).

### 5.6 Latency protocol

- Hardware: MacBook Air M1, 8 GB RAM, CPU only.
- **Detector load time:** measured once, separately from inference.
- **Warm-up:** one synthetic 320×240 RGB image is run before
  measurement; its latency is discarded.
- **Per-image latency:** the median and mean over the held-out set
  of clean val images, in seconds.
- We report both detectors under the same conditions.

### 5.7 Fair-comparison checks

- **No resizing of the source image** before passing to the detector
  (Torchvision Faster R-CNN accepts any size; YOLOv8n does its own
  letterbox resize internally — we pass the raw PIL image).
- **Same score threshold** for both detectors within each operating
  point.
- **Same IoU threshold and matching protocol** (single implementation
  in `src/metrics.py`).
- **Same ground truth** (the same 202-image held-out set).
- **Same inference batch size** (1).
- **Same random seed for any stochastic pre-processing** (seed = 42).

## 6. What we are NOT allowed to do

- Tune the threshold or any hyperparameter on `val`.
- Re-train either detector.
- Cherry-pick thresholds by re-running the dev sweep after seeing val
  results.
- Substitute a different protocol silently.
- Claim that dev-selected thresholds generalise beyond the
  documented evaluation set.

## 7. Outputs

1. `reports/research_question_and_protocol.md` (this file)
2. `artifacts/threshold_transfer.json` — machine-readable
3. `artifacts/threshold_transfer.csv` — long-form table
4. `artifacts/plots/threshold_transfer.png` — single publication-quality figure
5. `reports/threshold_transfer_study.md` — narrative report
6. Updated `reports/paper_draft.md`
7. Updated `README.md`
8. Updated `reports/progress.md` and `reports/experiment_log.md`

## 8. Pre-registration signature

This protocol is fixed before running. Any deviation after results are
seen is a protocol violation and is to be flagged explicitly in the
final report.
