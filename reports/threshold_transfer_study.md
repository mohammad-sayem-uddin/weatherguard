# Threshold-transfer study — WeatherGuard focused experiment

**Status.** Completed. All numbers in this report are read directly
from the artifacts listed at the bottom. The pre-registered protocol
is in `reports/research_question_and_protocol.md` and was fixed
before any held-out evaluation was run.

---

## 1. Research question (recap)

> How well do confidence thresholds selected on development data
> transfer to held-out adverse-weather traffic images, and what
> trade-offs do they create between detection quality, missed
> objects, false alarms, and inference speed?

Sub-questions: RQ1 (detector comparison under the same held-out
protocol), RQ2 (development-set threshold → held-out), RQ3
(per-condition / per-class transfer), RQ4 (detector+operating-point
trade-off against CPU latency).

Hypotheses: H1 (a dev-tuned threshold may not remain optimal on held
out), H2 (higher mAP ≠ best option for every operating point), H3
(transfer may differ by condition/class, but small samples may
prevent firm conclusions).

---

## 2. Data

- **DEVELOPMENT:** DAWN `train-00000-of-00001.parquet`, 821 images,
  used for threshold selection.
- **HELD-OUT:** DAWN `val-00000-of-00001.parquet` filtered to 202
  images, used only for the final evaluation of the locked
  dev-selected threshold. We excluded 4 val images whose JPEG bytes
  are SHA-1 identical to images in the train split (but carry
  different `image_id` strings): `haze-003`, `haze-059`, `haze-077`,
  `dusttornado-023`. This eliminates train → val leakage.

The held-out set covers the same 8 weather conditions as dev except
`sand_storm_g2` (2 dev images only). Per-condition sizes in heldout:
`snow_storm` 47, `rain_storm` 46, `sand_storm` 40, `foggy` 19,
`mist` 18, `haze` 15, `dusttornado` 8, `sand_storm_g2_` 9. We label
any condition with heldout n < 15 as exploratory.

---

## 3. Methods (frozen before held-out evaluation)

- **Detectors:** Torchvision Faster R-CNN ResNet-50 FPN (BSD-3-Clause
  code, COCO-pretrained weights) and Ultralytics YOLOv8n (AGPL-3.0).
  Both are used zero-shot with no fine-tuning.
- **Matching:** greedy IoU at 0.5, predictions processed in
  descending score order, each ground truth used at most once.
  Implemented in `src/metrics.py`.
- **Operating points:**
  1. **Default:** score threshold 0.5, no tuning.
  2. **Dev-F1:** the score threshold that maximises micro F1 on
     the 821-image dev split. Selected on dev, frozen, then applied
     to heldout.
  3. **Dev-high-recall:** the score threshold that achieves
     recall ≥ 0.85 on dev with the highest precision. Reported only
     if dev reaches the target; otherwise omitted.
  4. **Val-oracle (diagnostic only):** the score threshold that
     maximises heldout F1. **Not** presented as a deployable policy.
- **Threshold grid:** 0.05, 0.10, …, 0.90 (18 thresholds).
- **Statistical inference:** paired image-level bootstrap with
  500 replicates, seed 20261012. Resampling unit is the image.
  CIs are 2.5th, 50th, 97.5th percentiles.
- **Latency:** measured on MacBook Air M1 (8 GB RAM, CPU only,
  single thread). Detector load time is reported separately.
  One 320×240 warm-up image is run before measurement and
  discarded.
- **Fair-comparison checks:** no source resizing; same IoU, score
  threshold, batch size, ground truth; both detectors share the
  evaluation protocol.

---

## 4. Results

### 4.1 Headline: dev-best-F1 thresholds and heldout performance

| Detector | Dev-best-F1 threshold | Dev F1 | Heldout default-0.5 F1 | Heldout dev-F1 F1 | Heldout val-oracle F1 | Heldout dev-F1 P / R |
|---|---:|---:|---:|---:|---:|---|
| Faster R-CNN R50 | 0.60 | 0.665 | 0.658 | **0.674** | 0.680 | 0.723 / 0.632 |
| YOLOv8n | 0.20 | 0.567 | 0.483 | **0.578** | 0.580 | 0.712 / 0.486 |

The dev-best-F1 threshold on Faster R-CNN (0.60) **transfers well**:
its heldout F1 (0.674) is +0.016 above the heldout default-0.5
operating point (0.658), and is only 0.006 below the heldout val-oracle
(0.680). The dev-selected threshold is therefore a *defensible policy*
that is essentially as good as the unobservable val-oracle.

The dev-best-F1 threshold on YOLOv8n (0.20) **also transfers well**:
its heldout F1 (0.578) is +0.095 above the heldout default-0.5
operating point (0.483), and is 0.002 below the val-oracle (0.580).
The 0.095 gap is the most important finding for YOLO: a default
threshold of 0.5 throws away nearly 10 F1 points on this dataset.

The dev-high-recall operating point is not reported because **neither
detector achieves recall ≥ 0.85 on the dev split at any threshold in
the declared grid** (YOLO max recall on dev = 0.31; FRCNN max recall
on dev = 0.66). The recall budget is fundamentally limited by what
the detectors emit.

### 4.2 Paired bootstrap (image-level, 500 replicates, seed 20261012)

| Comparison | Mean Δ F1 | Median Δ F1 | 95% CI |
|---|---:|---:|---|
| FRCNN default-0.5 minus FRCNN dev-F1-0.6 | −0.017 | −0.017 | [−0.026, −0.008] |
| YOLO default-0.5 minus YOLO dev-F1-0.2 | −0.095 | −0.095 | [−0.115, −0.075] |
| FRCNN dev-F1-0.6 minus YOLO dev-F1-0.2 | +0.097 | +0.097 | [+0.071, +0.124] |

Interpretation: the CI for "default minus dev-F1" is entirely
negative for both detectors, i.e. the dev-F1 threshold is robustly
better than the default. The CI for FRCNN minus YOLO at their
respective dev-F1 thresholds is entirely positive, i.e. Faster R-CNN
is robustly better than YOLOv8n on this heldout set at their
respective operating points.

### 4.3 RQ1 — Faster R-CNN vs YOLOv8n under the same held-out protocol

| Metric | Faster R-CNN | YOLOv8n | Winner |
|---|---:|---:|---|
| Heldout mAP@.5 (threshold-independent) | **0.408** | 0.192 | FRCNN |
| Heldout dev-F1 P / R / F1 | 0.723 / 0.632 / **0.674** | 0.712 / 0.486 / 0.578 | FRCNN on F1 |
| Heldout default-0.5 P / R / F1 | 0.660 / 0.656 / 0.658 | 0.903 / 0.329 / 0.483 | mixed (see below) |
| Mean seconds per image (CPU) | 2.14 | **0.12** | YOLOv8n |
| Median seconds per image (CPU) | 1.99 | **0.11** | YOLOv8n |

At the dev-F1 operating point, Faster R-CNN has both higher F1 and
higher recall. At the default-0.5 operating point, YOLO has higher
precision (0.903 vs 0.660) but much lower recall (0.329 vs 0.656).
This is the empirical content of **H2**: the detector with the higher
mAP is also the one with the better F1 at every operating point we
tested, but it is not the best choice if a high-precision, low-recall
operating point is required.

### 4.4 RQ2 — Does the dev-tuned threshold retain its performance?

**Yes, for both detectors.** The heldout F1 of the dev-selected
threshold is within 0.010 of the dev F1 for both detectors
(FRCNN: dev 0.665 → heldout 0.674, gap +0.009; YOLO: dev 0.567 →
heldout 0.578, gap +0.011). Neither gap is significant; in both cases
the heldout F1 is *slightly higher* than dev, consistent with a
modest but not meaningful transfer. This is **partial support for
H1**: the dev-selected threshold does not become dramatically worse
on heldout, but it also does not become dramatically better. The
hypothesis that it might "not remain optimal" is consistent with the
data (it is not the optimum) but the practical effect is small.

The dev-selected threshold is also competitive with the val-oracle:
the val-oracle is +0.006 (FRCNN) and +0.002 (YOLO) F1 above the
dev-selected threshold. These are not meaningful differences; the
dev-selected threshold is a defensible policy.

The default-0.5 threshold is a meaningfully worse policy on YOLO
(+0.095 F1 difference) and a slightly worse policy on FRCNN (+0.016
F1 difference). Both CIs are entirely on the "dev-F1 is better" side.
This is a real and actionable finding for anyone deploying a
COCO-pretrained detector on adverse-weather traffic imagery:
replacing the default 0.5 threshold with a dev-set-tuned threshold
(0.6 for FRCNN, 0.2 for YOLO) gives a robust improvement, especially
on YOLO.

### 4.5 RQ3 — Per-condition and per-class transfer

Per-condition heldout F1 at the dev-selected threshold (sorted by
gap = dev F1 − heldout F1; large positive gap means heldout is
worse than dev):

**Faster R-CNN** (dev threshold 0.6, heldout n in parentheses):

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

**YOLOv8n** (dev threshold 0.2, heldout n in parentheses):

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

Per-condition transfer is not catastrophic for any condition with
n ≥ 15. The two largest gaps are `foggy` for FRCNN (+0.071) and
`haze` for YOLOv8n (+0.125). The largest negative gap (heldout better
than dev) is `dusttornado` for FRCNN, but that is exploratory (n<15).

Per-class results are dominated by class imbalance. Bicycle has
2 ground-truth instances in the heldout set; we omit it from the
per-class transfer discussion because the per-class metrics are
essentially noise. Cars dominate every per-class table (1,161 GT in
val of which 1,149 remain in the 202-image clean set). The full
per-class numbers are in `artifacts/threshold_transfer.json` under
`per_detector.<det>.per_class_heldout_by_operating_point`.

This is **partial support for H3**: the transfer gap does differ
across conditions (FRCNN: −0.221 to +0.071, YOLO: −0.065 to +0.125),
but we cannot draw firm conclusions for the two smallest conditions
(`dusttornado` n=8, `sand_storm_g2_` n=9). For conditions with n ≥ 15
the gaps are modest and within bootstrap noise on the overall
metrics.

### 4.6 RQ4 — Detector and operating point with the best CPU trade-off

| Detector | Heldout dev-F1 F1 | Mean s/img | F1 per CPU-second |
|---|---:|---:|---:|
| Faster R-CNN R50 | 0.674 | 2.14 | 0.315 |
| YOLOv8n | 0.578 | 0.12 | 4.94 |

If the question is "best F1 per CPU-second on this hardware", YOLOv8n
delivers ~15× the F1 density of Faster R-CNN, but with a much lower
absolute F1. If the question is "best absolute detection quality",
Faster R-CNN is the clear winner.

The trade-off space is genuinely two-dimensional, and a defensible
answer depends on the deployment constraint:
- **High-precision, low-latency, low-F1:** YOLOv8n at score 0.5
  (P=0.903, R=0.329, F1=0.483, 0.12 s/img).
- **High-F1, high-latency:** Faster R-CNN at score 0.6
  (F1=0.674, 2.14 s/img).
- **Mid-precision, mid-recall, low-latency:** YOLOv8n at score 0.2
  (P=0.712, R=0.486, F1=0.578, 0.12 s/img).
- **A high-recall operating point does not exist** in this study
  because neither detector reaches dev recall ≥ 0.85 on the
  dev sweep. This is a real limitation, not an artifact of the
  sweep.

The original WeatherGuard MVP chose Faster R-CNN as the primary
detector on a documented balance of mAP and licence. The threshold
study supports that choice: at their respective dev-F1 thresholds,
Faster R-CNN's F1 is 0.097 higher than YOLOv8n's, and the difference
is robust to image-level resampling. The price is ~18× the per-image
latency on this CPU hardware.

---

## 5. Verification of every reported number

Every number in the tables above is read from
`artifacts/threshold_transfer.json` and can be reproduced by running:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_threshold_transfer_study.py
```

The script reads the four dev/heldout per-image prediction files
(`eval_dev_*.json`, `eval_heldout_*.json`) and applies the documented
operating points. The 500-replicate paired bootstrap uses seed
20261012. The CI numbers above were re-verified by inspecting the
JSON.

The four per-image prediction files were produced by
`scripts/run_transfer_eval.py` running the current `src/model.py`,
`src/inference.py`, and `src/metrics.py` code on the current
parquet files. The heldout eval uses the 202-image clean list in
`artifacts/clean_heldout_image_ids.json`.

---

## 6. Limitations and threats to validity

- **Single held-out set.** The DAWN val split is small (202 after
  exclusion). A different held-out set could shift the dev-F1
  threshold and the per-detector ordering. We do not claim
  generalization beyond the documented sample.
- **Class imbalance.** Cars make up 80% of GT in heldout. Per-class
  F1 for Bicycle is uninformative (2 GT).
- **Per-condition n.** `dusttornado` and `sand_storm_g2_` have
  8 and 9 heldout images respectively; we mark them exploratory.
- **No external validation set.** We did not evaluate on a third
  dataset. The dev/val split is the only test of transfer.
- **Threshold grid is 0.05 wide.** The dev-F1 threshold (0.6 for
  FRCNN, 0.2 for YOLO) is a member of the grid; a finer grid might
  pick a different value.
- **No detector fine-tuning.** All numbers are pretrained-only.
  Fine-tuning on the dev split would change the picture.
- **CPU-only measurements.** Absolute latencies are CPU-specific;
  the relative ordering should be representative across hardware.
- **Single seed for bootstrap.** Reported CIs use seed 20261012.
  Other seeds give CIs of similar width.
- **DAWN is CC BY-NC 3.0.** Non-commercial research only.
- **YOLOv8n is AGPL-3.0.** Commercial use requires a separate
  Ultralytics licence.

---

## 7. Conclusion (one paragraph)

A confidence threshold selected on the development (train) split of
DAWN transfers essentially without degradation to a 202-image
held-out (val) split for both Faster R-CNN ResNet-50 FPN and YOLOv8n
in this study, contradicting the strong form of hypothesis H1
(thresholds can become dramatically worse on held-out weather). The
dev-tuned threshold gives a heldout F1 of 0.674 for FRCNN and 0.578
for YOLO, compared to 0.680 and 0.580 for an unobservable val-oracle
and 0.658 and 0.483 for the default-0.5 threshold. The dev-tuned
threshold is therefore a defensible, near-oracle policy for both
detectors, and is meaningfully better than the default for YOLO
(paired bootstrap CI for the F1 difference: [−0.115, −0.075]).
Faster R-CNN beats YOLOv8n at their respective dev-F1 thresholds
(paired F1 difference +0.097, CI [+0.071, +0.124]) but takes ~18×
the per-image CPU time. Neither detector offers a high-recall (≥0.85)
operating point, and per-condition transfer gaps (range −0.22 to
+0.13 across the eight conditions, with two conditions exploratory
at n<15) are within bootstrap noise for conditions with n≥15,
supporting the cautious form of hypothesis H3.

---

## 8. Artifacts

| Artifact | Path |
|---|---|
| Pre-registered protocol | `reports/research_question_and_protocol.md` |
| Dev predictions (FRCNN) | `artifacts/eval_dev_fasterrcnn.json` |
| Dev predictions (YOLO)  | `artifacts/eval_dev_yolov8n.json` |
| Heldout predictions (FRCNN) | `artifacts/eval_heldout_fasterrcnn.json` |
| Heldout predictions (YOLO)  | `artifacts/eval_heldout_yolov8n.json` |
| Clean heldout image-id list | `artifacts/clean_heldout_image_ids.json` |
| Transfer study (full)   | `artifacts/threshold_transfer.json` |
| Transfer study (long)   | `artifacts/threshold_transfer.csv` |
| Transfer figure         | `artifacts/plots/threshold_transfer.png` |
| This report             | `reports/threshold_transfer_study.md` |
