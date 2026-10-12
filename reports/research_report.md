# WeatherGuard — Research Report

**Project:** WeatherGuard — a reproducible CPU-friendly toolkit for
evaluating pretrained object detectors on real adverse-weather traffic
imagery.

**Author:** WeatherGuard contributors (autonomous build session).
**Date:** 2026-10-12.
**Status:** Completed end-to-end. All measurements in this report come
from the scripts in `scripts/` and are reproducible via the commands in
the README.

---

## 1. Research question

**Primary.** How does an existing object detector perform on real-world
traffic imagery captured in adverse weather, and what failure patterns
can be observed across the available weather categories?

**Secondary.**
1. Which object classes are most difficult to detect?
2. What false-positive and false-negative patterns occur?
3. How does confidence relate to correctness?
4. Which controlled image corruptions cause the largest performance
   changes?

We do **not** claim that these questions are novel. The DAWN dataset
itself is a published contribution to adverse-weather detection, and
robustness under controlled corruptions has been studied extensively
(Hendrycks & Dietterich, 2019, *Benchmarking Neural Network Robustness
to Common Corruptions and Perturbations*). WeatherGuard's contribution
is the **toolkit** that lets a researcher reproduce a small, honest,
CPU-friendly evaluation on the DAWN benchmark without training
anything.

---

## 2. Dataset

- **DAWN** — `Maxim37/dawn-dataset` on Hugging Face.
- 1,027 images total across train (821) and val (206) splits.
- 6 object classes: Truck, Person, Bicycle, Car, Motorcycle, Bus.
- 9 weather-condition buckets encoded as a prefix in `image_id`:
  `foggy`, `haze`, `mist`, `dusttornado`, `rain_storm`, `sand_storm`,
  `sand_storm_g2`, `sand_storm_g2_`, `snow_storm`. (The two
  `sand_storm_g2*` variants are preserved as observed in the source;
  one appears to be a minor upstream typo.)
- License: **CC BY-NC 3.0** — non-commercial research use only.
- All 1,438 bounding boxes in the val split pass geometric validation
  (no negatives, no out-of-bounds).
- See `reports/dawn_feasibility_report.md` for the full feasibility
  analysis and `reports/error_analysis_*.md` for per-condition
  breakdowns.

---

## 3. Models

| Model                       | Framework   | License (code)   | License (weights)              |
|-----------------------------|-------------|------------------|--------------------------------|
| Faster R-CNN ResNet-50 FPN  | Torchvision | BSD-3-Clause     | Torchvision model registry     |
| YOLOv8n                     | Ultralytics | AGPL-3.0         | AGPL-3.0 (per Ultralytics)     |

Both detectors are pretrained on **COCO 2017**. We never fine-tune. We
reduce the 80-class COCO output to the 6-class DAWN label space via
`COCO_TO_DAWN` in `src/dataset.py`. Out-of-vocabulary predictions are
ignored, not counted as FP.

---

## 4. Evaluation protocol

- **Matching:** greedy IoU at threshold 0.5; each GT used at most
  once; predictions processed in descending score order.
- **TP/FP/FN:** count predictions with score ≥ 0.5.
- **Per-class AP@.5:** 101-point interpolation over the sorted
  predictions; classes with no GT contribute NaN and are excluded
  from the mAP average.
- **mAP@.5:** macro-average of per-class AP@.5 over classes with ≥1
  GT.
- **Calibration:** Brier score and 10-bin reliability table over
  score≥0.5 predictions. We explicitly do **not** claim detector
  scores are calibrated probabilities.
- All scripts use `seed=42` where randomness is involved.

---

## 5. Results

### 5.1 Model comparison on the full 206-image val split

| Detector           | n  | mAP@.5 | Micro P | Micro R | Micro F1 | Mean ms/img | Median ms/img |
|--------------------|---:|-------:|--------:|--------:|---------:|------------:|--------------:|
| Faster R-CNN R50   | 206 |  0.410 |   0.662 |   0.659 |    0.660 |        1431 |          1266 |
| YOLOv8n            | 206 |  0.192 |   0.902 |   0.328 |    0.481 |         108 |            99  |

Selected: **Faster R-CNN ResNet-50 FPN**, on the documented balance of
mAP@.5 (substantially higher) and license (fully permissive).
YOLOv8n is reported as a fast reference detector; it is ~13× faster but
loses ~53% mAP and ~22 F1 points.

### 5.2 Per-class AP@.5 (Faster R-CNN)

| Class      | n_gt | n_pred | TP | FP | FN | P    | R    | F1   | AP@.5 |
|------------|-----:|-------:|---:|---:|---:|-----:|-----:|-----:|------:|
| Car        | 1161 |  1091  |804 |287 |357 |0.737 |0.693 |0.714 | 0.651 |
| Person     |  111 |   164  | 83 | 81 | 28 |0.506 |0.748 |0.604 | 0.704 |
| Motorcycle |   15 |    11  |  8 |  3 |  7 |0.727 |0.533 |0.615 | 0.520 |
| Bus        |   34 |    22  | 14 |  8 | 20 |0.636 |0.412 |0.500 | 0.371 |
| Truck      |  115 |   140  | 38 |102 | 77 |0.271 |0.330 |0.298 | 0.212 |
| Bicycle    |    2 |     2  |  0 |  2 |  2 |0.000 |0.000 |0.000 | 0.000 |

Failure pattern: **Trucks** are systematically misclassified as Cars
(102 FP vs 38 TP). Person is high-recall but high-FP (likely detector
picks up riders and pedestrians beyond the GT coverage). Bicycle has
only 2 GT and is uninformative.

### 5.3 Per-condition mAP@.5 (Faster R-CNN)

| Condition         | n   | mAP@.5 | P    | R    | F1   |
|-------------------|----:|-------:|-----:|-----:|-----:|
| snow_storm        |  47 |  0.627 |0.613 |0.724 |0.664 |
| sand_storm        |  40 |  0.544 |0.659 |0.684 |0.672 |
| dusttornado       |   9 |  0.478 |0.846 |0.688 |0.759 |
| sand_storm_g2_    |   9 |  0.485 |0.718 |0.635 |0.674 |
| foggy             |  19 |  0.401 |0.659 |0.560 |0.605 |
| rain_storm        |  46 |  0.362 |0.673 |0.651 |0.662 |
| haze              |  18 |  0.336 |0.685 |0.585 |0.631 |
| mist              |  18 |  0.299 |0.674 |0.609 |0.640 |

`snow_storm` is the easiest condition (the high-contrast white-on-white
texture appears to help the detector), while `mist` and `haze` are the
hardest (atmospheric scattering reduces local contrast). Conditions
with fewer than 15 val images (`dusttornado`, `sand_storm_g2_`) are
exploratory; confidence intervals are not reported because no
defensible resampling unit is available.

### 5.4 Controlled corruption experiment (78-image stratified subset, 10 per condition)

Same detector (Faster R-CNN), same images, same annotations, same
evaluation protocol. Photometric corruptions do not change object
geometry, so the original bounding boxes remain valid.

| Corruption         | mAP@.5 | Δ vs baseline | Micro P | Micro R | Micro F1 |
|--------------------|-------:|--------------:|--------:|--------:|---------:|
| identity (baseline)|  0.377 |          —    |  0.707  |  0.658  |  0.682   |
| brightness × 0.5   |  0.342 |         -9.3% |  0.768  |  0.584  |  0.664   |
| contrast × 0.5     |  0.335 |        -11.2% |  0.750  |  0.600  |  0.667   |
| gaussian noise σ=25|  0.201 |        -46.7% |  0.666  |  0.477  |  0.556   |
| blur r=4           |  0.154 |        -59.2% |  0.735  |  0.301  |  0.427   |

The same pattern holds for YOLOv8n on the same subset (see
`artifacts/corruption_yolov8n_corruption.json`): blur and noise are
the worst, brightness and contrast are milder degradations.

> **Caveat.** Photometric corruptions are *not* equivalent to real
> adverse weather. They are a controlled robustness benchmark; they
> should not be used to claim real-world weather robustness.

### 5.5 Confidence calibration (Faster R-CNN)

Brier score (lower = better calibration) over all score≥0.5
predictions: **0.161**. Per class:

| Class      | Brier | n     |
|------------|------:|------:|
| Motorcycle | 0.091 |   9   |
| Bus        | 0.105 |  16   |
| Car        | 0.156 | 1090  |
| Person     | 0.164 |  120  |
| Truck      | 0.260 |   69  |

Reliability bins (10 equal-width confidence bins):

| bin      | count | mean conf | frac correct |
|----------|------:|----------:|-------------:|
| 0.0–0.5  |     0 |         — |            — |
| 0.5–0.6  |   142 |     0.549 |        0.261 |
| 0.6–0.7  |   117 |     0.652 |        0.368 |
| 0.7–0.8  |   147 |     0.749 |        0.503 |
| 0.8–0.9  |   183 |     0.855 |        0.727 |
| 0.9–1.0  |   715 |     0.974 |        0.923 |

The detector is **systematically overconfident** in the low-score bins
(0.5–0.6: 0.55 mean conf, 0.26 actual accuracy). Even at the top bin
the mean confidence (0.97) exceeds the empirical accuracy (0.92).
Detector scores should be treated as relative ranks, not calibrated
probabilities.

### 5.6 Failure examples

`artifacts/qualitative/` contains 8 saved visualizations (not committed
to git; see `.gitignore`) showing ground truth (green) and predictions
(red) for a few representative images, including one rendered above in
the README and the failure-analysis page of the dashboard.

---

## 6. Limitations & threats to validity

- **Class imbalance** — Cars dominate (~83% of instances). Bicycle
  (2 GT) and Motorcycle (15 GT) per-class metrics are noisy.
- **Sample sizes per condition** range from 9 to 47. The smaller
  conditions are reported as exploratory.
- **Photometric corruption ≠ real weather.** The controlled
  corruption results describe a *robustness* benchmark, not a
  prediction of real-world weather performance.
- **DAWN license is CC BY-NC 3.0** — non-commercial research only.
- **No calibration claim.** Detector scores are not calibrated
  probabilities; downstream consumers should calibrate before use.
- **YOLOv8n is AGPL-3.0** — commercial use requires a separate
  Ultralytics licence.
- **No training, no fine-tuning.** All numbers are pretrained-only
  zero-shot evaluations.
- **CPU-only hardware** — absolute latencies are slower than what a
  GPU would deliver; relative ordering should be representative.

---

## 7. Reproducibility

```bash
# environment
cd /Users/sayemuddin/Desktop/WeatherGuard
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install ultralytics  # for YOLOv8n path

# run all experiments
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
PYTHONPATH=. .venv/bin/python scripts/run_full_eval.py --tag fasterrcnn_val_full
PYTHONPATH=. .venv/bin/python scripts/run_full_eval.py --tag yolov8n_val_full --detector yolov8n
PYTHONPATH=. .venv/bin/python scripts/run_corruption_experiment.py --tag fasterrcnn_corruption --limit 10
PYTHONPATH=. .venv/bin/python scripts/run_corruption_experiment.py --tag yolov8n_corruption --detector yolov8n --limit 10
PYTHONPATH=. .venv/bin/python scripts/analyze_confidence.py
PYTHONPATH=. .venv/bin/python scripts/summarize_comparison.py
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests

# launch the dashboard
PYTHONPATH=. .venv/bin/streamlit run app.py
```

All scripts use fixed seeds; results are deterministic on a single
host. The same script can be re-run after upgrading torch / torchvision
versions and the result will shift only insofar as the upstream
pretrained weights change.

---

## 8. Future work

1. **Severity within a condition** — DAWN does not provide severity
   labels. Pairing DAWN with a separate severity taxonomy (e.g. WMO
   weather codes) would allow degradation curves per condition.
2. **Cross-dataset robustness** — repeat the corruption experiment on
   a second weather dataset to disentangle detector brittleness from
   dataset artefacts.
3. **Detector ensemble** — fuse Faster R-CNN and YOLOv8n predictions
   and measure the change in mAP@.5 vs. single-detector baselines.
4. **Probability calibration** — apply temperature scaling or Platt
   scaling on a held-out fold and report post-calibration Brier and
   reliability.
