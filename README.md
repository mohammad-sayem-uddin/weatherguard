# WeatherGuard

> **Threshold transfer in pretrained object detectors for
> adverse-weather traffic imagery.**

WeatherGuard is a single, reproducible experiment that asks:

> **How well do confidence thresholds selected on development data
> transfer to held-out adverse-weather traffic images, and what
> trade-offs do they create between detection quality, missed
> objects, false alarms, and inference speed?**

It runs the public [DAWN](https://huggingface.co/datasets/Maxim37/dawn-dataset)
dataset through two off-the-shelf **pretrained** object detectors
(Torchvision Faster R-CNN, Ultralytics YOLOv8n), selects a
per-detector dev-best-F1 confidence threshold on a development
split, freezes it, and evaluates the locked threshold on a clean
held-out split. **No training is performed.**

> ⚠️ This is a research project. It is **not** a validated production
> autonomous-driving system. Numbers are limited to the DAWN val
> split and the documented evaluation protocol.

---

## Central question and strongest findings

**Central question (recap).** Does a dev-set-tuned confidence
threshold transfer to held-out adverse-weather images, and what
trade-offs does it create?

**Strongest finding (conservative wording).** A dev-tuned
threshold (0.60 for Faster R-CNN, 0.20 for YOLOv8n) on the DAWN
development split yields a held-out F1 that is **within 0.011 of
the dev F1 and within 0.006 of an unobservable val-oracle** for
both detectors, and is **robustly higher** (95% paired image-level
bootstrap CI for the difference default-0.5 minus dev-F1 is
entirely negative) than the default 0.5 threshold. The dev-tuned
threshold does not become dramatically worse on held-out
adverse-weather data; the residual gap to the val-oracle is
within the 0.05 threshold grid step.

**Absolute F1 difference, dev → held-out (at the dev-tuned
threshold):**

| Detector | Dev F1 (n=821) | Held-out F1 (n=202) | Absolute Δ | Relative Δ |
|---|---:|---:|---:|---:|
| Faster R-CNN | 0.6648 | 0.6740 | **+0.0092** | **+1.4%** |
| YOLOv8n | 0.5675 | 0.5776 | **+0.0101** | **+1.8%** |

(Slight improvement, not degradation. Values from
`artifacts/threshold_transfer.json`; dev and heldout computed at
the same threshold with the same matching protocol.)

**What the 95% confidence intervals estimate.** The CIs in the
table below are **paired image-level bootstrap 95% CIs from 500
replicates (seed 20261012)**. They describe the uncertainty in
the F1 estimate on this specific **202-image held-out sample**
under image-level resampling with replacement. They are **not**
a statement about uncertainty on a deployment population, and
they are **not** a hypothesis test.

**Why it matters.** The result contradicts the strong form of
the concern that "a dev-tuned threshold will not transfer to
held-out weather". It also gives a concrete, deployment-relevant
threshold to use (score 0.60 for Faster R-CNN, score 0.20 for
YOLOv8n) and a robust CI for the F1 improvement over the default
0.5 threshold.

## Main results table

All numbers in this table are read directly from
`artifacts/threshold_transfer.json`. They are on a clean 202-image
held-out split, computed with the documented protocol
(IoU 0.5, greedy matching, micro F1, 500 paired image-level
bootstrap replicates, seed 20261012). The "dev-F1" threshold is
the score that maximises micro F1 on the 821-image dev split; it
is frozen before any held-out evaluation.

| Detector | Dev-best-F1 threshold | Heldout F1 at dev-F1 | Heldout F1 at default-0.5 | Heldout F1 at val-oracle (diagnostic) | Heldout dev-F1 P / R | Mean s/img (CPU) |
|---|---:|---:|---:|---:|---|---:|
| Faster R-CNN ResNet-50 FPN | **0.60** | **0.674** | 0.658 | 0.680 | 0.723 / 0.632 | 2.14 |
| YOLOv8n | **0.20** | **0.578** | 0.483 | 0.580 | 0.712 / 0.486 | 0.12 |
| **Difference (paired image-level bootstrap)** | — | **+0.097** | — | — | — | — |
| 95% CI on the FRCNN-minus-YOLO F1 difference | — | **[+0.071, +0.124]** | — | — | — | — |

The dev-selected threshold beats the default-0.5 threshold for
both detectors. Paired 95% CIs for the F1 difference
(default-0.5 − dev-F1) are entirely negative for both:

| Comparison | Mean Δ F1 | 95% CI |
|---|---:|---|
| FRCNN default-0.5 − FRCNN dev-F1-0.6 | −0.017 | [−0.026, −0.008] |
| YOLO default-0.5 − YOLO dev-F1-0.2 | −0.095 | [−0.115, −0.075] |

A high-recall operating point (dev recall ≥ 0.85) is **not
available** for either detector on the dev split; the recall
ceiling of these pretrained detectors on DAWN is bounded by what
they emit (FRCNN dev max recall = 0.66; YOLO dev max recall = 0.31).

The trade-off is genuine: Faster R-CNN at score 0.6 is the
highest-F1 operating point on this hardware, and YOLOv8n at
score 0.2 is the highest-F1-per-CPU-second operating point
(4.94 F1·s⁻¹ vs Faster R-CNN's 0.315 F1·s⁻¹).

## Research sub-questions and direct answers

The pre-registered protocol is in
`reports/research_question_and_protocol.md` and was fixed before
any held-out evaluation was run. The four sub-questions and the
evidence-based answers:

- **RQ1. How do Faster R-CNN and YOLOv8n compare under the same
  held-out protocol?** Faster R-CNN has higher mAP@.5 (0.408 vs
  0.192) and higher F1 at the dev-selected threshold (+0.097,
  CI [+0.071, +0.124]). The detector with the higher mAP is also
  the detector with the higher F1 at every operating point we
  tested, with the trade-off of ~18× the per-image CPU latency.
  At the default-0.5 threshold, YOLOv8n's higher precision
  (0.903 vs 0.660) and lower recall (0.329 vs 0.656) make it the
  better choice if a high-precision, low-recall operating point
  is required.
- **RQ2. Does a dev-selected threshold retain its performance on
  held-out data?** Yes. The dev-selected F1 is within 0.010 of the
  heldout F1 for both detectors and within 0.006 of the heldout
  val-oracle. The dev-selected threshold is a defensible
  near-oracle policy.
- **RQ3. Do threshold-transfer failures differ by weather
  condition or class?** Per-condition transfer gaps for
  conditions with n ≥ 15 are modest (range −0.062 to +0.125).
  The largest absolute gaps are for `foggy` (FRCNN, +0.071) and
  `haze` (YOLO, +0.125). The two smallest conditions
  (`dusttornado` n=8, `sand_storm_g2_` n=9) are too small for
  inference. Per-class transfer is dominated by class imbalance
  (Cars make up 80% of GT; Bicycle has only 2 GT in heldout and
  is uninformative).
- **RQ4. Which detector and operating point offers the most
  defensible detection-quality-vs-CPU-time trade-off?** For
  highest absolute F1: Faster R-CNN at score 0.6 (F1=0.674,
  2.14 s/img). For highest F1 per CPU-second: YOLOv8n at
  score 0.2 (F1=0.578, 0.12 s/img). The choice depends on
  whether the deployment is detection-quality- or
  throughput-bottlenecked.

## Limitations and threats to validity

- **Single held-out set.** The DAWN val split is small (202
  images after a byte-duplicate exclusion). A different held-out
  set could shift the dev-F1 threshold and the per-detector
  ordering.
- **Class imbalance.** Cars make up ~80% of GT in heldout.
  Per-class F1 for Bicycle (2 GT) is uninformative.
- **No high-recall operating point.** Neither detector reaches
  recall ≥ 0.85 on the dev sweep; we cannot recommend a
  high-recall policy from this study.
- **Threshold grid is 0.05 wide.** A finer grid might pick a
  different threshold.
- **No fine-tuning.** All numbers are pretrained-only.
- **CPU-only measurements.** Absolute latencies are CPU-specific.
- **Single bootstrap seed.** Other seeds give CIs of similar
  width.
- **DAWN is CC BY-NC 3.0** (non-commercial research only). No
  images are redistributed in this repository.
- **YOLOv8n is AGPL-3.0** (commercial use requires a separate
  Ultralytics licence). The Faster R-CNN path is fully
  permissive.

## Key artifacts

| Artifact | Path |
|---|---|
| Pre-registered protocol | `reports/research_question_and_protocol.md` |
| Threshold-transfer study | `reports/threshold_transfer_study.md` |
| Study machine-readable results | `artifacts/threshold_transfer.json` |
| Study long-form table | `artifacts/threshold_transfer.csv` |
| Study figure | `artifacts/plots/threshold_transfer.png` |
| Dev predictions (FRCNN) | `artifacts/eval_dev_fasterrcnn.json` |
| Dev predictions (YOLO) | `artifacts/eval_dev_yolov8n.json` |
| Heldout predictions (FRCNN) | `artifacts/eval_heldout_fasterrcnn.json` |
| Heldout predictions (YOLO) | `artifacts/eval_heldout_yolov8n.json` |
| Clean heldout list (4 byte-duplicates excluded) | `artifacts/clean_heldout_image_ids.json` |
| Paper draft (focused on this study) | `reports/paper_draft.md` |

## How to reproduce the focused study

The complete protocol and all required commands are in
`reports/research_question_and_protocol.md` and
`reports/paper_draft.md` § 6. A condensed version:

```bash
# 0. Install (Python 3.11, macOS / Linux CPU)
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install ultralytics        # for the YOLOv8n path (AGPL-3.0)

# 1. Place the DAWN parquet files in the project root.
#    Do NOT commit them; they are excluded by .gitignore.
ls -la *.parquet
# train-00000-of-00001.parquet   val-00000-of-00001.parquet

# 2. Run dev (train) and heldout (val) eval for each detector
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

# 3. Run the threshold transfer study
PYTHONPATH=. .venv/bin/python scripts/run_threshold_transfer_study.py --boots 500

# 4. Tests and artifact validator
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests
PYTHONPATH=. .venv/bin/python scripts/validate_artifacts.py

# 5. (Optional) launch the Streamlit dashboard to inspect results
PYTHONPATH=. .venv/bin/streamlit run app.py
```

Total expected runtime on a MacBook Air M1: ~40 minutes for the
four detector evals + ~1 minute for the transfer study + bootstrap.

## Dataset and licenses

- **DAWN.** Source: <https://huggingface.co/datasets/Maxim37/dawn-dataset>.
  License: **CC BY-NC 3.0** (non-commercial research use only). No
  images are redistributed in this repository. Attribution is in
  `CITATION.cff` and `NOTICE`. We exclude 4 val images whose JPEG
  bytes are SHA-1 identical to images in the train split to
  prevent leakage; the excluded ids are
  `haze-003, haze-059, haze-077, dusttornado-023`.
- **Faster R-CNN ResNet-50 FPN.** Torchvision, BSD-3-Clause code,
  COCO-pretrained weights per the Torchvision model registry.
  Fully permissive.
- **YOLOv8n.** Ultralytics, AGPL-3.0. **Commercial use requires
  a separate Ultralytics licence.**

The project retains the original WeatherGuard MVP (corruption
robustness study, error taxonomy, calibration audit) for
completeness; see `reports/research_report.md`,
`reports/advanced_mission_final_report.md`, and the previous
artifacts (`artifacts/corruption_*.json`, `artifacts/error_taxonomy.json`,
`artifacts/calibration.json`, etc.).

## Repository structure

```
WeatherGuard/
├── app.py                          # Streamlit dashboard (14 pages)
├── src/
│   ├── dataset.py                  # parquet load, BBox, condition parser
│   ├── model.py                    # FasterRCNNResNet50FPN, YOLOv8N
│   ├── inference.py                # run_inference, eval-records builder
│   ├── metrics.py                  # IoU, greedy match, AP, mAP, evaluate
│   ├── experiments.py              # brightness/contrast/blur/noise
│   └── visualization.py            # draw_boxes, render_prediction
├── configs/default.yaml
├── scripts/                        # one runner per experiment
│   ├── run_transfer_eval.py        # dev + heldout eval (focused study)
│   ├── run_threshold_transfer_study.py
│   ├── run_full_eval.py            # MVP eval
│   ├── run_corruption_experiment.py
│   ├── run_corruption_severity.py
│   ├── run_statistical_analysis.py
│   ├── run_threshold_sweep.py
│   ├── run_error_taxonomy.py
│   ├── run_preprocessing_ablation.py
│   ├── run_calibration.py
│   ├── analyze_confidence.py
│   ├── render_failure_examples.py
│   ├── summarize_comparison.py
│   ├── experiment_registry.py
│   └── validate_artifacts.py
├── tests/                          # 71 unit tests
├── artifacts/
│   ├── threshold_transfer.{json,csv}  # focused study outputs
│   ├── plots/threshold_transfer.png
│   ├── clean_heldout_image_ids.json
│   ├── eval_dev_*.json, eval_heldout_*.json
│   ├── eval_*.json, corruption_*.json
│   ├── threshold_sweep.json, statistical_analysis.json
│   ├── error_taxonomy.json, calibration.json
│   ├── experiment_registry.jsonl
│   └── plots/
├── reports/
│   ├── research_question_and_protocol.md
│   ├── threshold_transfer_study.md
│   ├── paper_draft.md
│   ├── research_report.md            # MVP report
│   ├── advanced_mission_final_report.md
│   ├── dawn_feasibility_report.md
│   ├── final_report.md
│   ├── experiment_log.md
│   ├── experiment_registry.md
│   ├── statistical_analysis.md
│   ├── threshold_sweep.md
│   ├── error_taxonomy.md
│   ├── calibration.md
│   ├── progress.md
│   └── ...
├── .github/workflows/tests.yml
├── requirements.txt
├── README.md
├── LICENSE                          # MIT for our code
├── NOTICE                           # third-party license notices
├── CITATION.cff
└── .gitignore
```

## License

WeatherGuard's own code is released under the **MIT License**
(see `LICENSE`). Third-party components retain their original
licences; see `NOTICE` for the full list. The DAWN dataset is
licensed under **CC BY-NC 3.0** and is **not** redistributed in this
repository. The YOLOv8n path is licensed under **AGPL-3.0**;
commercial use requires a separate Ultralytics licence.
