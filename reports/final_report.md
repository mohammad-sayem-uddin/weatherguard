# WeatherGuard — Final Completion Report

**Project:** WeatherGuard
**Build session:** 2026-10-12 (autonomous, single session)
**Status:** ✅ Complete and reproducible.

---

## 1. What works

| Component | Status | Where |
|---|---|---|
| Parquet DAWN loader with condition parser | ✅ | `src/dataset.py` |
| Faster R-CNN ResNet-50 FPN detector wrapper | ✅ | `src/model.py` |
| YOLOv8n detector wrapper | ✅ | `src/model.py` |
| Greedy IoU match, 101-point AP, mAP@.5 | ✅ | `src/metrics.py` |
| Photometric corruption grid | ✅ | `src/experiments.py` |
| Streamlit dashboard, 8 pages | ✅ | `app.py` |
| Unit tests (48) | ✅ all pass | `tests/` |
| GitHub Actions workflow | ✅ | `.github/workflows/tests.yml` |
| README, LICENSE, NOTICE, CITATION.cff | ✅ | repo root |
| DAWN feasibility report | ✅ | `reports/dawn_feasibility_report.md` |
| Smoke test report | ✅ | `reports/smoke_test_report.md` |
| Research report | ✅ | `reports/research_report.md` |
| Error analysis (both detectors) | ✅ | `reports/error_analysis_*.md` |
| Qualitative failure visualizations | ✅ | `artifacts/qualitative/*.png` |

## 2. Approaches actually compared

| Detector | License | mAP@.5 (val 206) | Mean ms/img |
|---|---|---:|---:|
| Faster R-CNN ResNet-50 FPN (Torchvision) | BSD-3-Clause | **0.410** | 1431 |
| YOLOv8n (Ultralytics) | AGPL-3.0 | 0.192 | 108 |

**Selected:** Faster R-CNN, on a documented balance of mAP and
licensing. YOLOv8n is preserved as a speed reference.

## 3. Experiments actually executed

1. **Smoke test** — 8 deterministic val images, Faster R-CNN.
   `artifacts/smoke_test_report.json`,
   `reports/smoke_test_report.md`.
2. **Full val evaluation** — 206 images, both detectors. Per-image
   predictions, per-class metrics, per-condition metrics. JSON outputs
   under `artifacts/eval_*_val_full.json`.
3. **Controlled corruption experiment** — 78-image stratified subset
   (10 per condition, 8 conditions), 5 corruptions (identity,
   brightness, contrast, blur, noise). Both detectors. JSON + CSV
   under `artifacts/corruption_*`.
4. **Confidence / calibration analysis** — Brier score, 10-bin
   reliability table, per-class and per-condition breakdown. Reports
   under `reports/error_analysis_*.md`, CSV under
   `artifacts/reliability_*.csv`.
5. **Qualitative failure gallery** — 8 images rendered with
   ground-truth + prediction overlays, saved to
   `artifacts/qualitative/`.

## 4. Where the results live

- `artifacts/eval_fasterrcnn_val_full.json` (1.8 MB) — full val
  per-image predictions + per-class + per-condition metrics.
- `artifacts/eval_yolov8n_val_full.json` (17.7 MB).
- `artifacts/corruption_fasterrcnn_corruption.json` / `.csv` — 5
  corruptions.
- `artifacts/corruption_yolov8n_corruption.json` / `.csv`.
- `artifacts/reliability_*.csv` — 10-bin reliability data.
- `artifacts/model_comparison.csv` / `.json` — single comparison
  summary table.
- `artifacts/qualitative/*.png` — 8 failure examples (gitignored).
- `artifacts/smoke_test_report.json` / `reports/smoke_test_report.md`.

## 5. Test results

- **48 unit tests, all passing** (run with
  `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests`).
  Coverage:
  - `tests/test_metrics.py` — 22 tests (IoU, greedy match, AP,
    evaluation pipeline, score threshold, OOV handling, error
    reporting).
  - `tests/test_dataset.py` — 10 tests (parquet schema, condition
    parsing, sampling, summary, errors).
  - `tests/test_experiments.py` — 8 tests (corruption correctness,
    determinism, defaults, error reporting).
  - `tests/test_visualization_inference.py` — 7 tests (postprocessing,
    rendering, score filtering).
  - `tests/test_yolo_smoke.py` — 1 test (YOLOv8n import + predict,
    skipped if ultralytics is unavailable).

## 6. Resource observations

- **Disk:** venv 1.4 GB, artifacts 19 MB, dataset 135 MB. Plenty of
  headroom on a 256 GB MacBook Air (system was 95% full at start, so
  we kept artifacts small and put nothing in the git tree).
- **Memory:** no observed OOM. Peak working set for Faster R-CNN
  inference was well under 4 GB.
- **Latency (CPU, M1):** Faster R-CNN ≈ 1.3 s/image; YOLOv8n ≈ 0.1 s/image.
- **YOLOv8n weights:** 6.2 MB. Faster R-CNN weights: ~160 MB,
  cached at `~/.cache/torch/hub/checkpoints/`.

## 7. Remaining limitations

- Per-class metrics for Bicycle (2 GT) and Motorcycle (15 GT) are
  noisy.
- Per-condition metrics for `dusttornado` (9 val images) and
  `sand_storm_g2_` (9 val images) are exploratory.
- No calibration — detector scores are not calibrated probabilities.
- No GPU baselines; absolute latencies are CPU-specific. Relative
  ordering should be representative.
- YOLOv8n path is AGPL-3.0 and therefore inappropriate for
  commercial deployment; the Faster R-CNN path is fully permissive.

## 8. Exact commands

```bash
# Setup
cd /Users/sayemuddin/Desktop/WeatherGuard
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install ultralytics    # optional, for YOLO

# Reproduce all measurements
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
PYTHONPATH=. .venv/bin/python scripts/run_full_eval.py --tag fasterrcnn_val_full
PYTHONPATH=. .venv/bin/python scripts/run_full_eval.py --tag yolov8n_val_full --detector yolov8n
PYTHONPATH=. .venv/bin/python scripts/run_corruption_experiment.py --tag fasterrcnn_corruption --limit 10
PYTHONPATH=. .venv/bin/python scripts/run_corruption_experiment.py --tag yolov8n_corruption --detector yolov8n --limit 10
PYTHONPATH=. .venv/bin/python scripts/analyze_confidence.py
PYTHONPATH=. .venv/bin/python scripts/summarize_comparison.py
PYTHONPATH=. .venv/bin/python scripts/render_failure_examples.py
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests

# Launch the dashboard
PYTHONPATH=. .venv/bin/streamlit run app.py
```

## 9. Next three highest-value improvements

1. **Probability calibration** — apply temperature scaling on a
   held-out fold and report post-calibration Brier + reliability.
   Inputs needed: 1–2 hours of work; no new model downloads.
2. **Detector ensemble** — fuse Faster R-CNN and YOLOv8n predictions
   (e.g. confidence-weighted NMS) and measure the change in mAP@.5
   vs. single-detector baselines.
3. **Cross-dataset corruption** — repeat the corruption experiment on
   a second weather dataset (e.g. the adverse-weather subset of BDD100K)
   to disentangle detector brittleness from DAWN-specific artefacts.
