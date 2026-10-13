# WeatherGuard — Experiment Log

Chronological log of executed experiments. The canonical
machine-readable record is `artifacts/experiment_registry.jsonl`.

| # | Time (UTC) | Experiment | Detector | n_images | Status | Runtime (s) | Artifact |
|--:|---|---|---|---:|---|--:|---|
| 1 | 2026-10-12 18:00Z | smoke_test | FRCNN | 8 | completed | ~9 | `artifacts/smoke_test_report.json` |
| 2 | 2026-10-12 18:57Z | full_eval_fasterrcnn_val | FRCNN | 206 | completed | 298 | `artifacts/eval_fasterrcnn_val_full.json` |
| 3 | 2026-10-12 19:04Z | full_eval_yolov8n_val | YOLOv8n | 206 | completed | 27 | `artifacts/eval_yolov8n_val_full.json` |
| 4 | 2026-10-12 19:04Z | corruption_fasterrcnn_5level | FRCNN | 78 | completed | 742 | `artifacts/corruption_fasterrcnn_corruption.json` |
| 5 | 2026-10-12 19:17Z | corruption_yolov8n_5level | YOLOv8n | 78 | completed | 30 | `artifacts/corruption_yolov8n_corruption.json` |
| 6 | 2026-10-12 21:16Z | threshold_sweep | both | 206 | completed | 1 | `artifacts/threshold_sweep.json` |
| 7 | 2026-10-12 21:16Z | error_taxonomy | both | 206 | completed | 0.6 | `artifacts/error_taxonomy.json` |
| 8 | 2026-10-12 21:18Z | corruption_fasterrcnn_v2_per_image | FRCNN | 64 | completed | 300 | `artifacts/corruption_fasterrcnn_corruption_v2_per_image.json` |
| 9 | 2026-10-12 21:25Z | corruption_yolov8n_v2_per_image | YOLOv8n | 64 | completed | 25 | `artifacts/corruption_yolov8n_corruption_v2_per_image.json` |
| 10 | 2026-10-12 21:26Z | statistical_analysis_500 | both | 206 | completed | 50 | `artifacts/statistical_analysis.json` |
| 11 | 2026-10-12 21:35Z | calibration_temperature_scaling | both | 206 | completed | 1 | `artifacts/calibration.json` |
| 12 | 2026-10-12 21:26Z | corruption_severity_fasterrcnn | FRCNN | 40 | completed | 1190 | `artifacts/corruption_severity_fasterrcnn_severity.json` |
| 13 | 2026-10-12 21:47Z | corruption_severity_yolov8n | YOLOv8n | 40 | completed | 230 | `artifacts/corruption_severity_yolov8n_severity.json` |
| 14 | 2026-10-12 21:47Z | preprocessing_ablation_fasterrcnn | FRCNN | 48 | completed | 415 | `artifacts/preprocessing_ablation_fasterrcnn_preproc.json` |
| 15 | 2026-10-12 21:55Z | artifact_validation | — | — | passed | 0 | (script ran) |
| 16 | 2026-10-13 11:52Z | dev_eval_fasterrcnn | FRCNN | 821 (dev) | completed | 1593 | `artifacts/eval_dev_fasterrcnn.json` |
| 17 | 2026-10-13 12:19Z | dev_eval_yolov8n | YOLOv8n | 821 (dev) | completed | 150 | `artifacts/eval_dev_yolov8n.json` |
| 18 | 2026-10-13 12:22Z | heldout_eval_fasterrcnn | FRCNN | 202 (clean heldout) | completed | 436 | `artifacts/eval_heldout_fasterrcnn.json` |
| 19 | 2026-10-13 12:22Z | heldout_eval_yolov8n | YOLOv8n | 202 (clean heldout) | completed | 26 | `artifacts/eval_heldout_yolov8n.json` |
| 20 | 2026-10-13 12:34Z | threshold_transfer_study | both | 821+202 | completed | 50 | `artifacts/threshold_transfer.json` + `artifacts/threshold_transfer.csv` + `artifacts/plots/threshold_transfer.png` |

**Detector load times (CPU, M1):**
- Faster R-CNN ResNet-50 FPN: ~13.6s first run (cold cache),
  ~0.6s on subsequent warm loads.
- YOLOv8n: ~0.6s on first run, ~0.03s on warm loads.

**Failed or blocked experiments:** none.
