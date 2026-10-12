# WeatherGuard — Experiment Registry Summary

This is a human-readable summary of the machine-readable experiment
registry at `artifacts/experiment_registry.jsonl`. Both files are
append-only — existing entries are never overwritten.

| ID | Detector | Dataset | n | Status | Runtime (s) | Artifacts |
|---|---|---|---:|---|--:|---|
| smoke_test | FRCNN | val | 8 | completed | ~9 | `artifacts/smoke_test_report.json`, `reports/smoke_test_report.md` |
| full_eval_fasterrcnn_val | FRCNN | val | 206 | completed | 298 | `artifacts/eval_fasterrcnn_val_full.json` |
| full_eval_yolov8n_val | YOLOv8n | val | 206 | completed | 27 | `artifacts/eval_yolov8n_val_full.json` |
| corruption_fasterrcnn_5level | FRCNN | val (78 stratified) | 78 | completed | 742 | `corruption_fasterrcnn_corruption.{json,csv}` |
| corruption_yolov8n_5level | YOLOv8n | val (78 stratified) | 78 | completed | 30 | `corruption_yolov8n_corruption.{json,csv}` |
| corruption_fasterrcnn_v2_per_image | FRCNN | val (64 stratified) | 64 | completed | 300 | `corruption_fasterrcnn_corruption_v2_per_image.json` |
| corruption_yolov8n_v2_per_image | YOLOv8n | val (64 stratified) | 64 | completed | 25 | `corruption_yolov8n_corruption_v2_per_image.json` |
| statistical_analysis_500 | both | val | 206 | completed | 50 | `statistical_analysis.{json,csv}`, `reports/statistical_analysis.md` |
| threshold_sweep | both | val | 206 | completed | 1 | `threshold_sweep.{json,csv}`, `reports/threshold_sweep.md` |
| error_taxonomy | both | val | 206 | completed | 1 | `error_taxonomy.{json,csv}`, `reports/error_taxonomy.md` |
| corruption_severity_fasterrcnn | FRCNN | val (40 stratified) | 40 | completed | 1190 | `corruption_severity_fasterrcnn_severity.{json,csv}`, `reports/corruption_severity_fasterrcnn_severity.md` |
| corruption_severity_yolov8n | YOLOv8n | val (40 stratified) | 40 | completed | 230 | `corruption_severity_yolov8n_severity.{json,csv}`, `reports/corruption_severity_yolov8n_severity.md` |
| preprocessing_ablation_fasterrcnn | FRCNN | val (48 stratified) | 48 | completed | 415 | `preprocessing_ablation_fasterrcnn_preproc.{json,csv}`, `reports/preprocessing_ablation_fasterrcnn_preproc.md` |
| calibration_temperature_scaling | both | val | 206 | completed | 1 | `calibration.json`, `reports/calibration.md` |

## Validator

`scripts/validate_artifacts.py` checks that every artifact referenced
by the registry exists and has a non-zero size, that no parquet / yolo
weights are tracked by git, and that all required reports are
present. Run it any time with:

```bash
PYTHONPATH=. .venv/bin/python scripts/validate_artifacts.py
```

The most recent run reported **All checks passed.**
