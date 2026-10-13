# WeatherGuard — Progress Log

Updated by the autonomous build session. Tracks which stages are complete
and what is currently in flight, so a future run can resume without
repeating finished work.

## MVP (commit `dd51841`) — complete

| Stage | Name                              | Status        |
|-------|-----------------------------------|---------------|
| 0     | Environment & feasibility         | ✅ done       |
| 1     | Environment & inference baseline  | ✅ done       |
| 2     | Dataset & evaluation pipeline     | ✅ done       |
| 3     | Model comparison                  | ✅ done       |
| 4     | Weather & corruption experiments  | ✅ done       |
| 5     | Error analysis & research report  | ✅ done       |
| 6     | Streamlit dashboard               | ✅ done       |
| 7     | Tests & hardening                 | ✅ done       |
| 8     | Final polish & release            | ✅ done       |

## Advanced mission — complete (commit `4503f03`)

| Workstream | Description                          | Status        |
|------------|--------------------------------------|---------------|
| A          | Statistical reliability (bootstrap)  | ✅ done       |
| B          | Advanced detection error taxonomy    | ✅ done       |
| C          | Confidence-threshold sweep           | ✅ done       |
| D          | Deeper robustness (severity curves)  | ✅ done       |
| E          | Preprocessing ablation               | ✅ done       |
| F          | Calibration improvements              | ✅ done       |
| G          | Streamlit dashboard upgrade          | ✅ done       |
| H          | Experiment registry & validator      | ✅ done       |
| I          | Research paper draft                 | ✅ done       |
| J          | Final integration & QA               | ✅ done       |

## Headline advanced-mission numbers

- **FRCNN mAP@.5** = 0.410, 95% paired bootstrap CI [0.34, 0.52].
- **YOLOv8n mAP@.5** = 0.192, 95% paired bootstrap CI [0.16, 0.25].
- **Paired mAP diff (FRCNN − YOLO)** = +0.217, 95% CI [0.15, 0.30].
- **Car/Truck confusion** dominates: 47 GT Cars predicted as Trucks.
- **Blur r=4 paired mAP drop CI** = [−0.30, −0.12] (entirely negative).
- **Noise σ=25 paired mAP drop CI** = [−0.26, −0.07] (entirely negative).
- **CLAHE preprocessing** drops FRCNN mAP by 0.097.
- **FRCNN temperature scaling** worsens Brier (0.052 → 0.073);
  YOLO halves it (0.048 → 0.024).
- **71 unit tests pass** (up from 48).
