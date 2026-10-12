# WeatherGuard — Progress Log

Updated by the autonomous build session. Tracks which stages are complete
and what is currently in flight, so a future run can resume without
repeating finished work.

| Stage | Name                              | Status        |
|-------|-----------------------------------|---------------|
| 0     | Environment & feasibility         | ✅ done       |
| 1     | Environment & inference baseline  | ✅ done       |
| 2     | Dataset & evaluation pipeline     | 🔄 in progress|
| 3     | Model comparison                  | ⏳ pending    |
| 4     | Weather & corruption experiments  | ⏳ pending    |
| 5     | Error analysis & research report  | ⏳ pending    |
| 6     | Streamlit dashboard               | ⏳ pending    |
| 7     | Tests & hardening                 | ⏳ pending    |
| 8     | Final polish & release            | ⏳ pending    |

## Stage 1 — completed
- Python 3.11.15 venv, torch 2.14.1 + torchvision 0.29.1 (CPU).
- Faster R-CNN ResNet-50 FPN loaded in 13.6s on CPU.
- 8-image smoke test: 214 detections, median 1.1 s/image.
- mAP@.5 = 0.561 (Cars P=0.756, R=0.721; Person 1/1; Truck 0/4 — useful failure pattern).
- Report: `reports/smoke_test_report.md` + `artifacts/smoke_test_report.json`.
