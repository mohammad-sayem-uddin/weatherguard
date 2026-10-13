# WeatherGuard — Progress Log

## MVP (commit `dd51841`) — complete

| Stage | Name | Status |
|---|---|---|
| 0 | Environment & feasibility | done |
| 1 | Environment & inference baseline | done |
| 2 | Dataset & evaluation pipeline | done |
| 3 | Model comparison | done |
| 4 | Weather & corruption experiments | done |
| 5 | Error analysis & research report | done |
| 6 | Streamlit dashboard | done |
| 7 | Tests & hardening | done |
| 8 | Final polish & release | done |

## Advanced mission (commit `4503f03`) — complete

All 10 workstreams done; see `reports/advanced_mission_final_report.md`.

## Focused research mission — in flight

Primary research question:

> How well do confidence thresholds selected on development data transfer
> to held-out adverse-weather traffic images, and what trade-offs do
> they create between detection quality, missed objects, false alarms,
> and inference speed?

Sub-questions: RQ1 (Faster R-CNN vs YOLOv8n under same protocol),
RQ2 (development-set threshold → held-out), RQ3 (per-condition /
per-class transfer), RQ4 (detector+operating-point trade-off against
CPU latency).

| Phase | Activity | Status |
|---|---|---|
| 1 | Audit (splits, code, prior results, fair-comparison) | in flight |
| 2 | Re-run dev + val eval on current code, freeze cached predictions | pending |
| 3 | Threshold selection on dev, locked before val evaluation | pending |
| 4 | Held-out evaluation; per-condition, per-class, bootstrap | pending |
| 5 | Latency benchmark (warm-up vs steady-state) | pending |
| 6 | Reports and figures | pending |
| 7 | Paper draft + README rewrite | pending |
