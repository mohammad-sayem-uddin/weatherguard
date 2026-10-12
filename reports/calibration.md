# Calibration audit and temperature scaling

## Definitions

- *Correct* prediction: class matches the matched ground truth AND IoU ≥ 0.5.
- Score threshold for inclusion: 0.5.
- Each detection is one observation in the reliability table.
- Temperature scaling fits a single T on the **fit half** of the val set and is evaluated on the **eval half**.

## Results

### Faster R-CNN ResNet-50 FPN

- n pairs total: 1304, fit: 652, eval: 652
- Fitted T: **3.104**
- Brier (eval half, pre): **0.0520** → post: **0.0725**
- NLL (eval half, pre): **0.2234** → post: **0.2923**

Reliability (post-calibration, 10 bins):

| bin | count | mean conf | frac correct | gap |
|---|---:|---:|---:|---:|
| 0.0–0.1 | 0 | - | - | - |
| 0.1–0.2 | 0 | - | - | - |
| 0.2–0.3 | 0 | - | - | - |
| 0.3–0.4 | 0 | - | - | - |
| 0.4–0.5 | 0 | - | - | - |
| 0.5–0.6 | 0 | - | - | - |
| 0.6–0.7 | 15 | 0.697 | 0.867 | -0.169 |
| 0.7–0.8 | 318 | 0.752 | 0.918 | -0.166 |
| 0.8–0.9 | 295 | 0.845 | 0.976 | -0.131 |
| 0.9–1.0 | 24 | 0.915 | 0.958 | -0.043 |
### YOLOv8n

- n pairs total: 498, fit: 249, eval: 249
- Fitted T: **0.214**
- Brier (eval half, pre): **0.0478** → post: **0.0241**
- NLL (eval half, pre): **0.2273** → post: **0.1822**

Reliability (post-calibration, 10 bins):

| bin | count | mean conf | frac correct | gap |
|---|---:|---:|---:|---:|
| 0.0–0.1 | 0 | - | - | - |
| 0.1–0.2 | 0 | - | - | - |
| 0.2–0.3 | 0 | - | - | - |
| 0.3–0.4 | 0 | - | - | - |
| 0.4–0.5 | 0 | - | - | - |
| 0.5–0.6 | 0 | - | - | - |
| 0.6–0.7 | 0 | - | - | - |
| 0.7–0.8 | 0 | - | - | - |
| 0.8–0.9 | 0 | - | - | - |
| 0.9–1.0 | 249 | 0.998 | 0.976 | 0.022 |