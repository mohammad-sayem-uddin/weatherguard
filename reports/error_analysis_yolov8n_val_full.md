# Error analysis — YOLOv8n on DAWN val (n=206)

## Brier score (lower = better calibration)

- Overall (score>=0.5, IoU>=0.5): **0.1058**  
- Per class:
  - Person: 0.0857  (n=52)
  - Car: 0.1026  (n=415)
  - Bus: 0.1055  (n=11)
  - Truck: 0.2252  (n=20)

> Detector confidences are **not** calibrated probabilities. Treat them as relative scores only.

## Reliability bins (10 equal-width confidence bins)

| bin | count | mean conf | frac correct |
|---|---:|---:|---:|
| 0.0–0.1 | 0 | - | - |
| 0.1–0.2 | 0 | - | - |
| 0.2–0.3 | 0 | - | - |
| 0.3–0.4 | 0 | - | - |
| 0.4–0.5 | 0 | - | - |
| 0.5–0.6 | 102 | 0.549 | 0.892 |
| 0.6–0.7 | 105 | 0.655 | 0.905 |
| 0.7–0.8 | 127 | 0.752 | 0.992 |
| 0.8–0.9 | 134 | 0.850 | 0.963 |
| 0.9–1.0 | 30 | 0.913 | 1.000 |

## Per-condition TP / FP / FN

| condition | TP | FP | FN | P | R |
|---|---:|---:|---:|---:|---:|
| snow_storm | 141 | 5 | 207 | 0.966 | 0.405 |
| sand_storm | 93 | 8 | 173 | 0.921 | 0.350 |
| rain_storm | 87 | 7 | 191 | 0.926 | 0.313 |
| mist | 45 | 1 | 111 | 0.978 | 0.288 |
| sand_storm_g2_ | 31 | 2 | 65 | 0.939 | 0.323 |
| foggy | 30 | 2 | 70 | 0.938 | 0.300 |
| haze | 29 | 2 | 101 | 0.935 | 0.223 |
| dusttornado | 15 | 0 | 49 | 1.000 | 0.234 |
