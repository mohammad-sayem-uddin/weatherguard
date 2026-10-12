# Error analysis — Faster R-CNN on DAWN val (n=206)

## Brier score (lower = better calibration)

- Overall (score>=0.5, IoU>=0.5): **0.1609**  
- Per class:
  - Motorcycle: 0.0912  (n=9)
  - Bus: 0.1053  (n=16)
  - Car: 0.1556  (n=1090)
  - Person: 0.1639  (n=120)
  - Truck: 0.2604  (n=69)

> Detector confidences are **not** calibrated probabilities. Treat them as relative scores only.

## Reliability bins (10 equal-width confidence bins)

| bin | count | mean conf | frac correct |
|---|---:|---:|---:|
| 0.0–0.1 | 0 | - | - |
| 0.1–0.2 | 0 | - | - |
| 0.2–0.3 | 0 | - | - |
| 0.3–0.4 | 0 | - | - |
| 0.4–0.5 | 0 | - | - |
| 0.5–0.6 | 142 | 0.549 | 0.261 |
| 0.6–0.7 | 117 | 0.652 | 0.368 |
| 0.7–0.8 | 147 | 0.749 | 0.503 |
| 0.8–0.9 | 183 | 0.855 | 0.727 |
| 0.9–1.0 | 715 | 0.974 | 0.923 |

## Per-condition TP / FP / FN

| condition | TP | FP | FN | P | R |
|---|---:|---:|---:|---:|---:|
| snow_storm | 252 | 127 | 96 | 0.665 | 0.724 |
| sand_storm | 182 | 70 | 84 | 0.722 | 0.684 |
| rain_storm | 181 | 66 | 97 | 0.733 | 0.651 |
| mist | 95 | 29 | 61 | 0.766 | 0.609 |
| haze | 76 | 22 | 54 | 0.776 | 0.585 |
| sand_storm_g2_ | 61 | 18 | 35 | 0.772 | 0.635 |
| foggy | 56 | 18 | 44 | 0.757 | 0.560 |
| dusttornado | 44 | 7 | 20 | 0.863 | 0.688 |
