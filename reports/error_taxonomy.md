# Error taxonomy — Faster R-CNN vs YOLOv8n

IoU=0.5, score≥0.5.

## Faster R-CNN ResNet-50 FPN (n=206)

### Per-class predictions

| class | TP | Localization | Class_confusion | Duplicate | FP_other |
|---|---:|---:|---:|---:|---:|
| Car | 784 | 54 | 19 | 130 | 104 |
| Person | 82 | 6 | 2 | 15 | 59 |
| Bus | 13 | 1 | 4 | 4 | 0 |
| Truck | 35 | 7 | 52 | 41 | 5 |
| Motorcycle | 7 | 1 | 0 | 2 | 1 |
| Bicycle | 0 | 0 | 0 | 0 | 2 |

### Per-class ground truth

| class | Detected | Missed_local | Missed_class | Total |
|---|---:|---:|---:|---:|
| Car | 793 | 0 | 368 | 1161 |
| Bus | 13 | 0 | 21 | 34 |
| Person | 83 | 0 | 28 | 111 |
| Truck | 35 | 0 | 80 | 115 |
| Motorcycle | 8 | 0 | 7 | 15 |
| Bicycle | 0 | 0 | 2 | 2 |

### Class confusion (GT class → predicted class, IoU≥0.5)

| GT | Predicted | n |
|---|---|---:|
| Car | Truck | 47 |
| Truck | Car | 15 |
| Bus | Truck | 5 |
| Bus | Car | 4 |
| Truck | Bus | 3 |
| Car | Person | 1 |
| Car | Bus | 1 |
| Truck | Person | 1 |

### Per-size predictions (binned by nearest GT area)

| size | TP | Localization | Class_confusion | Duplicate | FP_other |
|---|---:|---:|---:|---:|---:|
| small | 79 | 7 | 1 | 5 | 6 |
| medium | 378 | 38 | 29 | 64 | 24 |
| large | 464 | 24 | 47 | 123 | 55 |
| n/a | 0 | 0 | 0 | 0 | 86 |

### Per-condition FP rate (predictions flagged FP_other / total predictions)

| condition | FP_other | total | FP rate |
|---|---:|---:|---:|
| snow_storm | 51 | 411 | 0.124 |
| sand_storm | 43 | 276 | 0.156 |
| rain_storm | 35 | 269 | 0.130 |
| mist | 9 | 141 | 0.064 |
| haze | 17 | 111 | 0.153 |
| foggy | 4 | 85 | 0.047 |
| sand_storm_g2_ | 8 | 85 | 0.094 |
| dusttornado | 4 | 52 | 0.077 |
## YOLOv8n (n=206)

### Per-class predictions

| class | TP | Localization | Class_confusion | Duplicate | FP_other |
|---|---:|---:|---:|---:|---:|
| Car | 395 | 9 | 3 | 4 | 4 |
| Person | 52 | 0 | 1 | 0 | 0 |
| Truck | 13 | 0 | 24 | 2 | 2 |
| Bus | 10 | 0 | 3 | 0 | 0 |

### Per-class ground truth

| class | Detected | Missed_local | Missed_class | Total |
|---|---:|---:|---:|---:|
| Car | 395 | 0 | 766 | 1161 |
| Bus | 10 | 0 | 24 | 34 |
| Person | 52 | 0 | 59 | 111 |
| Truck | 13 | 0 | 102 | 115 |
| Motorcycle | 0 | 0 | 15 | 15 |
| Bicycle | 0 | 0 | 2 | 2 |

### Class confusion (GT class → predicted class, IoU≥0.5)

| GT | Predicted | n |
|---|---|---:|
| Car | Truck | 23 |
| Car | Bus | 3 |
| Truck | Car | 2 |
| Bus | Truck | 1 |
| Truck | Person | 1 |
| Bus | Car | 1 |

### Per-size predictions (binned by nearest GT area)

| size | TP | Localization | Class_confusion | Duplicate | FP_other |
|---|---:|---:|---:|---:|---:|
| small | 7 | 0 | 0 | 0 | 0 |
| medium | 142 | 3 | 6 | 0 | 1 |
| large | 321 | 6 | 25 | 6 | 3 |
| n/a | 0 | 0 | 0 | 0 | 2 |

### Per-condition FP rate (predictions flagged FP_other / total predictions)

| condition | FP_other | total | FP rate |
|---|---:|---:|---:|
| snow_storm | 1 | 161 | 0.006 |
| sand_storm | 2 | 105 | 0.019 |
| rain_storm | 2 | 96 | 0.021 |
| mist | 0 | 47 | 0.000 |
| foggy | 0 | 33 | 0.000 |
| sand_storm_g2_ | 1 | 33 | 0.030 |
| haze | 0 | 31 | 0.000 |
| dusttornado | 0 | 16 | 0.000 |