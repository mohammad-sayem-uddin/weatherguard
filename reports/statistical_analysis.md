# Statistical analysis — paired image-level bootstrap

- Replicates: **500**, seed: 20261012, IoU=0.5, score=0.5.
- Runtime: 49.2 s.

## Baseline (point estimates on full val set)

| Detector | n | mAP@.5 | Micro P | Micro R | Micro F1 |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN ResNet-50 FPN | 206 | 0.4097 | 0.6622 | 0.6586 | 0.6604 |
| YOLOv8n | 206 | 0.1924 | 0.9023 | 0.3275 | 0.4806 |

## Bootstrap 95% CIs (image-level resampling with replacement)

| Detector | Metric | Mean | Median | 95% CI |
|---|---|---:|---:|---|
| Faster R-CNN ResNet-50 FPN | mAP_50 | 0.4193 | 0.4181 | [0.3405, 0.5177] |
| Faster R-CNN ResNet-50 FPN | micro_precision | 0.6621 | 0.6618 | [0.6298, 0.6931] |
| Faster R-CNN ResNet-50 FPN | micro_recall | 0.6583 | 0.6582 | [0.6262, 0.6908] |
| Faster R-CNN ResNet-50 FPN | micro_f1 | 0.6601 | 0.6609 | [0.6350, 0.6853] |
| YOLOv8n | mAP_50 | 0.1997 | 0.1987 | [0.1596, 0.2459] |
| YOLOv8n | micro_precision | 0.9026 | 0.9034 | [0.8694, 0.9311] |
| YOLOv8n | micro_recall | 0.3293 | 0.3296 | [0.2978, 0.3609] |
| YOLOv8n | micro_f1 | 0.4824 | 0.4831 | [0.4465, 0.5172] |

## Paired difference (Faster R-CNN − YOLOv8n)

- Resampling is *joint*: the same image_ids are drawn for both detectors. Common n = 206.

| Metric | Mean | Median | 95% CI |
|---|---:|---:|---|
| mAP_50 | 0.2185 | 0.2167 | [0.1516, 0.3019] |
| micro_f1 | 0.1800 | 0.1804 | [0.1474, 0.2102] |

> Overlapping CIs are **not** a formal significance test. The paired bootstrap gives a defensible non-parametric interval for the detector-difference distribution.


## Per-condition mAP@.5 bootstrap CIs

| Condition | n_val | Faster R-CNN mAP (95% CI) | YOLOv8n mAP (95% CI) | Note |
|---|---:|---|---|---|
| dusttornado | 9 | 0.494 [0.295, 0.767] | 0.144 [0.047, 0.211] | exploratory (n<15) |
| foggy | 19 | 0.382 [0.117, 0.665] | 0.216 [0.050, 0.461] |  |
| haze | 18 | 0.354 [0.235, 0.500] | 0.200 [0.073, 0.376] |  |
| mist | 18 | 0.321 [0.235, 0.558] | 0.120 [0.068, 0.193] |  |
| rain_storm | 46 | 0.399 [0.280, 0.542] | 0.344 [0.187, 0.490] |  |
| sand_storm | 40 | 0.538 [0.395, 0.687] | 0.222 [0.130, 0.339] |  |
| sand_storm_g2_ | 9 | 0.485 [0.312, 0.637] | 0.230 [0.163, 0.359] | exploratory (n<15) |
| snow_storm | 47 | 0.631 [0.474, 0.777] | 0.396 [0.251, 0.580] |  |

## Warnings

- Condition 'dusttornado' has only 9 val images; bootstrap CIs are exploratory and should not be used for inference.
- Condition 'sand_storm_g2_' has only 9 val images; bootstrap CIs are exploratory and should not be used for inference.