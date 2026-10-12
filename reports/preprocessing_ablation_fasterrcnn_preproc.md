# Preprocessing ablation

Detector: `fasterrcnn_resnet50_fpn`, 48 stratified source images (seed=42).

All preprocessing is photometric. Source images on disk are not modified; only the in-memory copy that is fed to the detector is preprocessed.

## Aggregate

| preprocessor | mAP@.5 | P | R | F1 | TP | FP | FN | pre s | inf s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity (Original baseline) | 0.462 | 0.674 | 0.655 | 0.665 | 232 | 112 | 122 | 2.8 | 87.9 |
| clahe_mild (Mild CLAHE (L channel, tile 64, cutoff 2)) | 0.365 | 0.630 | 0.582 | 0.605 | 206 | 121 | 148 | 4.4 | 87.7 |
| gamma_brighten (Mild gamma brighten (γ=1.2)) | 0.474 | 0.697 | 0.650 | 0.673 | 230 | 100 | 124 | 3.4 | 83.8 |
| gamma_darken (Mild gamma darken (γ=0.8)) | 0.472 | 0.671 | 0.664 | 0.668 | 235 | 115 | 119 | 2.7 | 60.6 |
| smoothing_r1 (Light Gaussian blur preprocessing (r=1)) | 0.476 | 0.703 | 0.630 | 0.665 | 223 | 94 | 131 | 3.3 | 58.4 |

## Per-condition F1

| preprocessor | dusttornado | foggy | haze | mist | rain_storm | sand_storm | sand_storm_g2_ | snow_storm |
|---|---|---|---|---|---|---|---|---|
| identity | 0.697 | 0.600 | 0.667 | 0.692 | 0.606 | 0.619 | 0.688 | 0.672 |
| clahe_mild | 0.649 | 0.400 | 0.636 | 0.620 | 0.517 | 0.605 | 0.605 | 0.672 |
| gamma_brighten | 0.667 | 0.644 | 0.694 | 0.699 | 0.635 | 0.634 | 0.675 | 0.682 |
| gamma_darken | 0.696 | 0.645 | 0.613 | 0.699 | 0.606 | 0.651 | 0.684 | 0.687 |
| smoothing_r1 | 0.645 | 0.610 | 0.714 | 0.639 | 0.645 | 0.619 | 0.680 | 0.698 |

> Validation-selected comparisons: the same images and detector are used for every preprocessor, so differences are attributable to the preprocessor (modulo the small sample).
