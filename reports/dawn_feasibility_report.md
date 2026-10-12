# DAWN Dataset Feasibility Report

**Project:** WeatherGuard
**Author:** Autonomous AI research engineer
**Date:** 2026-10-12
**Status:** ✅ Dataset validated for use in WeatherGuard.

---

## 1. Source

- **Repository:** <https://huggingface.co/datasets/Maxim37/dawn-dataset>
- **Local files inspected:**
  - `train-00000-of-00001.parquet` — 105 MB (101 MiB on disk)
  - `val-00000-of-00001.parquet` — 34.2 MB (33 MiB on disk)
- **License:** CC BY-NC 3.0 (as listed on the Hugging Face card).
  Non-commercial research use only. Images will not be redistributed in the
  WeatherGuard repository. Attribution will be preserved in
  `CITATION.cff` and `README.md`.
- **Authentication:** No authentication required to download from
  Hugging Face public storage.

## 2. Schema

The two parquet files share an identical Apache Parquet schema:

| Column   | Type                                                                                              |
|----------|------------------------------------------------------------------------------------------------------|
| image_id | string                                                                                              |
| image    | struct<bytes: binary, path: string>                                                                 |
| width    | int64                                                                                               |
| height   | int64                                                                                               |
| objects  | list<struct<class: int64, class_name: string, height: double, width: double, x_min: double, y_min: double>> |

- `image.bytes` is JPEG-encoded (JFIF header confirmed in samples).
- Bounding boxes use the COCO `[x_min, y_min, width, height]` convention in
  absolute pixel coordinates.
- The `class` integer and the `class_name` string are consistent across both
  splits.

### Class mapping (train ∪ val)

| class_id | class_name  |
|----------|-------------|
| 0        | Truck       |
| 1        | Person     |
| 2        | Bicycle    |
| 3        | Car        |
| 4        | Motorcycle |
| 5        | Bus        |

This is a 6-class subset aligned with common COCO categories, which makes
it directly compatible with the COCO-pretrained detectors we plan to use.
The detector's label space must be reduced to these 6 classes during
evaluation; raw COCO predictions outside this set are reported as
"out-of-vocabulary" detections.

### Weather-condition labels

There is **no dedicated `weather` field**. The condition is encoded as a
delimited prefix in the `image_id` string, before the first `-`. Examples:
`foggy-001`, `sand_storm-003`, `dusttornado-015`, `rain_storm-007`,
`snow_storm-002`, `haze-005`, `mist-009`.

Conditions found:

| Condition        | Train | Val |
|------------------|------:|----:|
| sand_storm       |   192 |  40 |
| snow_storm       |   157 |  47 |
| rain_storm       |   154 |  46 |
| haze             |    96 |  18 |
| foggy            |    85 |  19 |
| mist             |    64 |  18 |
| sand_storm_g2_   |    44 |   9 |
| dusttornado      |    27 |   9 |
| sand_storm_g2    |     2 |   0 |

**Note on `sand_storm_g2_`:** the trailing underscore is present in the
source dataset and is preserved verbatim by WeatherGuard. The two-image
`sand_storm_g2` bucket (no underscore) appears to be a minor dataset
authoring typo; it is preserved as its own condition for transparency.
The dataset authors should be cited as the authoritative source.

## 3. Sample statistics

| Split | Images | Object instances | Mean objects/image | Image W range | Image H range |
|-------|------:|-----------------:|-------------------:|---------------|---------------|
| train |   821 |            6,407 |               7.80 |   254..2,667  |   161..1,950  |
| val   |   206 |            1,438 |               6.98 |   261..2,390  |   187..1,542  |

All 1,438 bounding boxes in the validation split were checked for
geometric validity: 0 boxes had negative coordinates, 0 boxes had
non-positive width or height, and 0 boxes extended outside the image
canvas. All 1,438 are usable as ground-truth annotations for object
detection evaluation.

Three sample images were decoded from raw bytes with Pillow. All decoded
successfully as RGB JPEGs and the rendered bounding boxes visually align
with the annotated objects (see `scratchpad/sample_images/*.jpg`).

## 4. Suitability for the WeatherGuard research questions

| Research question                                                      | Feasible? | Notes |
|------------------------------------------------------------------------|-----------|-------|
| Detect objects on real adverse-weather traffic imagery                 | ✅ Yes    | 1,027 images total, six COCO-compatible classes. |
| Which classes are hardest to detect?                                   | ✅ Yes    | Class imbalance is real (Cars dominate), but rare classes (Bicycle: 26 total) can still be reported qualitatively. |
| False-positive / false-negative patterns                                | ✅ Yes    | Ground truth + image IDs support both. |
| Confidence–correctness relationship                                     | ✅ Yes    | Detector scores are produced for every prediction; can correlate with TP/FP. |
| Controlled corruption robustness experiment                             | ✅ Yes    | The image geometry is preserved exactly; we can corrupt pixel values without modifying annotations. Annotation geometry is unchanged by brightness/contrast/noise/blur. |
| Condition-wise comparison                           | ⚠ Partial | Some conditions (`dusttornado`, `sand_storm_g2`) have ≤ 9 val images. We will use the val split for condition-wise statistics with appropriate confidence-interval caveats. |
| Claiming novel research findings                                        | ❌ No     | A small literature review will be included in the README, and we will not claim novelty beyond what the literature review supports. |

## 5. Risks / threats to validity

1. **Class imbalance.** Cars make up ~83% of instances. Per-class metrics
   on rare classes (Bicycle, Motorcycle) will be noisy and reported with
   caveats.
2. **Small validation split.** 206 images yields limited statistical
   power for fine-grained condition breakdowns.
3. **No severity labels within a condition.** "rain_storm" covers a range
   of intensities; we treat the condition as a categorical label only.
4. **Synthetic corruption ≠ real weather.** We will explicitly state that
   the corruption experiment is a controlled-robustness benchmark, not a
   real-world weather-robustness claim.
5. **Dataset license.** CC BY-NC 3.0 forbids commercial use; downstream
   WeatherGuard deployment must respect this.

## 6. Decision

**Proceed with DAWN.** The dataset is accessible without authentication,
the schema is well-formed, the annotations are geometrically valid, and
the labels support all four primary and secondary research questions.
WeatherGuard will use:

- `train-00000-of-00001.parquet` as the training/qualitative pool (we
  will *not* retrain on it; we use it for additional prediction
  inference and visualization).
- `val-00000-of-00001.parquet` as the held-out evaluation set for
  metrics, condition-wise breakdowns, and the corruption robustness
  experiment.

Attribution will be added to `CITATION.cff`. No dataset images will be
checked into the WeatherGuard git repository.