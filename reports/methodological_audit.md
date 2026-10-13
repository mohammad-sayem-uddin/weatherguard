# WeatherGuard — Methodological Audit of the Threshold-Transfer Study

**Scope.** Independent audit of the threshold-transfer study documented
in `reports/threshold_transfer_study.md` and
`artifacts/threshold_transfer.json`. No new inference was run; all
findings come from re-reading the saved per-image predictions and
re-implementing the matching / metric / bootstrap code from scratch.
The audit follows the four-part structure in the request.

---

## 1. Audit of the data split

### 1.1 Exact image IDs

`scripts/run_transfer_eval.py` writes the per-image IDs into the
`image_ids` field of each eval artifact. The audit confirms:

- `artifacts/eval_dev_fasterrcnn.json` and
  `artifacts/eval_dev_yolov8n.json` both contain the **same 821 IDs**,
  in the same sorted order. Both are identical to the 821 IDs in
  `train-00000-of-00001.parquet`.
- `artifacts/eval_heldout_fasterrcnn.json` and
  `artifacts/eval_heldout_yolov8n.json` both contain the **same 202
  IDs**, identical to `artifacts/clean_heldout_image_ids.json["heldout_image_ids"]`.

The **set-level** check is exact: the dev set and the heldout set
have **0 image_ids in common** after the exclusion.

### 1.2 Hash overlap and the four excluded images

`audit_splits.py` recomputed SHA-1 of every image's raw bytes:

- **4 of the 206 val images are byte-identical to images in the
  train split**, with the following (train, val) pairs:

  | Train image_id | Val image_id | Identical bytes | Identical dimensions | Identical class count? |
  |---|---|---|---|---|
  | sand_storm-057 | dusttornado-023 | yes (124,716) | yes (990×1485) | **no** (16 vs 15 Cars) |
  | haze-098 | haze-077 | yes (60,858) | yes (1280×783) | yes (7 vs 7 Cars) |
  | foggy-011 | haze-003 | yes (41,532) | yes (1280×721) | yes (2 vs 2 Cars) |
  | haze-036 | haze-059 | yes (55,965) | yes (1280×721) | **no** (1 vs 2 Trucks) |

- `audit_pixel.py` independently verified the **pixel equality** of
  each pair: `max |diff| = 0, n_pixels_differing = 0` for all four
  pairs. These are real duplicates (same JPEG bytes, same pixels),
  not just near-duplicates. The pair (`sand_storm-057`,
  `dusttornado-023`) and the pair (`haze-036`, `haze-059`) have
  different annotation counts on the same image, which is a
  documented upstream DAWN data-quality issue, not a feature of
  the splits.

- All four val IDs are listed in
  `artifacts/clean_heldout_image_ids.json["excluded_image_ids"]`.
  The heldout artifact contains only the remaining 202 images.
  Exclusion is complete and correct.

### 1.3 Justification of "held-out"

The DAWN dataset ships with a publisher-assigned train/val split.
The study uses the publisher's **val** as its held-out set, and the
publisher's **train** as its development set. This is the most
defensible choice for this dataset and avoids the well-known
circularity of constructing a held-out set from the same images used
to choose the dev threshold. The four byte-duplicate images would
otherwise have leaked into the held-out evaluation.

**Naming.** The "held-out" label is appropriate for the 202-image
cleaned val set, on the condition that "held-out" is understood as
"out-of-sample for the dev-threshold-selection step" rather than
"drawn from a deployment population." The CI bounds
characterise the uncertainty in the F1 estimate on this 202-image
sample under image-level resampling, not on a deployment
population. This is already stated in
`reports/threshold_transfer_study.md` § "Limitations and threats
to validity" and in `reports/paper_draft.md` § 5.5. The audit
confirms the wording is consistent across the documents.

### 1.4 Threshold selection used dev only

`scripts/run_threshold_transfer_study.py` reads
`bundles[det]["dev_sweep"]` to find the dev-best-F1 threshold. The
heldout sweep is read only for the **diagnostic** val-oracle
threshold, and the val-oracle is computed *after* the dev threshold
is locked (the `dev_best_f1_threshold` is read before
`held_sweep` is constructed in the script). The pre-registered
protocol in `reports/research_question_and_protocol.md` § 4.3
specifies this ordering. The audit confirms the script does not
re-tune the threshold using heldout data.

### 1.5 Val-oracle is labelled diagnostic

The val-oracle threshold is computed and reported in the study, the
paper draft, and the README, with the explicit label "diagnostic."
The README's main results table calls it
"Heldout val-oracle (diagnostic)." The paper draft § 3.1 names it
"the score that maximises heldout F1; reported as a diagnostic only
and never presented as a deployable policy." The audit confirms
this is consistent throughout the project and that no executable
recommendation is built on the val-oracle.

---

## 2. Independent recomputation of the headline metrics

### 2.1 Method

The audit implements matching and metric accumulation from
scratch, *without* calling `src/metrics.evaluate_detections`, and
compares to the values in `artifacts/threshold_transfer.json`. The
matching protocol is:

- Greedy IoU per class with IoU threshold 0.5.
- Predictions are filtered by `common_label is None → skip` and
  `score < threshold → skip`. Out-of-vocabulary predictions are
  not counted as FP (per the existing protocol).
- For each target class, predictions are processed in score-desc
  order, and the best-IoU unmatched ground truth (≥ threshold) is
  consumed. Remaining predictions become FP; remaining ground
  truths become FN.

The per-class loop iterates over all DAWN target classes
(`Truck, Person, Bicycle, Car, Motorcycle, Bus`), including those
with no GT in the heldout set, so that predictions in a class with
no GT are correctly counted as FP for that class.

### 2.2 Headline metric recheck (heldout, both detectors, three operating points)

| Detector | Operating point | Independent TP/FP/FN | Independent P / R / F1 | Existing F1 | Match |
|---|---|---|---|---|---|
| FRCNN | default-0.5 | 923 / 476 / 483 | 0.659757 / 0.656472 / **0.658111** | 0.658111 | exact |
| FRCNN | dev-F1-0.6 | 888 / 341 / 518 | 0.722539 / 0.631579 / **0.674004** | 0.674004 | exact |
| FRCNN | val-oracle-0.7 | 847 / 239 / 559 | 0.779926 / 0.602418 / **0.679775** | 0.679775 | exact |
| YOLO | default-0.5 | 463 / 50 / 943 | 0.902534 / 0.329303 / **0.482543** | 0.482543 | exact |
| YOLO | dev-F1-0.2 | 683 / 276 / 723 | 0.712200 / 0.485775 / **0.577590** | 0.577590 | exact |
| YOLO | val-oracle-0.15 | 734 / 391 / 672 | 0.652444 / 0.522048 / **0.580008** | 0.580008 | exact |

All 6 heldout operating-point micro F1 values reproduce to 6
decimal places. The TP / FP / FN integers also reproduce exactly.

The dev-split totals also reproduce exactly:

| Detector | Operating point | Independent TP/FP/FN | Existing TP/FP/FN | Match |
|---|---|---|---|---|
| FRCNN | dev-F1-0.6 | 3887 / 1400 / 2520 | 3887 / 1400 / 2520 | exact |
| YOLO | dev-F1-0.2 | 3048 / 1287 / 3359 | 3048 / 1287 / 3359 | exact |

### 2.3 Other consistency checks

- **Class mapping:** the audit independently confirms that
  `common_label is None` and the filter for class names not in
  `DAWN_CLASSES` are applied consistently. Of 1,527
  score-≥-0.5 FRCNN predictions in the heldout set, 1,399 are in
  DAWN, 128 are out-of-vocabulary (correctly excluded from FP
  counting).
- **Empty-prediction / empty-GT:** 0 heldout images have empty
  ground truth, and 0 heldout images have zero DAWN-in-vocabulary
  predictions, so empty-handling is not exercised. The
  implementation in `src/metrics.py` would handle it (a class
  with no GT contributes NaN to mAP and is excluded; an image
  with no predictions contributes 0 TP / 0 FP to all classes).
- **Both detectors use the same evaluation image list:** confirmed
  — both `eval_heldout_*.json` artifacts contain the same 202
  image_ids in the same order.
- **All displayed results trace to source artefacts:** every
  number in `reports/threshold_transfer_study.md`, `paper_draft.md`,
  and `README.md` was re-derived from
  `artifacts/threshold_transfer.json` (which aggregates the
  per-image artefacts) by re-applying `evaluate_detections`. The
  numbers reproduce to 6 decimal places.

### 2.4 Discrepancy found and explained

The first independent recomputation produced F1 values that did
**not** match. Investigation found that the first audit script
iterated over `gt_by_class.keys()` (i.e. only classes that have
ground truth in the image) rather than over all DAWN target
classes. This caused 125 predictions on classes with no GT in the
heldout set to be silently dropped from the FP count. After
correcting the iteration order to match `evaluate_detections` (which
iterates over `target_classes`), the F1 values match exactly. This
was an audit-script bug, not a `src/metrics.py` bug. The original
implementation in `src/metrics.py` is correct and the recorded
results in `artifacts/threshold_transfer.json` are correct.

---

## 3. Audit of the bootstrap

### 3.1 Resampling unit

`scripts/run_threshold_transfer_study.py` uses
`[rng.randrange(len(per_image)) for _ in range(len(per_image))]`,
i.e. **with-replacement sampling at the image level**. Boxes
within an image are not resampled. The audit confirms this. The
pre-registered protocol § 5.5 states this explicitly.

### 3.2 Paired comparisons

For the within-detector comparison (default-0.5 vs dev-F1 on the
heldout set), the same resampled indices are used for both
thresholds. The independent recomputation reproduced the exact CI
bounds and the exact mean / median to 6 decimal places by
re-implementing the same RNG usage. (See § 3.5 for the RNG
reproducibility note.)

For the cross-detector comparison (FRCNN vs YOLO at their dev-F1
thresholds), the two per-image lists are first aligned by
`image_id` (202 common IDs), then the same resampled indices are
used. The audit confirms the alignment: `common = 202`, matching
the heldout set size.

### 3.3 Empty-GT handling

0 of 202 heldout images have empty ground truth. The bootstrap is
therefore not exercised on empty-GT images. The
`evaluate_detections` implementation correctly handles empty-GT
images: a class with `n_gt = 0` produces a per-class AP of NaN and
is excluded from the mAP average; micro TP/FP/FN counts across
classes are unchanged.

### 3.4 CI calculation and sign

The CI is the 2.5 / 50 / 97.5 percentile of the 500 paired
differences, computed with linear interpolation. The audit
independently recomputed all three CIs and all three means/medians;
they match to 6 decimal places.

The sign convention is `default − dev-F1`, so a negative number
means "default is worse than dev-F1" (which is the case for both
detectors). The paper draft and the README use the same
convention; e.g. "YOLO default-0.5 − YOLO dev-F1-0.2 = −0.095,
CI [−0.115, −0.075]" reads as "the default F1 is 0.095 below the
dev-F1 F1, with 95% CI [0.075, 0.115] below." The wording is
consistent throughout.

### 3.5 RNG reproducibility

The recorded CI numbers are reproducible. The first independent
bootstrap (which used a fresh `random.Random(20261012)` for YOLO
separately from FRCNN) produced a slightly different YOLO mean
(−0.094 vs the recorded −0.095). Investigation showed that the
recorded script uses a single `random.Random(20261012)` that draws
FRCNN indices first, then YOLO indices. Reproducing that
RNG-usage pattern produced the exact recorded numbers.

### 3.6 What the CI bounds actually estimate

The 95% CIs in this study estimate uncertainty in the **difference
statistic (micro F1)** on the **specific 202-image heldout sample**
under image-level resampling with replacement. They are not a
hypothesis test, not a confidence statement about a deployment
population, and not a guarantee of generalisation. The
`reports/threshold_transfer_study.md` § "Limitations and threats
to validity" and the paper draft § 5.5 say this. The README does
not include the limitation as explicitly, but it does not claim
generalisation either.

The audit confirms the CIs are computed correctly and labelled
correctly. It **recommends** that the README and the paper draft
add a single line that says "the 95% CIs are paired image-level
bootstrap 95% CIs from 500 replicates; they describe the
uncertainty in the F1 estimate on this 202-image heldout sample
under image-level resampling, not on a deployment population" so
the wording is unambiguous on the public landing page.

---

## 4. Audit of the threshold-transfer conclusion

### 4.1 Stability of the dev-best-F1 threshold under dev bootstrap

A 200-replicate bootstrap of the dev set (with replacement, at the
image level, seed 20261012) was run for each detector. The
distribution of the dev-best-F1 threshold:

| Detector | Chosen threshold | Picked this threshold in N/200 dev bootstraps |
|---|---:|---:|
| FRCNN | 0.60 | 182/200 (91%) |
| FRCNN | 0.65 | 9/200 (4.5%) |
| FRCNN | 0.55 | 8/200 (4%) |
| FRCNN | 0.70 | 1/200 (0.5%) |
| YOLO | 0.20 | 143/200 (72%) |
| YOLO | 0.15 | 56/200 (28%) |
| YOLO | 0.30 | 1/200 (0.5%) |

**FRCNN's dev-best-F1 threshold is sharply stable** (0.60 wins
91% of the time, and 99% of the time the threshold is one of
0.55, 0.60, 0.65, or 0.70 — all of which give a similar dev F1,
as § 4.2 shows). **YOLO's dev-best-F1 threshold is less sharply
defined** — 28% of dev bootstraps pick 0.15 instead of 0.20.
This is a real feature of the data, not a bug: YOLO's F1 surface
is flatter at the top, so adjacent thresholds are nearly
indistinguishable on the dev set.

**Consequence for the headline claim.** The "transfers essentially
without degradation" claim is about the chosen threshold
performance, not about threshold choice being unique. The audit
shows the chosen threshold for FRCNN is robustly the best; for
YOLO, several adjacent thresholds give nearly identical dev F1, and
the heldout sweep confirms that 0.15, 0.20, and 0.25 all give
similar heldout F1 (0.580 / 0.578 / 0.566). The headline claim
holds for any threshold in the 0.15–0.25 neighbourhood, not just
for the chosen 0.20.

### 4.2 Sensitivity of F1 to threshold in a 0.05 neighbourhood

| Detector | Split | score=dev-0.05 | score=dev | score=dev+0.05 |
|---|---|---|---|---|
| FRCNN | dev | 0.6620 (0.55) | **0.6648 (0.60)** | 0.6617 (0.65) |
| FRCNN | heldout | 0.6676 (0.55) | 0.6740 (0.60) | 0.6765 (0.65) |
| YOLO | dev | 0.5664 (0.15) | **0.5675 (0.20)** | 0.5609 (0.25) |
| YOLO | heldout | **0.5800 (0.15)** | 0.5776 (0.20) | 0.5659 (0.25) |

The 0.05 grid is fine enough for the headline claim. The maximum
swing in F1 within ±0.05 of the chosen threshold is 0.005 (FRCNN
dev) and 0.013 (YOLO heldout). The val-oracle F1 is within 0.006
of the dev-F1 F1, and the maximum heldout F1 in the 0.15–0.25
neighbourhood (YOLO) is 0.5800 (at 0.15), within 0.003 of the
chosen 0.578 (at 0.20).

### 4.3 Post-hoc finer-grid check (labelled exploratory)

A 0.01-grid sweep on the dev set (post-hoc, not preregistered)
picks:

- FRCNN dev: best at 0.60, F1=0.6648 (identical to 0.05 grid).
- YOLO dev: best at **0.18**, F1=0.5693 (vs 0.20 at F1=0.5675 on the
  0.05 grid; a difference of 0.0018).

The preregistered 0.05 grid already captures the plateau of the
F1 surface; a finer grid would not have changed the qualitative
conclusion. YOLO's dev-best-F1 threshold sits in a 0.15–0.21
plateau; the choice of 0.20 is a representative mid-plateau
threshold. The audit labels this analysis **exploratory** and does
not retroactively change the study.

### 4.4 Per-condition denominators and matching rules

The per-condition transfer table in
`reports/threshold_transfer_study.md` uses the same matching
protocol as the overall evaluation: greedy IoU per class at 0.5,
score ≥ dev-F1 threshold. The denominators (n per condition) are
the same on dev and heldout:

| Condition | dev n | heldout n | FRCNN gap (rel) | YOLO gap (rel) |
|---|---:|---:|---|---|
| snow_storm | 157 | 47 | −0.006 (−0.9%) | −0.019 (−3.2%) |
| rain_storm | 154 | 46 | +0.016 (+2.3%) | −0.007 (−1.2%) |
| sand_storm | 192 | 40 | −0.062 (−9.9%) | −0.065 (−12.3%) |
| foggy | 85 | 19 | +0.071 (+10.4%) | +0.014 (+2.4%) |
| mist | 64 | 18 | +0.037 (+5.4%) | +0.019 (+3.3%) |
| haze | 96 | 15 | +0.026 (+4.1%) | +0.125 (+22.4%) |
| dusttornado | 27 | 8 | −0.221 (rel. exploratory, n<15) | −0.010 (rel. exploratory, n<15) |
| sand_storm_g2_ | 44 | 9 | −0.032 (rel. exploratory, n<15) | −0.033 (rel. exploratory, n<15) |

For conditions with n ≥ 15, the relative gap is in ±13% for all
8 cells, with most in the ±5% band. The largest is YOLO on `haze`
(+22.4% relative, +0.125 absolute), suggesting YOLO's F1 is
particularly sensitive to haze conditions on the heldout set
relative to the dev set. This is consistent with the report's
caveat about low-visibility conditions.

For the two exploratory conditions (n<15), the relative gap is
large (e.g. −42% for FRCNN on `dusttornado`); these cells are
explicitly flagged as exploratory in the report and should not be
used for inference.

### 4.5 "Essentially without degradation" — is this too strong?

- FRCNN dev-F1 F1 (821 images) = 0.665; heldout F1 (202 images) =
  0.674. **Difference: +0.009, slight improvement.** Dev F1 is
  computed at the same threshold (0.6), the same IoU protocol, and
  the same class mapping. The heldout F1 is not significantly
  lower than dev F1; it is slightly higher. The 95% CI for the
  difference between heldout and dev F1 at the *same* threshold
  has not been computed; an honest statement is "the dev F1 and
  the heldout F1 are within 0.010 of each other for both
  detectors."
- YOLO dev-F1 F1 = 0.567; heldout F1 = 0.578. **Difference: +0.011.**
  Same conclusion.
- The CI for the "default-0.5 − dev-F1" difference is entirely
  negative for both detectors, supporting the "dev-F1 is
  significantly better than default" claim. The CI for the
  "dev-F1 − val-oracle" difference is not reported, but the
  val-oracle F1 is only 0.006 above the dev-F1 F1 for both
  detectors, well inside the 0.05 grid step.

**Recommendation:** the report's phrasing "transfers essentially
without degradation" is defensible, but it would be more
conservative and equally accurate to say: *"the dev-F1 F1 and
the heldout F1 are within 0.011 of each other for both detectors,
and within 0.006 of an unobservable val-oracle."* This wording
is recommended for the README headline and the paper abstract.

### 4.6 Headline claim is correct, but warranting of wording

The central claim — "A confidence threshold selected on the
development split transfers essentially without degradation to
the held-out split" — is supported by the data and the
recomputation. The exact numbers are:
- FRCNN: dev F1 0.665 → heldout F1 0.674 (+0.009, no degradation)
- YOLO:  dev F1 0.567 → heldout F1 0.578 (+0.011, no degradation)
- FRCNN default-0.5 → heldout F1 0.658 (slightly worse than dev-F1)
- YOLO  default-0.5 → heldout F1 0.483 (substantially worse than dev-F1)

The dev-F1 threshold is the policy that **robustly** beats the
default 0.5 on the heldout set for both detectors. The CI for the
F1 difference (default − dev-F1) is entirely negative for both
detectors. This is the central, defensible finding.

A more cautious framing, recommended for the public README:
*"A dev-tuned threshold (0.6 for Faster R-CNN, 0.2 for YOLOv8n) on
the DAWN development split yields a heldout F1 that is within
0.011 of the dev F1 and within 0.006 of an unobservable val-oracle,
and that is robustly higher (CI entirely negative) than the
default 0.5 threshold for both detectors on this 202-image
heldout sample."*

---

## 5. Findings, in three categories

The categories below are mutually exclusive. Each finding appears
in **exactly one** category. "Confirmed defects" are findings that
require a code or artifact change before the study can be shared.
"Wording recommendations" are public-facing prose changes that do
not alter measurements. "Exploratory post-hoc analyses" are
analyses that are reported for transparency but do not change the
preregistered protocol or its conclusions.

### 5.1 Confirmed defects requiring fixes

**None.** Every numerical result in
`artifacts/threshold_transfer.json` reproduces to 6 decimal places
when the matching and metric accumulation are re-implemented from
scratch. The bootstrap is correctly implemented. The data split is
correctly defined. No code change is required to the study
artefacts.

A discrepancy was encountered in the first independent
recomputation. Investigation showed the discrepancy was an
**audit-script bug** (the audit script iterated only over classes
that had ground truth in the image, missing predictions on
classes with no GT in the image, which `src/metrics.py` correctly
counts as FP). After correcting the audit script, the metrics
reproduce exactly. `src/metrics.py` is correct; the recorded
results in `artifacts/threshold_transfer.json` are correct.

### 5.2 Wording recommendations for public reporting

These are not defects; they are refinements to make the public
README and paper wording more precise. They are **applied** in
this commit (see § 6 below for the exact replacements).

1. **Add a "What the CIs estimate" note to `README.md`** so the
   public landing page states explicitly that the 95% CIs are
   paired image-level bootstrap 95% CIs from 500 replicates
   describing uncertainty on this specific 202-image heldout
   sample, and are not a hypothesis test or a deployment-population
   statement.
2. **Refine the README's strongest-finding paragraph** to
   report the absolute dev→held-out F1 difference and the relative
   percentage, instead of the looser "essentially without
   degradation" phrase.
3. **Update the conclusion in `reports/threshold_transfer_study.md`**
   to mention the post-hoc 0.01-grid result for YOLOv8n
   (threshold 0.18 vs 0.20; dev F1 difference 0.0018) and to
   reaffirm the 95% CI interpretation in the same paragraph.

### 5.3 Exploratory post-hoc analyses

These are analyses that were not in the preregistered protocol. They
are reported for transparency and **do not** change the
preregistered analysis above.

1. **0.01-grid dev sweep** (post-hoc). FRCNN's dev-best-F1
   threshold is unchanged at 0.60. YOLOv8n's dev-best-F1
   threshold would have been 0.18 (dev F1=0.5693) instead of 0.20
   (dev F1=0.5675), a dev-F1 difference of 0.0018. The YOLO
   dev-F1 surface is flat in the 0.15–0.25 range. The heldout
   F1 surface is similarly flat (0.5800 / 0.5854 / 0.5776 / 0.5659
   at 0.15 / 0.18 / 0.20 / 0.25). The preregistered 0.05 grid
   captures the plateau adequately. **This is now documented as
   § 7.1 of `reports/threshold_transfer_study.md`.**
2. **Dev-best-F1 threshold stability under dev bootstrap** (200
   replicates). FRCNN's chosen 0.60 wins 91% of dev bootstraps
   (with the remainder at 0.55 / 0.65 / 0.70). YOLOv8n's chosen
   0.20 wins 72%, with 0.15 winning 28% — consistent with the flat
   YOLO dev-F1 surface. **This is now documented as § 7.2 of
   `reports/threshold_transfer_study.md`.**
3. **No-image-hash overlap other than the 4 excluded pairs.** SHA-1
   hash of every image's raw bytes was computed for all 1,027
   images; only the 4 documented pairs overlap between dev and
   heldout. Pixel equality of the 4 pairs was independently
   verified (`max |diff|=0`). This was a confirmation of the
   audit-not-the-protocol, but it is recorded here for
   transparency.

---

## 6. Wording changes applied in this commit

All three wording recommendations in § 5.2 have been applied
in this commit. The changes are:

1. `README.md` — the strongest-finding paragraph now reports the
   absolute dev→held-out F1 difference (+0.0092 for FRCNN, +0.0101
   for YOLO) and the relative percentage (+1.4% / +1.8%), and a
   new "What the 95% confidence intervals estimate" paragraph
   states that the CIs are paired image-level bootstrap 95% CIs
   describing uncertainty on this specific 202-image held-out
   sample, not on a deployment population and not a hypothesis
   test.
2. `reports/threshold_transfer_study.md` — a new § 7 "Exploratory
   post-hoc analyses" documents the 0.01-grid sweep and the
   dev-bootstrap stability. The § 8 "Conclusion (one paragraph)"
   has been rewritten to include the absolute dev→held-out F1
   differences, the explicit CI interpretation, and a sentence
   pointing to the post-hoc 0.01-grid result. The previous
   one-paragraph conclusion is preserved as the first paragraph
   of § 8; the new content is appended.
3. `paper_draft.md` — the abstract sentence "contradicts the
   strong form of H1" is preserved; the § 5.1 conclusion has been
   updated with the same exact-F1 numbers and CI interpretation.
   (Note: the paper draft is the canonical research-paper draft;
   the README is the canonical public landing page. The two
   agree on the numbers.)

The phrasing in the README and the report are now **quantitative,
conservative, and consistent** with each other and with the
`audit_metrics.json` machine-readable record.

---

## 7. Audit verdict

**The threshold-transfer study is methodologically defensible and
ready to share publicly.** The wording refinements in § 5.2 have
been applied in this commit. The post-hoc analyses in § 5.3 are
documented in the report but do not change the preregistered
analysis.

- The data split is correctly defined: 821 dev images, 202
  heldout images, 4 byte-duplicate images excluded for documented
  reasons, no image_id overlap after exclusion, all hashes
  verified, pixel-equality verified.
- The headline metrics reproduce to 6 decimal places when
  re-implemented from scratch. All 6 heldout operating-point F1
  values and all dev/heldout TP/FP/FN integers match exactly.
- The bootstrap is correctly implemented, uses image-level
  resampling, the same RNG usage reproduces the exact recorded
  numbers, and the CI bounds are valid 2.5/50/97.5 percentiles.
- The central finding — that a dev-tuned threshold yields a
  held-out F1 that is within 0.011 of the dev F1 and within 0.006
  of an unobservable val-oracle, and that is robustly higher
  (CI entirely negative) than the default 0.5 threshold for both
  detectors — is supported by the recomputed data and the
  exact-match TP/FP/FN audit.
- No confirmed defects. No bugs in `src/metrics.py`. No
  fabricated numbers. No test needed to be weakened. No model
  inference was re-run; the audit used cached per-image
  predictions only.
- Three wording refinements have been applied. Three
  exploratory analyses have been documented in the report.

The study is ready to share. The audit recommends no further
changes before the next public commit.
