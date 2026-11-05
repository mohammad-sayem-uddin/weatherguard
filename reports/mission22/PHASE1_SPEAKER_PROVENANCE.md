# MISSION 22 — PHASE 1: Reconcile Audio Speaker and Source Provenance

**Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** `https://github.com/mohammad-sayem-uddin/weatherguard`  
**Branch:** `master`  
**Manifest:** `artifacts/mission22/audio_provenance_manifest.csv`  
**Audit Artifact:** `artifacts/mission22/speaker_provenance_audit.json`  
**Date:** November 5, 2026  

---

## 1. Executive Summary

Phase 1 performed an exhaustive provenance trace across all **1,500 audio recordings** in the Rescue Drone corpus and audited the **745 unique parent recordings**.

### Key Findings & Historical Downgrade:
1. **Verification of the "33 Unseen Human Speakers" Claim:**  
   Mission 21 claimed that *"33 human speakers in the test set are completely novel and unseen, establishing speaker-independent generalization."*  
   **Forensic Verification:** **UNCONFIRMED / METHODOLOGICALLY MISATTRIBUTED**.  
   - Of the 34 human speakers in the test split not present in the training set, exactly **33 were NEGATIVE-CLASS speakers** from Mozilla Common Voice (ordinary non-distress speech).
   - Only **ONE human speaker** (`pos_spk_wa_2026-09-01_s2_early`) represented a novel, unseen distress vocalization.
   - All other positive distress cries in the locked test set originated from human speakers or neural text-to-speech (TTS) engines already present in the training split.
   - Therefore, true **speaker-independent generalization on emergency distress vocalizations** was evaluated on only one human individual, and cannot be scientifically claimed as confirmed.
2. **Performance Drop on Novel Speakers:**  
   When evaluated strictly on the subset of truly unseen speakers ($N=62$), model performance dropped from $F_1 = 0.9267$ down to **$F_1 = 0.7143$**, and the false negative rate (missed distress calls) surged from **$7.33\%$ to $28.57\%$** (2 out of 7 novel distress cries were missed).
3. **Traceability Achieved:**  
   All 1,500 samples are now indexed in `artifacts/mission22/audio_provenance_manifest.csv` with parent recording IDs, session IDs, verified speaker IDs, speech modalities, and augmentation parameters.

---

## 2. Dataset Composition & Modality Breakdown

The Rescue Drone 1,500-sample corpus is partitioned across three splits with zero parent recording leakage:

| Split | Total Samples | Owner Field Recordings (Human Distress) | Public Common Voice (Human Normal Speech) | Synthetic TTS (Neural Emergency & Control) | Parent Roots |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 900 | 339 | 307 | 254 (111 pos, 143 neg) | 446 |
| **Val** | 300 | 34 | 53 | 213 (116 pos, 97 neg) | 92 |
| **Test** | 300 | 35 | 55 | 210 (115 pos, 95 neg) | 207 |
| **TOTAL** | **1,500** | **408** | **415** | **677** | **745** |

### Speech Modality Distribution:
- **Human Speech:** 823 samples (54.87%) — 408 positive distress vocalizations, 415 negative normal speech samples.
- **Synthetic Neural Speech:** 677 samples (45.13%) — 342 positive distress vocalizations, 335 negative control phrases across 6 neural voice engines (`bn-BD-NabanitaNeural`, `bn-IN-TanishaaNeural`, `bn-BD-PradeepNeural`, `bn-IN-BashkarNeural`, `Piya`, and `gTTS_bn_standard`).

---

## 3. Speaker & Session Overlap Analysis

### 3.1 Parent Recording Isolation (PASSED)
- **Train vs. Test Parent Recording Overlap:** **0 roots** (Zero leakage).
- Every augmented derivative remains strictly in the same split as its parent source recording.

### 3.2 Speaker Identity Overlap Between Train and Test
- **Synthetic Neural Engines:** All 6 synthetic voice profiles are present in both train and test splits.
- **Owner Human Distress Speakers:**
  - 19 human recording sessions in train.
  - 6 human recording sessions in test.
  - **Shared between Train and Test:** 5 sessions (`pos_spk_clip_f01_c02`, `pos_spk_clip_f02_c07`, `pos_spk_clip_f03_c09`, `pos_spk_wa_2026-09-06_s8_s1`, `pos_spk_wa_2026-09-06_s9_s2`).
  - **Unseen Novel Distress Speaker:** Exactly **1 session** (`pos_spk_wa_2026-09-01_s2_early`, 7 clips).
- **Mozilla Common Voice Negative Speakers:**
  - 33 speakers in test were unseen in train.
  - These speakers contributed 55 negative samples (correctly rejected ordinary Bengali sentences).

---

## 4. Empirical Evaluation by Provenance Subset

Evaluating the fine-tuned 2D CNN (`AudioBaselinePyTorchCNN`) on the 300 locked test samples reveals substantial performance divergence across provenance subsets:

| Evaluation Subset | Sample Count | Accuracy | Precision | Recall | F1 Score | False Alarm Rate (FPR) | Miss Rate (FNR) | Confusion Matrix |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Authentic Field Only** (`owner_collected`) | 38 | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **0.00%** | **0.00%** | $\text{TN}=34, \text{FP}=0, \text{FN}=0, \text{TP}=4$ |
| **Augmented Only** (`augmented_real_world`) | 262 | 91.60% | 0.9247 | 0.9247 | 0.9247 | 9.48% | 7.53% | $\text{TN}=105, \text{FP}=11, \text{FN}=11, \text{TP}=135$ |
| **Combined Locked Test** | 300 | 92.67% | 0.9267 | 0.9267 | 0.9267 | 7.33% | 7.33% | $\text{TN}=139, \text{FP}=11, \text{FN}=11, \text{TP}=139$ |
| **Truly Unseen Speakers Subset** | 62 | 93.55% | 0.7143 | 0.7143 | **0.7143** | 3.64% | **28.57%** | $\text{TN}=53, \text{FP}=2, \text{FN}=2, \text{TP}=5$ |

### Scientific Interpretation:
1. **Authentic Field Test Limitation:** While the model achieved 100% on the 38 authentic field samples, the test set contained only **4 positive emergency cries** (and 34 negative controls). A sample size of $N=4$ positive events has a wide 95% Wilson score binomial confidence interval ($[51.0\%, 100.0\%]$), which cannot support definitive operational certification.
2. **Speaker Generalization Gap:** On the 62 samples from speakers not seen during training, the miss rate increased nearly four-fold ($7.33\% \to 28.57\%$). Two out of seven emergency cries from the unseen human victim were misclassified as non-distress speech.
3. **Operational Consequence:** Future data collection must prioritize diverse, unaugmented human distress vocalizations across a minimum of 30 independent human speakers in the test split before claiming robust speaker-independent rescue capability.

---

## 5. Provenance Manifest Schema

The traceable manifest has been written to `artifacts/mission22/audio_provenance_manifest.csv`. Its schema conforms to:

```csv
sample_id,filename,parent_recording_id,speaker_id,session_id,source_type,authentic_vs_augmented,augmentation_parameters,noise_condition,snr_db,split_assignment,positive_negative_label,speech_modality,sha256_hash
```

Phase 1 is complete.
