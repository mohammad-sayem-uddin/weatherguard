# Phase 1 — Dataset Independence and Speaker Provenance Audit

**Project:** WeatherGuard + Rescue Drone  
**Author:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Mission:** 21 — Multi-Modal Acoustic-Visual Rescue Intelligence & Independent Validation  
**Date:** 2026-11-05  
**Artifact:** `reports/mission21/PHASE1_INDEPENDENCE_AUDIT.md`  

---

## 1. Executive Summary

This audit establishes the rigorous empirical boundary between independent physical captures, augmented derivatives, recording sessions, and speaker identities across the WeatherGuard (vision) and Rescue Drone (audio) corpora.

### Key Audit Findings:
1. **WeatherGuard Vision Corpus (1,000 frames):**
   - **Sequence-Level Split Isolation:** Zero sequence leakage ($\text{Sequences}(\text{Train}) \cap \text{Sequences}(\text{Test}) = \emptyset$).
   - **Locked Test Set:** 160 authentic real camera frames (100%), 0 augmented derivative frames (0%).
   - **Near-Duplicate Audit:** Exactly 0 cross-split frame pairs with dHash Hamming distance $\le 4$.
2. **Rescue Drone Audio Corpus (1,500 WAVs):**
   - **Recording-Level Isolation:** Exactly 0 parent recording roots leak across splits ($\text{Roots}(\text{Train}) \cap \text{Roots}(\text{Test}) = \emptyset$).
   - **Derivative Integrity:** All augmented derivatives reside strictly in the same split as their parent source recording (0 derivative split violations).
   - **Speaker vs. Recording Truth:** All 1,500 samples trace to verified voice profiles or speaker IDs. 6 synthetic neural voice engines and 5 human speakers span multiple recording sessions across train and test. 33 human speakers in the locked test set are completely novel and unseen.

---

## 2. WeatherGuard Vision Independence Audit

### 2.1 Root Sequence Partitioning
The 1,000-frame WeatherGuard V2 corpus spans 13 root driving video sequences and stationary adverse-weather capture domains:

```
Total Sequences: 13
├── Train Partition (5 sequences, 667 frames):
│   ├── bdd_video_b1d0a191: 138 frames (Clear / Night highway driving)
│   ├── bdd_video_b1e1a7b8: 195 frames (Daylight suburban driving)
│   ├── bdd_video_b23c9e00: 150 frames (Daylight clear expressway driving)
│   ├── dawn_overcast:      135 frames (Authentic DAWN overcast driving)
│   └── dawn_sandstorm:      49 frames (Authentic DAWN sandstorm driving)
│   [381 real camera frames + 286 Koschmieder fog / rain streak augmented derivatives]
│
├── Val Partition (4 sequences, 173 frames):
│   ├── bdd_video_b1c66a42:  50 frames (Daylight city street driving)
│   ├── bdd_video_b1f4491b:  13 frames (Daylight expressway driving)
│   ├── bdd_video_b24380ab:  65 frames (Daylight urban boulevard driving)
│   └── dawn_rain:           45 frames (Authentic DAWN rain driving)
│   [159 real camera frames + 14 augmented derivatives]
│
└── Locked Test Partition (4 sequences, 160 frames):
    ├── bdd_video_b24d283f:  51 frames (Authentic rain windshield camera driving)
    ├── bdd_video_b255cd6c:  43 frames (Authentic nighttime low-light driving)
    ├── dawn_dense_fog:      19 frames (Authentic DAWN extreme dense fog)
    └── dawn_snow:           47 frames (Authentic DAWN heavy snowfall driving)
    [160 authentic real camera frames, 0 augmented derivative frames (100% pure!)]
```

### 2.2 Mathematical Leakage Verification
- **Cross-Split Sequence Overlap:**
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- **Perceptual Duplicate Audit (dHash):**
  Across all $\approx 106,720$ cross-split frame pairs between Train and Test, the minimum Hamming distance is $d_H \ge 6$. Exactly 0 pairs meet or exceed the near-duplicate threshold ($d_H \le 4$).

---

## 3. Rescue Drone Audio Independence and Speaker Audit

### 3.1 Recording Root Allocation
The 1,500-sample corpus comprises 745 unique parent source recordings:
- **Train Split ($N=900$):** 331 unique parent roots (422 collected originals + 478 augmented derivatives).
- **Val Split ($N=300$):** 207 unique parent roots (40 collected originals + 260 augmented derivatives).
- **Test Split ($N=300$):** 207 unique parent roots (38 collected originals + 262 augmented derivatives).

### 3.2 Verification of Zero Parent Recording Leakage
$$\text{Roots}(\text{Train}) \cap \text{Roots}(\text{Val}) = \emptyset$$
$$\text{Roots}(\text{Train}) \cap \text{Roots}(\text{Test}) = \emptyset$$
$$\text{Roots}(\text{Val}) \cap \text{Roots}(\text{Test}) = \emptyset$$
$$\text{Derivative Leakage Count} = 0$$

Every augmented clip resides strictly within the partition of its parent recording. Neither the raw audio nor any pitch-shifted, speed-perturbed, or noise-injected version of a training utterance enters the locked test set.

### 3.3 Complete Speaker and Voice Provenance Matrix

Every audio sample was tracked to its origin through `dataset_v2_metadata.csv`, `original_dataset_inventory.csv`, and engine manifests:

| Audio Category | Total Samples in Corpus | Test Set Representation | Unique Voice / Speaker IDs | Cross-Split Overlap Status |
| :--- | :---: | :---: | :---: | :--- |
| **Neural TTS Synthesizer Voices** (`bn-BD-NabanitaNeural`, `bn-BD-PradeepNeural`, `bn-IN-TanishaaNeural`, `bn-IN-BashkarNeural`, `gTTS_bn_standard`, `Piya`) | 500 base + 500 aug = 1,000 | 210 samples (70.0%) | 6 distinct voices | **Shared Voice Profile Across Splits:** Base utterances are strictly disjoint, but the underlying synthetic vocal timbre is present in train, val, and test. |
| **Owner-Collected & Human Bengalis (CommonVoice / Session recordings)** | 500 samples | 90 samples (30.0%) | 38 human speakers in test | **Partially Disjoint:** 33 human speakers are completely novel and unseen in training; 5 human speakers (`s8_s1`, `s9_s2`, `c02`, `c07`, `c09`) have different recording sessions in train and test. |

### 3.4 Methodological Mandates for Mission 21
To prevent overstating generalization:
1. **Never conflate root recordings with independent biological speakers.** The 745 roots represent 745 unique audio utterances, not 745 distinct human beings.
2. In subsequent evaluation phases, results must be reported stratified by:
   - **Subset 1: Owner-Collected-Only Test Subset ($N=38$):** Authentic physical microphone captures.
   - **Subset 2: Augmented-Only Test Subset ($N=262$):** Parent recordings with calibrated noise/SNR/distance transforms.
   - **Subset 3: Novel Unseen Human Speakers ($N=45$):** Human utterances from individuals never encountered in training.
   - **Subset 4: Neural Voice Profiles ($N=210$):** High-fidelity synthetic vocalizations with disjoint phrases.
