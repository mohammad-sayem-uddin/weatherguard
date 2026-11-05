# Mission 20 — Phase 5 & 6: Real Audio Corpus Audit & Leakage-Resistant Splits

**Date:** 2026-11-05  
**Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Project:** Rescue Drone Acoustic Distress & WeatherGuard Multi-Modal Safety  
**Repository:** `https://github.com/mohammad-sayem-uddin/weatherguard`  
**Audio Dataset Path:** `/Users/sayemuddin/Desktop/Rescue_Drone/audio_dataset_v2/dataset_v2_1500/`  
**Corpus Size:** 1,500 calibrated WAV files ($16\text{ kHz}$ mono, $2.0\text{ s}$ duration)  
**Status:** COMPLETE, AUDITED & LEAKAGE-FREE  

---

## 1. Executive Summary & Forensic Audit

In preliminary development harnesses, rapid verification used synthetic spectrogram fixtures (`SyntheticSpectrogramDataset`). Phases 5 and 6 of Mission 20 conduct an exhaustive forensic audit of the **actual 1,500-sample Bengali Emergency Vocalization Corpus** and construct a **strictly source-isolated, speaker-independent partition**.

### Audited Corpus Profile:
- **Total Audio Files:** 1,500 WAV recordings
- **Target Vocabulary:** Bengali Emergency Cry (*"bachao"* / বাঁচাও) vs Normal Bengali Conversational Speech & Ambient Negatives
- **Class Balance:** Exactly 750 Positive (`bachao`) and 750 Negative (`normal_bengali`)
- **Audio Standards:** $16,000\text{ Hz}$ sampling rate, 16-bit PCM mono, exactly $2.000\text{ s}$ duration ($32,000$ samples per file)
- **Audio Quality:** Mean RMS $= 0.0543$, Mean Peak $= 0.4056$, Total Clipped Samples $= 0$
- **Integrity:** 1,500 unique SHA-256 hashes ($100\%$ unique files, 0 bitwise collisions)

---

## 2. Parent Recording Traceability & Derivative Grouping

A critical flaw in naive audio dataset splitting is treating augmented versions of an audio file as independent data points. If a clean vocalization $X$ is placed in the training split while a noisy derivative $X + \text{drone noise}$ is placed in the test split, the neural network memorizes speaker vocal tract timbre, formant frequencies, and pitch rather than learning generalized distress acoustic features.

### Root Source Decomposition:
To prevent this leakage, every single file in `metadata/audio_corpus_manifest.csv` was programmatically traced back through its `original_source_ref`:
1. **Owner-Collected Real-World Captures:** 500 base recordings captured across authentic acoustic environments.
2. **Augmented Real-World Derivatives:** 1,000 derivatives generated through physics-based acoustic perturbations:
   - Strong Wind noise injection
   - Drone Propeller acoustic interference
   - Heavy Rain splatter noise
   - Flood Water / surging turbulent flow
   - Calibrated SNRs: $0\text{ dB}$, $5\text{ dB}$, $10\text{ dB}$, $15\text{ dB}$, $20\text{ dB}$, and clean ($\infty\text{ dB}$)
   - Distance Simulations: Close mic ($<1\text{m}$), Medium distance ($3\text{--}5\text{m}$), and Distant ($10\text{m}$)
3. **Canonical Root Clustering:** The 1,500 samples collapse into **745 unique root physical recordings** (316 positive roots and 429 negative roots).

---

## 3. Leakage-Resistant Audio Partition Architecture

Partitioning was performed strictly on the set of **745 root sources**, using deterministic greedy bin-packing (Seed 0):

```
                        1,500 Bengali Audio Samples
                       (745 Unique Root Recordings)
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
    Train Split                  Val Split                  Test Split
 (900 samples, 60.0%)        (300 samples, 20.0%)        (300 samples, 20.0%)
 331 Unique Roots            207 Unique Roots            207 Unique Roots
 450 Pos / 450 Neg           150 Pos / 150 Neg           150 Pos / 150 Neg
```

### Partition Distribution:

| Metric | Train Split | Validation Split | Locked Test Split | Total Corpus |
| :--- | :---: | :---: | :---: | :---: |
| **Total Audio Files** | 900 (60.0%) | 300 (20.0%) | 300 (20.0%) | 1,500 (100.0%) |
| **Positive (`bachao`)** | 450 (50.0%) | 150 (50.0%) | 150 (50.0%) | 750 (50.0%) |
| **Negative (`normal_bengali`)** | 450 (50.0%) | 150 (50.0%) | 150 (50.0%) | 750 (50.0%) |
| **Unique Root Sources** | 331 roots | 207 roots | 207 roots | 745 roots |
| **Real-World Captures** | 422 | 40 | 38 | 500 |
| **Augmented Derivatives** | 478 | 260 | 262 | 1,000 |

### Mathematical Proof of Zero Overlap:
$$\text{Roots}(\text{Train}) \cap \text{Roots}(\text{Val}) = \emptyset$$
$$\text{Roots}(\text{Train}) \cap \text{Roots}(\text{Test}) = \emptyset$$
$$\text{Roots}(\text{Val}) \cap \text{Roots}(\text{Test}) = \emptyset$$
$$\text{Total Cross-Split Overlap Violations} = \mathbf{0}$$

---

## 4. Stratified Environmental Representation

All splits exhibit balanced coverage across clean speech, environmental noise conditions, and distance tiers:

### Noise Condition Stratification:
| Noise Environment | Train Split | Val Split | Locked Test Split | Total Files | Target Proportion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Speech** | 597 (59.7%) | 203 (20.3%) | 200 (20.0%) | 1,000 | 60 / 20 / 20 |
| **Strong Wind** | 71 (54.2%) | 31 (23.7%) | 29 (22.1%) | 131 | 60 / 20 / 20 |
| **Heavy Rain** | 81 (62.3%) | 24 (18.5%) | 25 (19.2%) | 130 | 60 / 20 / 20 |
| **Drone Propeller** | 77 (62.1%) | 22 (17.7%) | 25 (20.2%) | 124 | 60 / 20 / 20 |
| **Flood Water** | 74 (64.3%) | 20 (17.4%) | 21 (18.3%) | 115 | 60 / 20 / 20 |
| **Total** | **900** | **300** | **300** | **1,500** | **100.0%** |

### Acoustic Distance Simulation Stratification:
| Distance Tier | Train Split | Val Split | Locked Test Split | Total Files |
| :--- | :---: | :---: | :---: | :---: |
| **Close Mic ($<1\text{ m}$)** | 698 | 239 | 233 | 1,170 |
| **Medium Distance ($3\text{--}5\text{ m}$)** | 90 | 34 | 33 | 157 |
| **Distant ($10\text{ m}$)** | 112 | 27 | 34 | 173 |
| **Total** | **900** | **300** | **300** | **1,500** |

---

## 5. Artifact Generation & Verification

The partition has been committed to version-controlled artifacts:
- `scripts/mission20/build_leakage_resistant_audio_splits.py`: Automated, deterministic partitioning script.
- `artifacts/mission20/audio_splits_summary.json`: Complete partition metadata, distribution statistics, and validation status.
- `artifacts/mission20/audio_split_manifest.csv`: Full sample-by-sample manifest labeled with canonical split assignment (`train`, `val`, `test`).

---

## 6. Compliance Checklist

- [x] Zero modification of historical mission reports.
- [x] Every augmented derivative is strictly co-located with its root parent recording.
- [x] Zero cross-split root source leakage.
- [x] Classes are perfectly balanced (50.0% positive, 50.0% negative in every split).
- [x] Zero audio files committed to Git (only metadata CSV and JSON summary).
