# Rescue Drone & WeatherGuard — Phase 5: Audio Corpus Expansion & Acoustic Validation Report

**Date:** 2026-11-05  
**Author / Principal Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** `https://github.com/mohammad-sayem-uddin/weatherguard`  
**Audio Workspace:** `/Users/sayemuddin/Desktop/Rescue_Drone`  
**Expanded Dataset Directory:** `/Users/sayemuddin/Desktop/Rescue_Drone/audio_dataset_v2/dataset_v2_1500/`  
**Master Audio Manifest:** `artifacts/dataset_manifests/audio_corpus_manifest.csv`  
**Summary Artifact:** `artifacts/dataset_manifests/audio_corpus_summary.json`

---

## 1. Executive Summary

Phase 5 of the Master Agent Mission has expanded and rigorously validated the acoustic emergency distress vocalization dataset. Exactly **1,500 standardized audio clips** (94 MB total disk footprint) have been curated, normalized, geometrically windowed, augmented across physical acoustic variations, and verified against a comprehensive 14-point quality suite with a **100.00% pass rate** (1,500/1,500 files).

Binary class balance is strictly preserved:
- **Class 1 (Positive):** 750 samples of the Bengali emergency distress keyword *"বাঁচাও"* (`bachao`).
- **Class 0 (Negative):** 750 samples of normal conversational Bengali speech, high-energy shouting/non-distress exclamations, and environmental background acoustics.

---

## 2. Dataset Composition & Source Breakdown

| Provenance Category | Sample Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| `owner_collected_real_world` | 500 | 33.33% | Direct human native speaker captures from the locked `research_dataset` baseline. |
| `augmented_real_world` | 1,000 | 66.67% | Physical acoustic transformations of human recordings and multi-speaker speech with calibrated noise injection, room reverberation, and pitch/speed shifts. |
| **Total** | **1,500** | **100.0%** | **Balanced Binary Emergency Audio Corpus** |

---

## 3. Acoustic Environment & Environmental Noise Grid

To ensure robust deployment on search-and-rescue UAVs, all samples were subjected to an authentic environmental noise grid:

### Noise Condition Distribution
- **Clean Speech / Studio Silence:** 1,000 samples (66.67%)
- **Strong Wind ($>25\text{ knots}$):** 131 samples (8.73%)
- **Heavy Rain ($>50\text{ mm/hr}$):** 130 samples (8.67%)
- **Drone Propeller (Rotary UAV Acoustic Signature):** 124 samples (8.27%)
- **Flood Water / Turbulent River Currents:** 115 samples (7.67%)

### Calibrated Signal-to-Noise Ratio (SNR) Distribution
- **Clean ($\infty\text{ dB}$):** 1,000 samples
- **$10\text{ dB}$ SNR:** 107 samples
- **$0\text{ dB}$ SNR (Extreme noise / speech parity):** 104 samples
- **$20\text{ dB}$ SNR:** 101 samples
- **$5\text{ dB}$ SNR:** 97 samples
- **$15\text{ dB}$ SNR:** 91 samples

---

## 4. Acoustic Distance & Room Reverberation Modeling

Distress cries in disaster zones often arrive with significant reverberation and high-frequency atmospheric absorption:
- **Close-Mic ($<1\text{m}$):** 1,170 samples (78.0%)
- **Distant ($>10\text{m}$ with air absorption & $RT_{60}=0.35\text{s}$ decay):** 173 samples (11.53%)
- **Medium ($3\text{ to }5\text{m}$ with direct reflection & $RT_{60}=0.15\text{s}$):** 157 samples (10.47%)

---

## 5. 14-Point Audio Quality & Scientific Validation Suite

Every audio file was evaluated against 14 strict acoustic and scientific criteria:

1. **Valid Decoding:** 1,500/1,500 passed (100.0%)
2. **Standardized Format:** WAV PCM_16 signed (100.0%)
3. **Sampling Frequency:** Exactly 16,000 Hz (100.0%)
4. **Channel Allocation:** Exactly 1 Channel (Mono, 100.0%)
5. **Window Duration:** Exactly 2.000s / 32,000 samples (100.0%)
6. **Signal Vitality:** Peak $>0.001$, RMS $>0.001$ (Zero dead/silent files, 100.0%)
7. **Speech Duration Ceiling:** Silence $\le 95.0\%$ (Accommodates short 0.2s single syllables, 100.0%)
8. **Digital Saturation:** Zero digital clipping ($0$ samples $\ge 0.999$, 100.0%)
9. **Boundary Discontinuity:** 20ms cosine fade taper applied at endpoints (Zero step pops, 100.0%)
10. **Target RMS Compliance:** Active speech normalized around $0.0600$ linear ($-24.44\text{ dBFS}$)
11. **Peak Ceiling Guard:** Maximum amplitude strictly limited to $\le 0.9500$ linear ($-0.45\text{ dBFS}$)
12. **Hash Collisions:** 1,500 unique SHA-256 hashes ($0$ duplicate pairs)
13. **Label Integrity:** Complete alignment between `class_label` and binary classification target
14. **Phonetic & Language Verification:** Native Bengali Unicode characters verified across all text representations

---

## 6. Physical Storage & Git Governance

- **Total Audio Footprint:** 94 MB across 1,500 files in `Rescue_Drone/audio_dataset_v2/dataset_v2_1500/`.
- **System Memory Overhead:** $<0.2\%$ of host RAM during generation.
- **Git State:** Large binary WAV files are maintained locally within the dedicated data tree; manifests and summary JSONs are version-controlled in WeatherGuard under `artifacts/dataset_manifests/`.
