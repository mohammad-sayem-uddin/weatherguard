# WeatherGuard & Rescue Drone — Phases 6 & 7: Provenance Audit & Versioned Cryptographic Manifest Report

**Date:** 2026-11-05  
**Author / Principal Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** `https://github.com/mohammad-sayem-uddin/weatherguard`  
**Vision Manifest:** `artifacts/dataset_manifests/weatherguard_v2_manifest.json`  
**Audio Manifest:** `artifacts/dataset_manifests/audio_v2_manifest.json`  

---

## 1. Executive Summary

Phases 6 and 7 of the Master Agent Mission establish unified dataset provenance governance, eliminate ambiguity regarding synthetic versus real data, and generate immutable, version-controlled cryptographic ledgers for both the WeatherGuard Road Perception Corpus (Vision V2) and the Rescue Drone Acoustic Distress Corpus (Audio V2).

### Core Provenance Policy Statement
> *"WeatherGuard is an owner-curated, multi-source adverse-weather road-scene dataset. Its individual records retain their source provenance, including collected, public, generated, and augmented data."*

All documentation, README files, experiment registries, and dashboard interfaces have been audited and updated to ensure strict scientific transparency:
- Real-world captures are explicitly identified with their underlying physical capture source (`public_real_world` or `owner_collected_real_world`).
- Physical optical or acoustic transformations are categorized under `augmented_real_world`.
- Upcoming AI-generated imagery (WeatherGuard-D1 flow models) remains designated as `BLOCKED_PENDING_DATA` with zero fabrication.

---

## 2. Versioned Dataset Specifications & Cryptographic Manifests

### 2.1. WeatherGuard V2 Road Perception Corpus (Vision)
- **Version:** `v2.0.0`
- **Total Images:** 1,000 standardized frames ($1280 \times 720$)
- **Total Valid Objects:** 13,838 annotated bounding boxes
- **Master Dataset Checksum (SHA-256):** `aeca94514017...`
- **Class Taxonomy:** 6 canonical road classes (`truck`, `person`, `bicycle`, `car`, `motorcycle`, `bus`)
- **Scale Distribution:**
  - Ultra-Small ($<16\text{ px}$): 464 objects (3.35%)
  - Small ($16-32\text{ px}$): 3,311 objects (23.93%)
  - Medium ($32-96\text{ px}$): 6,637 objects (47.96%)
  - Large ($\ge 96\text{ px}$): 3,426 objects (24.76%)
- **Source Breakdown:** 700 `public_real_world` (DAWN: 205, BDD100K: 495), 300 `augmented_real_world` (Koschmieder fog, rain streaks), 0 `ai_generated`.

### 2.2. Rescue Drone Bengali Distress Vocalization Corpus (Audio)
- **Version:** `v2.0.0`
- **Total Samples:** 1,500 audio files ($2.0\text{s}$ duration, $32,000$ samples @ 16 kHz Mono PCM_16)
- **Master Dataset Checksum (SHA-256):** `0fe5cf238e70...`
- **Class Taxonomy:**
  - Class 1 (Positive): 750 samples of emergency cry *"বাঁচাও"* (`bachao`)
  - Class 0 (Negative): 750 samples of normal conversational Bengali speech, shouting, and background environmental acoustics
- **Source Breakdown:** 500 `owner_collected_real_world` (human native speakers), 1,000 `augmented_real_world` (physical environmental noise & room impulse reverberation).
- **Quality Compliance:** 100.00% pass rate across the 14-point scientific validation suite.

---

## 3. Storage and Repository Integrity

- **Local Storage Footprint:** 190 MB (Vision Corpus) + 94 MB (Audio Corpus) = 284 MB total.
- **Available Host Headroom:** $>5.8\text{ GiB}$ remaining on macOS APFS volume.
- **Git Separation:** Heavy raw image and WAV files reside in ignored paths (`data/` in WeatherGuard and local storage in Rescue_Drone); lightweight JSON manifests and CSV ledgers are versioned under `artifacts/dataset_manifests/`.
