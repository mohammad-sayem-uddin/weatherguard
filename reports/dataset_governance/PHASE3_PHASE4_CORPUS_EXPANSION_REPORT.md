# WeatherGuard — Phases 3 & 4: Image Corpus Expansion & Quality Governance Report

**Date:** 2026-11-05  
**Author / Principal Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** `https://github.com/mohammad-sayem-uddin/weatherguard`  
**Dataset Reference:** `data/weatherguard_corpus/`  
**Master Manifest:** `data/weatherguard_corpus/metadata/corpus_manifest.csv`  
**Summary Artifact:** `artifacts/dataset_manifests/weatherguard_corpus_summary.json`

---

## 1. Executive Summary

Phases 3 and 4 of the Master Agent Mission have been successfully executed on the WeatherGuard computer vision research system. A high-diversity, multi-source road perception corpus of exactly **1,000 standardized frames** and **13,838 validated object bounding boxes** has been curated, normalized, geometrically verified, and cataloged.

Strict provenance governance was enforced:
- **Zero mislabeling:** Real-world frames are explicitly designated as `public_real_world`; physically-grounded weather-augmented frames are explicitly designated as `augmented_real_world`.
- **Zero synthetic ambiguity:** No AI-generated or simulated frames are present in this corpus; when future generative models (e.g., Google Flow / Gemini Flash annotations) are introduced, their provenance will be segregated under `ai_generated`.

---

## 2. Dataset Composition & Source Breakdown

| Provenance Category | Frame Count | Percentage | Primary Sources | Description |
| :--- | :---: | :---: | :--- | :--- |
| `public_real_world` | 700 | 70.0% | DAWN (205), BDD100K Driving Sequences (495) | Real-world automotive dashcam captures across natural fog, snow, rain, night, sandstorm, and clear daytime. |
| `augmented_real_world` | 300 | 30.0% | Physically augmented BDD100K / DAWN frames | Real clear/overcast frames subjected to Koschmieder optical transmission fog ($\beta=0.85$, $A=0.88$) and heavy rain streaks ($N=600$). |
| `ai_generated` | 0 | 0.0% | N/A | Reserved for upcoming WeatherGuard-D1 flow syntheses. |
| **Total** | **1,000** | **100.0%** | **Multi-Source Curated** | **Comprehensive benchmark corpus** |

---

## 3. Environmental Condition Distribution

The 1,000 frames deliberately emphasize adverse weather, degraded visibility, and challenging illumination:

### Weather Conditions
- **Clear:** 263 frames (26.3%)
- **Night:** 181 frames (18.1%)
- **Dense Fog:** 169 frames (16.9%)
- **Heavy Rain:** 150 frames (15.0%)
- **Rain:** 96 frames (9.6%)
- **Sandstorm:** 49 frames (4.9%)
- **Snow:** 47 frames (4.7%)
- **Overcast:** 45 frames (4.5%)

### Lighting Conditions
- **Daylight:** 449 frames (44.9%)
- **Low-Light (Fog / Heavy Rain / Dusk):** 370 frames (37.0%)
- **Night (Artificial / Headlight illumination):** 181 frames (18.1%)

---

## 4. Object Scale Distribution & Annotation Validation

All bounding boxes underwent rigorous geometric sanitization:
1. **Coordinate Clamping:** All coordinates clamped to image boundaries $[0, W] \times [0, H]$.
2. **Degenerate Rejection:** Bounding boxes with zero area, inverted coordinates ($x_2 \le x_1$ or $y_2 \le y_1$), or subpixel dimensions ($< 2.0\text{ px}$) were discarded.
3. **Canonical Alignment:** Classes mapped strictly to the 6 target categories: `truck`, `person`, `bicycle`, `car`, `motorcycle`, `bus`.

### Scale Distribution (13,838 Total Objects)
- **Ultra-Small ($<16\text{ px}$ max dimension):** 464 objects (3.35%)
- **Small ($16\text{ to }32\text{ px}$):** 3,311 objects (23.93%)
- **Medium ($32\text{ to }96\text{ px}$):** 6,637 objects (47.96%)
- **Large ($\ge 96\text{ px}$):** 3,426 objects (24.76%)

### Manual Review Queue
- **352 candidate boxes** flagged in `data/weatherguard_corpus/metadata/manual_review_queue.csv` for human-in-the-loop audit (primarily ultra-small objects $<12\text{ px}$ or distant horizon targets).

---

## 5. Storage Footprint & System Integrity

- **Physical Storage Used:** 190 MB (total directory footprint including 1,000 images, 1,000 YOLO labels, sample visual overlays, and metadata).
- **Available Development Host Capacity:** 6.2 GiB remaining on Apple M1 internal storage.
- **Git State:** `data/` directory strictly excluded via `.gitignore`; metadata, scripts, and audit ledgers safely versioned under `artifacts/` and `reports/`.
