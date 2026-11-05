# WeatherGuard — Phase 2: Comprehensive Existing Image Dataset Audit Report

**Date:** 2026-11-05  
**Project Owner:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** [https://github.com/mohammad-sayem-uddin/weatherguard](https://github.com/mohammad-sayem-uddin/weatherguard)  
**Artifact:** [`artifacts/existing_dataset_audit.json`](file:///Users/sayemuddin/Desktop/WeatherGuard/artifacts/existing_dataset_audit.json)  

---

## 1. Executive Summary

Phase 2 audited all existing image datasets and video sequence partitions currently utilized by WeatherGuard. The audit surveyed image counts, sensor modalities, native resolutions, weather conditions, class distributions, and scale breakdowns according to the standardized WeatherGuard definitions:
- **Ultra-Small:** $< 16\text{ px}$ ($\text{area} < 256\text{ px}^2$)
- **Small:** $16\text{ px} \le \text{dim} < 32\text{ px}$
- **Medium:** $32\text{ px} \le \text{dim} < 96\text{ px}$
- **Large:** $\ge 96\text{ px}$

All historical evaluation splits and locked reference cohorts are preserved in an unmodified, frozen state to ensure complete experimental integrity.

---

## 2. Multi-Dataset Inventory Ledger

| Dataset Name | Source Category & Provenance | Sensor Modality | Image / Frame Count | Native Resolution | Primary Weather Conditions | Annotation Count | Status |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **DAWN** | Public Real-World (Kenan et al., 2020) | 3-Channel RGB | 206 frames | $1024 \times 682$ to $1200 \times 900$ | Fog (19), Rain (46), Snow (47), Sand (49), Other (45) | 1,438 boxes | 🟢 Intact |
| **FLIR ADAS v1** | Public Real-World (Teledyne FLIR) | Thermal LWIR ($8–14\,\mu\text{m}$) | 1,452 frames | $640 \times 512$ | Night, Overcast, Low-Contrast Thermal Roadways | 10,821 boxes | 🟢 Intact |
| **BDD100K Dev** | Public Real-World Video (Yu et al., 2020) | 3-Channel RGB | 405 frames | $1280 \times 720$ (16:9) | Rain (`b24380ab`), Night (`b255cd6c`) | 3,120 boxes | 🟢 Intact |
| **BDD100K Locked** | Public Real-World Video (Yu et al., 2020) | 3-Channel RGB | 808 frames | $1280 \times 720$ (16:9) | Overcast (`b1e1a7b8`), Night (`b1d0a191`), Rain (`b24d283f`, `b23c9e00`) | 7,458 boxes | 🟢 Locked Frozen |
| **Missions 14–15 Unseen** | Public Real-World Video (Yu et al., 2020) | 3-Channel RGB | 3,677 frames | $1280 \times 720$ (16:9) | Diverse rain, dusk, night, overcast highway sequences | 31,480 boxes | 🟢 Intact |

---

## 3. Scale and Class Distribution Breakdown

### 3.1. DAWN Adverse Weather Dataset
- **Canonical Classes:** `car` ($1,161$), `truck` ($115$), `person` ($111$), `bus` ($34$), `motorcycle` ($15$), `bicycle` ($2$).
- **Object Scales:** Ultra-Small: $6$ ($0.42\%$), Small: $113$ ($7.86\%$), Medium: $574$ ($39.92\%$), Large: $745$ ($51.81\%$).
- **Key Observation:** Severe scarcity of ultra-small targets ($<16\text{px}$) in DAWN; only 6 instances exist across 206 images, establishing the need for targeted expansion on distant, low-pixel road traffic.

### 3.2. FLIR ADAS Thermal Infrared Dataset
- **Canonical Classes:** `car` ($6,512$), `person` ($3,696$), `bicycle` ($613$).
- **Object Scales:** Ultra-Small: $377$ ($3.48\%$), Small: $3,255$ ($30.08\%$), Medium: $5,666$ ($52.36\%$), Large: $1,561$ ($14.43\%$).
- **Key Observation:** High density of small targets ($30.08\%$), providing an ideal benchmark for spatial contrast and gradient transfer across optical vs thermal modalities.

---

## 4. Preservation Invariants

1. **Zero Overwriting:** No label, image, or bounding box coordinates in `data/dawn_yolo/` or `data/bdd100k/` were modified or deleted.
2. **Locked Manifest Separation:** The 808-frame locked test manifest (`artifacts/mission13_ablation_results.json`) remains completely isolated from training data.
