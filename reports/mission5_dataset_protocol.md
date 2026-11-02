# WeatherGuard Mission 5: Dataset & Cohort Protocol

**Document:** `reports/mission5_dataset_protocol.md`  
**Date:** 2026-11-01  
**Project:** WeatherGuard (`/Users/sayemuddin/Desktop/WeatherGuard`)  
**Status:** Frozen & Preregistered  
**Artifact:** `artifacts/mission5_split_manifest.json`  

---

## 1. Executive Summary & Cohort Architecture

Mission 5 establishes a rigorous multi-cohort architecture designed to isolate model fine-tuning from out-of-domain evaluation and external test validation.

To prevent data leakage, training is strictly confined to Group A (DAWN). Group B (BDD100K Development) is used exclusively for out-of-domain model ranking and policy validation. Group C (BDD100K External Test) remains permanently locked until the final evaluation.

---

## 2. Cohort Definitions & Integrity Verification

| Cohort | Source Dataset | Size | Role & Constraints | Overlap with Others |
|---|---|---|---|---|
| **Group A (Train)** | DAWN Train (`train-00000-of-00001.parquet`) | $N = 821$ | Model fine-tuning, backbone freezing experiments, weather augmentation | Disjoint ($\cap B = \emptyset, \cap C = \emptyset$) |
| **Group A (Val)** | DAWN Val (`val-00000-of-00001.parquet`) | $N = 206$ | In-domain validation loss monitoring, mAP tracking | Disjoint ($\cap B = \emptyset, \cap C = \emptyset$) |
| **Group B (Dev)** | BDD100K Dev (`data/bdd100k/images`) | $N = 160$ | Cross-dataset development validation, checkpoint ranking, augmentation selection | Disjoint ($\cap A = \emptyset, \cap C = \emptyset$) |
| **Group C (Test)** | BDD100K Locked Test (`data/bdd100k/images`) | $N = 203$ | Permanently locked external evaluation test cohort (zero training, zero tuning) | Disjoint ($\cap A = \emptyset, \cap B = \emptyset$) |
| **Group D (BDD Train)** | BDD100K Supervised Train | $N = 0$ (Blocked) | Formally documented as unavailable locally; omitted to avoid external data downloads | N/A |

### Automated Integrity Checks (`artifacts/mission5_split_manifest.json`)
- `disjoint_A_and_B`: **True** (0 overlapping image IDs)
- `disjoint_A_and_C`: **True** (0 overlapping image IDs)
- `disjoint_B_and_C`: **True** (0 overlapping image IDs)
- `zero_leakage_verified`: **True**

---

## 3. Class Taxonomy Mapping

Both DAWN and BDD100K are mapped to the 6 canonical autonomous driving categories:
1. `Person` (COCO `person`, BDD `pedestrian`)
2. `Bicycle` (COCO `bicycle`, BDD `bicycle`)
3. `Car` (COCO `car`, BDD `car`)
4. `Motorcycle` (COCO `motorcycle`, BDD `motorcycle`)
5. `Bus` (COCO `bus`, BDD `bus`)
6. `Truck` (COCO `truck`, BDD `truck`)

Non-canonical classes in BDD100K (`rider`, `traffic light`, `traffic sign`, `train`, `trailer`, `other vehicle`, `other person`) are explicitly excluded.

---

## 4. Status of Group D (Cross-Domain BDD100K Training)

Per project instructions (Section 3 and Section 7):
> "If Group D cannot be obtained reliably, continue with DAWN-only training and clearly state that the experiment does not constitute BDD-supervised domain adaptation. ... Record it as blocked and continue with the remaining experiments."

Only the 363 BDD100K images allocated to Groups B and C are present locally. Downloading thousands of additional training images would violate local storage constraints and risk cohort contamination. Group D is formally marked as **BLOCKED / UNAVAILABLE**. Mission 5 focuses on supervised DAWN adaptation, weather-aware data augmentation, and zero-shot cross-dataset transfer to BDD100K.
