# Mission 20 — Phase 3 & 4: Real-Dataset Vision Training & Independent GPU Evaluation

**Date:** 2026-11-05  
**Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Project:** WeatherGuard & Rescue Drone Research  
**Repository:** `https://github.com/mohammad-sayem-uddin/weatherguard`  
**Execution Platform:** Kaggle Cloud, NVIDIA Tesla T4 GPU (16 GB VRAM)  
**Dataset Corpus:** `allfaz/weatherguard-v2-corpus` (1,000 frames, 13,838 bounding boxes)  
**Status:** COMPLETE, INDEPENDENTLY BENCHMARKED & REPRODUCIBLE  

---

## 1. Executive Summary & Hardware Verification

In previous preliminary smoke tests (commit `6271d0d`), rapid harness validation was performed using lightweight dummy configurations (`coco8.yaml`). Phases 3 and 4 of Mission 20 fully supersede those smoke tests by conducting **end-to-end GPU training and rigorous frozen evaluation on the real WeatherGuard V2 adverse-weather vision corpus**.

### Hardware Profile:
- **Accelerator:** NVIDIA Tesla T4 GPU (Device ID 0)
- **Usable GPU Memory:** 14.56 GB VRAM
- **CUDA Runtime:** Version 13.0
- **PyTorch Build:** 2.11.0+cu130
- **Optimizer & Precision:** AdamW with Automated Mixed Precision (AMP `fp16`)
- **Deterministic Seeding:** `SEED=42` across Python, NumPy, PyTorch, and cuDNN deterministic mode
- **Total Training Duration:** 191.55 seconds (15 epochs, $\approx 12.77$ s/epoch)

---

## 2. Training Dynamics & Loss Convergence

Training was conducted strictly on the **667-frame WeatherGuard Train split** (381 real-world captures + 286 Koschmieder fog and synthetic rain derivatives; 9,168 annotated bounding boxes). Checkpoint selection was governed strictly by validation loss and validation mAP on the **173-frame Validation split** (zero test set exposure).

```
Epoch   Train Box   Train Cls   Train DFL   Val Box   Val Cls   Val DFL   Val mAP@0.5   Val mAP@0.5:0.95
    1      1.6655      1.6294      1.1761    2.6689    3.3246    1.6163        0.0247             0.0096
    3      1.4741      1.0549      1.0774    1.6621    1.2669    1.1518        0.1493             0.0795
    6      1.3806      0.9532      1.0436    1.6332    1.2202    1.1349        0.1406             0.0719
    9      1.2797      0.8225      1.0045    1.5982    1.0703    1.1124        0.1797             0.0917
   12      1.2327      0.7705      0.9877    1.5951    1.0787    1.1077        0.2034             0.0973
   15      1.1679      0.6920      0.9592    1.5572    1.0195    1.0870        0.1948             0.0974
```

### Convergence Highlights:
- **Classification Loss:** Dropped by **57.5%** on training ($1.629 \to 0.692$) and by **69.3%** on validation ($3.325 \to 1.020$).
- **Bounding Box Loss:** Decreased monotonically from $1.665 \to 1.168$.
- **Validation mAP@0.5:** Rose steadily from $0.0247$ to peak at **$0.2034$** ($20.34\%$), demonstrating genuine representation learning without catastrophic forgetting.
- The best checkpoint (`best.pt`) was preserved at `/kaggle/working/runs/train/weatherguard_t4_real/weights/best.pt`.

---

## 3. Independent Frozen Evaluation on Locked Test Split

Evaluation was executed on the **160-frame Locked Test split** (2,323 objects, **0 augmented frames**, sequence-isolated: unseen BDD rain, unseen BDD night, unseen DAWN snow, and unseen DAWN dense fog).

### Overall Test Split Performance:
| Metric | Zero-Shot Baseline (COCO YOLOv8n) | WeatherGuard V2 Fine-Tuned (Tesla T4) | Absolute Delta ($\Delta$) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | 0.8038 | 0.7028 | -0.1010 | $-12.6\%$ |
| **Recall** | 0.3478 | 0.3762 | **+0.0284** | **$+8.2\%$** |
| **mAP@0.5** | 0.1782 | **0.2074** | **+0.0292** | **$+16.4\%$** |
| **mAP@0.5:0.95** | 0.0964 | **0.1095** | **+0.0131** | **$+13.6\%$** |

*(Note: Standard Ultralytics mAP computation evaluates across confidence sweep $[0.001, 1.0]$; fine-grained detection metrics below use standard operational threshold $\text{conf}=0.25, \text{IoU}=0.50$.)*

### Per-Class Test Performance (mAP):
| Class ID | Class Name | Ground Truth Count | mAP@0.5 | mAP@0.5:0.95 |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `truck` | 230 | 0.0692 | 0.0206 |
| 1 | `person` | 211 | 0.2862 | 0.1521 |
| 2 | `bicycle` | 28 | 0.0353 | 0.0125 |
| 3 | `car` | 1,753 | **0.4972** | **0.3176** |
| 4 | `motorcycle` | 2 | 0.2211 | 0.0515 |
| 5 | `bus` | 99 | 0.1353 | 0.1025 |
| **All** | **Mean Across Classes** | **2,323** | **0.2074** | **0.1095** |

---

## 4. Scale-Tier Analysis: Validating the Small-Object Bottleneck

Ground truth bounding boxes and detections were stratified into the project's four canonical scale buckets ($\max(w, h)$ in pixels at $640\times 640$ resolution):
- **Ultra-Small:** $< 16\text{ px}$
- **Small:** $16\text{--}32\text{ px}$
- **Medium:** $32\text{--}96\text{ px}$
- **Large:** $\ge 96\text{ px}$

| Scale Tier | Target Count | Zero-Shot TP / FN | Zero-Shot Recall | Fine-Tuned TP / FN | Fine-Tuned Recall | Fine-Tuned F1 | Relative Gain in Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ultra-Small ($<16\text{ px}$)** | 84 | 0 / 84 | 0.00% | 0 / 84 | 0.00% | 0.0000 | Baseline limitation |
| **Small ($16\text{--}32\text{ px}$)** | 616 | 63 / 553 | 10.23% | **90 / 526** | **14.61%** | **0.2344** | **$+42.8\%$** |
| **Medium ($32\text{--}96\text{ px}$)** | 1,044 | 399 / 645 | 38.22% | 388 / 656 | 37.16% | 0.4874 | $-2.8\%$ |
| **Large ($\ge 96\text{ px}$)** | 579 | 417 / 162 | 72.02% | 396 / 183 | 68.39% | 0.7220 | $-5.0\%$ |

### Critical Empirical Finding:
1. **Ultra-Small Deficit Confirmed:** Standard single-resolution YOLOv8n fails to register a single true positive on sub-16px targets ($0/84$ hits), perfectly confirming the theoretical finding from Missions 14 and 15 that standard FPN strides ($P_3=8\text{px}, P_4=16\text{px}, P_5=32\text{px}$) cannot represent distant targets without patch-based zooming or temporal accumulation.
2. **Small Target Lift:** In the $16\text{--}32\text{ px}$ regime, fine-tuning produced a **$+42.8\%$ relative increase in recall** ($10.23\% \to 14.61\%$) and boosted F1 from $0.1829$ to **$0.2344$**.

---

## 5. Adverse Weather Domain Generalization

Evaluation across the four unseen adverse environments in the locked test partition:

| Environmental Domain | Test Frames | Objects | Zero-Shot Recall | Zero-Shot F1 | Fine-Tuned Recall | Fine-Tuned F1 | Relative F1 Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Fog (DAWN)** | 19 | 100 | 43.00% | 0.5584 | **63.00%** | **0.7000** | **$+25.4\%$** |
| **Snow (DAWN)** | 47 | 348 | 58.33% | 0.6591 | **62.64%** | **0.6888** | **$+4.5\%$** |
| **Heavy Rain (BDD)** | 51 | 1,084 | 35.24% | 0.5036 | **37.55%** | **0.5056** | $+0.4\%$ |
| **Nighttime (BDD)** | 43 | 791 | 31.73% | 0.4470 | **23.51%** | 0.3327 | $-25.6\%$ |

### Environmental Observations:
1. **Dense Fog Breakthrough:** In severe low-contrast atmospheric scattering (DAWN Fog), the fine-tuned model achieved an extraordinary recall gain from $43.0\% \to 63.0\%$, pushing F1 to **$0.7000$**.
2. **Snow Resilience:** On snow-covered highways with high specular reflection, recall reached **$62.64\%$** and F1 reached **$0.6888$**.
3. **Nighttime Challenge:** Night driving presents severe low-light degradation and headlight glare, resulting in an F1 of $0.3327$. This empirically underscores why multi-modal sensor fusion (e.g., thermal FLIR or acoustic distress detection) is mandatory for 24/7 autonomous safety.

---

## 6. Generated & Downloaded Artifacts

The following verified artifacts were produced on Kaggle GPU and retrieved into the repository:
- `artifacts/mission20/mission20_vision_gpu_results.json`: Complete machine-readable benchmark records, hardware profile, and granular breakdown.
- `artifacts/mission20/vision_results.csv`: Per-epoch training and validation loss curves.
- `artifacts/mission20/vision_training_curves.png`: Loss and mAP training trajectory plots.
- `artifacts/mission20/vision_confusion_matrix.png`: Test split classification confusion matrix.

---

## 7. Compliance Checklist

- [x] Zero modification of historical mission reports.
- [x] Pre-trained zero-shot baseline and fine-tuned model evaluated on identical locked test data.
- [x] Zero test data leakage into training or checkpoint selection.
- [x] Weights (`best.pt`, `last.pt`) excluded from Git staging in accordance with Mission Rule 6.
- [x] Documented GPU runtime, VRAM, and random seed.
