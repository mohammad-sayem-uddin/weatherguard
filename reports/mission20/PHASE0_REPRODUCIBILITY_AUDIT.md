# WeatherGuard & Rescue Drone — Mission 20: Phase 0 Reproducibility Audit

**Date:** 2026-11-05  
**Author / Principal Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Commit Inspected:** `6271d0d` (`master`)  
**Target Architecture:** Kaggle Cloud NVIDIA Tesla T4 GPU (16 GB GDDR6 VRAM)  
**Objective:** Independent forensic audit of prior smoke-test experiments, dataset manifests, and cloud GPU executions.

---

## 1. Executive Summary

This audit independently inspects commit `6271d0d` and all associated artifacts to establish the exact provenance, limitations, and methodological boundaries of the experiments reported in the previous session.

The findings confirm that while the **cloud GPU orchestration** (Kaggle MCP integration, NVIDIA Tesla T4 initialization, AMP mixed-precision execution, and artifact retrieval) was successfully validated, the experimental evaluation workloads in that session were **preliminary infrastructure smoke-tests**, rather than full-corpus research benchmarks. Specifically:
1. The WeatherGuard vision training run utilized Ultralytics' standard 8-image smoke-test dataset (`coco8.yaml`) rather than the curated 1,000-frame WeatherGuard V2 corpus.
2. The Rescue Drone audio training run utilized a synthetic Gaussian-noise spectrogram tensor fixture (`SyntheticSpectrogramDataset`) rather than the curated 1,500 physical WAV recordings.
3. The reported 100% metrics in the audio smoke-test reflect the artificial simplicity of the synthetic fixture, not real-world distress detection capability.
4. The local 1,000-frame image corpus and 1,500-file audio corpus exist in full on disk and their cryptographic SHA-256 master digests match the documented manifests byte-for-byte.

The historical reports correctly logged these runs as infrastructure verifications; Mission 20 now formally replaces these initial harness checks with rigorous, leakage-resistant experiments on the actual datasets.

---

## 2. Forensic Investigation of Prior Benchmark Runs

### 2.1. Why the WeatherGuard Benchmark Used `coco8.yaml` Rather Than WeatherGuard V2
- **Investigation:** In `scripts/kaggle/generate_kaggle_notebook.py` (lines 88–96) and the submitted Kaggle notebook `allfaz/weatherguard-gpu-benchmark-tesla-t4`, the training invocation called:
  ```python
  results = run_training_experiment(
      data_yaml='coco8.yaml',
      weights_path='yolov8n.pt',
      epochs=5,
      imgsz=640,
      batch_size=16,
      output_dir='/kaggle/working/runs/train',
      experiment_id='weatherguard_t4_benchmark',
  )
  ```
- **Finding:** `coco8.yaml` was selected as a rapid, lightweight verification harness to test that the Kaggle Tesla T4 environment could clone the repository, install dependencies, allocate GPU memory, execute `model.train()` with AdamW and AMP, and output checkpoints to `/kaggle/working/`. The 1,000-frame WeatherGuard V2 dataset had not yet been packaged into a Kaggle dataset or hosted in a remote repository accessible to the Kaggle kernel. Consequently, the resulting metrics ($mAP@0.5 = 0.8970$) evaluate Ultralytics' 8-image COCO subset, **not** the WeatherGuard V2 adverse-weather road perception corpus.

### 2.2. Whether the YOLO Benchmark Outputs Correspond to the Five Recorded Epochs
- **Investigation:** The retrieved ledger [`artifacts/kaggle_gpu/results/weatherguard_t4_benchmark/results.csv`](file:///Users/sayemuddin/Desktop/WeatherGuard/artifacts/kaggle_gpu/results/weatherguard_t4_benchmark/results.csv) and associated loss curves (`results.png`, `BoxF1_curve.png`) were inspected row-by-row.
- **Finding:** Exactly 5 epochs were executed and recorded:
  - Epoch 1: Box loss 0.9474, Class loss 2.2287, mAP@0.5 = 0.89699, Cumulative time: 2.21s
  - Epoch 2: Box loss 0.9066, Class loss 1.9565, mAP@0.5 = 0.89694, Cumulative time: 2.75s
  - Epoch 3: Box loss 0.9666, Class loss 1.7241, mAP@0.5 = 0.89694, Cumulative time: 3.22s
  - Epoch 4: Box loss 0.9628, Class loss 2.5473, mAP@0.5 = 0.89694, Cumulative time: 3.67s
  - Epoch 5: Box loss 0.8287, Class loss 1.5764, mAP@0.5 = 0.88153, Cumulative time: 4.21s
- **Conclusion:** The outputs and generated plots accurately reflect the 5 completed training epochs on the smoke dataset.

### 2.3. Why the Audio Training Runner Uses `SyntheticSpectrogramDataset`
- **Investigation:** In `scripts/kaggle/audio_gpu_runner.py` (lines 50–95), the training routine instantiated:
  ```python
  train_ds = SyntheticSpectrogramDataset(num_samples=300, seed=SEED)
  val_ds = SyntheticSpectrogramDataset(num_samples=100, seed=SEED + 1)
  ```
  where `SyntheticSpectrogramDataset` generated random Gaussian tensors $\sim \mathcal{N}(0, 1)$ of shape $(1, 64, 126)$ and added $+1.5$ energy to the first 20 mel bins for class 1.
- **Finding:** This class was implemented to verify the PyTorch GPU training harness (DataLoader collation, mixed-precision loss backward pass, AdamW stepping, and ROC-AUC evaluation) without requiring a multi-megabyte audio dataset upload to Kaggle before the audio corpus was finalized and audited.

### 2.4. Whether the 4.47-Second Audio Runtime Covers All Five Epochs
- **Investigation:** The timing log and execution records in [`artifacts/kaggle_gpu/results/audio_gpu/audio_gpu_training_metrics.json`](file:///Users/sayemuddin/Desktop/WeatherGuard/artifacts/kaggle_gpu/results/audio_gpu/audio_gpu_training_metrics.json) were verified against PyTorch GPU execution on Tesla T4.
- **Finding:** With 300 synthetic samples and batch size 32, each training epoch contains only $\lceil 300 / 32 \rceil = 10$ mini-batches. Across 5 epochs, the model performed 50 forward/backward passes. On an NVIDIA Tesla T4, 50 forward/backward passes of a 3-layer 2D CNN take $\approx 0.8\text{ seconds}$ of pure CUDA execution. The recorded wall-clock time of **4.47 seconds** covered dataset tensor creation, 5 epochs of training, 5 validation passes (100 samples $\times$ 4 batches), metric computation, and model serialization to disk. All five epochs were genuinely executed.

### 2.5. Whether the 100-Sample Validation Set Is Independent of Training
- **Investigation:** `train_ds` was seeded with `SEED=42`, while `val_ds` was seeded with `SEED=43`.
- **Finding:** While the random number sequences were technically independent, both sets were generated by the exact same synthetic mathematical rule:
  $$\mathbf{X}_{c=1} = \mathbf{Z} + 1.5 \cdot \mathbf{1}_{[0:20, :]}, \quad \mathbf{X}_{c=0} = \mathbf{Z}, \quad \mathbf{Z} \sim \mathcal{N}(0, \mathbf{I})$$
  Because this decision boundary is linearly separable by simple thresholding of the low-frequency mels, the neural network achieved $1.0000$ accuracy, $1.0000$ F1, and $1.0000$ ROC-AUC by epoch 1. This test proved that the PyTorch training loop functions correctly, but provides **zero evidence** of acoustic distress classification in real audio.

### 2.6. Whether the Local Corpus and Manifest Checksums Match the Documented Versions
- **Investigation:** All local files on disk were audited against the published manifests using SHA-256 digests.
- **Finding:**
  1. **WeatherGuard Vision V2:**
     - Documented Checksum in `weatherguard_v2_manifest.json`: `aeca945140176861214088fb9e776993efb8e2e9c20a672322a36b5ec6324db6`
     - Computed Checksum from 1,000 image hashes: `aeca945140176861214088fb9e776993efb8e2e9c20a672322a36b5ec6324db6` (**Exact Match**)
     - Physical Files on Disk: 1,000 JPEG images in `data/weatherguard_corpus/images/`, 1,000 YOLO TXT labels in `data/weatherguard_corpus/labels/`.
  2. **Rescue Drone Audio V2:**
     - Documented Checksum in `audio_v2_manifest.json`: `0fe5cf238e70ba7f0b9f5462fffa5a6bb4feeb1db23cfbc8975a593441a1a547`
     - Computed Checksum from 1,500 audio hashes: `0fe5cf238e70ba7f0b9f5462fffa5a6bb4feeb1db23cfbc8975a593441a1a547` (**Exact Match**)
     - Physical Files on Disk: 750 positive WAV files and 750 negative WAV files in `/Users/sayemuddin/Desktop/Rescue_Drone/audio_dataset_v2/dataset_v2_1500/audio/`.

---

## 3. Methodological Correction Matrix

| Component | Previous Smoke-Test Implementation | Mission 20 Research Requirement |
| :--- | :--- | :--- |
| **Vision Training Data** | `coco8.yaml` (8 toy images) | Actual WeatherGuard V2 training split (curated road scenes) |
| **Vision Test Data** | `coco8.yaml` validation set (4 images) | Source-isolated locked test split |
| **Audio Training Data** | Synthetic Gaussian tensors ($N=300$) | Actual 16 kHz Mono WAV audio recordings |
| **Audio Test Data** | Synthetic Gaussian tensors ($N=100$) | Source-isolated locked audio test split |
| **Audio Classification Metric** | Trivial 1.0000 on synthetic tensors | Evaluated on real distress cries & physical noise tracks |
| **Split Isolation** | None (ad-hoc synthetic splits) | Strict source-aware grouping preventing augmented leakage |

---

## 4. Conclusion & Action Plan

The previous session established operational cloud GPU runners and produced verified dataset manifests. However, no scientific conclusions regarding object detection under adverse weather or acoustic distress detection can be drawn from smoke-test datasets.

Mission 20 proceeds directly to:
1. **Phase 1:** Comprehensive geometric and class verification of the 1,000-frame WeatherGuard corpus.
2. **Phase 2:** Building strictly source-isolated, leakage-resistant train/val/test splits.
3. **Phase 3 & 4:** Training and evaluating YOLOv8n on the real WeatherGuard dataset on Kaggle GPU.
4. **Phase 5 & 6:** Auditing real audio sources and constructing leakage-resistant audio splits.
5. **Phase 7 & 8:** Training and evaluating the CNN on real audio data on Kaggle GPU.
