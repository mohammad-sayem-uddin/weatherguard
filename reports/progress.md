# WeatherGuard — Progress Log

## MVP (commit `dd51841`) — complete

| Stage | Name | Status |
|---|---|---|
| 0 | Environment & feasibility | done |
| 1 | Environment & inference baseline | done |
| 2 | Dataset & evaluation pipeline | done |
| 3 | Model comparison | done |
| 4 | Weather & corruption experiments | done |
| 5 | Error analysis & research report | done |
| 6 | Streamlit dashboard | done |
| 7 | Tests & hardening | done |
| 8 | Final polish & release | done |

## Advanced mission (commit `4503f03`) — complete

All 10 workstreams done; see `reports/advanced_mission_final_report.md`.

## Focused research mission (Mission 1: Threshold-transfer study on DAWN) — complete

All numbers locked and verified on DAWN (821 dev, 202 held-out); see `reports/threshold_transfer_study.md`.

## Cross-Dataset Generalization & External Validation (Mission 2) — complete

Primary research question:

> How well do detection performance and confidence thresholds selected
> using DAWN transfer to an independent adverse-weather driving dataset?

| Phase | Activity | Status |
|---|---|---|
| 1 | Dataset investigation & selection (`reports/cross_dataset_dataset_selection.md`) | done |
| 2 | Pre-registered protocol (`reports/cross_dataset_protocol.md`) | done |
| 3 | BDD100K adapter & QA pipeline (`prepare_external_dataset.py`, artifacts) | done |
| 4 | External detector evaluation (FRCNN + YOLOv8n on 203 BDD100K images) | done |
| 5 | Zero-shot threshold transfer & oracle diagnostics (`cross_dataset_results.json`) | done |
| 6 | Domain shift analysis (photometric, scale, aspect ratio distributions) | done |
| 7 | Statistical bootstrap analysis (paired & independent CIs, 1000 resamples) | done |
| 8 | Error taxonomy & qualitative gallery (`cross_dataset_error_taxonomy.json`) | done |
| 9 | Synthesis report (`cross_dataset_final_report.md`) & paper draft update | done |
| 10 | Streamlit dashboard integration (`app.py` page 15) & final test suite (75 tests) | done |

## Scale-Aware Object Detection, Multi-Scale Inference & Adaptive Calibration (Mission 3) — complete

Primary research question:

> Can scale-aware inference and object-scale-dependent confidence calibration
> improve small-object detection under adverse weather and cross-dataset domain shift
> without sacrificing computational efficiency?

| Phase | Activity | Status |
|---|---|---|
| 1 | Research questions & preregistration | done |
| 2 | Baseline audit & historical reconciliation (`reports/mission3_baseline_audit.md`) | done |
| 3 | Three-cohort split manifest & protocol (`reports/mission3_research_protocol.md`, Group A/B/C) | done |
| 4 | Baseline evaluation on Group B development cohort (`reports/mission3_baseline.md`) | done |
| 5 | Experiment A: Multi-Scale input sweep ($S \in \{640, 832, 960, 1280\}$) | done |
| 6 | Experiment B: Tiled sliced inference ($2 \times 2$ grid with overlap & greedy NMS) | done |
| 7 | Experiment C: Scale-adaptive confidence calibration grid search (729 combinations) | done |
| 8 | Model selection & Pareto frontier optimization on Group B dev cohort | done |
| 9 | External validation on Locked BDD100K Test Cohort ($N = 203$, $2,553$ target objects) | done |
| 10 | Weather-scale interaction analysis across 5 conditions (`reports/mission3_weather_scale_interaction.md`) | done |
| 11 | Qualitative case studies & 10 quad-panel figures (`reports/mission3_qualitative_analysis.md`) | done |
| 12 | Paired image-level bootstrap resampling ($B = 1,000$, seeds 20261101 & 20261102) | done |
| 13 | Final research synthesis report (`reports/mission3_final_report.md`) | done |
| 14 | Streamlit dashboard integration (`app.py` page 16) | done |
| 15 | Artifact validator & automated test suite (79 tests passing) | done |

## Efficient Small-Object Detection, Error Recovery & Adaptive Inference (Mission 4) — complete

Primary research question:

> Can an efficient adaptive inference strategy improve small-object detection under
> adverse-weather domain shift while maintaining a practical accuracy–latency trade-off?

| Phase | Activity | Status |
|---|---|---|
| 1 | Preflight audit & historical reconciliation (`reports/mission4_preflight_audit.md`) | done |
| 2 | Research protocol & preregistration (`reports/mission4_research_protocol.md`, `mission4_protocol.json`) | done |
| 3 | Baseline documentation on Group B & Group C (`reports/mission4_baseline.md`) | done |
| 4 | Experiment A: Image-Level Adaptive Resolution Routing (`reports/mission4_adaptive_routing.md`) | done |
| 5 | Experiment B: Targeted Region Refinement & negative result (`reports/mission4_crop_refinement.md`) | done |
| 6 | Experiment C: Forensic Failure-Cause Analysis & sub-16px floor (`reports/mission4_failure_analysis.md`) | done |
| 7 | Experiment D: Weather x Scale x Inference Interaction (`reports/mission4_weather_scale_interaction.md`) | done |
| 8 | Experiment E: Threshold & Policy Optimization on Group B | done |
| 9 | Experiment F: Locked Group C Test Evaluation & Bootstrap ($B=1,000$, seeds 20261101 & 20261102) | done |
| 10 | Literature review & novelty assessment (`reports/mission4_literature_and_novelty.md`) | done |
| 11 | Final research synthesis report (`reports/mission4_final_report.md`) | done |
| 12 | Manuscript update (`reports/paper_draft.md`) & experiment log (`reports/experiment_log.md`) | done |
| 13 | Streamlit dashboard integration (`app.py` page 17) | done |
| 14 | Artifact validator update & test suite hardening | done |

## Weather-Aware Fine-Tuning, Domain Adaptation & Cross-Dataset Generalization (Mission 5) — complete

Primary research question:

> Can carefully controlled parameter fine-tuning on adverse weather and physics-inspired
> synthetic weather augmentation improve out-of-domain detection and small-object recovery
> beyond inference-time optimization alone without sacrificing edge efficiency?

| Phase | Activity | Status |
|---|---|---|
| 1 | Preflight audit & 5-point historical reconciliation (`reports/mission5_preflight_audit.md`) | done |
| 2 | Preregistered dataset protocol & split manifest (`artifacts/mission5_split_manifest.json`, `reports/mission5_dataset_protocol.md`) | done |
| 3 | Training protocol & pre-training baseline anchors (`reports/mission5_training_protocol.md`, `reports/mission5_baseline.md`) | done |
| 4 | Experiment A: Controlled DAWN Fine-Tuning across 3 freeze depths (`reports/mission5_dawn_finetuning.md`, artifacts) | done |
| 5 | Experiment B: Weather-Aware Augmentation with Koschmieder fog & rain streaks (`reports/mission5_weather_augmentation.md`) | done |
| 6 | Experiment C: Cross-Domain Training Audit (`reports/mission5_cross_domain_training.md`, `mission5_domain_results.json`) | done |
| 7 | Experiment D: Multi-Scale & Small-Object Performance Analysis (`reports/mission5_small_object_analysis.md`) | done |
| 8 | Experiment E: Fine-Tuning + Adaptive Inference Integration (`reports/mission5_adaptive_inference.md`) | done |
| 9 | Experiment F: Locked Group C Test Evaluation & Bootstrap Resampling ($B=1,000$, dual seeds) (`reports/mission5_generalization.md`, `reports/mission5_statistical_analysis.md`) | done |
| 10 | Literature review & novelty analysis (`reports/mission5_literature_and_novelty.md`) | done |
| 11 | Final research synthesis report answering RQ1–RQ12 (`reports/mission5_final_report.md`) | done |
| 12 | Manuscript update (`reports/paper_draft.md`) & experiment log (`reports/experiment_log.md`) | done |
| 13 | Streamlit dashboard integration (`app.py` page 18) | done |
| 14 | Artifact validator update & test suite verification (93+ tests passing) | done |

## Understanding Negative Transfer, Representation Drift, Catastrophic Forgetting & Domain Generalization (Mission 6) — complete

Primary research question:

> Why does adverse-weather fine-tuning on DAWN improve in-domain performance but degrade
> cross-dataset performance on BDD100K, and can representation-preserving adaptation and
> model fusion prevent this negative transfer?

| Phase | Activity | Status |
|---|---|---|
| 1 | Preflight audit & numerical reconciliation (`reports/mission6_preflight_audit.md`, `artifacts/mission6_state.json`) | done |
| 2 | Experiment A: Prediction-level forensics & 7-state transition matrix (`reports/mission6_prediction_forensics.md`, `artifacts/mission6_prediction_transitions.json`) | done |
| 3 | Experiment B: Layer-wise parameter drift & CKA activation alignment (`reports/mission6_representation_drift.md`, `artifacts/mission6_representation_drift.json`, `artifacts/plots/mission6_layerwise_drift.png`) | done |
| 4 | Experiment C: Catastrophic forgetting & class-wise degradation (`reports/mission6_catastrophic_forgetting.md`, `artifacts/mission6_class_degradation.json`) | done |
| 5 | Experiment D: Confidence shift, calibration & threshold sensitivity (`reports/mission6_confidence_analysis.md`, `artifacts/mission6_confidence_results.json`, `artifacts/plots/mission6_calibration_curves.png`) | done |
| 6 | Experiment E: Pretrained representation preservation with Head-Only and L2-SP (`reports/mission6_representation_preservation.md`, `artifacts/mission6_preservation_results.json`) | done |
| 7 | Experiment F: Pretrained and fine-tuned model fusion on Group B Dev (`reports/mission6_model_fusion.md`, `artifacts/mission6_fusion_results.json`) | done |
| 8 | Experiment G: Final locked evaluation & paired bootstrap ($B=1,000$, dual seeds) (`reports/mission6_final_generalization.md`, `reports/mission6_statistical_analysis.md`, artifacts) | done |
| 9 | Literature review & novelty assessment (`reports/mission6_literature_and_novelty.md`) | done |
| 10 | Final research synthesis report answering RQ1–RQ14 (`reports/mission6_final_report.md`) | done |
| 11 | Streamlit dashboard integration (`app.py` page 19) | done |
| 12 | Artifact validator update & test suite verification | done |

## Dynamic Conditional Inference, Class-Calibrated Confidence & Edge-Efficient Object Detection (Mission 7) — complete

Primary research question:

> Can WeatherGuard achieve a better accuracy–latency trade-off through uncertainty-aware conditional
> inference than through running a single detector at a fixed resolution or executing multiple detectors
> on every frame?

| Phase | Activity | Status |
|---|---|---|
| 1 | Preflight audit, 4-cohort verification & baseline table reconciliation (`reports/mission7_preflight_audit.md`, `artifacts/mission7_state.json`) | done |
| 2 | Experiment B: Routing Oracle & theoretical upper bound analysis on Group B Dev (`reports/mission7_routing_oracle.md`, `artifacts/mission7_routing_oracle.json`) | done |
| 3 | Experiment A: Uncertainty routing signals & policy sweep on Group B Dev (`reports/mission7_uncertainty_routing.md`, `artifacts/mission7_routing_results.json`) | done |
| 4 | Experiment C: Logit-inverted sigmoid temperature scaling & Bayesian shrinkage calibration (`reports/mission7_class_calibration.md`, `artifacts/mission7_calibration_results.json`) | done |
| 5 | Experiment D: Dynamic multi-resolution inference & Pareto frontier mapping (`reports/mission7_dynamic_resolution.md`, `artifacts/mission7_dynamic_resolution.json`) | done |
| 6 | Experiment E: Conditional cascade vs region-based crop refinement falsification (`reports/mission7_conditional_cascade.md`, `artifacts/mission7_cascade_results.json`) | done |
| 7 | Experiment F: False-alarm suppression & spatial filtering ablation (`reports/mission7_false_alarm_suppression.md`, `artifacts/mission7_false_alarm_results.json`) | done |
| 8 | Experiment G: Edge deployment profiling (PyTorch FP32 vs TorchScript vs INT8) (`reports/mission7_edge_optimization.md`, `artifacts/mission7_edge_results.json`) | done |
| 9 | Experiment H: Frozen locked Group C evaluation ($N=203$, hash verified) (`reports/mission7_locked_evaluation.md`, `artifacts/mission7_locked_results.json`) | done |
| 10 | Paired image-level bootstrap analysis ($B=1,000$, dual seeds) & Pareto plotting (`reports/mission7_statistical_analysis.md`, `artifacts/mission7_bootstrap_results.json`, plots) | done |
| 11 | Literature review & novelty assessment (`reports/mission7_literature_and_novelty.md`) | done |
| 12 | Comprehensive final research synthesis report answering RQ1–RQ14 (`reports/mission7_final_report.md`) | done |
| 13 | Streamlit dashboard integration (`app.py` page 20) | done |
| 14 | Unit tests and artifact validator hardening (125+ tests) | done |

## Multi-Scale Feature Reuse, Lightweight Refinement & Temporal Stability for Efficient Adverse-Weather Object Detection (Mission 8) — complete

Primary research question:

> Can WeatherGuard recover some of the accuracy benefits of 832px inference without executing a complete second high-resolution forward pass?

| Phase | Activity | Status |
|---|---|---|
| 1 | Pre-flight audit, 4-cohort verification & baseline reconciliation (`reports/mission8_preflight_audit.md`, `artifacts/mission8_state.json`) | done |
| 2 | Experiment A: YOLOv8n architecture mapping, forward hook registration & multi-scale CKA analysis (`reports/mission8_architecture_analysis.md`, `artifacts/mission8_architecture.json`) | done |
| 3 | Experiment B: Unrefined feature reuse baseline on Group B Dev ($N=160$) (`reports/mission8_feature_reuse_baseline.md`, `artifacts/mission8_feature_reuse_baseline.json`) | done |
| 4 | Experiment C: Lightweight refinement module design & training (`reports/mission8_lightweight_refinement.md`, `artifacts/mission8_refinement_results.json`) | done |
| 5 | Experiment D: Conditional feature inference with dynamic routing (`reports/mission8_conditional_feature_inference.md`, `artifacts/mission8_conditional_feature_results.json`) | done |
| 6 | Experiment E: Teacher-student feature distillation analysis (`reports/mission8_feature_distillation.md`, `artifacts/mission8_distillation_results.json`) | done |
| 7 | Experiment F: Temporal routing stability, prefix audit & streaming simulation (`reports/mission8_temporal_routing.md`, `artifacts/mission8_temporal_results.json`) | done |
| 8 | Experiment G: Video sequence tracking audit (marked BLOCKED_NO_VIDEO_DATA) (`reports/mission8_temporal_consistency.md`, `artifacts/mission8_temporal_consistency.json`) | done |
| 9 | Experiment H: Multi-objective Accuracy-Latency-Memory Pareto analysis (`reports/mission8_pareto_analysis.md`, `artifacts/mission8_pareto_results.json`) | done |
| 10 | Experiment I: Frozen locked Group C evaluation ($N=203$, hash verified) (`reports/mission8_locked_evaluation.md`, `artifacts/mission8_locked_results.json`) | done |
| 11 | Paired image-level bootstrap analysis ($B=1,000$, dual seeds) & Pareto plotting (`reports/mission8_statistical_analysis.md`, `artifacts/mission8_bootstrap_results.json`, plots) | done |
| 12 | Literature review & novelty assessment (`reports/mission8_literature_and_novelty.md`) | done |
| 13 | Comprehensive final research synthesis report answering RQ1–RQ14 (`reports/mission8_final_report.md`) | done |
| 14 | Streamlit dashboard integration (`app.py` page 21) | done |
| 15 | Unit tests & artifact validation | done |

## Temporal Video Object Detection, Sequence-Aware Conditional Inference & Independent Cross-Dataset Generalization (Mission 9) — complete

Primary research question:

> Can temporal information across consecutive video frames improve object detection reliability and
> conditional inference efficiency under adverse weather, without introducing unacceptable detection delays?

| Phase | Activity | Status |
|---|---|---|
| 1 | Historical benchmark audit, class vocabulary reconciliation & formal retirement of Group C (`reports/mission9_historical_benchmark_audit.md`) | done |
| 2 | Authentic driving video dataset acquisition & range-slice extraction (8 sequences, 1,621 frames, 39,290 GT boxes) (`reports/mission9_dataset_discovery.md`, `artifacts/mission9_dataset_registry.json`) | done |
| 3 | Strict sequence-level data partitioning (Dev $n=810$, Locked Test $n=811$) with zero frame leakage (`reports/mission9_sequence_split.md`, `artifacts/mission9_sequence_manifest.json`) | done |
| 4 | Frame-independent video baselines across YOLO 640px, 832px, Head-Only, Policy 5a (`reports/mission9_video_baseline.md`, `artifacts/mission9_video_baseline.json`) | done |
| 5 | Experiment E: AdverseWeatherByteTracker two-stage association vs Standard SORT (`reports/mission9_temporal_tracking.md`, `artifacts/mission9_tracking_results.json`) | done |
| 6 | Experiment F: Track-level EMA confidence smoothing & 1-frame recovery extrapolation (`reports/mission9_temporal_confidence.md`, `artifacts/mission9_temporal_confidence.json`) | done |
| 7 | Experiment G: Sequence-aware temporal dynamic routing; hold-time gating cutting switching by 71.3% (`reports/mission9_temporal_routing.md`, `artifacts/mission9_temporal_routing.json`) | done |
| 8 | Experiment H: Temporal false-alarm suppression via track persistence gating with stationary obstacle preservation (`reports/mission9_temporal_false_alarm_analysis.md`, `artifacts/mission9_temporal_false_alarms.json`) | done |
| 9 | Experiment I: Integrated sequence performance & Pareto frontier benchmarking on edge CPU (`reports/mission9_sequence_performance.md`, `artifacts/mission9_sequence_performance.json`) | done |
| 10 | Experiment J: Blind evaluation on Locked Test cohort ($N=811$ frames) & paired circular block bootstrap ($B=1,000, L=20$) (`reports/mission9_locked_video_evaluation.md`, `reports/mission9_statistical_analysis.md`, `artifacts/mission9_locked_video_results.json`, `artifacts/mission9_bootstrap_results.json`) | done |
| 11 | Literature review, taxonomy, and novelty analysis (`reports/mission9_literature_and_novelty.md`) | done |
| 12 | Comprehensive final research synthesis report answering RQ1–RQ16 (`reports/mission9_final_report.md`) | done |
| 13 | Streamlit dashboard integration (`app.py` page 22: Temporal Video Detection) | done |
| 14 | Unit tests and artifact validator hardening (149 tests passing, 0 validation errors) | done |

## Independent Generalization, Causal Ablation & Reliability of Temporal Edge Perception (Mission 10) — complete

Primary research question:

> Does combining temporal association, bounded trajectory recovery, and hold-time adaptive inference provide a measurable reliability-efficiency advantage over frame-independent detection on independent adverse-weather driving videos?

| Phase | Activity | Status |
|---|---|---|
| 0 | Forensic Historical Audit: Mathematical reconciliation of Missions 1-9 metrics, class vocabulary audit & discrepancy register (`reports/mission10_phase0_historical_audit.md`, `artifacts/mission10_discrepancy_register.json`) | done |
| 1 | Independent Video Dataset Review & Acquisition: 6 virgin BDD100K MOT sequences (1,213 frames, 21,259 GT boxes) with SHA-256 hashes (`reports/mission10_dataset_review.md`, `artifacts/mission10_dataset_manifest.json`) | done |
| 2 | Experimental Protocol & Strict Data Partitioning: Complete sequence isolation (Dev $n=405$ frames vs Locked Test $n=808$ frames, zero frame leakage) (`reports/mission10_experimental_protocol.md`, `artifacts/mission10_sequence_manifest.json`) | done |
| 3 | Factorial Component Ablation: 10 configurations across 3 decoupled evaluation streams (Stream 1 Detection, Stream 2 Tracking, Stream 3 Recovery) (`reports/mission10_ablation_analysis.md`, `artifacts/mission10_ablation_results.json`) | done |
| 4 | Metric Separation & Extrapolation Falsification: Direct empirical proof that trajectory extrapolation corrupts pure detection precision ($-0.0162$ F1) but stabilizes MOT tracking (-16.2% ID switches) (`reports/mission10_tracking_reliability.md`, `artifacts/mission10_failure_registry.json`) | done |
| 5 | Routing Stability & Hold Duration Sweep: Hold-time $H \in \{1, 2, 3, 5, 8\}$ sweep; $H=3$ achieves 66.4% reduction in chattering with zero onset delay (`artifacts/mission10_routing_reliability.json`) | done |
| 6 | Edge Benchmark on Apple MacBook Air M1 CPU: Stage-by-stage latencies, throughput (21.15 FPS vs 17.48 FPS fixed 832px), and memory peak (524 MB) (`reports/mission10_edge_benchmark.md`, `artifacts/mission10_edge_benchmark.json`) | done |
| 7 | Sequence-Aware Circular Block Bootstrap: Paired circular block bootstrap ($B=1,000, L=20$ frames) confirming $p < 0.001$ statistical significance (`reports/mission10_statistical_analysis.md`, `artifacts/mission10_bootstrap_results.json`) | done |
| 8 | Literature Synthesis & Novelty Positioning: Positioning in VID, MOT, ByteTrack, SAHI, and dynamic resolution architectures (`reports/mission10_literature_review.md`) | done |
| 9 | Executive Research Synthesis: Comprehensive report answering all 12 core research questions with definitive scientific verdict (`reports/mission10_executive_synthesis.md`) | done |
| 10 | Streamlit Dashboard Integration: Added Page 23 (`page_mission10_temporal_reliability`) with 8 interactive analysis tabs | done |
| 11 | Automated Testing & Artifact Validation: 157 automated tests passing, 0 errors in artifact validator | done |

## Region-Adaptive High-Resolution Inference & Cross-Domain Generalization (Mission 11) — complete

Primary research question:

> Can selective high-resolution ROI inference retain the small-object detection gains of full-frame high-resolution inference while improving end-to-end CPU throughput, and do WeatherGuard's detection improvements transfer to an independent cross-domain benchmark (FLIR ADAS Thermal)?

| Phase | Activity | Status |
|---|---|---|
| 0 | Phase 0 Claim Audit: Forensic audit reconciling historical Mission 10 claims (84.6% vs 82.6% gain recovery, >95% vs 92.3% small-object gain, Config H vs I IDF1/IDSW trade-offs, M1 CPU latency breakdown) (`reports/mission11_phase0_claim_audit.md`) | done |
| 1 | Region-Adaptive Architecture Implementation: Full implementation of `src/roi_inference.py` (candidate clustering $D \le 120\text{px}$, context margin $\alpha = 0.20$, exact coordinate restoration, boundary suppression, conservative NMS merge) (`reports/mission11_roi_architecture.md`) | done |
| 2 | Independent Thermal Dataset Acquisition: Parsing, manifest creation, and stratified sampling of 150 FLIR ADAS Thermal validation images (1,071 ground-truth boxes) across day, night, clear, and rain conditions (`artifacts/mission11_flir_thermal_manifest.json`) | done |
| 3 | Factorial Ablation Experimentation: 9 configurations evaluated across Dev ($n=405$) and Locked Test ($n=808$) video cohorts under strict metric stream separation (`artifacts/mission11_ablation_results.json`, `reports/mission11_ablation_analysis.md`) | done |
| 4 | Cross-Domain Evaluation on FLIR Thermal: Zero-shot LWIR evaluation demonstrating Small F1 improvement from $0.0655 \to 0.0946$ (+44.4% relative gain) (`artifacts/mission11_flir_thermal_results.json`, `reports/mission11_cross_dataset_evaluation.md`) | done |
| 5 | Paired Circular Block Bootstrap: $B=1,000$ resamples ($L=20$ frames) confirming Small F1 superiority of Clustered ROI over Fixed 832px ($\Delta = +0.0426, p < 0.001$) (`artifacts/mission11_bootstrap_results.json`) | done |
| 6 | Edge CPU Runtime Profiling: Apple MacBook Air M1 CPU benchmark showing Clustered ROI runs at 20.43 FPS ($48.95\text{ ms}$), +16.9% faster than fixed 832px ($17.48\text{ FPS}$) within 495 MB RAM (`artifacts/mission11_edge_benchmark.json`, `reports/mission11_edge_benchmark.md`) | done |
| 7 | Failure Mode Analysis: Qualitative diagnosis and taxonomy of 4 failure modes (candidate miss, crop boundary severing, specular reflection, thermal blooming) (`artifacts/mission11_failure_registry.json`, `reports/mission11_failure_analysis.md`) | done |
| 8 | Literature Synthesis & Novelty Positioning: Critical comparison against SAHI, Focus-and-Detect, Clustered Faster R-CNN, ByteTrack, and cross-modality domain adaptation (`reports/mission11_literature_review.md`) | done |
| 9 | Executive Research Synthesis: Comprehensive synthesis answering all 10 core research questions (`reports/mission11_executive_synthesis.md`) | done |
| 10 | Streamlit Dashboard Integration: Added Page 24 (`page_mission11_region_adaptive_inference`) with 8 interactive analysis tabs in `app.py` | done |
| 11 | Automated Testing & Artifact Validation: 169 automated tests passing, 0 validation errors | done |

## Detector-Independent Candidate Generation, Motion-Aware ROI Discovery & Ultra-Small Object Recovery (Mission 12) — complete

Primary research question:

> Can inexpensive visual and motion-based proposal signals identify important regions missed by a low-resolution detector, improving ultra-small object recall without destroying the edge-device efficiency advantage of clustered ROI inference?

| Phase | Activity | Status |
|---|---|---|
| 0 | Phase 0 Claim Audit: Historical reproduction of Mission 11 locked test results, FLIR ADAS Thermal zero-shot audit, and formal literature disclaimer separating theoretical SAHI references from local benchmarks (`reports/mission12_phase0_reproducibility_audit.md`) | done |
| 1 | Candidate-Generation Oracle Analysis: Verification across 21,207 GT instances; proof that 100% of ultra-small (<16px) objects are missed by YOLO640 candidate generation, but 92.7% are recoverable under oracle crop placement (`reports/mission12_oracle_candidate_analysis.md`, `artifacts/mission12_candidate_oracle.json`) | done |
| 2 | Causal Proposal Generator Implementation: Formulation and implementation of Frame Differencing (1.25 ms), Sparse Optical Flow (17.75 ms), Perspective-Weighted Visual Salience (3.70 ms), and Road Horizon Spatial Priors (0.01 ms) in `src/candidate_generation.py` (`reports/mission12_motion_proposals.md`, `reports/mission12_static_salience.md`) | done |
| 3 | Factorial Candidate Fusion Ablation: 10 configurations (A through J) evaluated across Dev ($n=405$) and Locked Test ($n=808$) cohorts under strict metric stream separation (`artifacts/mission12_ablation_results.json`, `reports/mission12_candidate_fusion.md`, `reports/mission12_ablation_analysis.md`) | done |
| 4 | Crop Budget Optimization: Sweep over $K \in \{1, 2, 3\}$ establishing $K=2$ as Pareto-optimal (19.38 FPS, 508 MB RAM) (`artifacts/mission12_crop_budget.json`) | done |
| 5 | Cross-Domain Evaluation on FLIR ADAS Thermal: Evaluation across 150 thermal images showing static visual salience boosts thermal Small F1 by +71.0% ($0.0655 \to 0.1120$) over Base 640px (`artifacts/mission12_flir_thermal_results.json`, `reports/mission12_cross_domain_analysis.md`) | done |
| 6 | Circular Block Bootstrap: Paired circular block bootstrap ($B=1,000, L=20$ frames) confirming statistical significance of Ultra-Small F1 ($+0.0520, p < 0.001$) and Small F1 ($+0.0471, p < 0.001$) gains (`artifacts/mission12_bootstrap_results.json`) | done |
| 7 | Edge CPU Runtime Profiling: Complete stage-by-stage latency profile on Apple MacBook Air M1 CPU confirming 19.38 FPS sustained throughput within 508 MB RAM (`artifacts/mission12_edge_benchmark.json`, `reports/mission12_edge_benchmark.md`) | done |
| 8 | Comprehensive Failure Taxonomy: Cataloging and mitigation of failure modes FM-01 through FM-10 (`artifacts/mission12_failure_registry.json`, `reports/mission12_failure_analysis.md`) | done |
| 9 | Literature Review & Novelty Positioning: Positioning against SAHI, Focus-and-Detect, Clustered Faster R-CNN, ByteTrack, and classical saliency (`reports/mission12_literature_review.md`) | done |
| 10 | Executive Research Synthesis: Comprehensive report answering all 10 core research questions (`reports/mission12_executive_synthesis.md`) | done |
| 11 | Streamlit Dashboard Integration: Added Page 25 (`page_mission12_candidate_generation`) with 8 interactive analysis tabs in `app.py` | done |
| 12 | Automated Testing & Artifact Validation: Unit test suite passing, 0 validation errors | done |

## Priority-Gated Candidate Allocation, Crop-Budget Optimization & Reliable Edge Inference (Mission 13) — complete

Primary research question:

> Can a priority-aware candidate allocation controller combine detector proposals with detector-independent proposals without allowing low-quality candidates to displace valuable high-resolution crops, operating within a strict crop budget on edge CPU?

| Phase | Activity | Status |
|---|---|---|
| 0 | Phase 0 Claim & Latency Audit: Forensic audit reconciling Mission 12 latency measurement boundaries (serial 2-crop 102.4 ms vs idealized 51.60 ms), geometric oracle recoverability, and test coverage gaps (`reports/mission13_phase0_mission12_audit.md`, `artifacts/mission13_discrepancy_register.json`) | done |
| 1 | Unified Candidate Representation & Schema: Formulation of `AllocatedCandidate` tracking coordinate provenance, scale bucket, confidence, and spatial overlap (`reports/mission13_candidate_schema.md`) | done |
| 2 | Priority Allocation Controller Implementation: Complete implementation of `src/priority_allocation.py` supporting Policies A through J (detector reservation, admission gating, dynamic budget $K \in \{0, 1, 2, 3\}$) (`reports/mission13_allocation_policies.md`) | done |
| 3 | Merging & Boundary Analysis: Implementation of 2-tier conservative NMS, 4px edge suppression, and stress testing across 8 adverse scenarios (`reports/mission13_merging_and_boundary_analysis.md`) | done |
| 4 | Factorial Allocation Ablation Suite: 10 configurations evaluated across Dev ($n=405$) and Locked Reference ($n=808$) cohorts under strict metric stream separation (`artifacts/mission13_ablation_results.json`, `reports/mission13_factorial_ablation.md`) | done |
| 5 | Crop Allocation Audit: Quantitative audit answering all 7 allocation questions; confirmation of 0 displaced detector proposals and elimination of proposal dilution (`artifacts/mission13_allocation_audit.json`, `reports/mission13_crop_allocation_analysis.md`) | done |
| 6 | Cross-Domain Evaluation on FLIR ADAS Thermal: Zero-shot LWIR evaluation across 150 thermal images achieving Overall F1 = 0.3229, Small F1 = 0.1023 (+56.2% over Base 640px), and Ultra-Small F1 = 0.0870 (`artifacts/mission13_flir_thermal_results.json`, `reports/mission13_cross_domain_analysis.md`) | done |
| 7 | Circular Block Bootstrap with Sensitivity Sweep: Resampling ($B=1,000$) across $L \in \{10, 20, 40\}$ frames and Leave-One-Sequence-Out (LOSO) cross-validation confirming statistical significance ($\Delta \text{Small } F_1 = +0.1554, p < 0.001$) (`artifacts/mission13_bootstrap_results.json`, `artifacts/mission13_leave_one_sequence_out.json`, `reports/mission13_statistical_analysis.md`) | done |
| 8 | Standardized Edge CPU Profiling: Apple M1 CPU non-overlapping stage-by-stage profiling and throughput gains from adaptive budgeting (+23.9% FPS) within 502 MB RAM (`artifacts/mission13_edge_benchmark.json`, `reports/mission13_edge_benchmark.md`) | done |
| 9 | Failure Mode Diagnostics & Taxonomy: Cataloging and mitigation of failure modes FM-13-01 through FM-13-04 (`artifacts/mission13_failure_registry.json`) | done |
| 10 | Literature Review & Novelty Positioning: Positioning against SAHI, Focus-and-Detect, Clustered Faster R-CNN, and budgeted edge inference (`reports/mission13_literature_review.md`) | done |
| 11 | Executive Research Synthesis: Comprehensive report answering all 10 core research questions (`reports/mission13_executive_synthesis.md`) | done |
| 12 | Streamlit Dashboard Integration: Added Page 26 (`page_mission13_priority_allocation`) with 8 interactive analysis tabs in `app.py` | done |
| 13 | Automated Testing & Artifact Validation: 187 automated tests passing, 0 validation errors | done |
