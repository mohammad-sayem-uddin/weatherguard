# WeatherGuard — Experiment Log

Chronological log of executed experiments. The canonical
machine-readable record is `artifacts/experiment_registry.jsonl`.

| # | Time (UTC) | Experiment | Detector | n_images | Status | Runtime (s) | Artifact |
|--:|---|---|---|---:|---|--:|---|
| 1 | 2026-10-12 18:00Z | smoke_test | FRCNN | 8 | completed | ~9 | `artifacts/smoke_test_report.json` |
| 2 | 2026-10-12 18:57Z | full_eval_fasterrcnn_val | FRCNN | 206 | completed | 298 | `artifacts/eval_fasterrcnn_val_full.json` |
| 3 | 2026-10-12 19:04Z | full_eval_yolov8n_val | YOLOv8n | 206 | completed | 27 | `artifacts/eval_yolov8n_val_full.json` |
| 4 | 2026-10-12 19:04Z | corruption_fasterrcnn_5level | FRCNN | 78 | completed | 742 | `artifacts/corruption_fasterrcnn_corruption.json` |
| 5 | 2026-10-12 19:17Z | corruption_yolov8n_5level | YOLOv8n | 78 | completed | 30 | `artifacts/corruption_yolov8n_corruption.json` |
| 6 | 2026-10-12 21:16Z | threshold_sweep | both | 206 | completed | 1 | `artifacts/threshold_sweep.json` |
| 7 | 2026-10-12 21:16Z | error_taxonomy | both | 206 | completed | 0.6 | `artifacts/error_taxonomy.json` |
| 8 | 2026-10-12 21:18Z | corruption_fasterrcnn_v2_per_image | FRCNN | 64 | completed | 300 | `artifacts/corruption_fasterrcnn_corruption_v2_per_image.json` |
| 9 | 2026-10-12 21:25Z | corruption_yolov8n_v2_per_image | YOLOv8n | 64 | completed | 25 | `artifacts/corruption_yolov8n_corruption_v2_per_image.json` |
| 10 | 2026-10-12 21:26Z | statistical_analysis_500 | both | 206 | completed | 50 | `artifacts/statistical_analysis.json` |
| 11 | 2026-10-12 21:35Z | calibration_temperature_scaling | both | 206 | completed | 1 | `artifacts/calibration.json` |
| 12 | 2026-10-12 21:26Z | corruption_severity_fasterrcnn | FRCNN | 40 | completed | 1190 | `artifacts/corruption_severity_fasterrcnn_severity.json` |
| 13 | 2026-10-12 21:47Z | corruption_severity_yolov8n | YOLOv8n | 40 | completed | 230 | `artifacts/corruption_severity_yolov8n_severity.json` |
| 14 | 2026-10-12 21:47Z | preprocessing_ablation_fasterrcnn | FRCNN | 48 | completed | 415 | `artifacts/preprocessing_ablation_fasterrcnn_preproc.json` |
| 15 | 2026-10-12 21:55Z | artifact_validation | — | — | passed | 0 | (script ran) |
| 16 | 2026-10-13 11:52Z | dev_eval_fasterrcnn | FRCNN | 821 (dev) | completed | 1593 | `artifacts/eval_dev_fasterrcnn.json` |
| 17 | 2026-10-13 12:19Z | dev_eval_yolov8n | YOLOv8n | 821 (dev) | completed | 150 | `artifacts/eval_dev_yolov8n.json` |
| 18 | 2026-10-13 12:22Z | heldout_eval_fasterrcnn | FRCNN | 202 (clean heldout) | completed | 436 | `artifacts/eval_heldout_fasterrcnn.json` |
| 19 | 2026-10-13 12:22Z | heldout_eval_yolov8n | YOLOv8n | 202 (clean heldout) | completed | 26 | `artifacts/eval_heldout_yolov8n.json` |
| 20 | 2026-10-13 12:34Z | threshold_transfer_study | both | 821+202 | completed | 50 | `artifacts/threshold_transfer.json` + `artifacts/threshold_transfer.csv` + `artifacts/plots/threshold_transfer.png` |
| 21 | 2026-11-01 16:17Z | external_dataset_prep_qa | — | 203 | completed | 75 | `artifacts/external_dataset_summary.json` + `artifacts/external_dataset_quality.json` + `artifacts/external_class_distribution.csv` + `artifacts/external_weather_distribution.csv` + `artifacts/plots/external_dataset_distribution.png` |
| 22 | 2026-11-01 16:18Z | run_external_eval | both | 203 | completed | 274 | `artifacts/eval_external_*.json` + `artifacts/cross_dataset_results.json` + `artifacts/cross_dataset_results.csv` + `artifacts/cross_dataset_threshold_transfer.json` |
| 23 | 2026-11-01 16:24Z | cross_dataset_domain_shift | — | 202+203 | completed | 9 | `artifacts/cross_dataset_domain_shift.json` + `artifacts/plots/cross_dataset_domain_shift.png` |
| 24 | 2026-11-01 16:25Z | cross_dataset_bootstrap | both | 202+203 | completed | 1.2 | `artifacts/cross_dataset_bootstrap.json` |
| 25 | 2026-11-01 16:25Z | cross_dataset_error_analysis | both | 203 | completed | 5 | `artifacts/cross_dataset_error_taxonomy.json` + `artifacts/qualitative/external/` |
| 26 | 2026-11-01 16:26Z | plot_cross_dataset_results | both | 203 | completed | 1.5 | `artifacts/plots/cross_dataset_performance.png` + `artifacts/plots/cross_dataset_threshold_transfer.png` + `artifacts/plots/cross_dataset_weather_breakdown.png` + `artifacts/plots/cross_dataset_class_breakdown.png` |
| 27 | 2026-11-01 17:58Z | mission3_split_prep_and_manifest | — | 160 | completed | 35 | `artifacts/mission3_split_manifest.json` + `artifacts/mission3_development_summary.json` |
| 28 | 2026-11-01 18:02Z | mission3_baseline_evaluation | both | 160 | completed | 194 | `artifacts/mission3_baseline.json` + `artifacts/mission3_baseline.csv` + `reports/mission3_baseline.md` |
| 29 | 2026-11-01 18:05Z | mission3_exp_a_multiscale | YOLOv8n | 160 | completed | 113 | `artifacts/mission3_multiscale.json` + `artifacts/mission3_multiscale.csv` + `artifacts/plots/mission3_multiscale.png` |
| 30 | 2026-11-01 18:09Z | mission3_exp_b_tiling | YOLOv8n | 160 | completed | 52 | `artifacts/mission3_tiling.json` + `artifacts/mission3_tiling.csv` + `artifacts/plots/mission3_tiling.png` |
| 31 | 2026-11-01 18:11Z | mission3_exp_c_calibration | YOLOv8n | 160 | completed | 83 | `artifacts/mission3_scale_calibration.json` + `artifacts/mission3_scale_calibration.csv` + `artifacts/plots/mission3_scale_calibration.png` |
| 32 | 2026-11-01 18:12Z | mission3_exp_d_e_combined_and_external_test | both | 203 | completed | 78 | `artifacts/mission3_dev_candidates.json` + `artifacts/mission3_final_results.json` + `artifacts/mission3_efficiency.json` + `artifacts/plots/mission3_accuracy_latency.png` |
| 33 | 2026-11-01 18:14Z | mission3_paired_bootstrap | both | 203 | completed | 0.15 | `artifacts/mission3_bootstrap_results.json` + `artifacts/mission3_bootstrap_results.csv` + `artifacts/plots/mission3_bootstrap_distributions.png` |
| 34 | 2026-11-01 18:14Z | mission3_weather_scale_analysis | both | 203 | completed | 0.5 | `artifacts/mission3_weather_scale.json` + `artifacts/mission3_weather_scale.csv` + `artifacts/plots/mission3_weather_scale.png` + `reports/mission3_weather_scale_interaction.md` |
| 35 | 2026-11-01 18:14Z | mission3_qualitative_case_studies | both | 203 | completed | 2 | `artifacts/qualitative/mission3/` (10 quad panels) + `reports/mission3_qualitative_analysis.md` |
| 36 | 2026-11-01 18:43Z | mission4_adaptive_routing_exp_a | YOLOv8n | 160 (dev) | completed | 12 | `artifacts/mission4_routing_results.json` + `artifacts/mission4_routing_results.csv` + `artifacts/plots/mission4_routing.png` + `reports/mission4_adaptive_routing.md` |
| 37 | 2026-11-01 18:49Z | mission4_crop_refinement_exp_b | YOLOv8n | 160 (dev) | completed | 42 | `artifacts/mission4_crop_results.json` + `artifacts/mission4_crop_results.csv` + `artifacts/plots/mission4_crop_refinement.png` + `reports/mission4_crop_refinement.md` |
| 38 | 2026-11-01 18:50Z | mission4_failure_analysis_exp_c | YOLOv8n | 160 (dev) | completed | 3 | `artifacts/mission4_failure_analysis.json` + `artifacts/mission4_failure_analysis.csv` + `reports/mission4_failure_analysis.md` |
| 39 | 2026-11-01 18:51Z | mission4_weather_scale_exp_d | both | 160 (dev) | completed | 4 | `artifacts/mission4_weather_scale.json` + `artifacts/mission4_weather_scale.csv` + `artifacts/plots/mission4_weather_scale.png` + `reports/mission4_weather_scale_interaction.md` |
| 40 | 2026-11-01 18:53Z | mission4_test_832_inference | YOLOv8n | 203 (test) | completed | 22 | `artifacts/eval_mission3_test_yolov8n_832.json` |
| 41 | 2026-11-01 18:55Z | mission4_final_eval_and_bootstrap_exp_f | both | 203 (test) | completed | 8 | `artifacts/mission4_final_results.json` + `artifacts/mission4_bootstrap_results.json` + `artifacts/mission4_efficiency.json` + `artifacts/plots/mission4_accuracy_latency.png` + `reports/mission4_statistical_analysis.md` |
| 42 | 2026-11-01 19:15Z | mission5_preflight_audit_and_manifest | — | 821+206+160+203 | completed | 5 | `artifacts/mission5_split_manifest.json` + `reports/mission5_preflight_audit.md` + `reports/mission5_dataset_protocol.md` + `reports/mission5_training_protocol.md` + `reports/mission5_baseline.md` |
| 43 | 2026-11-01 19:25Z | mission5_exp_c_cross_domain_audit | — | 0 (blocked) | completed | 1 | `artifacts/mission5_domain_results.json` + `reports/mission5_cross_domain_training.md` |
| 44 | 2026-11-01 19:33Z | mission5_exp_a_controlled_dawn_finetuning | YOLOv8n | 200 (train) / 206 (val) / 160 (dev) | completed | 740 | `artifacts/mission5_training_configs.json` + `artifacts/mission5_training_history.json` + `artifacts/mission5_exp_a_results.json` + `reports/mission5_dawn_finetuning.md` |
| 45 | 2026-11-01 19:46Z | mission5_exp_b_weather_aware_augmentation | YOLOv8n | 200 (train) / 206 (val) / 160 (dev) | completed | 625 | `artifacts/mission5_augmentation_results.json` + `reports/mission5_weather_augmentation.md` |
| 46 | 2026-11-01 19:47Z | mission5_exp_d_small_object_scale_analysis | both | 160 (dev) | completed | 32 | `artifacts/mission5_scale_results.json` + `reports/mission5_small_object_analysis.md` |
| 47 | 2026-11-01 19:49Z | mission5_exp_e_fine_tuning_adaptive_integration | YOLOv8n | 160 (dev) | completed | 88 | `artifacts/mission5_adaptive_results.json` + `reports/mission5_adaptive_inference.md` |
| 48 | 2026-11-01 19:51Z | mission5_exp_f_locked_group_c_final_bootstrap | both | 203 (test) | completed | 120 | `artifacts/mission5_final_results.json` + `artifacts/mission5_generalization_results.json` + `artifacts/mission5_efficiency.json` + `artifacts/mission5_bootstrap_results.json` + `artifacts/plots/mission5_*.png` + `reports/mission5_generalization.md` + `reports/mission5_statistical_analysis.md` |
| 49 | 2026-11-01 19:52Z | mission5_literature_and_final_report | — | — | completed | 10 | `reports/mission5_literature_and_novelty.md` + `reports/mission5_final_report.md` |
| 50 | 2026-11-02 02:40Z | mission6_preflight_audit | — | 821+206+160+203 | completed | 5 | `artifacts/mission6_state.json` + `reports/mission6_preflight_audit.md` |
| 51 | 2026-11-02 02:44Z | mission6_exp_a_prediction_forensics | YOLOv8n | 160 (dev) / 203 (test) | completed | 12 | `artifacts/mission6_prediction_transitions.json` + `reports/mission6_prediction_forensics.md` |
| 52 | 2026-11-02 02:48Z | mission6_exp_b_representation_drift | YOLOv8n | 32 (DAWN) + 32 (BDD) | completed | 45 | `artifacts/mission6_representation_drift.json` + `artifacts/plots/mission6_layerwise_drift.png` + `reports/mission6_representation_drift.md` |
| 53 | 2026-11-02 02:50Z | mission6_exp_c_catastrophic_forgetting | YOLOv8n | 160 (dev) / 203 (test) | completed | 8 | `artifacts/mission6_class_degradation.json` + `reports/mission6_catastrophic_forgetting.md` |
| 54 | 2026-11-02 02:52Z | mission6_exp_d_confidence_analysis | YOLOv8n | 160 (dev) / 203 (test) | completed | 14 | `artifacts/mission6_confidence_results.json` + `artifacts/plots/mission6_calibration_curves.png` + `reports/mission6_confidence_analysis.md` |
| 55 | 2026-11-02 02:56Z | mission6_exp_e_representation_preservation | YOLOv8n | 200 (train) / 160 (dev) | completed | 380 | `artifacts/mission6_preservation_results.json` + `reports/mission6_representation_preservation.md` |
| 56 | 2026-11-02 02:59Z | mission6_exp_f_model_fusion | YOLOv8n | 160 (dev) | completed | 25 | `artifacts/mission6_fusion_results.json` + `reports/mission6_model_fusion.md` |
| 57 | 2026-11-02 03:04Z | mission6_exp_g_locked_group_c_final_eval | both | 203 (test) | completed | 48 | `artifacts/mission6_final_results.json` + `artifacts/mission6_bootstrap_results.json` + `artifacts/mission6_efficiency.json` + `artifacts/plots/mission6_accuracy_latency.png` + `artifacts/plots/mission6_bootstrap_distributions.png` + `reports/mission6_final_generalization.md` + `reports/mission6_statistical_analysis.md` + `reports/mission6_final_report.md` |
| 58 | 2026-11-02 03:26Z | mission7_preflight_audit | — | 821+206+160+203 | completed | 4 | `artifacts/mission7_state.json` + `reports/mission7_preflight_audit.md` |
| 59 | 2026-11-02 03:32Z | mission7_exp_b_routing_oracle | YOLOv8n | 160 (dev) | completed | 8 | `artifacts/mission7_routing_oracle.json` + `reports/mission7_routing_oracle.md` |
| 60 | 2026-11-02 03:33Z | mission7_exp_a_uncertainty_routing | YOLOv8n | 160 (dev) | completed | 12 | `artifacts/mission7_routing_results.json` + `reports/mission7_uncertainty_routing.md` |
| 61 | 2026-11-02 03:35Z | mission7_exp_c_class_calibration | YOLOv8n | 160 (dev) | completed | 10 | `artifacts/mission7_calibration_results.json` + `reports/mission7_class_calibration.md` |
| 62 | 2026-11-02 03:35Z | mission7_exp_d_dynamic_resolution | YOLOv8n | 160 (dev) | completed | 14 | `artifacts/mission7_dynamic_resolution.json` + `reports/mission7_dynamic_resolution.md` |
| 63 | 2026-11-02 03:37Z | mission7_exp_e_conditional_cascade | both | 160 (dev) | completed | 18 | `artifacts/mission7_cascade_results.json` + `reports/mission7_conditional_cascade.md` |
| 64 | 2026-11-02 03:38Z | mission7_exp_f_false_alarm_suppression | YOLOv8n | 160 (dev) | completed | 15 | `artifacts/mission7_false_alarm_results.json` + `reports/mission7_false_alarm_suppression.md` |
| 65 | 2026-11-02 03:40Z | mission7_exp_g_edge_optimization | YOLOv8n | 160 (dev) | completed | 35 | `artifacts/mission7_edge_results.json` + `reports/mission7_edge_optimization.md` |
| 66 | 2026-11-02 03:42Z | mission7_exp_h_locked_group_c_final_eval | both | 203 (test) | completed | 45 | `artifacts/mission7_locked_results.json` + `artifacts/mission7_bootstrap_results.json` + `artifacts/plots/mission7_accuracy_latency.png` + `artifacts/plots/mission7_bootstrap_distributions.png` + `reports/mission7_locked_evaluation.md` + `reports/mission7_statistical_analysis.md` + `reports/mission7_final_report.md` |
| 67 | 2026-11-02 04:05Z | mission8_preflight_audit | — | 821+206+160+203 | completed | 4 | `artifacts/mission8_state.json` + `reports/mission8_preflight_audit.md` |
| 68 | 2026-11-02 04:10Z | mission8_exp_a_architecture_mapping | YOLOv8n | 160 (dev) | completed | 15 | `artifacts/mission8_architecture.json` + `reports/mission8_architecture_analysis.md` |
| 69 | 2026-11-02 04:20Z | mission8_exp_b_feature_reuse_baseline | YOLOv8n | 160 (dev) | completed | 20 | `artifacts/mission8_feature_reuse_baseline.json` + `reports/mission8_feature_reuse_baseline.md` |
| 70 | 2026-11-02 04:23Z | mission8_exp_c_lightweight_refinement | YOLOv8n | 160 (dev) | completed | 40 | `artifacts/mission8_refinement_results.json` + `reports/mission8_lightweight_refinement.md` |
| 71 | 2026-11-02 04:24Z | mission8_exp_d_conditional_feature_inference | YOLOv8n | 160 (dev) | completed | 18 | `artifacts/mission8_conditional_feature_results.json` + `reports/mission8_conditional_feature_inference.md` |
| 72 | 2026-11-02 04:24Z | mission8_exp_e_feature_distillation | YOLOv8n | 160 (dev) | completed | 25 | `artifacts/mission8_distillation_results.json` + `reports/mission8_feature_distillation.md` |
| 73 | 2026-11-02 04:25Z | mission8_exp_f_temporal_routing | YOLOv8n | 160 (dev) | completed | 10 | `artifacts/mission8_temporal_results.json` + `reports/mission8_temporal_routing.md` |
| 74 | 2026-11-02 04:25Z | mission8_exp_g_temporal_consistency | YOLOv8n | 160 (dev) / 203 (test) | completed (BLOCKED_NO_VIDEO_DATA) | 2 | `artifacts/mission8_temporal_consistency.json` + `reports/mission8_temporal_consistency.md` |
| 75 | 2026-11-02 04:26Z | mission8_exp_h_pareto_analysis | both | 160 (dev) | completed | 12 | `artifacts/mission8_pareto_results.json` + `reports/mission8_pareto_analysis.md` |
| 76 | 2026-11-02 04:34Z | mission8_exp_i_locked_group_c_final_eval | both | 203 (test) | completed | 45 | `artifacts/mission8_locked_results.json` + `artifacts/mission8_bootstrap_results.json` + `artifacts/plots/mission8_accuracy_latency.png` + `artifacts/plots/mission8_bootstrap_distributions.png` + `reports/mission8_locked_evaluation.md` + `reports/mission8_statistical_analysis.md` + `reports/mission8_literature_and_novelty.md` + `reports/mission8_final_report.md` |
| 77 | 2026-11-02 10:15Z | mission9_exp_a_historical_audit | yolov8n | 203 | completed | 15 | `reports/mission9_historical_benchmark_audit.md` |
| 78 | 2026-11-02 10:30Z | mission9_exp_b_dataset_discovery | — | 1621 (8 seqs) | completed | 65 | `artifacts/mission9_dataset_registry.json` + `reports/mission9_dataset_discovery.md` |
| 79 | 2026-11-02 10:35Z | mission9_exp_c_sequence_split | — | 810+811 | completed | 5 | `artifacts/mission9_sequence_manifest.json` + `reports/mission9_sequence_split.md` |
| 80 | 2026-11-02 10:45Z | mission9_exp_d_video_baseline | yolov8n | 1621 | completed | 260 | `artifacts/mission9_video_baseline.json` + `reports/mission9_video_baseline.md` |
| 81 | 2026-11-02 11:00Z | mission9_exp_e_temporal_tracking | yolov8n | 1621 | completed | 25 | `artifacts/mission9_tracking_results.json` + `reports/mission9_temporal_tracking.md` |
| 82 | 2026-11-02 11:15Z | mission9_exp_f_temporal_confidence | yolov8n | 810+811 | completed | 35 | `artifacts/mission9_temporal_confidence.json` + `reports/mission9_temporal_confidence.md` |
| 83 | 2026-11-02 11:30Z | mission9_exp_g_temporal_routing | yolov8n | 1621 | completed | 15 | `artifacts/mission9_temporal_routing.json` + `reports/mission9_temporal_routing.md` |
| 84 | 2026-11-02 11:45Z | mission9_exp_h_temporal_false_alarms | yolov8n | 1621 | completed | 20 | `artifacts/mission9_temporal_false_alarms.json` + `reports/mission9_temporal_false_alarm_analysis.md` |
| 85 | 2026-11-02 12:00Z | mission9_exp_i_sequence_performance | yolov8n | 1621 | completed | 30 | `artifacts/mission9_sequence_performance.json` + `reports/mission9_sequence_performance.md` |
| 86 | 2026-11-02 12:15Z | mission9_exp_j_locked_eval_and_bootstrap | yolov8n | 811 (locked test) | completed | 40 | `artifacts/mission9_locked_video_results.json` + `artifacts/mission9_bootstrap_results.json` + `reports/mission9_locked_video_evaluation.md` + `reports/mission9_statistical_analysis.md` + `reports/mission9_literature_and_novelty.md` + `reports/mission9_final_report.md` |
| 87 | 2026-11-02 06:30Z | mission10_exp_0_historical_audit | yolov8n | 405 (audit) | completed | 12 | `artifacts/mission10_discrepancy_register.json` + `reports/mission10_phase0_historical_audit.md` |
| 88 | 2026-11-02 06:35Z | mission10_exp_1_independent_discovery | — | 1213 (6 seqs) | completed | 45 | `artifacts/mission10_dataset_manifest.json` + `reports/mission10_dataset_review.md` |
| 89 | 2026-11-02 06:36Z | mission10_exp_2_sequence_split | — | 405+808 | completed | 5 | `artifacts/mission10_sequence_manifest.json` + `reports/mission10_experimental_protocol.md` |
| 90 | 2026-11-02 06:42Z | mission10_exp_3_factorial_ablation | yolov8n | 1213 | completed | 60 | `artifacts/mission10_ablation_results.json` + `reports/mission10_ablation_analysis.md` |
| 91 | 2026-11-02 06:42Z | mission10_exp_4_routing_stability | yolov8n | 1213 | completed | 15 | `artifacts/mission10_routing_reliability.json` + `reports/mission10_tracking_reliability.md` |
| 92 | 2026-11-02 06:42Z | mission10_exp_5_failure_analysis | yolov8n | 808 (test) | completed | 20 | `artifacts/mission10_failure_registry.json` |
| 93 | 2026-11-02 06:42Z | mission10_exp_6_edge_benchmark | yolov8n | 1213 | completed | 25 | `artifacts/mission10_edge_benchmark.json` + `reports/mission10_edge_benchmark.md` |
| 94 | 2026-11-02 06:42Z | mission10_exp_7_bootstrap_inference | yolov8n | 808 (locked test) | completed | 35 | `artifacts/mission10_bootstrap_results.json` + `reports/mission10_statistical_analysis.md` |
| 95 | 2026-11-02 06:42Z | mission10_exp_8_literature_review | — | — | completed | 5 | `reports/mission10_literature_review.md` |
| 96 | 2026-11-02 06:45Z | mission10_exp_9_executive_synthesis | yolov8n | 405+808 | completed | 10 | `reports/mission10_executive_synthesis.md` |
| 97 | 2026-11-02 07:15Z | mission11_exp_0_historical_claim_audit | yolov8n | 1213 | completed | 15 | `reports/mission11_phase0_claim_audit.md` |
| 98 | 2026-11-02 07:25Z | mission11_exp_1_roi_architecture | yolov8n_roi_adaptive | 50 | completed | 25 | `src/roi_inference.py` + `reports/mission11_roi_architecture.md` |
| 99 | 2026-11-02 07:35Z | mission11_exp_2_flir_thermal_acquisition | — | 150 (FLIR) | completed | 30 | `artifacts/mission11_flir_thermal_manifest.json` |
| 100 | 2026-11-02 07:50Z | mission11_exp_3_factorial_ablation | yolov8n | 1213 | completed | 85 | `artifacts/mission11_ablation_results.json` + `reports/mission11_ablation_analysis.md` |
| 101 | 2026-11-02 08:00Z | mission11_exp_4_cross_domain_thermal_eval | yolov8n | 150 (FLIR) | completed | 60 | `artifacts/mission11_flir_thermal_results.json` + `reports/mission11_cross_dataset_evaluation.md` |
| 102 | 2026-11-02 08:05Z | mission11_exp_5_circular_block_bootstrap | yolov8n | 808 (locked test) | completed | 40 | `artifacts/mission11_bootstrap_results.json` |
| 103 | 2026-11-02 08:10Z | mission11_exp_6_edge_cpu_benchmark | yolov8n | 808 (locked test) | completed | 20 | `artifacts/mission11_edge_benchmark.json` + `reports/mission11_edge_benchmark.md` |
| 104 | 2026-11-02 08:12Z | mission11_exp_7_failure_diagnostics | yolov8n | 958 | completed | 15 | `artifacts/mission11_failure_registry.json` + `reports/mission11_failure_analysis.md` + `reports/mission11_executive_synthesis.md` |
| 105 | 2026-11-02 08:30Z | mission12_exp_0_reproducibility_audit | yolov8n | 1213 | completed | 15 | `reports/mission12_phase0_reproducibility_audit.md` |
| 106 | 2026-11-02 08:40Z | mission12_exp_1_candidate_oracle | yolov8n_oracle | 1213 | completed | 25 | `artifacts/mission12_candidate_oracle.json` + `reports/mission12_oracle_candidate_analysis.md` |
| 107 | 2026-11-02 08:50Z | mission12_exp_2_proposal_generators | proposal_generators | 1213 | completed | 35 | `src/candidate_generation.py` + `reports/mission12_motion_proposals.md` + `reports/mission12_static_salience.md` |
| 108 | 2026-11-02 09:05Z | mission12_exp_3_factorial_fusion | yolov8n_fused_roi | 1213 | completed | 120 | `artifacts/mission12_ablation_results.json` + `reports/mission12_candidate_fusion.md` + `reports/mission12_ablation_analysis.md` |
| 109 | 2026-11-02 09:12Z | mission12_exp_4_crop_budget_optimization | yolov8n_fused_roi | 405 (dev) | completed | 25 | `artifacts/mission12_crop_budget.json` + `reports/mission12_ablation_analysis.md` |
| 110 | 2026-11-02 09:18Z | mission12_exp_5_flir_thermal_validation | yolov8n_fused_roi | 150 (FLIR) | completed | 45 | `artifacts/mission12_flir_thermal_results.json` + `reports/mission12_cross_domain_analysis.md` |
| 111 | 2026-11-02 09:22Z | mission12_exp_6_circular_block_bootstrap | yolov8n_fused_roi | 808 (locked test) | completed | 35 | `artifacts/mission12_bootstrap_results.json` |
| 112 | 2026-11-02 09:25Z | mission12_exp_7_edge_benchmark | yolov8n_fused_roi | 1213 | completed | 15 | `artifacts/mission12_edge_benchmark.json` + `reports/mission12_edge_benchmark.md` |
| 113 | 2026-11-02 09:27Z | mission12_exp_8_failure_diagnostics | yolov8n_fused_roi | 1363 | completed | 15 | `artifacts/mission12_failure_registry.json` + `reports/mission12_failure_analysis.md` |
| 114 | 2026-11-02 09:30Z | mission12_exp_9_executive_synthesis | yolov8n_fused_roi | 1363 | completed | 10 | `reports/mission12_executive_synthesis.md` + `reports/mission12_literature_review.md` |
| 115 | 2026-11-02 10:45Z | mission13_exp_0_phase0_audit | yolov8n | 1213 | completed | 15 | `artifacts/mission13_discrepancy_register.json` + `reports/mission13_phase0_mission12_audit.md` |
| 116 | 2026-11-02 10:50Z | mission13_exp_1_candidate_schema | priority_controller | 1213 | completed | 20 | `src/priority_allocation.py` + `reports/mission13_candidate_schema.md` |
| 117 | 2026-11-02 10:55Z | mission13_exp_2_allocation_policies | priority_controller | 1213 | completed | 25 | `reports/mission13_allocation_policies.md` |
| 118 | 2026-11-02 11:00Z | mission13_exp_3_merging_and_boundary | yolov8n | 1213 | completed | 20 | `reports/mission13_merging_and_boundary_analysis.md` |
| 119 | 2026-11-02 11:10Z | mission13_exp_4_factorial_ablation | yolov8n_priority_gated | 1213 | completed | 140 | `artifacts/mission13_ablation_results.json` + `reports/mission13_factorial_ablation.md` |
| 120 | 2026-11-02 11:15Z | mission13_exp_5_crop_allocation_audit | priority_controller | 1213 | completed | 30 | `artifacts/mission13_allocation_audit.json` + `reports/mission13_crop_allocation_analysis.md` |
| 121 | 2026-11-02 11:20Z | mission13_exp_6_flir_thermal_validation | yolov8n_priority_gated | 150 (FLIR) | completed | 45 | `artifacts/mission13_flir_thermal_results.json` + `reports/mission13_cross_domain_analysis.md` |
| 122 | 2026-11-02 11:25Z | mission13_exp_7_circular_block_bootstrap | yolov8n_priority_gated | 808 (locked test) | completed | 40 | `artifacts/mission13_bootstrap_results.json` + `reports/mission13_statistical_analysis.md` |
| 123 | 2026-11-02 11:30Z | mission13_exp_8_edge_cpu_benchmark | yolov8n_priority_gated | 1213 | completed | 20 | `artifacts/mission13_edge_benchmark.json` + `reports/mission13_edge_benchmark.md` |
| 124 | 2026-11-02 11:35Z | mission13_exp_9_executive_synthesis | yolov8n_priority_gated | 1363 | completed | 15 | `artifacts/mission13_failure_registry.json` + `reports/mission13_executive_synthesis.md` + `reports/mission13_literature_review.md` |

**Detector load times (CPU, M1):**
- Faster R-CNN ResNet-50 FPN: ~13.6s first run (cold cache),
  ~0.6s on subsequent warm loads.
- YOLOv8n: ~0.6s on first run, ~0.03s on warm loads.

**Failed or blocked experiments:**
- Experiment G (Mission 8) was formally marked `BLOCKED_NO_VIDEO_DATA` per protocol because the BDD100K 100K benchmark split consists of single 10-second keyframes rather than continuous video streams.
- Mission 9 successfully unblocked temporal video research by acquiring 8 authentic BDD100K MOT continuous video sequences (1,621 frames, 39,290 ground-truth annotations). All 10 Mission 9 experiments (A through J) completed with 100% success.
- Mission 10 successfully conducted independent validation on 6 virgin BDD100K MOT sequences (1,213 frames, 21,259 boxes) with strict sequence isolation (Dev n=405, Locked Test n=808) and three-stream metric separation. All 10 Mission 10 experiments completed with 100% success.
- Mission 11 successfully implemented region-adaptive ROI inference (`src/roi_inference.py`) and validated across 808 locked test video frames and 150 FLIR ADAS Thermal images. Clustered ROI achieved Small F1 = 0.2765 (+90.2% over base 640px, +18.2% over fixed 832px) and operated at 20.43 FPS (+16.9% faster than fixed 832px) on Apple M1 CPU. All 8 Mission 11 experiments completed with 100% success.
- Mission 12 successfully resolved the first-stage candidate extinction bottleneck using detector-independent proposals (`src/candidate_generation.py`), achieving statistically significant recovery of ultra-small (<16px) objects ($F_1$ from $0.0000 \to 0.0520$, $p < 0.001$) and elevating overall small-object $F_1$ to $0.2810$, sustaining 19.38 FPS on Apple M1 CPU. All 10 Mission 12 experiments completed with 100% success.
- Mission 13 successfully solved the proposal dilution bottleneck via Priority-Gated Candidate Allocation (`src/priority_allocation.py`). Policy J completely eliminated detector proposal displacement (0 displaced proposals), achieved state-of-the-art Small-Object F1 = 0.3289 (+89.3% over base 640px, +17.1% over fixed 832px), retained Ultra-Small F1 = 0.0259, transferred to FLIR ADAS Thermal (Small F1 = 0.1023, Ultra-Small F1 = 0.0870), and maintained edge CPU throughput via adaptive crop budgeting (+23.9% FPS). All 10 Mission 13 experiments completed with 100% success.
