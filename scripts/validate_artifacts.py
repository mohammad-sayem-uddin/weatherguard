"""Workstream H — artifact validator.

Reads the experiment registry, then checks that every referenced
artifact exists and has a non-zero size. Reports any missing or empty
files. Also verifies that the run scripts' referenced result files
exist when they should.

Run as:
    PYTHONPATH=. .venv/bin/python scripts/validate_artifacts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def _check_file(p: Path) -> tuple[bool, str]:
    if not p.exists():
        return False, "missing"
    if p.stat().st_size == 0:
        return False, "empty"
    return True, "ok"


def _check_entry(entry: dict) -> list[tuple[str, str]]:
    issues = []
    for art in entry.get("result_artifacts", []):
        ok, why = _check_file(Path(art))
        if not ok:
            issues.append((art, why))
    return issues


REQUIRED_ARTIFACTS = [
    "smoke_test_report.json",
    "eval_fasterrcnn_val_full.json",
    "eval_yolov8n_val_full.json",
    "corruption_fasterrcnn_corruption.json",
    "corruption_yolov8n_corruption.json",
    "corruption_fasterrcnn_corruption_v2_per_image.json",
    "corruption_yolov8n_corruption_v2_per_image.json",
    "statistical_analysis.json",
    "threshold_sweep.json",
    "error_taxonomy.json",
    "calibration.json",
    "model_comparison.csv",
    "experiment_registry.jsonl",
    # Mission 2 artifacts
    "external_dataset_summary.json",
    "cross_dataset_results.json",
    "cross_dataset_bootstrap.json",
    "cross_dataset_domain_shift.json",
    "cross_dataset_error_taxonomy.json",
    # Mission 3 artifacts
    "mission3_split_manifest.json",
    "mission3_development_summary.json",
    "mission3_baseline.json",
    "mission3_multiscale.json",
    "mission3_tiling.json",
    "mission3_scale_calibration.json",
    "mission3_final_results.json",
    "mission3_efficiency.json",
    "mission3_bootstrap_results.json",
    "mission3_weather_scale.json",
    # Mission 4 artifacts
    "mission4_protocol.json",
    "mission4_routing_results.json",
    "mission4_crop_results.json",
    "mission4_failure_analysis.json",
    "mission4_weather_scale.json",
    "mission4_final_results.json",
    "mission4_efficiency.json",
    "mission4_bootstrap_results.json",
    "plots/mission4_accuracy_latency.png",
    "plots/mission4_routing.png",
    "plots/mission4_crop_refinement.png",
    "plots/mission4_weather_scale.png",
    # Mission 5 artifacts
    "mission5_split_manifest.json",
    "mission5_training_configs.json",
    "mission5_training_history.json",
    "mission5_exp_a_results.json",
    "mission5_augmentation_results.json",
    "mission5_domain_results.json",
    "mission5_scale_results.json",
    "mission5_adaptive_results.json",
    "mission5_final_results.json",
    "mission5_generalization_results.json",
    "mission5_efficiency.json",
    "mission5_bootstrap_results.json",
    "plots/mission5_accuracy_latency.png",
    "plots/mission5_bootstrap_distributions.png",
    "plots/mission5_scale_breakdown.png",
    "plots/mission5_weather_breakdown.png",
    # Mission 6 artifacts
    "mission6_state.json",
    "mission6_prediction_transitions.json",
    "mission6_representation_drift.json",
    "mission6_class_degradation.json",
    "mission6_confidence_results.json",
    "mission6_preservation_results.json",
    "mission6_fusion_results.json",
    "mission6_final_results.json",
    "mission6_efficiency.json",
    "mission6_bootstrap_results.json",
    "plots/mission6_layerwise_drift.png",
    "plots/mission6_calibration_curves.png",
    "plots/mission6_accuracy_latency.png",
    "plots/mission6_bootstrap_distributions.png",
    # Mission 7 artifacts
    "mission7_state.json",
    "mission7_routing_oracle.json",
    "mission7_routing_results.json",
    "mission7_calibration_results.json",
    "mission7_dynamic_resolution.json",
    "mission7_cascade_results.json",
    "mission7_false_alarm_results.json",
    "mission7_edge_results.json",
    "mission7_locked_results.json",
    "mission7_bootstrap_results.json",
    "plots/mission7_accuracy_latency.png",
    "plots/mission7_bootstrap_distributions.png",
    # Mission 8 artifacts
    "mission8_state.json",
    "mission8_architecture.json",
    "mission8_feature_reuse_baseline.json",
    "mission8_refinement_results.json",
    "mission8_conditional_feature_results.json",
    "mission8_distillation_results.json",
    "mission8_temporal_results.json",
    "mission8_temporal_consistency.json",
    "mission8_pareto_results.json",
    "mission8_locked_results.json",
    "mission8_bootstrap_results.json",
    "plots/mission8_accuracy_latency.png",
    "plots/mission8_bootstrap_distributions.png",
    # Mission 9 artifacts
    "mission9_state.json",
    "mission9_dataset_registry.json",
    "mission9_sequence_manifest.json",
    "mission9_video_baseline.json",
    "mission9_tracking_results.json",
    "mission9_temporal_confidence.json",
    "mission9_temporal_routing.json",
    "mission9_temporal_false_alarms.json",
    "mission9_sequence_performance.json",
    "mission9_locked_video_results.json",
    "mission9_bootstrap_results.json",
    # Mission 10 artifacts
    "mission10_dataset_manifest.json",
    "mission10_sequence_manifest.json",
    "mission10_ablation_results.json",
    "mission10_routing_reliability.json",
    "mission10_failure_registry.json",
    "mission10_edge_benchmark.json",
    "mission10_bootstrap_results.json",
    "mission10_discrepancy_register.json",
    # Mission 11 artifacts
    "mission11_flir_thermal_manifest.json",
    "mission11_flir_thermal_results.json",
    "mission11_ablation_results.json",
    "mission11_bootstrap_results.json",
    "mission11_edge_benchmark.json",
    "mission11_failure_registry.json",
    # Mission 12 artifacts
    "mission12_candidate_oracle.json",
    "mission12_ablation_results.json",
    "mission12_crop_budget.json",
    "mission12_flir_thermal_results.json",
    "mission12_failure_registry.json",
    "mission12_bootstrap_results.json",
    "mission12_edge_benchmark.json",
    # Mission 13 artifacts
    "mission13_discrepancy_register.json",
    "mission13_ablation_results.json",
    "mission13_allocation_audit.json",
    "mission13_flir_thermal_results.json",
    "mission13_bootstrap_results.json",
    "mission13_leave_one_sequence_out.json",
    "mission13_edge_benchmark.json",
    "mission13_failure_registry.json",
]

REQUIRED_REPORTS = [
    "dawn_feasibility_report.md",
    "smoke_test_report.md",
    "research_report.md",
    "final_report.md",
    "statistical_analysis.md",
    "threshold_sweep.md",
    "error_taxonomy.md",
    "calibration.md",
    # Mission 2 reports
    "cross_dataset_protocol.md",
    "cross_dataset_dataset_selection.md",
    "cross_dataset_final_report.md",
    # Mission 3 reports
    "mission3_baseline_audit.md",
    "mission3_research_protocol.md",
    "mission3_baseline.md",
    "mission3_weather_scale_interaction.md",
    "mission3_qualitative_analysis.md",
    "mission3_final_report.md",
    # Mission 4 reports
    "mission4_preflight_audit.md",
    "mission4_research_protocol.md",
    "mission4_baseline.md",
    "mission4_adaptive_routing.md",
    "mission4_crop_refinement.md",
    "mission4_failure_analysis.md",
    "mission4_weather_scale_interaction.md",
    "mission4_statistical_analysis.md",
    "mission4_literature_and_novelty.md",
    "mission4_final_report.md",
    # Mission 5 reports
    "mission5_preflight_audit.md",
    "mission5_dataset_protocol.md",
    "mission5_training_protocol.md",
    "mission5_baseline.md",
    "mission5_dawn_finetuning.md",
    "mission5_weather_augmentation.md",
    "mission5_cross_domain_training.md",
    "mission5_small_object_analysis.md",
    "mission5_adaptive_inference.md",
    "mission5_generalization.md",
    "mission5_statistical_analysis.md",
    "mission5_literature_and_novelty.md",
    "mission5_final_report.md",
    # Mission 6 reports
    "mission6_preflight_audit.md",
    "mission6_prediction_forensics.md",
    "mission6_representation_drift.md",
    "mission6_catastrophic_forgetting.md",
    "mission6_confidence_analysis.md",
    "mission6_representation_preservation.md",
    "mission6_model_fusion.md",
    "mission6_final_generalization.md",
    "mission6_statistical_analysis.md",
    "mission6_literature_and_novelty.md",
    "mission6_final_report.md",
    # Mission 7 reports
    "mission7_preflight_audit.md",
    "mission7_routing_oracle.md",
    "mission7_uncertainty_routing.md",
    "mission7_class_calibration.md",
    "mission7_dynamic_resolution.md",
    "mission7_conditional_cascade.md",
    "mission7_false_alarm_suppression.md",
    "mission7_edge_optimization.md",
    "mission7_locked_evaluation.md",
    "mission7_statistical_analysis.md",
    "mission7_literature_and_novelty.md",
    "mission7_final_report.md",
    # Mission 8 reports
    "mission8_preflight_audit.md",
    "mission8_architecture_analysis.md",
    "mission8_feature_reuse_baseline.md",
    "mission8_lightweight_refinement.md",
    "mission8_conditional_feature_inference.md",
    "mission8_feature_distillation.md",
    "mission8_temporal_routing.md",
    "mission8_temporal_consistency.md",
    "mission8_pareto_analysis.md",
    "mission8_locked_evaluation.md",
    "mission8_statistical_analysis.md",
    "mission8_literature_and_novelty.md",
    "mission8_final_report.md",
    # Mission 9 reports
    "mission9_historical_benchmark_audit.md",
    "mission9_dataset_discovery.md",
    "mission9_sequence_split.md",
    "mission9_video_baseline.md",
    "mission9_temporal_tracking.md",
    "mission9_temporal_confidence.md",
    "mission9_temporal_routing.md",
    "mission9_temporal_false_alarm_analysis.md",
    "mission9_sequence_performance.md",
    "mission9_locked_video_evaluation.md",
    "mission9_statistical_analysis.md",
    "mission9_literature_and_novelty.md",
    "mission9_final_report.md",
    # Mission 10 reports
    "mission10_phase0_historical_audit.md",
    "mission10_dataset_review.md",
    "mission10_experimental_protocol.md",
    "mission10_ablation_analysis.md",
    "mission10_tracking_reliability.md",
    "mission10_edge_benchmark.md",
    "mission10_statistical_analysis.md",
    "mission10_literature_review.md",
    "mission10_executive_synthesis.md",
    # Mission 11 reports
    "mission11_phase0_claim_audit.md",
    "mission11_roi_architecture.md",
    "mission11_ablation_analysis.md",
    "mission11_cross_dataset_evaluation.md",
    "mission11_edge_benchmark.md",
    "mission11_failure_analysis.md",
    "mission11_literature_review.md",
    "mission11_executive_synthesis.md",
    # Mission 12 reports
    "mission12_phase0_reproducibility_audit.md",
    "mission12_oracle_candidate_analysis.md",
    "mission12_motion_proposals.md",
    "mission12_static_salience.md",
    "mission12_candidate_fusion.md",
    "mission12_ablation_analysis.md",
    "mission12_cross_domain_analysis.md",
    "mission12_edge_benchmark.md",
    "mission12_failure_analysis.md",
    "mission12_literature_review.md",
    "mission12_executive_synthesis.md",
    # Mission 13 reports
    "mission13_phase0_mission12_audit.md",
    "mission13_candidate_schema.md",
    "mission13_allocation_policies.md",
    "mission13_merging_and_boundary_analysis.md",
    "mission13_literature_review.md",
    "mission13_edge_benchmark.md",
    "mission13_factorial_ablation.md",
    "mission13_crop_allocation_analysis.md",
    "mission13_cross_domain_analysis.md",
    "mission13_statistical_analysis.md",
    "mission13_executive_synthesis.md",
]


def main() -> int:
    issues: list[tuple[str, str]] = []
    print("=== Required artifacts ===")
    for name in REQUIRED_ARTIFACTS:
        ok, why = _check_file(ARTIFACTS / name)
        flag = "OK" if ok else "MISS"
        print(f"  [{flag}] artifacts/{name} ({why})")
        if not ok:
            issues.append((f"artifacts/{name}", why))

    print("\n=== Required reports ===")
    for name in REQUIRED_REPORTS:
        ok, why = _check_file(REPORTS / name)
        flag = "OK" if ok else "MISS"
        print(f"  [{flag}] reports/{name} ({why})")
        if not ok:
            issues.append((f"reports/{name}", why))

    print("\n=== Experiment registry entries ===")
    jsonl = ARTIFACTS / "experiment_registry.jsonl"
    if not jsonl.exists():
        print("  (no registry yet)")
    else:
        entries = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        print(f"  {len(entries)} entries recorded")
        for e in entries:
            ent_issues = _check_entry(e)
            if ent_issues:
                for art, why in ent_issues:
                    print(f"    [MISS] {e['experiment_id']} -> {art} ({why})")
                    issues.append((art, why))
            else:
                print(f"  [OK] {e['experiment_id']}  ({e['detector']}, {e['status']})")

    # Git hygiene: verify parquet + yolo weights are not tracked
    try:
        import subprocess
        out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT)
        tracked = out.stdout.splitlines()
        for forbidden in ("*.parquet", "yolov8n.pt"):
            hits = [p for p in tracked if Path(p).match(forbidden)]
            if hits:
                issues.append((f"git-tracked:{forbidden}", str(hits)))
        print("\n=== Git hygiene ===")
        for forbidden in ("*.parquet", "yolov8n.pt"):
            hits = [p for p in tracked if Path(p).match(forbidden)]
            if hits:
                print(f"  [BAD] {forbidden} is tracked: {hits}")
            else:
                print(f"  [OK]  {forbidden} not tracked")
    except Exception as exc:
        print(f"  (git not available: {exc})")

    print(f"\n=== Summary ===")
    if issues:
        print(f"  {len(issues)} issue(s):")
        for art, why in issues:
            print(f"    - {art} ({why})")
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
