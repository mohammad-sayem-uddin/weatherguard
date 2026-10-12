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
