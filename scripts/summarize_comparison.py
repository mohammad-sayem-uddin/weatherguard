"""Build a model-comparison summary CSV/JSON from the two full eval
artifacts. Produces a single artifact the dashboard can read.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"

DETECTORS = [
    ("fasterrcnn_val_full", "Faster R-CNN ResNet-50 FPN (Torchvision)"),
    ("yolov8n_val_full", "YOLOv8n (Ultralytics)"),
]


def main() -> None:
    rows: list[dict] = []
    for tag, label in DETECTORS:
        path = ARTIFACTS / f"eval_{tag}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        e = d["eval_iou0.5_score0.5"]
        row = {
            "model": label,
            "tag": tag,
            "n_images": d["n_images"],
            "mAP_50": round(e["macro"]["mAP_50"], 4),
            "micro_precision": round(e["micro"]["precision"], 4),
            "micro_recall": round(e["micro"]["recall"], 4),
            "micro_f1": round(e["micro"]["f1"], 4),
            "load_seconds": round(d["detector_load_seconds"], 3),
            "inference_total_seconds": round(d["inference_total_seconds"], 2),
            "inference_mean_ms_per_image": round(d["inference_mean_seconds"] * 1000, 1),
            "inference_median_ms_per_image": round(d["inference_median_seconds"] * 1000, 1),
            "device": d.get("device", "cpu"),
            "weights_provenance": d.get("weights_provenance", ""),
            "license_note": d.get("license_note", ""),
        }
        rows.append(row)

    if not rows:
        print("No eval JSON files found; nothing to write.")
        return

    out_csv = ARTIFACTS / "model_comparison.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

    out_json = ARTIFACTS / "model_comparison.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
