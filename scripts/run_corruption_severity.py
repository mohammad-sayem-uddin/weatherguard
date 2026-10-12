"""Workstream D — corruption severity study.

Tests multiple severity levels for each of:
  * blur (radius 1, 2, 3, 4, 6, 8)
  * gaussian noise (sigma 5, 10, 15, 20, 25, 35, 50)
  * brightness (factor 0.9, 0.7, 0.5, 0.3, 0.15)
  * contrast (factor 0.9, 0.7, 0.5, 0.3, 0.15)

Uses the SAME source images (a deterministic stratified sample) for
every severity level. Photometric corruptions preserve ground truth
geometry so we reuse the original annotations.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from src.dataset import DawnDataset
from src.experiments import CorruptionConfig, corrupt_image
from src.inference import run_inference
from src.metrics import evaluate_detections
from src.model import build_detector

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
PLOTS = ARTIFACTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


SEVERITY_GRID: dict[str, list[CorruptionConfig]] = {
    "blur": [
        CorruptionConfig("blur", f"r={r}", {"radius": r})
        for r in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    ],
    "gaussian_noise": [
        CorruptionConfig("gaussian_noise", f"sigma={s}", {"sigma": float(s)})
        for s in [5, 10, 15, 20, 25, 35, 50]
    ],
    "brightness": [
        CorruptionConfig("brightness_reduce", f"factor={f}", {"factor": f})
        for f in [0.9, 0.7, 0.5, 0.3, 0.15]
    ],
    "contrast": [
        CorruptionConfig("contrast_reduce", f"factor={f}", {"factor": f})
        for f in [0.9, 0.7, 0.5, 0.3, 0.15]
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--limit", type=int, default=5, help="images per condition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="fasterrcnn_severity")
    args = parser.parse_args()

    ds = DawnDataset.from_parquet(ROOT / "val-00000-of-00001.parquet")
    sample = ds.stratified_condition_sample(per_condition=args.limit, seed=args.seed)
    print(f"Stratified sample: {len(sample)} images across {len({s.condition for s in sample})} conditions")

    detector = build_detector(args.detector, score_threshold=0.0)
    print(f"Loading detector: {detector.name}")
    detector.load()
    print(f"  loaded in {detector.load_seconds:.2f}s on {detector.device}")

    # Cache the corrupted PIL images per source record.
    print("Pre-computing corrupted images...")
    corrupted_pils: dict[tuple[str, str], object] = {}
    for r in sample:
        img = r.to_pil()
        for family, cfgs in SEVERITY_GRID.items():
            for cfg in cfgs:
                key = (r.image_id, cfg.name + "|" + json.dumps(cfg.params, sort_keys=True))
                corrupted_pils[key] = corrupt_image(img, cfg, seed=args.seed)

    rows: list[dict] = []
    summary: dict = {"detector": args.detector, "n_source_images": len(sample),
                     "per_corruption_family": {}}

    for family, cfgs in SEVERITY_GRID.items():
        family_results: list[dict] = []
        for cfg in cfgs:
            t0 = time.perf_counter()
            # Reconstruct the records for this corruption
            from src.dataset import DawnRecord
            import io as _io
            corrupted_records: list[DawnRecord] = []
            for r in sample:
                key = (r.image_id, cfg.name + "|" + json.dumps(cfg.params, sort_keys=True))
                img = corrupted_pils[key]
                buf = _io.BytesIO()
                img.save(buf, format="PNG")
                corrupted_records.append(
                    DawnRecord(
                        image_id=r.image_id,
                        image_bytes=buf.getvalue(),
                        width=img.width,
                        height=img.height,
                        objects=r.objects,
                    )
                )
            run = run_inference(detector, corrupted_records, score_threshold=0.0, warmup=False)
            from src.inference import predictions_to_eval_records
            eval_recs = predictions_to_eval_records(corrupted_records, run.predictions)
            res = evaluate_detections(eval_recs, iou_threshold=0.5, score_threshold=0.5)
            elapsed = time.perf_counter() - t0
            entry = {
                "family": family,
                "name": cfg.name,
                "description": cfg.description,
                "params": cfg.params,
                "n_images": res.n_images,
                "mAP_50": res.macro["mAP_50"],
                "micro_precision": res.micro["precision"],
                "micro_recall": res.micro["recall"],
                "micro_f1": res.micro["f1"],
                "micro_tp": res.micro["tp"],
                "micro_fp": res.micro["fp"],
                "micro_fn": res.micro["fn"],
                "inference_total_seconds": run.total_inference_seconds,
                "inference_mean_seconds": run.mean_inference_seconds,
                "inference_median_seconds": run.median_inference_seconds,
                "wall_seconds": elapsed,
            }
            family_results.append(entry)
            rows.append(entry)
            print(f"  {cfg.name:<22s} {cfg.description:<22s} "
                  f"mAP@.5={res.macro['mAP_50']:.3f} F1={res.micro['f1']:.3f} ({elapsed:.1f}s)")
        summary["per_corruption_family"][family] = family_results

    ARTIFACTS.mkdir(exist_ok=True)
    out_json = ARTIFACTS / f"corruption_severity_{args.tag}.json"
    out_json.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved {out_json}")

    out_csv = ARTIFACTS / f"corruption_severity_{args.tag}.csv"
    with out_csv.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Saved {out_csv}")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for family, entries in summary["per_corruption_family"].items():
            if not entries:
                continue
            param_keys = sorted({next(iter(e["params"].values())) for e in entries})
            xs = [next(iter(e["params"].values())) for e in entries]
            maps = [e["mAP_50"] for e in entries]
            f1s = [e["micro_f1"] for e in entries]
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(xs, maps, marker="o")
            ax[0].set_xlabel(f"{family} parameter")
            ax[0].set_ylabel("mAP@.5")
            ax[0].set_title(f"mAP@.5 vs {family} severity")
            ax[0].grid(True, alpha=0.3)
            ax[1].plot(xs, f1s, marker="o", color="tab:orange")
            ax[1].set_xlabel(f"{family} parameter")
            ax[1].set_ylabel("Micro F1")
            ax[1].set_title(f"Micro F1 vs {family} severity")
            ax[1].grid(True, alpha=0.3)
            fig.tight_layout()
            out = PLOTS / f"severity_{family}_{args.tag}.png"
            fig.savefig(out, dpi=110)
            plt.close(fig)
            print(f"  plot {out}")
    except Exception as exc:
        print(f"  (matplotlib not available or failed: {exc})")

    md = ["# Corruption severity curves\n",
          f"Detector: `{args.detector}`, "
          f"{len(sample)} stratified source images (seed={args.seed}).\n",
          "## Per-corruption-family results\n"]
    for family, entries in summary["per_corruption_family"].items():
        md.append(f"\n### {family}\n")
        md.append("| params | mAP@.5 | P | R | F1 | TP | FP | FN | wall s |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for e in entries:
            params_repr = json.dumps(e["params"])
            md.append(
                f"| {params_repr} | {e['mAP_50']:.3f} | {e['micro_precision']:.3f} | "
                f"{e['micro_recall']:.3f} | {e['micro_f1']:.3f} | {e['micro_tp']} | "
                f"{e['micro_fp']} | {e['micro_fn']} | {e['wall_seconds']:.1f} |"
            )
    md.append(
        "\n> Photometric corruptions are not equivalent to real adverse "
        "weather. These are controlled severity curves, not predictions of "
        "real-world weather performance.\n"
    )
    (REPORTS / f"corruption_severity_{args.tag}.md").write_text("\n".join(md))
    print(f"Saved {REPORTS}/corruption_severity_{args.tag}.md")


if __name__ == "__main__":
    main()
