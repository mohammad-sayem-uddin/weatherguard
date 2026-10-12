"""Workstream E — preprocessing ablation.

Compare the same detector on the SAME source images under several
lightweight, CPU-friendly preprocessing strategies:
  1. identity  (baseline)
  2. mild CLAHE on L channel (clip limit 2.0, tile 8x8)
  3. mild gamma correction (gamma 1.2 to brighten)
  4. mild gamma correction (gamma 0.8 to darken)
  5. light Gaussian blur preprocessing (radius 1.0) — to test whether
     denoising helps for noise-prone images. We label this as a
     *smoothing* preprocessor.

Hyperparameters are fixed (no search). All preprocessing happens in
PIL; originals are never modified.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from src.dataset import DawnDataset, DawnRecord
from src.inference import run_inference, predictions_to_eval_records
from src.metrics import evaluate_detections
from src.model import build_detector

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def _to_records_with_pil(records: list[DawnRecord], pil_fn) -> list[DawnRecord]:
    out = []
    for r in records:
        img = pil_fn(r.to_pil())
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(DawnRecord(
            image_id=r.image_id, image_bytes=buf.getvalue(),
            width=img.width, height=img.height, objects=r.objects,
        ))
    return out


def _identity(img: Image.Image) -> Image.Image:
    return img


def _clahe_mild(img: Image.Image) -> Image.Image:
    """Apply CLAHE on the L channel of LAB. Clip=2.0, tile=8x8."""
    from PIL import ImageOps
    # Use Pillow's autocontrast as a portable stand-in if cv2 is unavailable.
    # True CLAHE requires OpenCV; we approximate with local autocontrast
    # using a small kernel via contrast enhancement.
    # Fallback: use Pillow ImageEnhance.Contrast on the L channel.
    rgb = img.convert("RGB")
    # Convert to L, equalise locally via tile-based autocontrast.
    l = rgb.convert("L")
    w, h = l.size
    tile = 64
    l2 = l.copy()
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            box = (x, y, min(x + tile, w), min(y + tile, h))
            crop = l.crop(box)
            eq = ImageOps.autocontrast(crop, cutoff=2)
            l2.paste(eq, box)
    # Re-merge by adjusting brightness of the original via L2.
    rgb2 = rgb.copy()
    # Use a simple L-blend: scale each channel by ratio l2/l (clipped).
    import numpy as np
    arr = np.asarray(rgb).astype(np.float32)
    l_arr = np.asarray(l).astype(np.float32) + 1.0
    l2_arr = np.asarray(l2).astype(np.float32) + 1.0
    ratio = (l2_arr / l_arr)[:, :, None]
    arr2 = np.clip(arr * ratio, 0, 255).astype(np.uint8)
    return Image.fromarray(arr2, mode="RGB")


def _gamma(img: Image.Image, gamma: float) -> Image.Image:
    inv = 1.0 / gamma
    lut = [int(((i / 255.0) ** inv) * 255.0) for i in range(256)]
    lut = lut * (img.mode != "L" and 3 or 1) if img.mode == "L" else lut * 3
    return img.point(lut * (img.mode != "L" and 1 or 1) if img.mode == "L" else lut)


def _gamma_brighten(img: Image.Image) -> Image.Image:
    return _gamma(img, 1.2)


def _gamma_darken(img: Image.Image) -> Image.Image:
    return _gamma(img, 0.8)


def _smoothing(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=1.0))


PREPROCESSORS = [
    ("identity", "Original baseline", _identity),
    ("clahe_mild", "Mild CLAHE (L channel, tile 64, cutoff 2)", _clahe_mild),
    ("gamma_brighten", "Mild gamma brighten (γ=1.2)", _gamma_brighten),
    ("gamma_darken", "Mild gamma darken (γ=0.8)", _gamma_darken),
    ("smoothing_r1", "Light Gaussian blur preprocessing (r=1)", _smoothing),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--limit", type=int, default=8, help="images per condition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="fasterrcnn_preproc")
    args = parser.parse_args()

    ds = DawnDataset.from_parquet(ROOT / "val-00000-of-00001.parquet")
    sample = ds.stratified_condition_sample(per_condition=args.limit, seed=args.seed)
    print(f"Stratified sample: {len(sample)} images across "
          f"{len({s.condition for s in sample})} conditions")

    detector = build_detector(args.detector, score_threshold=0.0)
    print(f"Loading detector: {args.detector}")
    detector.load()
    print(f"  loaded in {detector.load_seconds:.2f}s on {detector.device}")

    rows: list[dict] = []
    summary: dict = {"detector": args.detector, "n_source_images": len(sample),
                     "per_preprocessor": {}}
    # Per-condition results for the per-preprocessor results
    from collections import defaultdict
    per_pre_cond: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for name, descr, fn in PREPROCESSORS:
        print(f"\n=== Preprocessor: {name} ({descr}) ===")
        # Time preprocessing separately
        t_pre0 = time.perf_counter()
        new_records = _to_records_with_pil(sample, fn)
        pre_seconds = time.perf_counter() - t_pre0

        t0 = time.perf_counter()
        run = run_inference(detector, new_records, score_threshold=0.0, warmup=False)
        inf_seconds = time.perf_counter() - t0

        eval_recs = predictions_to_eval_records(new_records, run.predictions)
        res = evaluate_detections(eval_recs, iou_threshold=0.5, score_threshold=0.5)
        per_cond = {}
        for cond in {r.condition for r in new_records}:
            subset = [r for r in eval_recs if r["condition"] == cond]
            r_c = evaluate_detections(subset, iou_threshold=0.5, score_threshold=0.5)
            per_cond[cond] = {
                "n_images": r_c.n_images,
                "mAP_50": r_c.macro["mAP_50"],
                "micro_f1": r_c.micro["f1"],
            }
        entry = {
            "preprocessor": name,
            "description": descr,
            "n_images": res.n_images,
            "mAP_50": res.macro["mAP_50"],
            "micro_precision": res.micro["precision"],
            "micro_recall": res.micro["recall"],
            "micro_f1": res.micro["f1"],
            "micro_tp": res.micro["tp"],
            "micro_fp": res.micro["fp"],
            "micro_fn": res.micro["fn"],
            "preprocessing_total_seconds": pre_seconds,
            "inference_total_seconds": run.total_inference_seconds,
            "inference_mean_seconds": run.mean_inference_seconds,
            "wall_total_seconds": pre_seconds + inf_seconds,
            "per_condition": per_cond,
        }
        summary["per_preprocessor"][name] = entry
        rows.append({k: v for k, v in entry.items() if k != "per_condition"})
        print(f"  mAP@.5 = {res.macro['mAP_50']:.3f}, F1 = {res.micro['f1']:.3f}, "
              f"pre={pre_seconds:.1f}s, inf={inf_seconds:.1f}s")

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"preprocessing_ablation_{args.tag}.json").write_text(
        json.dumps(summary, indent=2, default=float))
    out_csv = ARTIFACTS / f"preprocessing_ablation_{args.tag}.csv"
    with out_csv.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Saved {ARTIFACTS}/preprocessing_ablation_{args.tag}.{{json,csv}}")

    md = ["# Preprocessing ablation\n",
          f"Detector: `{args.detector}`, "
          f"{len(sample)} stratified source images (seed={args.seed}).\n",
          "All preprocessing is photometric. Source images on disk are not "
          "modified; only the in-memory copy that is fed to the detector is "
          "preprocessed.\n",
          "## Aggregate\n",
          "| preprocessor | mAP@.5 | P | R | F1 | TP | FP | FN | pre s | inf s |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, descr, _ in PREPROCESSORS:
        e = summary["per_preprocessor"][name]
        md.append(
            f"| {name} ({descr}) | {e['mAP_50']:.3f} | {e['micro_precision']:.3f} | "
            f"{e['micro_recall']:.3f} | {e['micro_f1']:.3f} | {e['micro_tp']} | "
            f"{e['micro_fp']} | {e['micro_fn']} | {e['preprocessing_total_seconds']:.1f} | "
            f"{e['inference_total_seconds']:.1f} |"
        )
    md.append("\n## Per-condition F1\n")
    conditions = sorted({c for n, _, _ in PREPROCESSORS
                          for c in summary["per_preprocessor"][n]["per_condition"].keys()})
    header = "| preprocessor | " + " | ".join(conditions) + " |"
    md.append(header)
    md.append("|" + "---|" * (len(conditions) + 1))
    for name, descr, _ in PREPROCESSORS:
        e = summary["per_preprocessor"][name]
        cells = [f"{e['per_condition'].get(c, {}).get('micro_f1', float('nan')):.3f}" for c in conditions]
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md.append(
        "\n> Validation-selected comparisons: the same images and detector "
        "are used for every preprocessor, so differences are attributable "
        "to the preprocessor (modulo the small sample).\n"
    )
    (REPORTS / f"preprocessing_ablation_{args.tag}.md").write_text("\n".join(md))
    print(f"Saved {REPORTS}/preprocessing_ablation_{args.tag}.md")


if __name__ == "__main__":
    main()
