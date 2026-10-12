"""Generate qualitative failure analysis images (small subset) for the
dashboard and the final report. Writes under artifacts/qualitative/ and
NOT into the git repository (see .gitignore).

Only saves the 12 most informative examples (mix of TP, FP, FN) to keep
storage small.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.dataset import DawnDataset
from src.model import build_detector
from src.inference import run_inference
from src.metrics import iou_xyxy
from src.visualization import render_prediction, pil_to_png_bytes

ROOT = Path("/Users/sayemuddin/Desktop/WeatherGuard")
ARTIFACTS = ROOT / "artifacts"
OUT_DIR = ARTIFACTS / "qualitative"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PATH = ARTIFACTS / "eval_fasterrcnn_val_full.json"
if not EVAL_PATH.exists():
    raise SystemExit(f"Missing {EVAL_PATH}. Run scripts/run_full_eval.py first.")

data = json.loads(EVAL_PATH.read_text())
# Build a quick per-image summary.
detector = build_detector("fasterrcnn_resnet50_fpn", score_threshold=0.0)
detector.load()

# Pick a handful of images with the most informative failure patterns.
records_by_id = {}
ds = DawnDataset.from_parquet(ROOT / "val-00000-of-00001.parquet")
for r in ds:
    records_by_id[r.image_id] = r

# Compute FN count per image
fn_counts = []
for er in data["per_image"]:
    gt_by_class = {}
    for g in er["ground_truth"]:
        gt_by_class.setdefault(g["class_name"], []).append(g["box_xyxy"])
    preds_by_class = {}
    for p in er["predictions"]:
        if p.get("common_label") is None or p["score"] < 0.5:
            continue
        preds_by_class.setdefault(p["common_label"], []).append(p)
    n_fn = 0
    for cn, gts in gt_by_class.items():
        gt_used = [False] * len(gts)
        for p in preds_by_class.get(cn, []):
            best = 0.0
            bi = -1
            for gi, g in enumerate(gts):
                if gt_used[gi]:
                    continue
                v = iou_xyxy(p["box_xyxy"], g)
                if v > best:
                    best, bi = v, gi
            if bi >= 0 and best >= 0.5:
                gt_used[bi] = True
        n_fn += sum(1 for u in gt_used if not u)
    fn_counts.append((er["image_id"], n_fn, er["condition"]))

# Top 6 by FN, then 6 with predictions
top_fn = sorted(fn_counts, key=lambda x: -x[1])[:6]
# Also: pick a few with many predictions
pred_counts = sorted(
    ((er["image_id"], sum(1 for p in er["predictions"] if p.get("common_label") and p["score"] >= 0.5),
      er["condition"]) for er in data["per_image"]),
    key=lambda x: -x[1]
)[:6]
sample = list({x[0] for x in top_fn + pred_counts})[:12]

# Re-run inference for these (the eval JSON has predictions; we can reuse)
preds_by_id = {er["image_id"]: er for er in data["per_image"]}
for image_id in sample:
    rec = records_by_id.get(image_id)
    if rec is None:
        continue
    er = preds_by_id[image_id]
    from src.model import Detection
    from src.dataset import BBox
    dets = [
        Detection(
            box_xyxy=tuple(p["box_xyxy"]),
            score=p["score"],
            label=p["label"],
            common_label=p["common_label"],
        )
        for p in er["predictions"]
    ]
    from src.model import Prediction
    pred = Prediction(image_id=image_id, width=rec.width, height=rec.height, detections=dets, inference_seconds=0.0)
    img = render_prediction(rec, pred, score_threshold=0.5)
    (OUT_DIR / f"{image_id}.png").write_bytes(pil_to_png_bytes(img))
    print(f"  wrote {OUT_DIR / (image_id + '.png')}")
print(f"Done. Wrote {len(sample)} examples under {OUT_DIR}/")
