"""WeatherGuard — Streamlit dashboard.

Pages:
  1. Overview
  2. Dataset & condition statistics
  3. Model comparison
  4. Weather-condition analysis
  5. Controlled corruption robustness
  6. Failure analysis gallery
  7. Single-image inference
  8. Configuration & reproducibility

Run with:
    PYTHONPATH=. .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.dataset import DAWN_CLASSES, DawnDataset, DawnRecord  # noqa: E402
from src.model import build_detector  # noqa: E402
from src.visualization import draw_boxes, render_prediction  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
VAL_PARQUET = ROOT / "val-00000-of-00001.parquet"
TRAIN_PARQUET = ROOT / "train-00000-of-00001.parquet"

st.set_page_config(
    page_title="WeatherGuard",
    page_icon="WG",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # surface in UI, not crash
        st.error(f"Failed to load {path.name}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def cached_load_dataset(path_str: str) -> dict:
    """Load DAWN dataset once and cache the summary."""
    path = Path(path_str)
    if not path.exists():
        return {"error": f"Parquet not found: {path}"}
    ds = DawnDataset.from_parquet(path)
    return {
        "n_images": len(ds),
        "summary": ds.summary(),
        "records": [
            {
                "image_id": r.image_id,
                "condition": r.condition,
                "width": r.width,
                "height": r.height,
                "n_objects": len(r.objects),
                "image_bytes": r.image_bytes,
            }
            for r in ds
        ],
    }


def _image_from_bytes(b: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(b)).convert("RGB")


def main() -> None:
    st.sidebar.title("WeatherGuard")
    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Overview",
            "2. Dataset statistics",
            "3. Model comparison",
            "4. Weather-condition analysis",
            "5. Corruption robustness",
            "6. Failure analysis",
            "7. Single-image inference",
            "8. Configuration & reproducibility",
        ],
    )

    if page.startswith("1."):
        page_overview()
    elif page.startswith("2."):
        page_dataset()
    elif page.startswith("3."):
        page_model_comparison()
    elif page.startswith("4."):
        page_weather()
    elif page.startswith("5."):
        page_corruption()
    elif page.startswith("6."):
        page_failures()
    elif page.startswith("7."):
        page_inference()
    elif page.startswith("8."):
        page_config()


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------

def page_overview() -> None:
    st.header("Overview")
    st.markdown(
        """
**WeatherGuard** is a research-oriented computer-vision toolkit that
evaluates pretrained object detectors on real adverse-weather traffic
imagery. It runs the existing **DAWN** dataset through openly licensed
pretrained detectors (Torchvision Faster R-CNN, Ultralytics YOLOv8n) and
reports standard detection metrics, per-condition breakdowns, and a
controlled corruption robustness benchmark.

> The dashboard displays **only measured results**. When an experiment
> has not been run yet, the relevant section says so and points at the
> exact command to run.
"""
    )
    st.subheader("Quick facts")
    st.markdown(
        f"""
- Project workspace: `{ROOT}`
- DAWN val split: `{VAL_PARQUET.name}` ({VAL_PARQUET.stat().st_size//1024//1024} MB)
- DAWN train split: `{TRAIN_PARQUET.name}` ({TRAIN_PARQUET.stat().st_size//1024//1024} MB)
- License of dataset: **CC BY-NC 3.0** (non-commercial research only).
- Python: `{sys.version.split()[0]}`
- Platform: `{sys.platform}`
"""
    )
    col1, col2, col3 = st.columns(3)
    if (ARTIFACTS / "eval_fasterrcnn_val_full.json").exists():
        data = _load_json(ARTIFACTS / "eval_fasterrcnn_val_full.json")
        col1.metric("Faster R-CNN mAP@.5 (val)", f"{data['eval_iou0.5_score0.5']['macro']['mAP_50']:.3f}")
        col1.metric("Faster R-CNN mean latency (ms)", f"{data['inference_mean_seconds']*1000:.0f}")
    if (ARTIFACTS / "eval_yolov8n_val_full.json").exists():
        data = _load_json(ARTIFACTS / "eval_yolov8n_val_full.json")
        col2.metric("YOLOv8n mAP@.5 (val)", f"{data['eval_iou0.5_score0.5']['macro']['mAP_50']:.3f}")
        col2.metric("YOLOv8n mean latency (ms)", f"{data['inference_mean_seconds']*1000:.0f}")
    if (ARTIFACTS / "corruption_fasterrcnn_corruption.json").exists():
        data = _load_json(ARTIFACTS / "corruption_fasterrcnn_corruption.json")
        if "identity" in data:
            base = data["identity"]
            col3.metric("Baseline mAP@.5", f"{base['macro']['mAP_50']:.3f}")


def page_dataset() -> None:
    st.header("Dataset statistics")
    ds_data = cached_load_dataset(str(VAL_PARQUET))
    if "error" in ds_data:
        st.error(ds_data["error"])
        return
    s = ds_data["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images (val)", s["n_images"])
    c2.metric("Object instances", s["n_objects"])
    c3.metric("Mean objects/image", f"{s['mean_objects_per_image']:.2f}")
    c4.metric("Conditions", len(s["per_condition_n_images"]))

    st.subheader("Per-condition counts (val)")
    cond_df = pd.DataFrame(
        {
            "n_images": s["per_condition_n_images"],
            "n_objects": s["per_condition_n_objects"],
        }
    ).sort_values("n_images", ascending=False)
    st.dataframe(cond_df, use_container_width=True)
    st.bar_chart(cond_df["n_images"])

    st.subheader("Per-class counts (val)")
    cls_df = pd.DataFrame(
        {"n_objects": s["per_class_n_objects"]}
    ).sort_values("n_objects", ascending=False)
    st.dataframe(cls_df, use_container_width=True)
    st.bar_chart(cls_df)


def _display_eval_block(data: dict, label: str) -> None:
    st.subheader(label)
    e = data["eval_iou0.5_score0.5"]
    cls_df = pd.DataFrame(e["per_class"])
    st.markdown(
        f"**mAP@.5 (score>=0.5):** {e['macro']['mAP_50']:.3f} &nbsp;&nbsp; "
        f"**Micro** P={e['micro']['precision']:.3f} R={e['micro']['recall']:.3f} F1={e['micro']['f1']:.3f}"
    )
    st.dataframe(cls_df, use_container_width=True)
    st.bar_chart(cls_df.set_index("class_name")[["precision", "recall", "f1"]])

    if "inference_mean_seconds" in data:
        st.markdown(
            f"**Inference:** mean {data['inference_mean_seconds']*1000:.0f} ms, "
            f"median {data['inference_median_seconds']*1000:.0f} ms, "
            f"total {data['inference_total_seconds']:.1f} s on {data.get('device','cpu')}."
        )


def page_model_comparison() -> None:
    st.header("Model comparison")
    rows: list[dict] = []
    for tag, label in [
        ("fasterrcnn_val_full", "Faster R-CNN ResNet-50 FPN (Torchvision)"),
        ("yolov8n_val_full", "YOLOv8n (Ultralytics)"),
    ]:
        path = ARTIFACTS / f"eval_{tag}.json"
        if not path.exists():
            st.info(f"Result file not found: {path.name}. Run `scripts/run_full_eval.py --tag {tag} --detector <name>` to generate it.")
            continue
        d = _load_json(path)
        e = d["eval_iou0.5_score0.5"]
        rows.append(
            {
                "Model": label,
                "n_images": d["n_images"],
                "mAP@.5": e["macro"]["mAP_50"],
                "Micro P": e["micro"]["precision"],
                "Micro R": e["micro"]["recall"],
                "Micro F1": e["micro"]["f1"],
                "Mean ms/img": d["inference_mean_seconds"] * 1000,
                "Median ms/img": d["inference_median_seconds"] * 1000,
                "Load s": d["detector_load_seconds"],
            }
        )
        with st.expander(f"Per-class for {label}", expanded=False):
            _display_eval_block(d, label)
    if rows:
        st.subheader("Summary")
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
        # Export
        st.download_button(
            "Download comparison (CSV)",
            pd.DataFrame(rows).to_csv(index=False),
            "model_comparison.csv",
            "text/csv",
        )


def page_weather() -> None:
    st.header("Weather-condition analysis")
    # Use whichever detector is present
    for tag, label in [
        ("fasterrcnn_val_full", "Faster R-CNN"),
        ("yolov8n_val_full", "YOLOv8n"),
    ]:
        path = ARTIFACTS / f"eval_{tag}.json"
        if not path.exists():
            continue
        d = _load_json(path)
        st.subheader(label)
        per_cond = d.get("per_condition_iou0.5_score0.5", {})
        rows = []
        for cond, ed in per_cond.items():
            rows.append(
                {
                    "condition": cond,
                    "n_images": ed["n_images"],
                    "mAP@.5": ed["macro"]["mAP_50"],
                    "P": ed["micro"]["precision"],
                    "R": ed["micro"]["recall"],
                    "F1": ed["micro"]["f1"],
                }
            )
        df = pd.DataFrame(rows).sort_values("n_images", ascending=False)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("condition")["mAP@.5"])
        st.caption(
            "Small-n conditions (<15 images) are exploratory; confidence intervals "
            "are not reported because no defensible resampling unit is available."
        )
        st.download_button(
            f"Download {label} per-condition CSV",
            df.to_csv(index=False),
            f"per_condition_{tag}.csv",
            "text/csv",
            key=f"dl_cond_{tag}",
        )
        break  # show one detector to avoid clutter; user can switch via the cached result


def page_corruption() -> None:
    st.header("Controlled corruption robustness")
    for tag in ["fasterrcnn_corruption", "yolov8n_corruption"]:
        path = ARTIFACTS / f"corruption_{tag}.json"
        if not path.exists():
            st.info(
                f"Result file not found: {path.name}. "
                f"Run `scripts/run_corruption_experiment.py --tag {tag} --detector <name>`."
            )
            continue
        d = _load_json(path)
        st.subheader(tag)
        rows = []
        for name, ed in d.items():
            rows.append(
                {
                    "corruption": name,
                    "description": ed["description"],
                    "params": json.dumps(ed.get("params", {})),
                    "mAP@.5": ed["macro"]["mAP_50"],
                    "Micro F1": ed["micro"]["f1"],
                    "Mean ms/img": ed["inference_mean_seconds"] * 1000,
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("corruption")["mAP@.5"])
        st.download_button(
            f"Download {tag} CSV",
            df.to_csv(index=False),
            f"corruption_{tag}.csv",
            "text/csv",
            key=f"dl_corr_{tag}",
        )
    st.warning(
        "Photometric corruptions are *not* equivalent to real adverse weather. "
        "Use this as a controlled robustness benchmark, not as a claim of "
        "real-world weather robustness."
    )


def page_failures() -> None:
    st.header("Failure analysis")
    st.caption("Showing the highest-confidence false positives from the val split.")

    path = ARTIFACTS / "eval_fasterrcnn_val_full.json"
    if not path.exists():
        st.info("Run `scripts/run_full_eval.py --tag fasterrcnn_val_full` first.")
        return
    d = _load_json(path)
    rows: list[dict] = []
    for er in d["per_image"]:
        gt_classes = {g["class_name"] for g in er["ground_truth"]}
        for p in er["predictions"]:
            if p["common_label"] is None or p["score"] < 0.5:
                continue
            best_iou = 0.0
            for g in er["ground_truth"]:
                if g["class_name"] != p["common_label"]:
                    continue
                ax1, ay1, ax2, ay2 = p["box_xyxy"]
                bx1, by1, bx2, by2 = g["box_xyxy"]
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                b = max(0, bx2 - bx1) * max(0, by2 - by1)
                union = a + b - inter
                if union > 0:
                    best_iou = max(best_iou, inter / union)
            is_tp = best_iou >= 0.5
            is_fn_class_present = not is_tp and p["common_label"] in gt_classes
            tag = "TP" if is_tp else "FP"
            rows.append(
                {
                    "image_id": er["image_id"],
                    "condition": er["condition"],
                    "class": p["common_label"],
                    "score": p["score"],
                    "best_iou": best_iou,
                    "tag": tag,
                }
            )
    if not rows:
        st.info("No detections above score=0.5 found.")
        return
    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(50)
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download top-50 predictions (CSV)",
        df.to_csv(index=False),
        "top_predictions.csv",
        "text/csv",
    )

    # Confidence distribution
    st.subheader("Confidence distribution by correctness")
    df["correct"] = df["tag"] == "TP"
    chart_df = df.groupby(["class", "correct"]).size().unstack(fill_value=0)
    st.bar_chart(chart_df)


def page_inference() -> None:
    st.header("Single-image inference")
    st.caption("Run the selected detector on a user-uploaded image or a bundled DAWN val example.")

    detector_name = st.selectbox(
        "Detector",
        ["fasterrcnn_resnet50_fpn", "yolov8n"],
        help="Both are pretrained on COCO 2017. See the Configuration page for licensing.",
    )
    score_th = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)
    max_det = st.slider("Max detections to draw", 1, 50, 10)

    src = st.radio("Source", ["Bundled DAWN val example", "Upload your own"])
    image: Any = None
    image_id = "uploaded"
    if src == "Bundled DAWN val example":
        ds_data = cached_load_dataset(str(VAL_PARQUET))
        if "error" in ds_data:
            st.error(ds_data["error"])
            return
        ids = [r["image_id"] for r in ds_data["records"]]
        choice = st.selectbox("Pick a DAWN val image", ids[:80])
        rec = next(r for r in ds_data["records"] if r["image_id"] == choice)
        image = _image_from_bytes(rec["image_bytes"])
        image_id = choice
    else:
        uploaded = st.file_uploader("JPEG / PNG image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")

    if image is None:
        return

    if st.button("Run inference"):
        with st.spinner("Loading detector and running inference..."):
            det = build_detector(detector_name, score_threshold=0.0)
            det.load()
            pred = det.predict(image, image_id=image_id)
        st.success(f"Inference: {pred.inference_seconds*1000:.0f} ms, {len(pred.detections)} raw detections")
        top = sorted(pred.detections, key=lambda d: -d.score)[:max_det]
        boxes = [d.box_xyxy for d in top if d.score >= score_th]
        labels = [
            f"{d.common_label or d.label}:{d.score:.2f}"
            for d in top
            if d.score >= score_th
        ]
        rendered = draw_boxes(image, boxes, labels, color=(220, 30, 30))
        st.image(rendered, caption=f"{len(boxes)} detections above score {score_th}", use_container_width=True)
        # Export detections
        st.download_button(
            "Download detections (JSON)",
            json.dumps([d.to_dict() for d in pred.detections], indent=2),
            f"{image_id}_detections.json",
            "application/json",
        )


def page_config() -> None:
    st.header("Configuration & reproducibility")
    st.markdown(
        """
### Models
- **Faster R-CNN ResNet-50 FPN** — Torchvision, BSD-3-Clause code,
  COCO-pretrained weights per the Torchvision model registry.
- **YOLOv8n** — Ultralytics, AGPL-3.0 (commercial use requires a
  separate Ultralytics licence).

### Dataset
- DAWN: <https://huggingface.co/datasets/Maxim37/dawn-dataset>.
- License: **CC BY-NC 3.0** (non-commercial research use only).
- DAWN class space (6 classes): `Truck`, `Person`, `Bicycle`, `Car`,
  `Motorcycle`, `Bus`. We map detector COCO labels into this space via
  `COCO_TO_DAWN` in `src/dataset.py`. Out-of-vocabulary predictions are
  ignored in evaluation.

### Evaluation protocol
- IoU threshold for matching predictions to ground truth: **0.5**
  (`DEFAULT_IOU_THRESHOLD` in `src/metrics.py`).
- Confidence threshold for TP/FP counting: **0.5**
  (`DEFAULT_SCORE_THRESHOLD`).
- Per-class AP: 101-point interpolation, with no GT → NaN (excluded
  from the mAP average).
- Micro metrics: pool all TP/FP/FN across classes.
- mAP@.5: macro-average of per-class AP@.5 over classes that have ≥1 GT.
- All measurements are deterministic for a fixed seed and a fixed
  detector.

### How to run the experiments
```
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
PYTHONPATH=. .venv/bin/python scripts/run_full_eval.py --tag fasterrcnn_val_full
PYTHONPATH=. .venv/bin/python scripts/run_full_eval.py --tag yolov8n_val_full --detector yolov8n
PYTHONPATH=. .venv/bin/python scripts/run_corruption_experiment.py --tag fasterrcnn_corruption
PYTHONPATH=. .venv/bin/python scripts/run_corruption_experiment.py --tag yolov8n_corruption --detector yolov8n
```

### How to launch the dashboard
```
PYTHONPATH=. .venv/bin/streamlit run app.py
```
"""
    )


if __name__ == "__main__":
    main()
