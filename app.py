"""WeatherGuard — Streamlit research dashboard (advanced edition).

Pages:
  1. Overview
  2. Dataset statistics
  3. Model comparison
  4. Weather-condition analysis
  5. Statistical reliability
  6. Confidence-threshold sweep
  7. Error taxonomy
  8. Robustness laboratory (severity curves + original corruption)
  9. Preprocessing ablation
  10. Calibration
  11. Failure analysis
  12. Single-image inference
  13. Configuration & reproducibility
  14. Experiment registry & artifact validator

Run with:
    PYTHONPATH=. .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.dataset import DAWN_CLASSES, DawnDataset, DawnRecord  # noqa: E402
from src.model import build_detector  # noqa: E402
from src.visualization import draw_boxes  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
PLOTS = ARTIFACTS / "plots"
VAL_PARQUET = ROOT / "val-00000-of-00001.parquet"
TRAIN_PARQUET = ROOT / "train-00000-of-00001.parquet"

st.set_page_config(
    page_title="WeatherGuard Research Dashboard",
    page_icon="WG",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        st.error(f"Failed to load {path.name}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def cached_load_dataset(path_str: str) -> dict:
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


def _has(name: str) -> bool:
    return (ARTIFACTS / name).exists()


def main() -> None:
    st.sidebar.title("WeatherGuard")
    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Overview",
            "2. Dataset statistics",
            "3. Model comparison",
            "4. Weather-condition analysis",
            "5. Statistical reliability",
            "6. Confidence-threshold sweep",
            "7. Error taxonomy",
            "8. Robustness laboratory",
            "9. Preprocessing ablation",
            "10. Calibration",
            "11. Failure analysis",
            "12. Single-image inference",
            "13. Configuration & reproducibility",
            "14. Experiment registry",
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
        page_stats()
    elif page.startswith("6."):
        page_threshold()
    elif page.startswith("7."):
        page_error_taxonomy()
    elif page.startswith("8."):
        page_robustness()
    elif page.startswith("9."):
        page_preprocessing()
    elif page.startswith("10."):
        page_calibration()
    elif page.startswith("11."):
        page_failures()
    elif page.startswith("12."):
        page_inference()
    elif page.startswith("13."):
        page_config()
    elif page.startswith("14."):
        page_registry()


def page_overview() -> None:
    st.header("Overview")
    st.markdown(
        """
**WeatherGuard** evaluates pretrained object detectors on the public
**DAWN** adverse-weather traffic dataset. This dashboard reads real
result files from `artifacts/`. When a result file is missing, the
relevant section says so and points at the exact command to run it.

> No metric in this dashboard is hardcoded. Every number comes from a
> JSON or CSV in `artifacts/`.
"""
    )
    cols = st.columns(4)
    if _has("eval_fasterrcnn_val_full.json"):
        d = _load_json(ARTIFACTS / "eval_fasterrcnn_val_full.json")
        cols[0].metric("FRCNN mAP@.5", f"{d['eval_iou0.5_score0.5']['macro']['mAP_50']:.3f}")
        cols[0].metric("FRCNN mean ms/img", f"{d['inference_mean_seconds']*1000:.0f}")
    if _has("eval_yolov8n_val_full.json"):
        d = _load_json(ARTIFACTS / "eval_yolov8n_val_full.json")
        cols[1].metric("YOLOv8n mAP@.5", f"{d['eval_iou0.5_score0.5']['macro']['mAP_50']:.3f}")
        cols[1].metric("YOLOv8n mean ms/img", f"{d['inference_mean_seconds']*1000:.0f}")
    if _has("calibration.json"):
        d = _load_json(ARTIFACTS / "calibration.json")
        for tag, info in d["detectors"].items():
            label = info["label"]
            if "FRCNN" in label and len(cols[2].__dict__.get("_metric_objects", [])) < 2:
                cols[2].metric(f"{label[:14]} Brier (pre→post)",
                               f"{info['brier_pre']:.3f}→{info['brier_post']:.3f}")
            elif "YOLO" in label and len(cols[3].__dict__.get("_metric_objects", [])) < 2:
                cols[3].metric(f"{label[:14]} Brier (pre→post)",
                               f"{info['brier_pre']:.3f}→{info['brier_post']:.3f}")
    if _has("corruption_fasterrcnn_corruption.json"):
        d = _load_json(ARTIFACTS / "corruption_fasterrcnn_corruption.json")
        if "blur" in d:
            cols[2].metric("FRCNN mAP@.5 (blur r=4)", f"{d['blur']['macro']['mAP_50']:.3f}")
        if "gaussian_noise" in d:
            cols[3].metric("FRCNN mAP@.5 (noise σ=25)", f"{d['gaussian_noise']['macro']['mAP_50']:.3f}")

    st.subheader("Status of experiment artifacts")
    files = [
        ("smoke_test_report.json", "Smoke test"),
        ("eval_fasterrcnn_val_full.json", "FRCNN full val"),
        ("eval_yolov8n_val_full.json", "YOLOv8n full val"),
        ("corruption_fasterrcnn_corruption.json", "FRCNN corruption (5 levels)"),
        ("corruption_yolov8n_corruption.json", "YOLOv8n corruption (5 levels)"),
        ("corruption_severity_fasterrcnn_severity.json", "FRCNN severity curves"),
        ("corruption_severity_yolov8n_severity.json", "YOLOv8n severity curves"),
        ("statistical_analysis.json", "Statistical analysis"),
        ("threshold_sweep.json", "Threshold sweep"),
        ("error_taxonomy.json", "Error taxonomy"),
        ("preprocessing_ablation_fasterrcnn_preproc.json", "Preprocessing ablation"),
        ("calibration.json", "Calibration"),
        ("experiment_registry.jsonl", "Experiment registry"),
    ]
    rows = []
    for fn, label in files:
        p = ARTIFACTS / fn
        rows.append({"Artifact": fn, "Description": label,
                     "Status": "✅ present" if p.exists() else "❌ missing",
                     "Size (KB)": f"{p.stat().st_size/1024:.1f}" if p.exists() else "—"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


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
        {"n_images": s["per_condition_n_images"],
         "n_objects": s["per_condition_n_objects"]}
    ).sort_values("n_images", ascending=False)
    st.dataframe(cond_df, use_container_width=True)
    st.bar_chart(cond_df["n_images"])
    st.download_button("Download per-condition CSV", cond_df.to_csv(),
                       "per_condition_val.csv", "text/csv")

    st.subheader("Per-class counts (val)")
    cls_df = pd.DataFrame({"n_objects": s["per_class_n_objects"]}
                          ).sort_values("n_objects", ascending=False)
    st.dataframe(cls_df, use_container_width=True)
    st.bar_chart(cls_df)
    st.download_button("Download per-class CSV", cls_df.to_csv(),
                       "per_class_val.csv", "text/csv")


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
            st.info(f"Missing: {path.name}. Run `scripts/run_full_eval.py --tag {tag} --detector <name>`.")
            continue
        d = _load_json(path)
        e = d["eval_iou0.5_score0.5"]
        rows.append({
            "Model": label,
            "n_images": d["n_images"],
            "mAP@.5": e["macro"]["mAP_50"],
            "Micro P": e["micro"]["precision"],
            "Micro R": e["micro"]["recall"],
            "Micro F1": e["micro"]["f1"],
            "Mean ms/img": d["inference_mean_seconds"] * 1000,
            "Median ms/img": d["inference_median_seconds"] * 1000,
            "Load s": d["detector_load_seconds"],
        })
        with st.expander(f"Per-class for {label}", expanded=False):
            _display_eval_block(d, label)
    if rows:
        st.subheader("Summary")
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
        st.download_button("Download model_comparison.csv", pd.DataFrame(rows).to_csv(index=False),
                           "model_comparison.csv", "text/csv")


def page_weather() -> None:
    st.header("Weather-condition analysis")
    for tag, label in [("fasterrcnn_val_full", "Faster R-CNN"),
                       ("yolov8n_val_full", "YOLOv8n")]:
        path = ARTIFACTS / f"eval_{tag}.json"
        if not path.exists():
            st.info(f"Missing: {path.name}.")
            continue
        d = _load_json(path)
        st.subheader(label)
        per_cond = d.get("per_condition_iou0.5_score0.5", {})
        rows = []
        for cond, ed in per_cond.items():
            rows.append({"condition": cond, "n_images": ed["n_images"],
                         "mAP@.5": ed["macro"]["mAP_50"],
                         "P": ed["micro"]["precision"], "R": ed["micro"]["recall"],
                         "F1": ed["micro"]["f1"]})
        df = pd.DataFrame(rows).sort_values("n_images", ascending=False)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("condition")["mAP@.5"])
        st.download_button(f"Download per-condition ({label}) CSV", df.to_csv(index=False),
                           f"per_condition_{tag}.csv", "text/csv", key=f"cond_{tag}")


def page_stats() -> None:
    st.header("Statistical reliability (paired image-level bootstrap)")
    path = ARTIFACTS / "statistical_analysis.json"
    if not path.exists():
        st.info("Missing artifacts/statistical_analysis.json. Run `scripts/run_statistical_analysis.py`.")
        return
    d = _load_json(path)
    cfg = d["config"]
    st.markdown(
        f"**{cfg['replicates']} replicates**, seed {cfg['seed']}, "
        f"IoU={cfg['iou_threshold']}, score={cfg['score_threshold']}, "
        f"runtime {cfg['runtime_seconds']:.1f} s."
    )
    st.subheader("Per-detector bootstrap CIs")
    rows = []
    for tag, info in d["bootstrap"].items():
        for metric, summ in info.items():
            rows.append({"detector": tag, "metric": metric,
                         "mean": summ["mean"], "median": summ["median"],
                         "ci95_lo": summ["ci95"][0], "ci95_hi": summ["ci95"][1]})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download stats.csv", df.to_csv(index=False),
                       "statistical_analysis_filtered.csv", "text/csv")

    st.subheader("Paired difference (Faster R-CNN − YOLOv8n)")
    st.markdown(d["paired_difference"]["metric"])
    pd_rows = []
    for metric, summ in d["paired_difference"].items():
        if metric in ("n_common_images", "metric"):
            continue
        pd_rows.append({"metric": metric, "mean": summ["mean"],
                        "median": summ["median"],
                        "ci95_lo": summ["ci95"][0], "ci95_hi": summ["ci95"][1]})
    st.dataframe(pd.DataFrame(pd_rows), use_container_width=True)
    st.caption(
        "Overlapping CIs are not a formal significance test. The paired "
        "bootstrap provides a non-parametric interval for the "
        "detector-difference distribution."
    )

    if "corruption_bootstrap" in d and d["corruption_bootstrap"]:
        st.subheader("Corruption mAP-drop paired bootstrap CIs")
        for corr_tag, info in d["corruption_bootstrap"].items():
            st.markdown(f"**{corr_tag}**")
            crow = []
            for c, ci in info["per_corruption"].items():
                s = ci["mAP_drop_summary"]
                crow.append({"corruption": c, "n": ci["n_common_images"],
                             "mean_drop": s["mean"], "median_drop": s["median"],
                             "ci95_lo": s["ci95"][0], "ci95_hi": s["ci95"][1]})
            st.dataframe(pd.DataFrame(crow), use_container_width=True)
            st.download_button(f"Download {corr_tag} CSV", pd.DataFrame(crow).to_csv(index=False),
                               f"corruption_bootstrap_{corr_tag}.csv", "text/csv",
                               key=f"dl_cb_{corr_tag}")

    st.subheader("Per-condition bootstrap CIs")
    pc = d["per_condition_ci"]
    rows = []
    for cond, info in pc.items():
        for tag, di in info["by_detector"].items():
            s = di["summary"]
            rows.append({"condition": cond, "detector": tag, "n_val": info["n_val"],
                         "median": s["median"], "ci95_lo": s["ci95"][0],
                         "ci95_hi": s["ci95"][1]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.download_button("Download per-condition CI CSV",
                       pd.DataFrame(rows).to_csv(index=False),
                       "per_condition_ci.csv", "text/csv")
    for w in d.get("warnings", []):
        st.warning(w)


def page_threshold() -> None:
    st.header("Confidence-threshold sweep")
    path = ARTIFACTS / "threshold_sweep.json"
    if not path.exists():
        st.info("Missing artifacts/threshold_sweep.json. Run `scripts/run_threshold_sweep.py`.")
        return
    d = _load_json(path)
    for tag, info in d["detectors"].items():
        st.subheader(info["label"])
        rows = info["rows"]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        # Plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(df["score_threshold"], df["micro_precision"], marker="o", label="P")
            ax[0].plot(df["score_threshold"], df["micro_recall"], marker="o", label="R")
            ax[0].plot(df["score_threshold"], df["micro_f1"], marker="o", label="F1")
            ax[0].set_xlabel("Score threshold")
            ax[0].set_ylabel("Metric")
            ax[0].set_ylim(0, 1)
            ax[0].grid(True, alpha=0.3)
            ax[0].legend()
            ax[0].set_title(f"P/R/F1 vs threshold — {info['label']}")
            ax[1].plot(df["score_threshold"], df["mAP_50"], marker="o", color="tab:purple")
            ax[1].set_xlabel("Score threshold")
            ax[1].set_ylabel("mAP@.5")
            ax[1].grid(True, alpha=0.3)
            ax[1].set_title(f"mAP@.5 vs threshold — {info['label']}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Plot failed: {exc}")
        st.markdown(
            f"**Validation-selected F1 max:** score={info['best_micro_f1']['score_threshold']:.2f} → "
            f"P={info['best_micro_f1']['micro_precision']:.3f} "
            f"R={info['best_micro_f1']['micro_recall']:.3f} "
            f"F1={info['best_micro_f1']['micro_f1']:.3f}"
        )
        st.markdown(
            f"**Validation-selected mAP@.5 max:** score={info['best_mAP_50']['score_threshold']:.2f} → "
            f"mAP@.5={info['best_mAP_50']['mAP_50']:.4f}"
        )
        st.caption(
            "Selecting the best threshold on the validation set introduces "
            "selection bias. These are *validation-selected operating points*, "
            "not unbiased generalisation estimates."
        )
        st.download_button(f"Download {info['label']} sweep CSV",
                           df.to_csv(index=False), f"threshold_sweep_{tag}.csv",
                           "text/csv", key=f"dl_ts_{tag}")


def page_error_taxonomy() -> None:
    st.header("Detection error taxonomy")
    path = ARTIFACTS / "error_taxonomy.json"
    if not path.exists():
        st.info("Missing artifacts/error_taxonomy.json. Run `scripts/run_error_taxonomy.py`.")
        return
    d = _load_json(path)
    for tag, info in d.items():
        st.subheader(info["detector"])
        st.markdown(f"n={info['n_images']} val images, IoU=0.5, score>=0.5.")
        # Per-class predictions
        st.markdown("**Per-class predictions**")
        rows = []
        for cls, counts in info["per_class_pred"].items():
            row = {"class": cls}
            row.update(counts)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("class"), use_container_width=True)
        # Per-class GT
        st.markdown("**Per-class ground truth**")
        rows = []
        for cls, counts in info["per_class_gt"].items():
            row = {"class": cls}
            row.update(counts)
            row["Total"] = sum(counts.values())
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("class"), use_container_width=True)
        # Confusion
        if info["confusion"]:
            st.markdown("**Class confusion (GT class → predicted class, IoU≥0.5)**")
            cf_rows = []
            for k, n in sorted(info["confusion"].items(), key=lambda kv: -kv[1]):
                gt, pred = k.split("__", 1)
                cf_rows.append({"GT": gt, "Predicted": pred, "n": n})
            st.dataframe(pd.DataFrame(cf_rows), use_container_width=True)
        # Per-size
        st.markdown("**Per-size predictions (binned by nearest GT area)**")
        rows = []
        for size, counts in info["per_size_pred"].items():
            row = {"size": size}
            row.update(counts)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("size"), use_container_width=True)
        st.download_button(f"Download {info['detector']} taxonomy CSV",
                           pd.DataFrame(rows).to_csv(index=False),
                           f"error_taxonomy_{tag}.csv", "text/csv",
                           key=f"dl_et_{tag}")


def page_robustness() -> None:
    st.header("Robustness laboratory")
    st.markdown(
        "**Photometric corruptions are not equivalent to real adverse weather.** "
        "These are controlled robustness benchmarks."
    )
    for tag, label in [("fasterrcnn_corruption", "Faster R-CNN (5-level)"),
                       ("yolov8n_corruption", "YOLOv8n (5-level)")]:
        path = ARTIFACTS / f"corruption_{tag}.json"
        if not path.exists():
            st.info(f"Missing: {path.name}.")
            continue
        d = _load_json(path)
        st.subheader(label)
        rows = []
        for cname, ed in d.items():
            rows.append({"corruption": cname, "description": ed["description"],
                         "params": json.dumps(ed.get("params", {})),
                         "mAP@.5": ed["macro"]["mAP_50"],
                         "Micro F1": ed["micro"]["f1"],
                         "Mean ms/img": ed["inference_mean_seconds"]*1000})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.download_button(f"Download {label} CSV", pd.DataFrame(rows).to_csv(index=False),
                           f"corruption_summary_{tag}.csv", "text/csv", key=f"dl_cs_{tag}")
    # Severity
    for tag, label in [("fasterrcnn_severity", "Faster R-CNN (severity curves)"),
                       ("yolov8n_severity", "YOLOv8n (severity curves)")]:
        path = ARTIFACTS / f"corruption_severity_{tag}.json"
        if not path.exists():
            st.info(f"Missing: {path.name}. Run `scripts/run_corruption_severity.py --tag {tag} --detector <name>`.")
            continue
        d = _load_json(path)
        st.subheader(label)
        for family, entries in d["per_corruption_family"].items():
            st.markdown(f"**Family: {family}**")
            rows = []
            for e in entries:
                rows.append({"params": json.dumps(e["params"]),
                             "mAP@.5": e["mAP_50"],
                             "F1": e["micro_f1"],
                             "TP": e["micro_tp"], "FP": e["micro_fp"], "FN": e["micro_fn"],
                             "wall_s": e["wall_seconds"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            plot = PLOTS / f"severity_{family}_{tag}.png"
            if plot.exists():
                st.image(str(plot), caption=f"{family} severity curve")
            st.download_button(f"Download {family} ({label}) CSV",
                               pd.DataFrame(rows).to_csv(index=False),
                               f"severity_{family}_{tag}.csv", "text/csv",
                               key=f"dl_sv_{tag}_{family}")


def page_preprocessing() -> None:
    st.header("Preprocessing ablation")
    for tag, label in [("fasterrcnn_preproc", "Faster R-CNN")]:
        path = ARTIFACTS / f"preprocessing_ablation_{tag}.json"
        if not path.exists():
            st.info(f"Missing: {path.name}. Run `scripts/run_preprocessing_ablation.py --tag {tag}`.")
            continue
        d = _load_json(path)
        st.subheader(label)
        rows = []
        for name, info in d["per_preprocessor"].items():
            rows.append({"preprocessor": name, "description": info["description"],
                         "mAP@.5": info["mAP_50"], "F1": info["micro_f1"],
                         "pre s": info["preprocessing_total_seconds"],
                         "inf s": info["inference_total_seconds"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.download_button(f"Download {label} ablation CSV",
                           pd.DataFrame(rows).to_csv(index=False),
                           f"preprocessing_ablation_{tag}.csv", "text/csv",
                           key=f"dl_pp_{tag}")


def page_calibration() -> None:
    st.header("Calibration")
    path = ARTIFACTS / "calibration.json"
    if not path.exists():
        st.info("Missing artifacts/calibration.json. Run `scripts/run_calibration.py`.")
        return
    d = _load_json(path)
    for tag, info in d["detectors"].items():
        st.subheader(info["label"])
        st.markdown(
            f"- n pairs total: {info['n_pairs']}, fit: {info['n_fit']}, eval: {info['n_eval']}\n"
            f"- Fitted T: **{info['temperature']:.3f}**\n"
            f"- Brier (eval half): pre **{info['brier_pre']:.4f}** → post **{info['brier_post']:.4f}**\n"
            f"- NLL (eval half): pre **{info['nll_pre']:.4f}** → post **{info['nll_post']:.4f}**"
        )
        st.caption(info["caveat"])
        # Plot reliability
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            rel_pre = pd.DataFrame(info["reliability_pre"]).dropna()
            rel_post = pd.DataFrame(info["reliability_post"]).dropna()
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax[0].plot(rel_pre["mean_conf"], rel_pre["frac_correct"], marker="o", label="pre")
            ax[0].plot(rel_post["mean_conf"], rel_post["frac_correct"], marker="o", label="post")
            ax[0].set_xlabel("Mean confidence")
            ax[0].set_ylabel("Empirical accuracy")
            ax[0].set_title(f"Reliability — {info['label']} (pre)")
            ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
            ax[0].grid(True, alpha=0.3)
            ax[0].legend()
            ax[1].plot(rel_pre["bin_lo"], rel_pre["gap"], marker="o", label="pre gap")
            ax[1].plot(rel_post["bin_lo"], rel_post["gap"], marker="o", label="post gap")
            ax[1].axhline(0, color="k", lw=0.5)
            ax[1].set_xlabel("Bin lower edge")
            ax[1].set_ylabel("Confidence − accuracy (gap)")
            ax[1].set_title(f"Calibration gap — {info['label']}")
            ax[1].grid(True, alpha=0.3)
            ax[1].legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Plot failed: {exc}")
        # Per-bin tables
        st.markdown("**Reliability (post-calibration)**")
        st.dataframe(pd.DataFrame(info["reliability_post"]), use_container_width=True)
        st.download_button(f"Download {info['label']} pre reliability CSV",
                           pd.DataFrame(info["reliability_pre"]).to_csv(index=False),
                           f"reliability_{tag}_pre.csv", "text/csv",
                           key=f"dl_relpre_{tag}")
        st.download_button(f"Download {info['label']} post reliability CSV",
                           pd.DataFrame(info["reliability_post"]).to_csv(index=False),
                           f"reliability_{tag}_post.csv", "text/csv",
                           key=f"dl_relpost_{tag}")


def page_failures() -> None:
    st.header("Failure analysis")
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
            tag = "TP" if is_tp else "FP"
            rows.append({"image_id": er["image_id"], "condition": er["condition"],
                         "class": p["common_label"], "score": p["score"],
                         "best_iou": best_iou, "tag": tag})
    if not rows:
        st.info("No detections above score=0.5 found.")
        return
    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(100)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download top-100 predictions (CSV)", df.to_csv(index=False),
                       "top_predictions.csv", "text/csv")
    df["correct"] = df["tag"] == "TP"
    chart_df = df.groupby(["class", "correct"]).size().unstack(fill_value=0)
    st.bar_chart(chart_df)
    # Qualitative images
    qual_dir = ARTIFACTS / "qualitative"
    if qual_dir.exists():
        st.subheader("Saved qualitative examples")
        images = sorted(qual_dir.glob("*.png"))[:8]
        cols = st.columns(2)
        for i, p in enumerate(images):
            with cols[i % 2]:
                st.image(str(p), caption=p.name, use_container_width=True)


def page_inference() -> None:
    st.header("Single-image inference")
    st.caption("Run the selected detector on a user-uploaded image or a bundled DAWN val example.")
    detector_name = st.selectbox("Detector", ["fasterrcnn_resnet50_fpn", "yolov8n"])
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
        labels = [f"{d.common_label or d.label}:{d.score:.2f}" for d in top if d.score >= score_th]
        rendered = draw_boxes(image, boxes, labels, color=(220, 30, 30))
        st.image(rendered, caption=f"{len(boxes)} detections above score {score_th}",
                 use_container_width=True)
        st.download_button("Download detections (JSON)",
                           json.dumps([d.to_dict() for d in pred.detections], indent=2),
                           f"{image_id}_detections.json", "application/json")


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
  `Motorcycle`, `Bus`. Detector COCO labels are mapped into this space
  via `COCO_TO_DAWN` in `src/dataset.py`. Out-of-vocabulary predictions
  are ignored.

### Evaluation protocol
- IoU threshold: **0.5** (`DEFAULT_IOU_THRESHOLD`).
- Confidence threshold (default): **0.5** (`DEFAULT_SCORE_THRESHOLD`).
- Matching: greedy IoU, predictions processed in descending score
  order; each GT used at most once.
- AP per class: 101-point interpolation; classes with no GT
  contribute NaN and are excluded from the mAP average.
- mAP@.5: macro-average of per-class AP@.5 over classes with ≥1 GT.
- Out-of-DAWN-vocabulary predictions are ignored, not counted as FP.

### How to reproduce every experiment
```
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
PYTHONPATH=. .venv/bin/python scripts/run_full_eval.py --tag fasterrcnn_val_full
PYTHONPATH=. .venv/bin/python scripts/run_full_eval.py --tag yolov8n_val_full --detector yolov8n
PYTHONPATH=. .venv/bin/python scripts/run_corruption_experiment.py --tag fasterrcnn_corruption
PYTHONPATH=. .venv/bin/python scripts/run_corruption_experiment.py --tag yolov8n_corruption --detector yolov8n
PYTHONPATH=. .venv/bin/python scripts/run_corruption_experiment.py --tag fasterrcnn_corruption_v2 --limit 8
PYTHONPATH=. .venv/bin/python scripts/run_corruption_experiment.py --tag yolov8n_corruption_v2 --detector yolov8n --limit 8
PYTHONPATH=. .venv/bin/python scripts/run_corruption_severity.py --tag fasterrcnn_severity --limit 5
PYTHONPATH=. .venv/bin/python scripts/run_corruption_severity.py --detector yolov8n --tag yolov8n_severity --limit 5
PYTHONPATH=. .venv/bin/python scripts/run_statistical_analysis.py --replicates 500
PYTHONPATH=. .venv/bin/python scripts/run_threshold_sweep.py
PYTHONPATH=. .venv/bin/python scripts/run_error_taxonomy.py
PYTHONPATH=. .venv/bin/python scripts/run_preprocessing_ablation.py --tag fasterrcnn_preproc --limit 6
PYTHONPATH=. .venv/bin/python scripts/run_calibration.py
PYTHONPATH=. .venv/bin/python scripts/validate_artifacts.py
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests

# Launch the dashboard
PYTHONPATH=. .venv/bin/streamlit run app.py
```
"""
    )


def page_registry() -> None:
    st.header("Experiment registry & artifact validator")
    path = ARTIFACTS / "experiment_registry.jsonl"
    if not path.exists():
        st.info("No registry yet. Use `scripts/experiment_registry.py` to append entries.")
    else:
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            rows.append({
                "experiment_id": e["experiment_id"],
                "timestamp": e["timestamp"],
                "git_commit": e["git_commit"],
                "detector": e["detector"],
                "dataset_split": e["dataset_split"],
                "n_images": e["n_images"],
                "status": e["status"],
                "runtime_seconds": e["runtime_seconds"],
                "result_artifacts": ";".join(e["result_artifacts"]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.download_button("Download registry CSV", pd.DataFrame(rows).to_csv(index=False),
                           "experiment_registry.csv", "text/csv")

    st.subheader("Artifact validator")
    if st.button("Run artifact validation now"):
        with st.spinner("Validating..."):
            res = subprocess.run(
                [sys.executable, "scripts/validate_artifacts.py"],
                cwd=str(ROOT), capture_output=True, text=True,
            )
        st.text(res.stdout)
        if res.returncode != 0:
            st.error(f"Validator exited with code {res.returncode}")
        else:
            st.success("All checks passed.")


if __name__ == "__main__":
    main()
