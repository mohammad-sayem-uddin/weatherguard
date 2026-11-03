"""Locked Evaluation & Paired Sequence-Level Bootstrap for WeatherGuard Mission 15.

Module: scripts/run_mission15_locked_eval.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-03
Mission: Mission 15 — Temporal Evidence Accumulation & Horizon-Anchored Candidate Generation

Evaluates 4 frozen configurations across the 6 virgin Locked Test Sequences (1,214 frames, 14,025 GT boxes):
1. YOLO640 (Static baseline)
2. YOLO832 (Static high-resolution baseline)
3. Policy J (Historical frozen reference)
4. Policy K (Temporal-Horizon Priority Allocation, K=2)

Performs paired sequence-level bootstrapping (B=1,000 replicates) to compute:
- 95% Empirical Confidence Intervals
- Two-sided bootstrap p-values
- Sequence-level win/loss distributions

Outputs:
- artifacts/mission15/locked_evaluation.json
- artifacts/mission15/sequence_bootstrap.json
- reports/mission15/locked_evaluation.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mission15.policy_k import PolicyKConfig, PolicyKController
from src.priority_allocation import (
    DetectionBox,
    PriorityAllocationController,
    compute_iou,
)

ARTIFACTS_DIR = ROOT / "artifacts" / "mission15"
REPORTS_DIR = ROOT / "reports" / "mission15"
DATA_DIR = ROOT / "data" / "mission15_independent"
SEQ_DIR = DATA_DIR / "sequences"
LABELS_DIR = DATA_DIR / "labels"

ULTRA_SMALL_AREA_THRESHOLD = 256.0
SMALL_AREA_THRESHOLD = 1024.0
MEDIUM_AREA_THRESHOLD = 9216.0

CANONICAL_CLASSES = {
    "pedestrian": "person",
    "rider": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}

YOLO_TO_BDD = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class GTBox:
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    area: float
    scale: str
    track_id: int


def get_scale(area: float) -> str:
    if area < ULTRA_SMALL_AREA_THRESHOLD:
        return "ultra_small"
    elif area < SMALL_AREA_THRESHOLD:
        return "small"
    elif area < MEDIUM_AREA_THRESHOLD:
        return "medium"
    else:
        return "large"


def evaluate_detections(
    predictions: List[List[DetectionBox]],
    ground_truths: List[List[GTBox]],
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    scale_stats = {s: {"tp": 0, "fp": 0, "fn": 0} for s in ["ultra_small", "small", "medium", "large"]}

    for f_idx in range(len(ground_truths)):
        gts = ground_truths[f_idx]
        preds = predictions[f_idx]

        matched_gt = set()
        for p in preds:
            best_iou = 0.0
            best_gi = -1
            for gi, g in enumerate(gts):
                if gi in matched_gt or g.class_name != p.class_name:
                    continue
                iou = compute_iou((p.x1, p.y1, p.x2, p.y2), (g.x1, g.y1, g.x2, g.y2))
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            pw = p.x2 - p.x1
            ph = p.y2 - p.y1
            p_scale = get_scale(pw * ph)

            if best_iou >= iou_thresh and best_gi >= 0:
                tot_tp += 1
                matched_gt.add(best_gi)
                scale_stats[gts[best_gi].scale]["tp"] += 1
            else:
                tot_fp += 1
                scale_stats[p_scale]["fp"] += 1

        for gi, g in enumerate(gts):
            if gi not in matched_gt:
                tot_fn += 1
                scale_stats[g.scale]["fn"] += 1

    def calc_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return round(prec, 4), round(rec, 4), round(f1, 4)

    prec, rec, f1 = calc_prf(tot_tp, tot_fp, tot_fn)
    u_prec, u_rec, u_f1 = calc_prf(scale_stats["ultra_small"]["tp"], scale_stats["ultra_small"]["fp"], scale_stats["ultra_small"]["fn"])
    s_prec, s_rec, s_f1 = calc_prf(scale_stats["small"]["tp"], scale_stats["small"]["fp"], scale_stats["small"]["fn"])
    m_prec, m_rec, m_f1 = calc_prf(scale_stats["medium"]["tp"], scale_stats["medium"]["fp"], scale_stats["medium"]["fn"])
    l_prec, l_rec, l_f1 = calc_prf(scale_stats["large"]["tp"], scale_stats["large"]["fp"], scale_stats["large"]["fn"])

    return {
        "overall_precision": prec,
        "overall_recall": rec,
        "overall_f1": f1,
        "ultra_small_precision": u_prec,
        "ultra_small_recall": u_rec,
        "ultra_small_f1": u_f1,
        "ultra_small_tp": scale_stats["ultra_small"]["tp"],
        "small_precision": s_prec,
        "small_recall": s_rec,
        "small_f1": s_f1,
        "small_tp": scale_stats["small"]["tp"],
        "medium_f1": m_f1,
        "large_f1": l_f1,
        "total_tp": tot_tp,
        "total_fp": tot_fp,
        "total_fn": tot_fn,
    }


def merge_detections_nms(
    base_dets: List[DetectionBox],
    crop_dets: List[DetectionBox],
    iou_thresh: float = 0.50,
) -> List[DetectionBox]:
    if not crop_dets:
        return base_dets
    if not base_dets:
        return crop_dets

    all_dets = list(base_dets) + list(crop_dets)
    all_dets.sort(key=lambda d: d.confidence, reverse=True)

    keep = []
    while all_dets:
        cur = all_dets.pop(0)
        keep.append(cur)
        remaining = []
        for other in all_dets:
            if other.class_name == cur.class_name:
                iou = compute_iou((cur.x1, cur.y1, cur.x2, cur.y2), (other.x1, other.y1, other.x2, other.y2))
                if iou < iou_thresh:
                    remaining.append(other)
            else:
                remaining.append(other)
        all_dets = remaining

    return keep


def run_locked_evaluation():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(ARTIFACTS_DIR / "sequence_manifest.json"))
    locked_sids = [s["sequence_id"] for s in manifest["cohorts"]["locked_test_cohort"]["sequences"]]
    print(f"Executing Frozen Locked Evaluation across {len(locked_sids)} independent sequences...")

    weights_path = ROOT / "models" / "yolov8n.pt" if (ROOT / "models" / "yolov8n.pt").exists() else ROOT / "yolov8n.pt"
    yolo_model = YOLO(str(weights_path))

    # Load locked sequence data
    seq_frames: Dict[str, List[Dict[str, Any]]] = {}
    total_frames = 0
    total_gts = 0

    for sid in locked_sids:
        s_folder = SEQ_DIR / sid
        lbl_file = LABELS_DIR / f"{sid}.json"
        frame_files = sorted(list(s_folder.glob("*.jpg")))
        labels_json = json.load(open(lbl_file))
        lbl_lookup = {item["name"]: item for item in labels_json}

        frames = []
        for ff in frame_files:
            fname = ff.name
            raw_meta = lbl_lookup.get(fname, {})
            gt_boxes = []
            for lbl in raw_meta.get("labels", []):
                cat = lbl.get("category", "").lower()
                norm_cat = CANONICAL_CLASSES.get(cat, cat)
                b = lbl.get("box2d", {})
                x1, y1, x2, y2 = float(b.get("x1", 0)), float(b.get("y1", 0)), float(b.get("x2", 0)), float(b.get("y2", 0))
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                gt_boxes.append(GTBox(
                    class_name=norm_cat,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    area=area,
                    scale=get_scale(area),
                    track_id=int(lbl.get("id", -1)) if str(lbl.get("id", "")).isdigit() else -1,
                ))
            frames.append({"file_path": str(ff), "frame_name": fname, "gt_boxes": gt_boxes})
            total_gts += len(gt_boxes)
        seq_frames[sid] = frames
        total_frames += len(frames)

    print(f"Locked cohort loaded: {total_frames} frames, {total_gts} annotations.")

    # Storage for per-sequence predictions
    predictions: Dict[str, Dict[str, List[List[DetectionBox]]]] = {
        "A_YOLO640": {sid: [] for sid in locked_sids},
        "B_YOLO832": {sid: [] for sid in locked_sids},
        "C_Policy_J": {sid: [] for sid in locked_sids},
        "D_Policy_K": {sid: [] for sid in locked_sids},
    }

    timing_totals = defaultdict(float)

    def execute_crops(frame_rgb: np.ndarray, crops: List[Tuple[int, int, int, int]]) -> List[DetectionBox]:
        dets = []
        for cx1, cy1, cx2, cy2 in crops:
            crop_img = frame_rgb[cy1:cy2, cx1:cx2]
            if crop_img.shape[0] < 32 or crop_img.shape[1] < 32:
                continue
            res = yolo_model(crop_img, imgsz=448, conf=0.20, verbose=False)[0]
            for b in res.boxes:
                cls_id = int(b.cls[0])
                if cls_id in YOLO_TO_BDD:
                    xyxy = b.xyxy[0].cpu().numpy()
                    dets.append(DetectionBox(
                        class_name=YOLO_TO_BDD[cls_id],
                        confidence=float(b.conf[0]),
                        x1=float(xyxy[0] + cx1),
                        y1=float(xyxy[1] + cy1),
                        x2=float(xyxy[2] + cx1),
                        y2=float(xyxy[3] + cy1),
                    ))
        return dets

    # Initialize controllers
    ctrl_j = PriorityAllocationController(policy="policy_j_best_combination", max_crops=2)
    ctrl_k = PolicyKController(PolicyKConfig(max_crops=2, crop_size=448))

    for s_idx, sid in enumerate(locked_sids):
        print(f"[{s_idx+1}/{len(locked_sids)}] Evaluating sequence {sid} ({len(seq_frames[sid])} frames)...")
        ctrl_j.reset()
        ctrl_k.reset()

        for f_info in seq_frames[sid]:
            img_bgr = cv2.imread(f_info["file_path"])
            frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
            frame_rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 1. Config A: YOLO640
            t0 = time.perf_counter()
            res640 = yolo_model(f_info["file_path"], imgsz=640, conf=0.20, verbose=False)[0]
            boxes640 = []
            for b in res640.boxes:
                cls_id = int(b.cls[0])
                if cls_id in YOLO_TO_BDD:
                    xyxy = b.xyxy[0].cpu().numpy()
                    boxes640.append(DetectionBox(
                        class_name=YOLO_TO_BDD[cls_id],
                        confidence=float(b.conf[0]),
                        x1=float(xyxy[0]), y1=float(xyxy[1]), x2=float(xyxy[2]), y2=float(xyxy[3]),
                    ))
            timing_totals["A_YOLO640"] += time.perf_counter() - t0
            predictions["A_YOLO640"][sid].append(boxes640)

            # 2. Config B: YOLO832
            t0 = time.perf_counter()
            res832 = yolo_model(f_info["file_path"], imgsz=832, conf=0.20, verbose=False)[0]
            boxes832 = []
            for b in res832.boxes:
                cls_id = int(b.cls[0])
                if cls_id in YOLO_TO_BDD:
                    xyxy = b.xyxy[0].cpu().numpy()
                    boxes832.append(DetectionBox(
                        class_name=YOLO_TO_BDD[cls_id],
                        confidence=float(b.conf[0]),
                        x1=float(xyxy[0]), y1=float(xyxy[1]), x2=float(xyxy[2]), y2=float(xyxy[3]),
                    ))
            timing_totals["B_YOLO832"] += time.perf_counter() - t0
            predictions["B_YOLO832"][sid].append(boxes832)

            # 3. Config C: Policy J
            t0 = time.perf_counter()
            dec_j = ctrl_j.allocate_crops(img_bgr, detector_boxes=boxes640)
            crop_boxes_j = [(int(c.x1), int(c.y1), int(c.x2), int(c.y2)) for c in dec_j.allocated_crops]
            crop_dets_j = execute_crops(frame_rgb_full, crop_boxes_j)
            final_j = merge_detections_nms(boxes640, crop_dets_j)
            timing_totals["C_Policy_J"] += time.perf_counter() - t0
            predictions["C_Policy_J"][sid].append(final_j)

            # 4. Config D: Policy K
            t0 = time.perf_counter()
            crops_k, cands_k, _ = ctrl_k.process_and_allocate(frame_rgb_full, sequence_id=sid, detector_boxes=boxes640)
            crop_dets_k = execute_crops(frame_rgb_full, crops_k)
            final_k = merge_detections_nms(boxes640, crop_dets_k)
            timing_totals["D_Policy_K"] += time.perf_counter() - t0
            predictions["D_Policy_K"][sid].append(final_k)

    # Compute aggregate metrics
    locked_results = {"aggregate": {}, "per_sequence": {}}

    for cfg in ["A_YOLO640", "B_YOLO832", "C_Policy_J", "D_Policy_K"]:
        all_preds = [p for sid in locked_sids for p in predictions[cfg][sid]]
        all_gts = [f["gt_boxes"] for sid in locked_sids for f in seq_frames[sid]]
        m = evaluate_detections(all_preds, all_gts)
        mean_ms = (timing_totals[cfg] / total_frames) * 1000.0
        m["mean_latency_ms"] = round(mean_ms, 2)
        m["fps"] = round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0
        locked_results["aggregate"][cfg] = m

        # Per sequence metrics
        for sid in locked_sids:
            if sid not in locked_results["per_sequence"]:
                locked_results["per_sequence"][sid] = {}
            s_preds = predictions[cfg][sid]
            s_gts = [f["gt_boxes"] for f in seq_frames[sid]]
            locked_results["per_sequence"][sid][cfg] = evaluate_detections(s_preds, s_gts)

    # Save locked evaluation artifact
    locked_path = ARTIFACTS_DIR / "locked_evaluation.json"
    with open(locked_path, "w") as f:
        json.dump(locked_results, f, indent=2)
    print(f"Saved locked evaluation results to {locked_path}")

    # Paired sequence bootstrap analysis (B = 1,000)
    print("\nRunning Paired Sequence-Level Bootstrap (B=1,000)...")
    np.random.seed(42)
    B = 1000
    n_seqs = len(locked_sids)

    bootstrap_records = {}

    comparisons = [
        ("Policy_K_vs_YOLO640", "D_Policy_K", "A_YOLO640"),
        ("Policy_K_vs_Policy_J", "D_Policy_K", "C_Policy_J"),
        ("Policy_K_vs_YOLO832", "D_Policy_K", "B_YOLO832"),
    ]

    for comp_name, cfg_a, cfg_b in comparisons:
        diffs_small = []
        diffs_ultra = []
        diffs_overall = []

        wins_small = 0
        losses_small = 0
        ties_small = 0

        # Sequence-level win/loss count
        for sid in locked_sids:
            s_a = locked_results["per_sequence"][sid][cfg_a]
            s_b = locked_results["per_sequence"][sid][cfg_b]
            d_s = s_a["small_f1"] - s_b["small_f1"]
            if d_s > 0.0001:
                wins_small += 1
            elif d_s < -0.0001:
                losses_small += 1
            else:
                ties_small += 1

        for _ in range(B):
            sample_sids = np.random.choice(locked_sids, size=n_seqs, replace=True)
            # Flatten predictions and GTs for the resampled multiset
            b_preds_a = [p for s in sample_sids for p in predictions[cfg_a][s]]
            b_preds_b = [p for s in sample_sids for p in predictions[cfg_b][s]]
            b_gts = [f["gt_boxes"] for s in sample_sids for f in seq_frames[s]]

            res_a = evaluate_detections(b_preds_a, b_gts)
            res_b = evaluate_detections(b_preds_b, b_gts)

            diffs_small.append(res_a["small_f1"] - res_b["small_f1"])
            diffs_ultra.append(res_a["ultra_small_recall"] - res_b["ultra_small_recall"])
            diffs_overall.append(res_a["overall_f1"] - res_b["overall_f1"])

        def get_ci_and_p(diffs: List[float], point_est: float) -> Tuple[List[float], float]:
            diffs_sorted = np.sort(diffs)
            ci_low = float(np.percentile(diffs_sorted, 2.5))
            ci_high = float(np.percentile(diffs_sorted, 97.5))
            p_val = float(2.0 * min(np.mean(diffs_sorted <= 0), np.mean(diffs_sorted >= 0)))
            return [round(ci_low, 4), round(ci_high, 4)], min(1.0, round(p_val, 4))

        pt_small = locked_results["aggregate"][cfg_a]["small_f1"] - locked_results["aggregate"][cfg_b]["small_f1"]
        pt_ultra = locked_results["aggregate"][cfg_a]["ultra_small_recall"] - locked_results["aggregate"][cfg_b]["ultra_small_recall"]
        pt_overall = locked_results["aggregate"][cfg_a]["overall_f1"] - locked_results["aggregate"][cfg_b]["overall_f1"]

        ci_small, p_small = get_ci_and_p(diffs_small, pt_small)
        ci_ultra, p_ultra = get_ci_and_p(diffs_ultra, pt_ultra)
        ci_overall, p_overall = get_ci_and_p(diffs_overall, pt_overall)

        bootstrap_records[comp_name] = {
            "comparison": comp_name,
            "cfg_a": cfg_a,
            "cfg_b": cfg_b,
            "small_f1": {
                "point_estimate_delta": round(pt_small, 4),
                "ci_95": ci_small,
                "p_value": p_small,
                "sequence_win_rate": f"{wins_small}/{n_seqs} (wins/losses/ties: {wins_small}/{losses_small}/{ties_small})",
            },
            "ultra_small_recall": {
                "point_estimate_delta": round(pt_ultra, 4),
                "ci_95": ci_ultra,
                "p_value": p_ultra,
            },
            "overall_f1": {
                "point_estimate_delta": round(pt_overall, 4),
                "ci_95": ci_overall,
                "p_value": p_overall,
            },
        }

    bootstrap_artifact = {
        "mission": "Mission 15 — Paired Sequence Bootstrap Analysis",
        "num_bootstrap_replicates": B,
        "sample_size_sequences": n_seqs,
        "total_evaluation_frames": total_frames,
        "total_ground_truth_boxes": total_gts,
        "tests": bootstrap_records,
    }

    boot_path = ARTIFACTS_DIR / "sequence_bootstrap.json"
    with open(boot_path, "w") as f:
        json.dump(bootstrap_artifact, f, indent=2)
    print(f"Saved sequence bootstrap records to {boot_path}")


if __name__ == "__main__":
    run_locked_evaluation()
