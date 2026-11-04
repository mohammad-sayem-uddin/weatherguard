"""WeatherGuard-D1 Gemini Flash Annotation Validator & YOLO Label Converter.

Module: scripts/validate_d1_annotations.py
Author: Mohammad Sayem Uddin (<sayemwork41@gmail.com>)
Date: 2026-11-04
Mission: Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

Validates incoming Gemini Flash AI-assisted annotations against strict quality,
geometric, coordinate, and semantic rules, recalibrating object-size categories
and generating verified YOLO-format ground truth files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG_PATH = ROOT / "configs" / "weatherguard_d1.yaml"

VALID_CLASSES = {
    0: "truck",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "bus",
}
CLASS_NAME_TO_ID = {v: k for k, v in VALID_CLASSES.items()}


@dataclass
class ValidatedBox:
    box_id: str
    class_id: int
    class_name: str
    # Pixel coordinates [x1, y1, x2, y2]
    x1_px: float
    y1_px: float
    x2_px: float
    y2_px: float
    pixel_width: float
    pixel_height: float
    pixel_area: float
    max_dim_px: float
    # Object size category recalculated by WeatherGuard convention
    recalculated_size: str  # "ultra_small", "small", "medium", "large"
    gemini_supplied_size: Optional[str]
    size_disagreement: bool
    # YOLO normalized coordinates [x_center, y_center, w, h] (0.0 to 1.0)
    yolo_x_center: float
    yolo_y_center: float
    yolo_width: float
    yolo_height: float
    confidence: float
    occlusion: Optional[float] = 0.0
    truncation: Optional[float] = 0.0
    flags: List[str] = None


@dataclass
class ImageAnnotationResult:
    image_id: str
    filename: str
    condition: str
    is_valid: bool
    is_empty: bool
    status: str  # "ACCEPTED", "CONTAINS_INVALID_BOXES", "REJECTED_STRUCTURAL_ERROR", "EMPTY_ACCEPTED"
    valid_boxes: List[ValidatedBox]
    rejected_boxes: List[Dict[str, Any]]
    validation_flags: List[str]
    yolo_label_path: Optional[str] = None


def compute_iou_xyxy(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union between two (x1, y1, x2, y2) bounding boxes."""
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def classify_object_size(max_dim_px: float) -> str:
    """Classifies object size based on WeatherGuard convention: max(width_px, height_px).

    - Ultra-small: < 16 px
    - Small: 16 to < 32 px
    - Medium: 32 to < 96 px
    - Large: >= 96 px
    """
    if max_dim_px < 16.0:
        return "ultra_small"
    elif max_dim_px < 32.0:
        return "small"
    elif max_dim_px < 96.0:
        return "medium"
    else:
        return "large"


class WeatherGuardD1AnnotationValidator:
    """Validates Gemini Flash JSON annotations and converts to standard YOLO formats."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        img_props = self.config.get("image_properties", {})
        self.img_width = float(img_props.get("target_width", 1280.0))
        self.img_height = float(img_props.get("target_height", 720.0))

        qc = self.config.get("quality_control", {})
        self.min_dim_px = float(qc.get("min_bounding_box_dimension_px", 2.0))
        self.min_area_px = float(qc.get("min_bounding_box_area_px", 4.0))
        self.min_confidence = float(qc.get("min_annotation_confidence", 0.20))

    def validate_single_annotation_json(
        self,
        raw_json_data: Any,
        expected_image_id: Optional[str] = None,
        expected_condition: str = "day",
    ) -> ImageAnnotationResult:
        """Validates a parsed JSON dictionary representing Gemini Flash output for one image."""
        flags: List[str] = []
        rejected_boxes: List[Dict[str, Any]] = []
        valid_boxes: List[ValidatedBox] = []

        if not isinstance(raw_json_data, dict):
            return ImageAnnotationResult(
                image_id=expected_image_id or "unknown",
                filename="",
                condition=expected_condition,
                is_valid=False,
                is_empty=False,
                status="REJECTED_STRUCTURAL_ERROR",
                valid_boxes=[],
                rejected_boxes=[{"error": "Root JSON element is not an object/dictionary"}],
                validation_flags=["Root JSON element is not a dict"],
            )

        # Image ID resolution
        image_id = raw_json_data.get("image_id") or expected_image_id or "unknown"
        filename = raw_json_data.get("filename") or f"{image_id}.jpg"
        condition = raw_json_data.get("condition") or expected_condition

        if expected_image_id and image_id != expected_image_id:
            flags.append(f"Image ID mismatch: JSON '{image_id}' != Expected '{expected_image_id}'")

        # Extract annotations array
        annotations = raw_json_data.get("annotations", raw_json_data.get("objects", []))
        if not isinstance(annotations, list):
            return ImageAnnotationResult(
                image_id=image_id,
                filename=filename,
                condition=condition,
                is_valid=False,
                is_empty=False,
                status="REJECTED_STRUCTURAL_ERROR",
                valid_boxes=[],
                rejected_boxes=[{"error": "Annotations field is not a list"}],
                validation_flags=["Annotations field is not a list"],
            )

        # Empty image handling
        if len(annotations) == 0:
            return ImageAnnotationResult(
                image_id=image_id,
                filename=filename,
                condition=condition,
                is_valid=True,
                is_empty=True,
                status="EMPTY_ACCEPTED",
                valid_boxes=[],
                rejected_boxes=[],
                validation_flags=["Valid empty scene (0 annotations)"],
            )

        # Iterate and validate each bounding box
        for idx, item in enumerate(annotations):
            b_flags: List[str] = []
            box_id = item.get("box_id", f"{image_id}_box_{idx:03d}")

            # 1. Class ID and Name Verification
            cls_id = item.get("class_id")
            cls_name = item.get("class_name") or item.get("category") or item.get("label")

            if cls_id is None and cls_name in CLASS_NAME_TO_ID:
                cls_id = CLASS_NAME_TO_ID[cls_name]
            elif cls_name is None and cls_id in VALID_CLASSES:
                cls_name = VALID_CLASSES[cls_id]

            if cls_id not in VALID_CLASSES:
                b_flags.append(f"Invalid class_id: {cls_id} (allowed: 0..5)")
            elif cls_name != VALID_CLASSES[cls_id]:
                b_flags.append(f"Class mismatch: class_id {cls_id} does not match name '{cls_name}' (expected '{VALID_CLASSES[cls_id]}')")

            # 2. Coordinate Extraction & Validation
            # Gemini Flash returns [x_min, y_min, x_max, y_max] normalized 0..1000
            coords = item.get("box_2d") or item.get("bbox") or item.get("coordinates")
            x_min_norm = None
            y_min_norm = None
            x_max_norm = None
            y_max_norm = None

            if isinstance(coords, list) and len(coords) == 4:
                # Standard format [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax]
                # If explicit keys are provided in item:
                if "x_min" in item and "x_max" in item:
                    x_min_norm = float(item["x_min"])
                    y_min_norm = float(item["y_min"])
                    x_max_norm = float(item["x_max"])
                    y_max_norm = float(item["y_max"])
                else:
                    x_min_norm = float(coords[0])
                    y_min_norm = float(coords[1])
                    x_max_norm = float(coords[2])
                    y_max_norm = float(coords[3])
            elif all(k in item for k in ("x_min", "y_min", "x_max", "y_max")):
                x_min_norm = float(item["x_min"])
                y_min_norm = float(item["y_min"])
                x_max_norm = float(item["x_max"])
                y_max_norm = float(item["y_max"])
            else:
                b_flags.append("Missing or unparseable bounding box coordinates")

            if b_flags or x_min_norm is None:
                rejected_boxes.append({"box_index": idx, "raw": item, "rejection_reasons": b_flags})
                continue

            # Bounds checking in 0..1000 normalized range
            if not (0.0 <= x_min_norm <= 1000.0 and 0.0 <= x_max_norm <= 1000.0 and
                    0.0 <= y_min_norm <= 1000.0 and 0.0 <= y_max_norm <= 1000.0):
                b_flags.append(f"Coordinates out of [0, 1000] bounds: [{x_min_norm}, {y_min_norm}, {x_max_norm}, {y_max_norm}]")

            # Geometric non-inversion
            if x_min_norm >= x_max_norm:
                b_flags.append(f"Non-positive width: x_min ({x_min_norm}) >= x_max ({x_max_norm})")
            if y_min_norm >= y_max_norm:
                b_flags.append(f"Non-positive height: y_min ({y_min_norm}) >= y_max ({y_max_norm})")

            if b_flags:
                rejected_boxes.append({"box_index": idx, "raw": item, "rejection_reasons": b_flags})
                continue

            # Convert to pixel coordinates
            x1_px = (x_min_norm * self.img_width) / 1000.0
            y1_px = (y_min_norm * self.img_height) / 1000.0
            x2_px = (x_max_norm * self.img_width) / 1000.0
            y2_px = (y_max_norm * self.img_height) / 1000.0

            width_px = x2_px - x1_px
            height_px = y2_px - y1_px
            area_px = width_px * height_px
            max_dim_px = max(width_px, height_px)

            if width_px < self.min_dim_px or height_px < self.min_dim_px:
                b_flags.append(f"Sub-pixel dimensions: {width_px:.1f}x{height_px:.1f} < min {self.min_dim_px}px")
            if area_px < self.min_area_px:
                b_flags.append(f"Area {area_px:.1f}px^2 < min {self.min_area_px}px^2")

            # 3. Object Size Recalculation vs Gemini Supplying
            recalculated_size = classify_object_size(max_dim_px)
            gemini_size = item.get("object_size")
            size_disagreement = False
            if gemini_size and gemini_size.lower().replace("-", "_") != recalculated_size:
                size_disagreement = True
                b_flags.append(f"Gemini object_size '{gemini_size}' disagreed with recalculated '{recalculated_size}'")

            # 4. Confidence
            conf = float(item.get("confidence", 1.0))
            if conf < self.min_confidence:
                b_flags.append(f"Low annotation confidence: {conf:.2f} < threshold {self.min_confidence:.2f}")

            # 5. YOLO Format Conversion (Normalized 0.0 to 1.0)
            yolo_xc = round(((x1_px + x2_px) / 2.0) / self.img_width, 6)
            yolo_yc = round(((y1_px + y2_px) / 2.0) / self.img_height, 6)
            yolo_w = round(width_px / self.img_width, 6)
            yolo_h = round(height_px / self.img_height, 6)

            if b_flags and any("Sub-pixel" in fl or "Area" in fl for fl in b_flags):
                rejected_boxes.append({"box_index": idx, "raw": item, "rejection_reasons": b_flags})
                continue

            v_box = ValidatedBox(
                box_id=box_id,
                class_id=int(cls_id),
                class_name=cls_name,
                x1_px=round(x1_px, 2),
                y1_px=round(y1_px, 2),
                x2_px=round(x2_px, 2),
                y2_px=round(y2_px, 2),
                pixel_width=round(width_px, 2),
                pixel_height=round(height_px, 2),
                pixel_area=round(area_px, 2),
                max_dim_px=round(max_dim_px, 2),
                recalculated_size=recalculated_size,
                gemini_supplied_size=gemini_size,
                size_disagreement=size_disagreement,
                yolo_x_center=yolo_xc,
                yolo_y_center=yolo_yc,
                yolo_width=yolo_w,
                yolo_height=yolo_h,
                confidence=conf,
                occlusion=float(item.get("occlusion", 0.0)),
                truncation=float(item.get("truncation", 0.0)),
                flags=b_flags,
            )
            valid_boxes.append(v_box)

        # 6. Intra-Image Duplicate Annotation Detection (Same class, IoU >= 0.85)
        pruned_boxes: List[ValidatedBox] = []
        for i, b1 in enumerate(valid_boxes):
            is_dup = False
            for j in range(i + 1, len(valid_boxes)):
                b2 = valid_boxes[j]
                if b1.class_id == b2.class_id:
                    iou = compute_iou_xyxy((b1.x1_px, b1.y1_px, b1.x2_px, b1.y2_px),
                                           (b2.x1_px, b2.y1_px, b2.x2_px, b2.y2_px))
                    if iou >= 0.85:
                        is_dup = True
                        rejected_boxes.append({
                            "box_id": b2.box_id,
                            "rejection_reasons": [f"Intra-image duplicate of {b1.box_id} (IoU={iou:.2f})"]
                        })
                        flags.append(f"Detected duplicate box: {b1.box_id} and {b2.box_id} (IoU={iou:.2f})")
                        break
            if not is_dup:
                pruned_boxes.append(b1)

        valid_boxes = pruned_boxes

        status = "ACCEPTED"
        if rejected_boxes:
            status = "CONTAINS_INVALID_BOXES" if valid_boxes else "REJECTED_STRUCTURAL_ERROR"

        return ImageAnnotationResult(
            image_id=image_id,
            filename=filename,
            condition=condition,
            is_valid=(len(valid_boxes) > 0 or len(annotations) == 0),
            is_empty=(len(valid_boxes) == 0 and len(rejected_boxes) == 0),
            status=status,
            valid_boxes=valid_boxes,
            rejected_boxes=rejected_boxes,
            validation_flags=flags,
        )

    def write_yolo_label_file(self, result: ImageAnnotationResult, output_dir: Path) -> Path:
        """Writes standard YOLO format annotation text file (.txt)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        txt_path = output_dir / f"{result.image_id}.txt"
        lines = []
        for b in result.valid_boxes:
            lines.append(f"{b.class_id} {b.yolo_x_center:.6f} {b.yolo_y_center:.6f} {b.yolo_width:.6f} {b.yolo_height:.6f}")

        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        result.yolo_label_path = str(txt_path)
        return txt_path

    def validate_annotation_directory(
        self,
        json_input_dir: Path,
        yolo_output_dir: Optional[Path] = None,
        manifest_output_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Discovers, audits, and converts all Gemini JSON files in a directory."""
        if not json_input_dir.exists():
            return {
                "status": "NOT_AVAILABLE",
                "message": f"Annotations directory does not exist: {json_input_dir}",
                "total_files": 0,
            }

        json_files = sorted(list(json_input_dir.glob("*.json")))
        if not json_files:
            return {
                "status": "NO_ANNOTATIONS_FOUND",
                "message": f"No .json files found in {json_input_dir}",
                "total_files": 0,
            }

        results: List[ImageAnnotationResult] = []
        total_boxes = 0
        total_rejected = 0
        size_counts = {"ultra_small": 0, "small": 0, "medium": 0, "large": 0}
        class_counts = {name: 0 for name in VALID_CLASSES.values()}
        disagreement_count = 0

        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    raw_data = json.load(f)
                res = self.validate_single_annotation_json(raw_data, expected_image_id=jf.stem)
                if yolo_output_dir and res.is_valid:
                    self.write_yolo_label_file(res, yolo_output_dir)

                results.append(res)
                total_boxes += len(res.valid_boxes)
                total_rejected += len(res.rejected_boxes)

                for b in res.valid_boxes:
                    size_counts[b.recalculated_size] += 1
                    class_counts[b.class_name] += 1
                    if b.size_disagreement:
                        disagreement_count += 1
            except Exception as e:
                err_res = ImageAnnotationResult(
                    image_id=jf.stem,
                    filename=jf.name,
                    condition="unknown",
                    is_valid=False,
                    is_empty=False,
                    status="REJECTED_STRUCTURAL_ERROR",
                    valid_boxes=[],
                    rejected_boxes=[{"file_read_error": str(e)}],
                    validation_flags=[f"JSON parse error: {e}"],
                )
                results.append(err_res)

        summary = {
            "total_files_audited": len(results),
            "valid_images": sum(1 for r in results if r.is_valid),
            "empty_images": sum(1 for r in results if r.is_empty),
            "total_valid_boxes": total_boxes,
            "total_rejected_boxes": total_rejected,
            "object_size_distribution": size_counts,
            "class_distribution": class_counts,
            "gemini_size_disagreements": disagreement_count,
        }

        manifest = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": summary,
            "results": [asdict(r) for r in results],
        }

        if manifest_output_file:
            manifest_output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_output_file, "w") as f:
                json.dump(manifest, f, indent=2)

        return manifest


def main():
    parser = argparse.ArgumentParser(description="Validate Gemini Flash JSON annotations and produce YOLO ground truth.")
    parser.add_argument("--input-dir", type=str, default="data/WeatherGuard-D1/raw_annotations", help="Incoming JSON folder")
    parser.add_argument("--output-labels", type=str, default="data/WeatherGuard-D1/labels", help="Output YOLO text files folder")
    parser.add_argument("--manifest", type=str, default="data/WeatherGuard-D1/manifests/annotation_validation_manifest.json")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    validator = WeatherGuardD1AnnotationValidator(config_path=Path(args.config))
    res = validator.validate_annotation_directory(
        json_input_dir=Path(args.input_dir),
        yolo_output_dir=Path(args.output_labels),
        manifest_output_file=Path(args.manifest),
    )
    print(f"Validation complete: {res.get('summary', {}).get('total_valid_boxes', 0)} boxes validated.")


if __name__ == "__main__":
    main()
