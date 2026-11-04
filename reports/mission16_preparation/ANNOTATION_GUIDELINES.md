# WeatherGuard-D1 — Ground-Truth Annotation Guidelines
**Document:** `reports/mission16_preparation/ANNOTATION_GUIDELINES.md`  
**Date:** 2026-11-04  
**Author:** Mohammad Sayem Uddin (<sayemwork41@gmail.com>)  
**Mission:** Mission 16 Preparation — Synthetic-to-Real Generalization Infrastructure

---

## 1. Class Definitions & Ontological Boundaries

WeatherGuard-D1 standardizes detection onto six canonical road-user categories:

| Class ID | Canonical Name | Definition & Inclusions | Exclusions & Boundary Rules |
| :---: | :--- | :--- | :--- |
| **0** | `truck` | Heavy commercial cargo vehicles, semi-trailers, box trucks, flatbeds, dump trucks. | Pickup trucks and passenger SUVs must be labeled as `car`. |
| **1** | `person` | Pedestrians standing, walking, running; construction workers; officers. | Riders actively mounted on bicycles or motorcycles are labeled as `person` with rider flags, or vehicle riders. |
| **2** | `bicycle` | Human-powered bicycles, e-bikes, cargo cycles. | Motorcycles and mopeds are labeled as `motorcycle`. |
| **3** | `car` | Passenger sedans, hatchbacks, coupes, SUVs, minivans, station wagons, pickup trucks. | Commercial box trucks, buses, semi-cabs are excluded. |
| **4** | `motorcycle` | Two-wheeled motorized vehicles, scooters, mopeds. | Bicycles are excluded. |
| **5** | `bus` | Passenger transit buses, school buses, tour coaches, articulated buses. | Passenger minivans (under 10 seats) are labeled as `car`. |

---

## 2. Coordinate System & Conversion Standard

### 2.1 Gemini Flash Normalized Input Coordinate Standard
Annotations emitted by Gemini Flash use an integer range of $0$ to $1000$:
$$\text{Bounding Box: } [x_{min}, y_{min}, x_{max}, y_{max}], \quad 0 \le x, y \le 1000$$

### 2.2 Conversion to Native Pixel Grid ($1280 \times 720$)
$$x_{pixel} = x_{normalized} \times \frac{1280.0}{1000.0}$$
$$y_{pixel} = y_{normalized} \times \frac{720.0}{1000.0}$$

### 2.3 Conversion to YOLO Normalized Ground Truth ($0.0 \text{ to } 1.0$)
$$\text{width}_{norm} = \frac{x_{max\_px} - x_{min\_px}}{1280.0}, \quad \text{height}_{norm} = \frac{y_{max\_px} - y_{min\_px}}{720.0}$$
$$x_{center\_norm} = \frac{x_{min\_px} + x_{max\_px}}{2 \times 1280.0}, \quad y_{center\_norm} = \frac{y_{min\_px} + y_{max\_px}}{2 \times 720.0}$$
$$\text{Format: } \langle class\_id \rangle \quad \langle x_{center} \rangle \quad \langle y_{center} \rangle \quad \langle width \rangle \quad \langle height \rangle$$

---

## 3. Object Scale Classification Convention

To prevent ambiguity, WeatherGuard enforces the **maximum pixel dimension convention**:
$$\text{Scale Metric: } d_{max} = \max(\text{pixel width}, \text{pixel height})$$

| Scale Label | Range Condition | Operational Significance in WeatherGuard |
| :--- | :---: | :--- |
| **Ultra-Small** | $d_{max} < 16.0\text{ px}$ | Distant horizon vehicles; single-frame candidate bottleneck. |
| **Small** | $16.0\text{ px} \le d_{max} < 32.0\text{ px}$ | Critical perception target; high-resolution crop priority. |
| **Medium** | $32.0\text{ px} \le d_{max} < 96.0\text{ px}$ | Intermediate field targets; base 640px detector coverage. |
| **Large** | $d_{max} \ge 96.0\text{ px}$ | Near-field immediate collision hazards; base detector high recall. |

*Gemini Flash supplied `object_size` labels are mathematically audited and overridden whenever they disagree with $d_{max}$.*

---

## 4. Bounding Box Tightness, Occlusion & Quality Standards

1. **Tightness Invariant:** Boxes must bound the outermost visible extent of the target entity. Shadows on asphalt, headlight beams in fog, and exhaust plumes must not be included.
2. **Occlusion Handling:** If a vehicle is partially occluded by a guardrail or tree, bound only the visible portions if disjoint, or the complete body silhouette if continuous. Record occlusion score $\in [0.0, 1.0]$.
3. **Sub-Pixel Filtering:** Bounding boxes with width $< 2\text{px}$ or height $< 2\text{px}$ are rejected as noise artifacts.
4. **Intra-Image Deduplication:** Duplicate boxes of the same class sharing $\text{IoU} \ge 0.85$ are flagged and pruned to prevent artificial precision inflation.
