# WeatherGuard Cross-Dataset Evaluation: Dataset Investigation & Selection

**Date:** 2026-11-01  
**Author:** WeatherGuard Autonomous Research Agent  
**Status:** Frozen / Approved  

---

## 1. Executive Summary

WeatherGuard's initial benchmark evaluated object detection robustness and confidence-threshold transfer on the public **DAWN** (Detection in Adverse Weather Nature) dataset. While DAWN provides natural multi-hazard adverse weather images, relying on a single dataset creates key methodological limitations:
1. **Provenance bias:** DAWN images were aggregated from internet search engines (Google, Bing, YouTube) using adverse-weather query terms, introducing potential selection bias toward dramatic, high-contrast, or visually sensational scenes.
2. **Evaluation scope:** Operating points tuned on DAWN may overfit to web-scraped photographic distributions rather than real-world operational camera platforms.
3. **Dataset scale:** The held-out test split of DAWN comprises 202 images, leaving rare conditions (such as dust storms and tornadoes) with limited sample sizes.

To investigate **cross-dataset generalization**, **domain shift**, and **external operating-point transfer**, we conducted an investigation of candidate independent adverse-weather driving datasets. Based on systematic criteria—provenance independence, native 2D bounding boxes, multi-condition weather labeling, taxonomic compatibility with COCO classes, and download/storage feasibility—we select **BDD100K** as the **primary external dataset** and **RTTS (RESIDE)** as the **secondary reference dataset**.

---

## 2. Candidate Datasets Comparison

The following table provides an exhaustive, verifiable comparison across the primary candidate datasets and the baseline DAWN benchmark.

| Dimension | **DAWN (Baseline)** | **BDD100K (Primary Candidate)** | **RTTS / RESIDE (Secondary Candidate)** | **Foggy Cityscapes (Synthetic)** | **ACDC (Semantic Focus)** |
|---|---|---|---|---|---|
| **Official Citation** | Rashid et al. (2020), *IEEE ICCVW* | Yu et al. (2020), *CVPR* | Li et al. (2019), *IEEE TIP* | Sakaridis et al. (2018), *IJCV* | Sakaridis et al. (2021), *ICCV* |
| **Official Source** | Hugging Face / Mendeley | UC Berkeley (`bdd-data.berkeley.edu`, `doc.bdd100k.com`) | RESIDE repository / GitHub / Google Drive | Cityscapes Consortium (`cityscapes-dataset.com`) | ETH Zurich (`acdc.vision.ee.ethz.ch`) |
| **Official License** | CC BY-NC 3.0 | Custom Berkeley DeepDrive Academic License (Free for non-commercial research) | MIT / Open Academic License | Cityscapes License (Non-commercial research, account required) | Non-commercial Academic License |
| **Redistribution Prohibited?** | Non-commercial sharing permitted with attribution | Yes (Redistribution of images/annotations prohibited) | Redistribution restricted per source host | Yes (Redistribution strictly prohibited) | Yes (Redistribution strictly prohibited) |
| **Dataset Provenance** | Web scraping (Google, Bing, YouTube queries) | Forward-facing dashcam video in vehicles (US: NY, SF, Bay Area) | Web searches (Google, Bing, Baidu) + outdoor webcams | Clear-weather German city driving + simulated optical fog | On-road driving in Switzerland (adverse weather matched pairs) |
| **Independence from DAWN** | N/A (Original study) | **Genuinely Independent** (Zero overlap; in-vehicle mobile video capture) | **Questionable** (Shares web-crawled search queries: Google, Bing) | Fully Independent (Synthetic fog) | Fully Independent |
| **Sensor / Resolution** | Variable consumer / web camera resolutions | **Uniform 1280 × 720 HD** (30 fps dashcam video keyframes) | Variable web and camera resolutions | Uniform 2048 × 1024 | Uniform 1920 × 1080 |
| **Weather Conditions** | 8 categories: rain, snow, fog, sand, mist, haze, dusttornado, sand_storm_g2 | 6 categories: clear, overcast, partly cloudy, rainy, snowy, foggy | 1 category: fog / haze only | 1 category: synthetic fog ($\beta \in \{0.005, 0.01, 0.02\}$) | 4 categories: fog, rain, snow, night |
| **Adverse-Weather Volume** | 1,023 total (821 dev, 202 heldout) | **Validation split (10,000 images):** 769 snowy, 738 rainy, 13 foggy, 1,239 overcast | 4,322 annotated foggy/hazy images | 2,975 train, 500 val (all synthetic fog) | 1,000 per condition (4,006 total) |
| **Bounding Box Annotations** | Native 2D bounding boxes | Native 2D bounding boxes | Native 2D bounding boxes (PASCAL VOC XML) | Derived from polygonal instance segmentation | Primary task is semantic / panoptic segmentation |
| **Compatible Target Classes** | 6 classes: Person, Car, Truck, Bus, Motorcycle, Bicycle | 6 target classes: pedestrian, car, truck, bus, bicycle, motorcycle (+ rider, train, sign, light) | 5 classes: person, car, bus, bicycle, motorbike (**Truck absent!**) | 8 classes: person, rider, car, truck, bus, train, motorcycle, bicycle | 19 Cityscapes classes (no native 2D bounding boxes) |
| **Download / Storage Profile** | 139 MB (Parquet) | 68 MB (Validation JSON annotations) + ~15 MB for 200–250 sampled images | ~1.4 GB (Full tar archive) | ~12 GB (Cityscapes base + fog depth maps) | ~32 GB (RGB + masks) |

---

## 3. Deep-Dive Candidate Assessment

### 3.1 BDD100K (Selected Primary Dataset)
* **Provenance and Independence:** BDD100K was collected by the Berkeley DeepDrive Consortium using forward-facing cameras in passenger vehicles driven in diverse US regions (San Francisco, New York, San Jose, and surrounding highways). It is completely independent of DAWN: zero images originate from web-crawler queries.
* **Taxonomy & Class Alignment:** BDD100K explicitly distinguishes `pedestrian` from `rider`, and annotates `car`, `truck`, `bus`, `motorcycle`, and `bicycle`. This maps directly to our 6 COCO-pretrained detector target classes without semantic ambiguity.
* **Adverse Weather Fidelity:** Contains real-world rain (738 val images) and snow (769 val images) with natural droplets, windshield wiper streaks, road spray, diffuse lighting, and backscatter. Foggy images are present (13 in validation), alongside 1,239 overcast images (which represent low-contrast diffuse adverse driving conditions) and 5,346 clear-weather images (which serve as a vital control).
* **Technical Feasibility:** The full 10,000-image validation split annotations are distributed as a structured JSON file (67.8 MB), enabling exact deterministic stratification across weather and time-of-day before downloading the required evaluation images. Fetching ~200 targeted images consumes under 20 MB of storage, fitting effortlessly within our local hardware envelope (M1 Mac, 8 GB RAM).

### 3.2 RTTS / RESIDE (Selected Secondary Dataset)
* **Provenance:** Sourced as part of the REalistic Single Image DEhazing (RESIDE) benchmark (Li et al., 2019). Contains 4,322 real-world foggy/hazy images annotated in PASCAL VOC format.
* **Strengths:** Excellent depth of coverage for fog and haze (4,322 images).
* **Critical Limitations:** 
  1. Only covers fog/haze; lacks rain, snow, or clear controls.
  2. **Missing truck class:** RTTS only annotates 5 classes (`person`, `car`, `bus`, `bicycle`, `motorbike`). Commercial freight trucks are either unannotated or merged into `bus`/`car`, making it impossible to perform a 6-class detection transfer study.
  3. **Independence concern:** Many RTTS images were obtained by crawling Google, Bing, and Baidu using search queries, creating potential shared-provenance overlap with DAWN.

### 3.3 Foggy Cityscapes (Rejected)
* **Limitation:** Foggy Cityscapes applies a synthetic optical fog simulator based on depth maps to clear-weather German urban driving scenes. Synthetic fog does not replicate natural adverse weather phenomena such as wiper artifacts, water droplets, snowflake occlusions, or vehicle spray.
* **Licensing:** Requires academic portal registration with strict distribution bans; large download footprint (>10 GB).

### 3.4 ACDC (Adverse Conditions Dataset with Correspondences) (Rejected)
* **Limitation:** While ACDC offers real adverse weather (fog, night, rain, snow), its native benchmark is designed for **semantic segmentation** and **panoptic segmentation**, not 2D object detection. Converting semantic masks to bounding boxes introduces boundary artifacts and segmentation-specific errors that would violate our fair detection evaluation protocol.
* **Storage Footprint:** Requires downloading multi-gigabyte high-resolution image and mask archives (>30 GB).

---

## 4. Selection Decision & Resource Budget

### Primary Dataset: BDD100K
We select **BDD100K** as the primary cross-dataset generalization benchmark.

### Resource & Hardware Budget Estimation:
- **Annotation Metadata:** `samples.json` (67.8 MB, downloaded and validated).
- **Evaluation Sample Target:** $N = 203$ images (stratified: 60 rainy, 60 snowy, 13 foggy, 35 overcast, 35 clear control).
- **Image Storage Footprint:** $203 \times 55\text{ KB} \approx 11.2\text{ MB}$.
- **RAM Footprint:** Images processed sequentially in memory (single-image batching); peak memory $< 800\text{ MB}$.
- **CPU Inference Runtime (Apple M1 CPU):**
  - YOLOv8n: $\approx 203 \times 0.12\text{ s} \approx 25\text{ seconds}$.
  - Faster R-CNN ResNet-50 FPN: $\approx 203 \times 2.1\text{ s} \approx 426\text{ seconds}$ ($\approx 7.1\text{ minutes}$).
  - Total evaluation runtime: under 8 minutes.

All original images and raw annotation files will reside in `data/bdd100k/` (strictly excluded via `.gitignore`).
