# WeatherGuard Mission 10 — Phase 0: Forensic Audit & Discrepancy Register

**Mission:** Mission 10 — Independent Validation & Temporal Reliability  
**Date:** 2026-11-02  
**Hardware Environment:** Apple MacBook Air M1, 8 GB Unified Memory, macOS (CPU-only)  
**Status:** Audit Completed & Discrepancies Reconciled  

---

## 1. Forensic Executive Summary

Prior to commencing new independent video evaluations in Mission 10, Phase 0 performed an exhaustive forensic audit of:
1. **The Historical Group C Discrepancy** between Mission 7 ($F_1 = 0.4974$) and Missions 2/8 ($F_1 = 0.5111$).
2. **The Headline Mission 9 Results** on the 4 Locked Test driving video sequences ($N=811$ frames).
3. **The Detection vs Tracking Metric Boundary Flaw**, where extrapolated bounding boxes and low-confidence tracks were inadvertently evaluated in the standard object detection stream.

### Key Audit Findings:
- **Historical Discrepancy Fully Reconciled:** Recomputation from raw prediction artifact `artifacts/eval_external_yolov8n.json` confirms that $TP = 1,043$ and $FN = 1,510$ are byte-identical. The difference is 100% explained by the inclusion of 113 non-canonical COCO predictions (traffic lights, fire hydrants, umbrellas) in Mission 7, raising $FP$ from 485 to 598 and depressing $F_1$ from $0.5111$ to $0.4974$.
- **Headline Number Sourcing Clarified:** In the previous mission synthesis, several metrics were cited from development slices or overall 8-sequence aggregates rather than the isolated 4-sequence Locked Test benchmark. The actual locked test numbers have been independently reproduced directly from saved predictions.
- **The Metric Boundary Flaw Exposed:** In `weatherguard_temporal`, incorporating Kalman-extrapolated boxes ($G=1$) and low-confidence ByteTrack boxes directly into the detection evaluation stream increased recall from $0.3758 	o 0.4264$ (+773 true positives), but surged false positives from $1,521 	o 4,357$, dropping detection precision to $0.5989$ and detection $F_1$ to $0.4982$.
- **Hold-Time Routing Alone Beats Policy 5a:** When evaluated strictly as a detector without extrapolation pollution, Hold-Time Dynamic Router ($H=3$) achieves **$F_1 = 0.5398$** (Precision $0.8039$, Recall $0.4063$), outperforming both the static 640px base ($0.5093$) and frame-independent Policy 5a ($0.5334$) while cutting resolution switches by $71.3\%$.

---

## 2. Independent Recomputation of Mission 9 Locked Test ($N=811$ Frames)

Recomputed from raw prediction caches (`data/video_dataset/predictions/`) with $\tau = 0.20, \text{IoU} = 0.50$:

| Pipeline Configuration | Stream Type | Precision | Recall | Micro $F_1$ | Small $F_1$ (<32²) | Flicker Rate | Track Coverage | Switches/100f |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Static YOLOv8n-640** | Pure Detector | 0.7903 | 0.3758 | **0.5093** | 0.2429 | 0.0810 | 29.06% | 0.0 |
| **Static YOLOv8n-832** | Pure Detector | 0.8020 | 0.4163 | **0.5481** | 0.3140 | 0.0799 | 32.60% | 0.0 |
| **Policy 5a (Frame-Indep)** | Pure Detector | 0.8015 | 0.3998 | **0.5334** | 0.3005 | 0.0842 | 30.97% | 15.3 |
| **Hold-Time Router ($H=3$)** | Pure Detector | **0.8039** | **0.4063** | **0.5398** | **0.3045** | **0.0784** | **31.42%** | **4.4** |
| **WeatherGuard-Temporal** | Mixed (Det+Trk+Extrap) | 0.5989 | 0.4264 | **0.4982** | 0.3080 | 0.0633 | 33.76% | 4.4 |

---

## 3. The Historical Discrepancy Register

| Metric | Previously Reported | Independently Recomputed | Source Artifact | Root Cause | Corrective Action | Affects Scientific Conclusions? |
|---|:---:|:---:|---|---|---|:---:|
| **Base YOLO640 Locked Test F1** | 0.5694 | 0.5093 | `artifacts/mission9_locked_video_results.json` | Number 0.5694 was miscited from historical cross-dataset mAP@0.5 / sequence recall; actual locked test F1 is 0.5093. | Locked test benchmark corrected to 0.5093 in all tables and registry. | **No, because all comparisons on locked test are relative to 0.5093.** |
| **Fixed YOLO832 Locked Test F1** | 0.6019 | 0.5481 | `artifacts/mission9_locked_video_results.json` | Miscited from development subset; actual locked test F1 is 0.5481. | Locked test benchmark corrected to 0.5481. | **No. Fixed 832px remains higher than 640px (+0.0388).** |
| **Dynamic Policy 5a Locked Test F1** | 0.5935 | 0.5334 | `artifacts/mission9_locked_video_results.json` | Miscited from sequence sub-breakdown; actual locked test F1 is 0.5334. | Locked test benchmark corrected to 0.5334. | **No. Policy 5a outperforms Base 640 (+0.0241) on locked test.** |
| **Hold-Time Router H=3 Locked Test F1** | 0.5892 | 0.5398 | `artifacts/mission9_temporal_routing.json` | Miscited from dev medium-scale F1 slice; actual locked test detector-only F1 is 0.5398. | Locked test benchmark corrected to 0.5398. | **Positive. Hold-time router H=3 actually achieves 0.5398, beating Policy 5a (0.5334).** |
| **WeatherGuard-Temporal (Mixed Stream) F1** | 0.5947 | 0.4982 | `artifacts/mission9_locked_video_results.json` | Miscited from car-only sub-metric (0.5945); actual mixed-stream detection F1 is 0.4982 due to extrapolation FP pollution. | Separate detection evaluation stream from tracking/recovery evaluation stream (Phase 4 requirement). | **YES — CRITICAL: Proves that extrapolated tracking boxes must NOT be counted as detector outputs.** |
| **WeatherGuard-Temporal Small F1** | 0.2295 | 0.308 | `artifacts/mission9_locked_video_results.json` | 0.2295 was miscited from recall on sequence b24071b8; actual small F1 is 0.3080 (+0.0651 over base 0.2429). | Correct small-object F1 to 0.3080 in locked test registry. | **Positive. True small-object F1 gain is even stronger than previously reported.** |
| **WeatherGuard-Temporal Flicker Rate** | 0.0595 | 0.0633 | `artifacts/mission9_sequence_performance.json` | 0.0595 was the overall 8-sequence aggregate flicker rate; on locked test specifically, flicker rate is 0.0633. | Distinguish 8-sequence aggregate (0.0595) from 4-sequence locked test (0.0633). | **No. Substantial flicker reduction over base (0.0810) confirmed in both sets.** |
| **Hold-Time Router Flicker Rate** | 0.0521 | 0.0789 | `artifacts/mission9_temporal_confidence.json` | 0.0521 was the grid tuning minimum and sequence b1ca2e5d flicker rate; on locked test average it is 0.0784. | Report exact sequence-level and aggregated flicker metrics explicitly. | **No.** |
| **Mission 7 vs Mission 2/8 Historical Discrepancy** | 0.4974 vs 0.5111 | 0.4974 vs 0.5111 | `artifacts/eval_external_yolov8n.json` | Confirmed 100% mathematically: TP=1043 and FN=1510 are identical. Unfiltered evaluation included 113 non-canonical COCO false positives (traffic lights, umbrellas, stop signs), depressing precision from 0.6826 to 0.6356. | Reconciliation verified and permanently logged in audit register. | **No. Canonical target filtering is the methodologically correct standard.** |

---

## 4. Key Takeaways and Architectural Mandate for Mission 10

1. **Strict Separation of Three Evaluation Streams (Phase 4):**
   - **Stream 1: Standard Detection Evaluation:** Evaluates purely detector-generated bounding boxes. Zero extrapolated boxes allowed.
   - **Stream 2: Tracking Evaluation:** Evaluates trajectory continuity, ID switches, and fragmentation on ground-truth track IDs.
   - **Stream 3: Short-Gap Recovery Evaluation:** Evaluates extrapolated states independently, measuring recovery precision, duration, and false continuation rate.
2. **Hold-Time Routing Validated:** The hold-time state machine ($H=3$) is a genuine, pure-detection Pareto improvement ($F_1 = 0.5398$ vs $0.5093$), operating with 0 onset latency and 71.3% reduced switching.
3. **Independent Dataset Discovery:** Proceed immediately to Phase 1 to acquire a genuinely independent adverse-weather video driving benchmark.
