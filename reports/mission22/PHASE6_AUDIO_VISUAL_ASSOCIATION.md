# MISSION 22 — PHASE 6: Audiovisual Association Mechanics & Spatial Ambiguity Handling

**Investigator:** Mohammad Sayem Uddin (`sayemwork41@gmail.com`)  
**Repository:** `https://github.com/mohammad-sayem-uddin/weatherguard`  
**Branch:** `master`  
**Module:** `src/multimodal/association.py`  
**Unit Tests:** `tests/test_audio_visual_association.py`  
**Date:** November 5, 2026  

---

## 1. Executive Summary

Phase 6 engineered a physically grounded cross-modal association mechanism (`src/multimodal/association.py`) to prevent false identity linking in search-and-rescue UAV operations.

### Key Association Rules:
1. **Never Assume Visible Person is the Distress Source:**  
   In adverse weather, an acoustic distress cry may originate from an occluded or distant victim behind vegetation or rubble, while a completely different, silent bystander is visible in the camera frame.
2. **Strict Spatial Grounding Rules:**  
   - Under single-channel mono audio, **direction of arrival (DOA) and spatial coordinates are mathematically underdetermined**.
   - The engine explicitly sets `target_box = None` and `is_spatially_grounded = False` whenever distress audio occurs without confirmed visual single-target corroboration.
   - Acoustic evidence functions strictly as an **omnidirectional search trigger**, commanding the UAV to initiate an optical spiral search pattern (`ENGAGE_SPIRAL_SEARCH_PATTERN`), not as a localization fix.
3. **Multiple-Person Disambiguation:**  
   If distress audio co-occurs with multiple visible persons ($N > 1$), the engine refuses to arbitrarily assign distress to a random bounding box. It marks the state as `AMBIGUOUS_MULTIPLE_PERSONS`, flags `requires_human_verification = True`, and commands the UAV to hover and maintain surveillance.
4. **Temporal Window Gating:**  
   A maximum temporal discrepancy threshold of $\Delta t \le 1,500\text{ ms}$ rejects asynchronous misalignments, flagging them as `TEMPORAL_MISALIGNMENT`.

---

## 2. Association State Taxonomy & Action Matrix

```
                                  [ Incoming Frame & Audio Buffer ]
                                                  |
                                   (Is |Delta T| <= 1500 ms?)
                                      /               \
                                    No                 Yes
                                   /                     \
                   [ TEMPORAL_MISALIGNMENT ]       (Audio Distress?)
                                                   /              \
                                                 No                Yes
                                                /                    \
                                       (Visible Person?)       (Visible Person?)
                                         /            \          /            \
                                       No             Yes      No             Yes
                                       /                \      /                \
                               [ NO_TARGET ]      [ VISION_ONLY ]  [ AUDIO_ONLY ]   (Person Count?)
                                                                                     /           \
                                                                                  Count=1      Count > 1
                                                                                    /             \
                                                                    [ CORROBORATED ]  [ AMBIGUOUS_MULTI ]
```

| Operational State | Visual Condition | Acoustic Condition | Spatial Coordinates | Human Verification | Commanded Flight Action |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`NO_TARGET`** | No persons visible | No distress audio | `None` | Not Required | `CONTINUE_GRID_SEARCH` |
| **`UNASSOCIATED_AUDIO_ONLY`** | No persons visible | Confident distress cry | `None` | **REQUIRED** | `ENGAGE_SPIRAL_SEARCH_PATTERN` |
| **`UNASSOCIATED_VISION_ONLY`** | 1+ persons visible | No distress audio | Candidate Boxes | If $N > 1$ | `ENGAGE_VISUAL_TRACKING` |
| **`CORROBORATED_SINGLE_PERSON`** | Exactly 1 person visible | Synchronous distress cry | Single Bounding Box | Not Required | `CONFIRMED_VICTIM_LOCATED_DISPATCH_RESCUE` |
| **`AMBIGUOUS_MULTIPLE_PERSONS`** | 2+ persons visible | Synchronous distress cry | `None` (All preserved) | **REQUIRED** | `HOVER_AND_MAINTAIN_VISUAL_SURVEILLANCE` |
| **`TEMPORAL_MISALIGNMENT`** | Any | $|\Delta t| > 1500\text{ ms}$ | `None` | **REQUIRED** | `RESYNCHRONIZE_BUFFERS` |
| **`SENSOR_FAULT`** | Camera/mic failure | Camera/mic failure | `None` | **REQUIRED** | `HOLD_POSITION_MONITOR_SENSORS` |

---

## 3. Mono vs. Microphone-Array Hardware Disambiguation

### 3.1 What Current Mono Audio CAN Do
- Detect emergency Bengali distress cries (*"বাঁচাও"*) with high accuracy ($92.67\%$ F1).
- Detect presence within an omnidirectional acoustic radius ($30-75\text{ meters}$, depending on wind and engine noise).
- Alert the flight controller to switch from fast transit mode ($15\text{ m/s}$) to low-altitude optical spiral search ($2\text{ m/s}$).

### 3.2 What Current Mono Audio CANNOT Do (Non-Negotiable)
- Cannot compute Azimuth or Elevation angle.
- Cannot compute Range or victim GPS coordinates.
- Cannot determine which person in a group is screaming.
- Cannot localize occluded victims under collapsed structures.

### 3.3 Future Microphone-Array Roadmap (Proposed, NOT Implemented)
A 4-channel non-coplanar MEMS microphone array running Steered Response Power with Phase Transform (SRP-PHAT) is documented as a hardware roadmap item. It must **never be represented as an active feature** of the current system.

Phase 6 is complete.
