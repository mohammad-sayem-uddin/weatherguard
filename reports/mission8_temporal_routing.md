# WeatherGuard Mission 8: Temporal Routing Stability Analysis

**Document ID:** `WG-M8-TEMP-001`  
**Date:** November 2026  
**Status:** Completed (with documented dataset limitation)  
**Evaluation Scope:** BDD100K Dataset Video Structure Audit & Controlled Stream Simulation  

---

## 1. Executive Summary & Dataset Governance Audit

Experiment F investigated whether continuous temporal information can stabilize dynamic resolution routing decisions and prevent high-frequency switching oscillations.

### Data Continuity Audit:
- **Group B Development ($N=160$):** Contains 148 unique 8-character video prefix IDs (86.9% singletons).
- **Group C Locked Test ($N=203$):** Contains 197 unique 8-character video prefix IDs (94.1% singletons).
- **Factual Limitation:** BDD100K 100K detection images are single 10-second keyframes sampled from independent driving journeys, not continuous video streams. Evaluating temporal routing directly on Group B or Group C would violate scientific integrity by pretending independent driving trips are sequential video frames.
- **Protocol Adherence:** Following Section 7 of the protocol, the data limitation is documented honestly, and algorithms are evaluated via controlled temporal stream analysis.

---

## 2. Temporal Routing Algorithms & Stability Analysis

We evaluated five temporal stabilization policies on a 100-frame fluctuating driving stream:

| Routing Policy | Escalation Rate | Switching Frequency | Total Switches | Mean Escalated Run | Stability Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Frame-Independent (Policy 5a)** | 50.0% | 10.1% | 10 | 10.0 frames | High switching jitter (toggles rapidly) |
| **Moving Average (W=5)** | 50.0% | 6.1% | 6 | 16.7 frames | Moderate smoothing |
| **Exponential Moving Avg (EMA)** | 50.0% | 4.0% | 4 | 25.0 frames | Strong smoothing |
| **Hysteresis (Schmitt Trigger)** | 67.0% | 4.0% | 4 | 33.5 frames | **Optimal:** Drastically cuts switches |
| **Hold-Time Gating (hold=3)** | 63.0% | 6.1% | 6 | 21.0 frames | Guaranteed minimum stability |

---

## 3. Reaction Speed vs. Stability Trade-Off (Step Response)

When a sudden adverse-weather condition or small vehicle appears abruptly (tested via a clean step function at Frame 40):
- **Frame-Independent (Policy 5a):** Reacts instantaneously (**0 frames delay**, $0.0\text{ ms}$).
- **Hold-Time Gating:** Reacts instantaneously (**0 frames delay**, $0.0\text{ ms}$) on onset, while holding state to prevent flapping.
- **Hysteresis:** Reacts instantaneously (**0 frames delay**) because onset requires crossing $\tau_{\text{high}} = 1.0$.
- **Moving Average ($W=5$):** Introduces a **2 frames delay** (66.6 ms delay at 30 FPS) while the sliding window fills with high-uncertainty values.
- **EMA ($\alpha = 0.35$):** Introduces a **1 frames delay** (33.3 ms delay).

### Deployment Recommendation:
**Hysteresis Routing** is strictly superior to Moving Average or EMA filtering for safety-critical edge driving. It eliminates high-frequency switching oscillations without introducing any onset delay when a small pedestrian or obstacle suddenly emerges.
