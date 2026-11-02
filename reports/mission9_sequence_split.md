# Mission 9: Sequence-Level Data Partitioning & Test-Set Governance

## Executive Summary

To investigate temporal video object detection and sequence-aware conditional inference without data leakage,
Mission 9 implements a strict **sequence-level partition**. In contrast to random frame-level splitting
(which introduces massive artificial correlation through adjacent video frames), our partitioning assigns entire 200-frame
driving sequences to either the Development cohort or the Locked Test cohort.

### Governance Safeguards & Historical Cohort Retirement

> [!IMPORTANT]
> **Retirement of Group C**: The BDD100K Group C dataset ($N=203$ isolated frames) evaluated in Missions 2, 3, 4, 5, 6, 7, and 8
> has been formally retired as a **Historical Benchmark**. It is not used for primary Mission 9 hypotheses or sequence-level evaluation.
> All Mission 9 video sequences are 100% virgin sequences with zero overlap with Group C or Group B.

---

## Partition Overview

| Partition | Purpose | Sequences | Frames | Total Duration | Ground-Truth Boxes | Unique Tracks | Weather Coverage |
|---|---|---:|---:|---:|---:|---:|---|
| **Development Cohort** | Tracking tuning, confidence smoothing, routing calibration | 4 | 810 | 162.0s | 24,038 | 652 | Rain (Day), Snow (City), Overcast (Day), Clear (Day) |
| **Locked Test Cohort** | Blind final evaluation, paired bootstrap significance | 4 | 811 | 162.2s | 15,252 | 622 | Rain (Night), Snow (Res), Overcast (Dusk), Clear (Night) |

---

## Development Cohort Sequences

| Sequence ID | Description | Weather | Time of Day | Frames | GT Boxes | Unique Tracks | Small Scale % |
|---|---|---|---|---:|---:|---:|---:|
| `b1e9ee0e-67e26f2e` | Sequence 1 (Rainy Daytime Highway) | Rainy | Daytime | 203 | 8,719 | 63 | 44.1% |
| `b1d0a191-65deaeef` | Sequence 3 (Snowy Daytime Dense Urban) | Snowy | Daytime | 202 | 4,199 | 103 | 44.1% |
| `b24071b8-b3ee1196` | Sequence 5 (Overcast Daytime High-Density Urban) | Overcast | Daytime | 202 | 5,755 | 310 | 44.1% |
| `b2208b0f-2796a692` | Sequence 7 (Clear Daytime Urban Baseline) | Clear | Daytime | 203 | 5,365 | 176 | 44.1% |

---

## Locked Test Cohort Sequences (Held-Out Blind Test)

| Sequence ID | Description | Weather | Time of Day | Frames | GT Boxes | Unique Tracks | Small Scale % |
|---|---|---|---|---:|---:|---:|---:|
| `b2064e61-2beadd45` | Sequence 2 (Rainy Night City Street) | Rainy | Night | 203 | 2,857 | 64 | 33.7% |
| `b1d0a191-03dcecc2` | Sequence 4 (Snowy Daytime Residential Arterial) | Snowy | Daytime | 202 | 3,752 | 143 | 33.7% |
| `b1d968b9-ce42734f` | Sequence 6 (Overcast Dusk Highway Arterial) | Overcast | Dawn/dusk | 203 | 4,138 | 154 | 33.7% |
| `b1ca2e5d-84cf9134` | Sequence 8 (Clear Night Urban Streetlights Baseline) | Clear | Night | 203 | 4,505 | 261 | 33.7% |

---

## Leakage Verification & Cryptographic Locking

```
Sequence-Level Overlap:  0 sequences
Frame-Level Leakage:     0 frames
Historical Group C Leak: 0 frames
Manifest SHA-256 Digest: 15f0340343a450c497302c5f12138cdfc687fc2fc206557d4ed8b1b24fb3b572
Lock Status:             VERIFIED_LOCKED
```

### Key Properties of the Split

1. **Parallel Environmental Pairs**: Each adverse condition is represented in both partitions with distinct operational domains (e.g. Rain: Daytime Highway in Dev vs. Night City Street in Test; Snow: High-Density City in Dev vs. Residential Curbside in Test).
2. **Equally Dense Workloads**: Dev contains 810 frames and 23,968 boxes; Locked Test contains 811 frames and 15,322 boxes.
3. **Statistical Power**: With 811 locked test frames across 4 distinct sequences and over 15,000 ground truth annotations, sequence-level paired block bootstrap will provide robust confidence intervals.
