# WeatherGuard — Mission 11: Structured Failure Analysis

## Audited Failure Conditions

### [FM-01] First-Stage Candidate Miss (Vanishing Point Truncation)
- **Behavior:** Ultra-distant sub-16px vehicles along vanishing line produce zero activations at 640px; candidate generator proposes no crop.
- **Mitigation:** Fixed-grid horizon strip or hybrid road band prior.
- **Impact:** Caps candidate-driven ROI small-object recall at 28.5%.

### [FM-02] Crop Boundary Truncation Artifacts
- **Behavior:** Vehicles severed in half by crop boundary trigger partial-car false detections.
- **Mitigation:** Boundary margin suppression (margin=4px, conf<0.35).
- **Impact:** Eliminated 84.6% of boundary false positives.

### [FM-03] Specular Water Reflection Hallucination
- **Behavior:** High-magnification crops on wet asphalt magnify puddle reflections into car-like bounding boxes.
- **Mitigation:** Conservative merge requiring new_crop_conf >= 0.35.
- **Impact:** Suppressed 74 false alarms on wet pavement.

### [FM-04] Thermal Contrast Inversion & Hot Exhaust Blooming
- **Behavior:** Hot car exhausts and warm engine hoods bloom into saturated thermal blobs, washing out wheel and windshield contours.
- **Mitigation:** Multi-scale feature normalization; thermal class prior.
- **Impact:** Depresses visible-to-thermal zero-shot transfer precision.

