# WeatherGuard — Mission 11: Literature Review & Novelty Positioning

## 1. High-Resolution & Slicing SOTA
- **SAHI (Akyon et al., 2022):** Slicing Aided Hyper Inference uses static grid slicing. It suffers from high computational overhead (4-9x inference cost), rendering it unusable for real-time edge CPU.
- **Clustered Object Detection (Yang et al., ICCV 2019):** Explores candidate clustering but relies on complex region search networks. WeatherGuard demonstrates lightweight connected-component clustering with 0.35ms CPU overhead.
## 2. Thermal ADAS Transfer (FLIR, Teledyne)
- Thermal object detection in adverse weather provides complementary modalities but suffers domain shift due to loss of photometric color and high-beam blooming.
