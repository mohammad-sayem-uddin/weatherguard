# Smoke Test Report

- Detector: `fasterrcnn_resnet50_fpn`
- Device: `cpu`
- Weights provenance: Torchvision pretrained weights (COCO 2017 train). Model architecture: Faster R-CNN with ResNet-50 FPN backbone, shipped via torchvision.models.detection.
- License note: Torchvision code: BSD-3-Clause. Pretrained COCO weights: research-use terms per torchvision model registry. See https://github.com/pytorch/vision/blob/main/LICENSE for code and https://pytorch.org/vision/stable/models.html for weight notes.
- n_images: 8 (deterministic, seed=42)
- Detector load seconds: 13.64
- Total inference seconds: 9.32
- Mean per-image inference: 1155.1 ms
- Median per-image inference: 1090.1 ms
- Total detections: 214

## Per-class (IoU=0.5, score>=0.5)

| class | GT | PRED | TP | FP | FN | P | R | F1 | AP@.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Truck | 4 | 3 | 0 | 3 | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| Person | 1 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| Bicycle | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | nan |
| Car | 43 | 41 | 31 | 10 | 12 | 0.756 | 0.721 | 0.738 | 0.683 |
| Motorcycle | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | nan |
| Bus | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | nan |
| **mAP@.5** |  |  |  |  |  |  |  |  | **0.561** |

**Micro** P=0.711 R=0.667 F1=0.688
