# Container Intelligent Defect Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://github.com/ultralytics/ultralytics)

## Project Overview
This project is a solution for the **2025 MathorCup Big Data Challenge (Track A)**. It utilizes the state-of-the-art **YOLOv11** model to perform intelligent defect detection and classification on the outer surfaces of shipping containers. 

The model addresses complex background interference (ports, skies, machinery) and multi-scale defect detection, identifying three main types of damage:
- `dent` (Class ID: 0)
- `hole` (Class ID: 1)
- `rusty` (Class ID: 2)

## Features
- **Dual Tasks**: Supports both Binary Classification (Damaged vs. Undamaged) and Object Detection (Localization and specific defect classification).
- **Data Augmentation**: Integrates `Albumentations` for dynamic pixel-level and spatial-level transformations, improving model robustness against varying illuminations and angles.
- **Model Architecture Optimization**: Uses SPPF, C2PSA (spatial attention), and C3k2 modules native to YOLOv11 to enhance feature extraction for both large and micro defects.

## Dataset Preparation
The dataset consists of 3713 images formatted in standard YOLO structure. Ensure your directory structure looks like this:

```text
dataset_dir/
 ├── images/
 │   ├── train/
 │   └── val/
 └── labels/
     ├── train/
     └── val/
```

The `data.yaml` configuration should be:

```yaml
train: /path/to/dataset/images/train
val: /path/to/dataset/images/val
nc: 3
names: ['dent', 'hole', 'rusty']
```

## Installation
```bash
# Clone the repository
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name

# Install dependencies
pip install -r requirements.txt
# Ensure ultralytics is installed
pip install ultralytics albumentations opencv-python
```

## Usage

**1. Train the Model**
```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11m.pt")
    model.train(
        data="data.yaml",
        epochs=300,
        imgsz=640,
        batch=16,
        project="runs/train",
        name="container_defect"
    )
```

**2. Inference / Prediction**
```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/train/container_defect/weights/best.pt")
    results = model.predict(source="path/to/test/images", save=True, conf=0.25)
```

## Results
- **Binary Classification Accuracy**: 84.26% (F1-Score: 0.9146)
- **Object Detection**: mAP@0.5 = 0.541 (Hole detection reached mAP@0.5 of 0.649)
