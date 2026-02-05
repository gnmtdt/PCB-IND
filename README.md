# PCB-IND

## Overview
PCB-IND is a real-world industrial printed circuit board (PCB) surface defect dataset collected from mass production lines. The dataset captures complex illumination variations, background clutter, and large-scale differences, providing a realistic benchmark for PCB defect detection research.

The dataset is designed for object detection tasks and is fully compatible with YOLO-based detection frameworks.

## Data Source
The raw images were collected from real industrial production lines provided by **Fujian Fuqiang Delicate Circuit Plate Co., Ltd.**, using automated optical inspection (AOI) systems.

## Dataset Structure
```
PCB-IND/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

## Annotation Format
All annotations follow the YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```
Bounding box coordinates are normalized by image width and height.

## Defect Categories
| ID | Category |
|----|----------|
| 0 | Missing Hole |
| 1 | Mouse Bite |
| 2 | Open Circuit |
| 3 | Short |
| 4 | Spur |
| 5 | Spurious Copper |
| 6 | Pin Hole |
| 7 | Scratch |

## Usage
Example training command using YOLOv8:
```
yolo detect train data=data.yaml model=yolov8n.pt
```

## License
This dataset is released under the **CC BY 4.0** license.


---
Generated on 2025-12-21
