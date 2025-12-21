"""
Training script for YOLOv8 models on the PCB-IND dataset.

This script reproduces the benchmark results reported in the paper.
All models are initialized with COCO-pretrained weights.

Author: PCB-IND Dataset Contributors
License: MIT
"""

import warnings
from ultralytics import YOLO

warnings.filterwarnings("ignore")


def main():
    # -------------------------------------------------
    # Model configuration
    # -------------------------------------------------
    # Available options: yolov8n.pt, yolov8s.pt, yolov8m.pt
    model_name = "yolov8m.pt"
    model = YOLO(model_name)

    # -------------------------------------------------
    # Dataset configuration
    # -------------------------------------------------
    # The data.yaml file should follow the standard YOLO format
    data_yaml = "data/data.yaml"

    # -------------------------------------------------
    # Training configuration
    # -------------------------------------------------
    model.train(
        data=data_yaml,
        imgsz=640,
        epochs=300,
        batch=32,
        workers=8,
        device=0,
        optimizer="SGD",
        cache=False,

        # Augmentation strategy
        mosaic=1.0,
        close_mosaic=10,
        copy_paste=0.5,

        # Regularization and stability
        patience=50,

        # Logging
        project="runs/train",
        name="yolov8m_pcbind",
    )


if __name__ == "__main__":
    main()
