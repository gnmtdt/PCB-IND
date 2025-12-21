"""
Evaluation script for YOLOv8 models on the PCB-IND dataset.

This script evaluates trained models on the held-out test set and
automatically generates Precision–Recall curves and normalized
confusion matrices for technical validation.

Author: PCB-IND Dataset Contributors
License: MIT
"""

import warnings
from ultralytics import YOLO
from pathlib import Path

warnings.filterwarnings("ignore")


def main():
    # -------------------------------------------------
    # Model configuration
    # -------------------------------------------------
    # Path to the trained model weights (best.pt)
    model_path = Path("runs/train/yolov8n_pcbind/weights/best.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = YOLO(model_path)

    # -------------------------------------------------
    # Dataset configuration
    # -------------------------------------------------
    data_yaml = Path("data/data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    # -------------------------------------------------
    # Evaluation configuration
    # -------------------------------------------------
    save_dir = Path("runs/test")
    experiment_name = "yolov8n_pcbind_test"

    print("Evaluating model on the PCB-IND test set...")
    print(f"Model: {model_path}")
    print(f"Dataset config: {data_yaml}")

    metrics = model.val(
        data=str(data_yaml),
        split="test",          # evaluate strictly on the test set
        imgsz=640,
        batch=32,
        conf=0.001,            # low confidence threshold for full PR curves
        iou=0.6,
        project=str(save_dir),
        name=experiment_name,
        save_json=True,        # COCO-style results for further analysis
        plots=True,            # generate PR curves and confusion matrix
        exist_ok=True,
    )

    # -------------------------------------------------
    # Report overall metrics
    # -------------------------------------------------
    print("\nEvaluation Results (Overall)")
    print("-" * 40)
    print(f"mAP@0.50:       {metrics.box.map50:.4f}")
    print(f"mAP@0.50:0.95:  {metrics.box.map:.4f}")

    # -------------------------------------------------
    # Report per-class mAP
    # -------------------------------------------------
    print("\nPer-Class mAP@0.50:0.95")
    print("-" * 40)
    if metrics.box.maps is not None and len(metrics.box.maps) > 0:
        for idx, ap in enumerate(metrics.box.maps):
            class_name = metrics.names.get(idx, f"class_{idx}")
            print(f"{class_name:<20}: {ap:.4f}")
    else:
        print("Per-class metrics are not available in this run.")

    # -------------------------------------------------
    # Output location
    # -------------------------------------------------
    output_dir = save_dir / experiment_name
    print("\nEvaluation artifacts have been saved to:")
    print(output_dir.resolve())
    print("- Precision–Recall curves")
    print("- Normalized confusion matrix")
    print("- COCO-format prediction results (JSON)")


if __name__ == "__main__":
    main()
