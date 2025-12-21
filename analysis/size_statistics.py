"""
Object size statistics analysis based on MS COCO definitions.

This script analyzes bounding box width–height distributions and categorizes
object instances into Small, Medium, and Large groups according to MS COCO
area thresholds.

Expected dataset structure:
dataset_root/
├── data.yaml
└── labels/
    ├── train/
    ├── val/
    └── test/
"""

import argparse
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Plot style (publication-ready, non-bold)
# ----------------------------------------------------------------------
def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.labelweight": "normal",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    })


# ----------------------------------------------------------------------
# Dataset utilities
# ----------------------------------------------------------------------
def load_dataset_info(data_yaml: Path):
    """
    Load dataset root path and image size from data.yaml.
    """
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data["path"])

    # image size (assume square or first value if list)
    imgsz = data.get("imgsz", 300)
    if isinstance(imgsz, (list, tuple)):
        img_w, img_h = imgsz[0], imgsz[0]
    else:
        img_w, img_h = imgsz, imgsz

    return dataset_root, img_w, img_h


def load_yolo_boxes_pixel(labels_root: Path, img_w: int, img_h: int):
    """
    Load YOLO-format bounding boxes and convert them to pixel scale.
    """
    boxes = []

    for split in ["train", "val", "test"]:
        split_dir = labels_root / split
        if not split_dir.exists():
            continue

        for label_file in split_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    _, _, _, w, h = map(float, parts)

                    boxes.append([w * img_w, h * img_h])

    return np.asarray(boxes)


# ----------------------------------------------------------------------
# MS COCO size categorization
# ----------------------------------------------------------------------
def coco_size_category(w_px: float, h_px: float) -> str:
    area = w_px * h_px

    if area < 32 ** 2:
        return "Small"
    elif area < 96 ** 2:
        return "Medium"
    else:
        return "Large"


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
def plot_wh_distribution_coco(boxes: np.ndarray, save_path: Path):
    set_publication_style()

    size_groups = {"Small": [], "Medium": [], "Large": []}

    for w_px, h_px in boxes:
        size_groups[coco_size_category(w_px, h_px)].append((w_px, h_px))

    total = len(boxes)

    colors = {
        "Small": "#d62728",
        "Medium": "#2ca02c",
        "Large": "#1f77b4",
    }

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for size, pts in size_groups.items():
        if len(pts) == 0:
            continue

        pts = np.asarray(pts)
        ratio = len(pts) / total * 100

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=10,
            alpha=0.5,
            color=colors[size],
            label=f"{size} ({ratio:.2f}%)",
        )

    ax.set_xlabel("Bounding Box Width (pixels)")
    ax.set_ylabel("Bounding Box Height (pixels)")

    max_dim = max(boxes[:, 0].max(), boxes[:, 1].max()) * 1.05
    ax.set_xlim(0, max_dim)
    ax.set_ylim(0, max_dim)

    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    # Print statistics for paper consistency check
    print("[INFO] Object size distribution (MS COCO definition):")
    for size in ["Small", "Medium", "Large"]:
        count = len(size_groups[size])
        ratio = count / total * 100
        print(f"  {size:<6}: {count:6d} ({ratio:.2f}%)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze object size distribution using MS COCO criteria."
    )
    parser.add_argument(
        "--data_yaml",
        type=Path,
        required=True,
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/size_distribution_wh_coco.png"),
        help="Output figure path",
    )

    args = parser.parse_args()

    dataset_root, img_w, img_h = load_dataset_info(args.data_yaml)
    labels_root = dataset_root / "labels"

    boxes = load_yolo_boxes_pixel(labels_root, img_w, img_h)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_wh_distribution_coco(boxes, args.output)


if __name__ == "__main__":
    main()
