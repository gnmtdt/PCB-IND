"""
Class distribution analysis for object detection datasets.

This script computes and visualizes the class-wise instance distribution
across training, validation, and test splits under the YOLO-style directory
structure.

Expected dataset structure:
dataset_root/
├── data.yaml
└── labels/
    ├── train/
    ├── val/
    └── test/

The generated figure is suitable for publication-quality dataset analysis.
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Plot style (journal-ready, non-bold)
# ----------------------------------------------------------------------
def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.labelweight": "normal",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    })


# ----------------------------------------------------------------------
# Dataset utilities
# ----------------------------------------------------------------------
def load_dataset_info(data_yaml: Path):
    """
    Load dataset root path and class names from data.yaml.
    """
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data["path"])
    class_names = data["names"]

    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in range(len(class_names))]

    return dataset_root, class_names


def count_instances(label_dir: Path) -> Counter:
    """
    Count instances per class in a YOLO label directory.
    """
    counter = Counter()

    if not label_dir.exists():
        return counter

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0:
                    try:
                        cls_id = int(parts[0])
                        counter[cls_id] += 1
                    except ValueError:
                        continue

    return counter


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
def plot_class_distribution(
    train_counts: Counter,
    val_counts: Counter,
    test_counts: Counter,
    class_names: list,
    save_path: Path,
) -> None:
    set_publication_style()

    num_classes = len(class_names)
    train_vals = np.array([train_counts.get(i, 0) for i in range(num_classes)])
    val_vals = np.array([val_counts.get(i, 0) for i in range(num_classes)])
    test_vals = np.array([test_counts.get(i, 0) for i in range(num_classes)])

    x = np.arange(num_classes)
    width = 0.65

    colors = ["#2594d9", "#f2e291", "#f2b441"]

    fig, ax = plt.subplots(figsize=(12, 7))

    bars_train = ax.bar(
        x, train_vals, width,
        label="Train",
        color=colors[0],
        edgecolor="none",
        zorder=3,
    )

    bars_val = ax.bar(
        x, val_vals, width,
        bottom=train_vals,
        label="Validation",
        color=colors[1],
        edgecolor="none",
        zorder=3,
    )

    bars_test = ax.bar(
        x, test_vals, width,
        bottom=train_vals + val_vals,
        label="Test",
        color=colors[2],
        edgecolor="none",
        zorder=3,
    )

    # Annotate segment values (only if large enough)
    threshold = max(train_vals.max(), 1) * 0.025

    def annotate(bars):
        for bar in bars:
            h = bar.get_height()
            if h > threshold:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + h / 2,
                    f"{int(h)}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                )

    annotate(bars_train)
    annotate(bars_val)
    annotate(bars_test)

    # Total instances per class
    totals = train_vals + val_vals + test_vals
    for i, total in enumerate(totals):
        ax.text(
            x[i],
            total + totals.max() * 0.01,
            f"{total}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.set_ylabel("Number of Instances")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [name.replace("_", " ").title() for name in class_names],
        rotation=35,
        ha="right",
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Figure saved to: {save_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize class distribution for YOLO-style datasets."
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
        default=Path("analysis_outputs/class_distribution.png"),
        help="Output figure path",
    )

    args = parser.parse_args()

    dataset_root, class_names = load_dataset_info(args.data_yaml)
    labels_root = dataset_root / "labels"

    train_counts = count_instances(labels_root / "train")
    val_counts = count_instances(labels_root / "val")
    test_counts = count_instances(labels_root / "test")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(
        train_counts,
        val_counts,
        test_counts,
        class_names,
        args.output,
    )


if __name__ == "__main__":
    main()
