"""
t-SNE visualization of instance-level visual features.

This script extracts deep visual embeddings from cropped defect regions
using a pretrained ResNet-18 backbone and visualizes their distribution
in a 2D space via t-SNE.

The visualization aims to qualitatively analyze inter-class separability
and intra-class compactness of PCB surface defect categories.
"""

import argparse
from pathlib import Path
import random

import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.manifold import TSNE


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Plot style (publication-ready)
# ----------------------------------------------------------------------
def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.labelweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    })


# ----------------------------------------------------------------------
# Dataset utilities
# ----------------------------------------------------------------------
def load_dataset_info(data_yaml: Path):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data["path"])
    class_names = data["names"]

    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in range(len(class_names))]

    return dataset_root, class_names


# ----------------------------------------------------------------------
# Feature extractor
# ----------------------------------------------------------------------
def build_feature_extractor(device: torch.device):
    """
    Load ImageNet-pretrained ResNet-18 and remove classification head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model


def build_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def extract_instance_features(
    dataset_root: Path,
    class_names: list,
    model,
    preprocess,
    device,
    max_samples_per_class: int,
):
    features = []
    labels = []

    sample_counter = {i: 0 for i in range(len(class_names))}

    for split in ["train", "val", "test"]:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

        for img_path in tqdm(img_files, desc=f"Extracting ({split})"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            img_w, img_h = image.size

            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                cls, cx, cy, w, h = map(float, line.strip().split())
                cls = int(cls)

                if sample_counter[cls] >= max_samples_per_class:
                    continue

                x1 = int((cx - w / 2) * img_w)
                y1 = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)

                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue

                crop = image.crop((x1, y1, x2, y2))
                input_tensor = preprocess(crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    feat = model(input_tensor).cpu().numpy().squeeze()

                features.append(feat)
                labels.append(cls)
                sample_counter[cls] += 1

    return np.asarray(features), np.asarray(labels)


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
def plot_tsne(features, labels, class_names, save_path: Path):
    set_publication_style()

    tsne = TSNE(
        n_components=2,
        perplexity=40,
        n_iter=1500,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )

    embedding = tsne.fit_transform(features)

    colors = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
        "#CCB974", "#64B5CD",
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    for cls_id, cls_name in enumerate(class_names):
        idx = labels == cls_id
        if np.sum(idx) == 0:
            continue

        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s=28,
            alpha=0.7,
            color=colors[cls_id % len(colors)],
            label=cls_name,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(
        title="Defect Categories",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"[INFO] t-SNE visualization saved to: {save_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of PCB defect visual features."
    )
    parser.add_argument("--data_yaml", type=Path, required=True)
    parser.add_argument(
        "--max_samples_per_class", type=int, default=300,
        help="Maximum number of instances per class."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/tsne_visual_features.png"),
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root, class_names = load_dataset_info(args.data_yaml)

    model = build_feature_extractor(device)
    preprocess = build_preprocess()

    features, labels = extract_instance_features(
        dataset_root,
        class_names,
        model,
        preprocess,
        device,
        args.max_samples_per_class,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_tsne(features, labels, class_names, args.output)


if __name__ == "__main__":
    main()
