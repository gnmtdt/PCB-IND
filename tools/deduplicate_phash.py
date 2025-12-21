"""
Perceptual-hash-based deduplication tool for YOLO-style datasets.

This script removes near-duplicate images across train/val/test splits
to prevent data leakage. When duplicates are detected, images from splits
with higher priority (e.g., train > val > test) are retained.

Author: PCB-IND Dataset Contributors
License: MIT
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

import yaml
import imagehash
from PIL import Image
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

# Path to the source YOLO data.yaml (relative or absolute)
SOURCE_YAML = Path("data/source/data.yaml")

# Output root directory for the deduplicated dataset
TARGET_ROOT = Path("data/deduplicated")

# Hamming distance threshold for perceptual hash
# Images with distance <= threshold are considered duplicates
HAMMING_THRESHOLD = 3

# Split priority: higher value = higher priority to keep
# Used to avoid train-test leakage
SPLIT_PRIORITY = {
    "train": 3,
    "val": 2,
    "test": 1,
}

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}


# ============================================================
# Utility Functions
# ============================================================

def load_dataset_yaml(yaml_path: Path):
    """Load YOLO dataset configuration."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = yaml_path.parent
    root = Path(cfg.get("path", base_dir))

    splits = {}
    for split in ("train", "val", "test"):
        if split in cfg:
            p = Path(cfg[split])
            splits[split] = p if p.is_absolute() else root / p

    class_names = cfg.get("names", {})
    return splits, class_names


def list_images(directory: Path) -> List[Path]:
    """Recursively list all image files under a directory."""
    return [
        p for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def find_label(image_path: Path) -> Path | None:
    """
    Locate the corresponding YOLO label file for an image.
    Assumes standard images/labels directory structure.
    """
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        if label_path.exists():
            return label_path

    alt = image_path.with_suffix(".txt")
    return alt if alt.exists() else None


# ============================================================
# Main Procedure
# ============================================================

def main():
    source_splits, class_names = load_dataset_yaml(SOURCE_YAML)
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    all_items = []

    print("Step 1: Computing perceptual hashes...")
    for split, split_dir in source_splits.items():
        images = list_images(split_dir)
        for img_path in tqdm(images, desc=f"Hashing [{split}]"):
            try:
                with Image.open(img_path) as img:
                    ph = imagehash.phash(img)

                batch_id = img_path.stem[:4]  # batch identifier heuristic

                all_items.append({
                    "path": img_path,
                    "hash": ph,
                    "split": split,
                    "priority": SPLIT_PRIORITY.get(split, 0),
                    "batch": batch_id,
                    "name": img_path.name,
                })
            except Exception:
                continue

    print("Step 2: Deduplication with split-priority constraints...")
    all_items.sort(key=lambda x: (x["batch"], -x["priority"]))

    kept_items = []
    seen_hashes: Dict[str, List] = {}
    removed = 0

    for item in tqdm(all_items, desc="Deduplicating"):
        batch = item["batch"]
        h = item["hash"]

        if batch not in seen_hashes:
            seen_hashes[batch] = []

        duplicate = any((h - eh) <= HAMMING_THRESHOLD for eh in seen_hashes[batch])

        if duplicate:
            removed += 1
        else:
            kept_items.append(item)
            seen_hashes[batch].append(h)

    print(f"Total images scanned: {len(all_items)}")
    print(f"Images retained: {len(kept_items)}")
    print(f"Duplicates removed: {removed}")

    print("Step 3: Copying retained images and labels...")
    yaml_splits = {}

    for item in tqdm(kept_items, desc="Copying files"):
        split = item["split"]

        img_dst = TARGET_ROOT / "images" / split
        lbl_dst = TARGET_ROOT / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        shutil.copy2(item["path"], img_dst / item["name"])

        label_path = find_label(item["path"])
        if label_path:
            shutil.copy2(label_path, lbl_dst / label_path.name)

        yaml_splits[split] = f"images/{split}"

    print("Step 4: Writing new data.yaml...")
    new_yaml = {
        "path": str(TARGET_ROOT.as_posix()),
        **yaml_splits,
        "names": class_names,
    }

    with open(TARGET_ROOT / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(new_yaml, f, sort_keys=False)

    print("Deduplicated dataset is ready.")
    print(f"Output directory: {TARGET_ROOT.resolve()}")


if __name__ == "__main__":
    main()
