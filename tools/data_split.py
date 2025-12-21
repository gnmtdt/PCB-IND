#!/usr/bin/env python3
"""
Batch-aware dataset splitting with rarity preservation.

This script splits a YOLO-format dataset into train/val/test subsets while:
1. Preserving production batch and PCB side consistency.
2. Enforcing minimum instance constraints for rare defect classes.
3. Approximating target split ratios using fixed random seeds.

Author: PCB-IND Dataset Contributors
License: MIT
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-aware train/val/test split with rarity constraints"
    )
    parser.add_argument(
        "--src-images",
        type=Path,
        required=True,
        help="Path to source images directory"
    )
    parser.add_argument(
        "--src-labels",
        type=Path,
        required=True,
        help="Path to source YOLO labels directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root directory (YOLO format)"
    )
    parser.add_argument(
        "--splits",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios (default: 0.8 0.1 0.1)"
    )
    parser.add_argument(
        "--seed-range",
        type=int,
        default=500,
        help="Number of random seeds to search (default: 500)"
    )
    return parser.parse_args()


def read_label_counts(label_path: Path) -> Counter:
    counts = Counter()
    if not label_path.exists():
        return counts
    for line in label_path.read_text().splitlines():
        if line.strip():
            cls_id = int(line.split()[0])
            counts[cls_id] += 1
    return counts


def build_groups(
    image_dir: Path,
    label_dir: Path
) -> Tuple[Dict[str, List[Path]], Dict[str, Counter]]:
    """
    Group images by (batch_id, pcb_side).
    """
    groups = defaultdict(list)
    group_boxes = defaultdict(Counter)

    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        name_parts = img_path.stem.split("_")
        if len(name_parts) < 2:
            continue

        batch_id, side = name_parts[0], name_parts[1]
        group_id = f"{batch_id}_{side}"

        groups[group_id].append(img_path)
        label_path = label_dir / f"{img_path.stem}.txt"
        group_boxes[group_id].update(read_label_counts(label_path))

    return groups, group_boxes


def main() -> None:
    args = parse_args()

    split_names = ["train", "val", "test"]
    split_ratios = dict(zip(split_names, args.splits))

    rare_classes = {2, 6, 7}          # scratch, short, open
    min_instances = {7: 10}           # hard constraint for "open"

    groups, group_boxes = build_groups(args.src_images, args.src_labels)
    group_ids = list(groups.keys())
    total_images = sum(len(v) for v in groups.values())

    def count_images(gids: List[str]) -> int:
        return sum(len(groups[g]) for g in gids)

    best_result = None

    for seed in range(args.seed_range):
        random.seed(seed)
        shuffled = group_ids[:]
        random.shuffle(shuffled)

        splits = {k: [] for k in split_names}
        counts = {k: 0 for k in split_names}
        targets = {k: split_ratios[k] * total_images for k in split_names}

        for gid in shuffled:
            n_imgs = len(groups[gid])
            placed = False
            for k in split_names:
                if counts[k] + n_imgs <= targets[k]:
                    splits[k].append(gid)
                    counts[k] += n_imgs
                    placed = True
                    break
            if not placed:
                splits["train"].append(gid)
                counts["train"] += n_imgs

        val_boxes, test_boxes = Counter(), Counter()
        for g in splits["val"]:
            val_boxes.update(group_boxes[g])
        for g in splits["test"]:
            test_boxes.update(group_boxes[g])

        if any(val_boxes[c] < min_instances.get(c, 0) or
               test_boxes[c] < min_instances.get(c, 0)
               for c in min_instances):
            continue

        rarity_score = sum(min(val_boxes[c], test_boxes[c]) for c in rare_classes)
        ratio_penalty = sum(
            abs(counts[k] / total_images - split_ratios[k])
            for k in split_names
        )

        score = rarity_score - 50 * ratio_penalty

        if best_result is None or score > best_result["score"]:
            best_result = {
                "seed": seed,
                "splits": splits,
                "counts": counts,
                "score": score,
                "val_boxes": val_boxes,
                "test_boxes": test_boxes
            }

    if best_result is None:
        raise RuntimeError("No valid split found under given constraints.")

    for split in split_names:
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

        for gid in best_result["splits"][split]:
            for img in groups[gid]:
                shutil.copy2(img, args.output / "images" / split / img.name)
                label = args.src_labels / f"{img.stem}.txt"
                if label.exists():
                    shutil.copy2(label, args.output / "labels" / split / label.name)

    print("Dataset split completed.")
    print(f"Selected seed: {best_result['seed']}")
    print("Image counts:", best_result["counts"])
    print("Rare class instances (val):",
          {c: best_result["val_boxes"][c] for c in rare_classes})
    print("Rare class instances (test):",
          {c: best_result["test_boxes"][c] for c in rare_classes})


if __name__ == "__main__":
    main()
