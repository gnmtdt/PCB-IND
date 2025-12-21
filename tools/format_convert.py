"""
Dataset format conversion utility for PCB-IND.

This script converts a YOLO-format dataset into:
- YOLO (clean copy with relative paths)
- MS COCO format
- Pascal VOC format

Author: PCB-IND Dataset Contributors
License: MIT
"""

import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml
from PIL import Image

# ============================================================
# Configuration
# ============================================================

# Root directory of the source YOLO dataset
# Expected structure:
# dataset_root/
# ├── images/{train,val,test}
# └── labels/{train,val,test}
SRC_ROOT = Path("data/yolo")

# Output directory for exported datasets
EXPORT_ROOT = Path("dataset_export")

CLASS_NAMES = [
    "mouse_bite",
    "missing_copper",
    "scratch",
    "spurious_copper",
    "copper_burr",
    "stain",
    "short",
    "open",
]

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ============================================================
# Directory Initialization
# ============================================================

YOLO_DIR = EXPORT_ROOT / "YOLO"
COCO_DIR = EXPORT_ROOT / "COCO"
VOC_DIR = EXPORT_ROOT / "VOC"

for split in SPLITS:
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    (COCO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)

(COCO_DIR / "annotations").mkdir(parents=True, exist_ok=True)
(VOC_DIR / "JPEGImages").mkdir(parents=True, exist_ok=True)
(VOC_DIR / "Annotations").mkdir(parents=True, exist_ok=True)
(VOC_DIR / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)

# ============================================================
# YOLO Copy
# ============================================================

def copy_yolo_split(split: str):
    img_src = SRC_ROOT / "images" / split
    lbl_src = SRC_ROOT / "labels" / split

    for img_path in img_src.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        shutil.copy2(img_path, YOLO_DIR / "images" / split / img_path.name)

        label_path = lbl_src / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, YOLO_DIR / "labels" / split / label_path.name)


for split in SPLITS:
    copy_yolo_split(split)

# ============================================================
# YOLO data.yaml
# ============================================================

data_yaml = {
    "path": ".",
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": CLASS_NAMES,
}

with open(YOLO_DIR / "data.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False)

# ============================================================
# YOLO → COCO
# ============================================================

def export_coco(split: str):
    images = []
    annotations = []
    ann_id = 0
    img_id = 0

    img_dir = SRC_ROOT / "images" / split
    lbl_dir = SRC_ROOT / "labels" / split

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        width, height = Image.open(img_path).size

        images.append({
            "id": img_id,
            "file_name": f"images/{split}/{img_path.name}",
            "width": width,
            "height": height,
        })

        with open(label_path) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                x = (xc - bw / 2) * width
                y = (yc - bh / 2) * height
                bw *= width
                bh *= height

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls),
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1

        shutil.copy2(img_path, COCO_DIR / "images" / split / img_path.name)
        img_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)
        ],
    }


for split in SPLITS:
    coco_dict = export_coco(split)
    with open(COCO_DIR / "annotations" / f"{split}.json", "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2)

# ============================================================
# YOLO → VOC
# ============================================================

def yolo_to_voc(img_path: Path, label_path: Path):
    width, height = Image.open(img_path).size

    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = img_path.name

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    with open(label_path) as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())
            xmin = int((xc - bw / 2) * width)
            ymin = int((yc - bh / 2) * height)
            xmax = int((xc + bw / 2) * width)
            ymax = int((yc + bh / 2) * height)

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = CLASS_NAMES[int(cls)]
            box = ET.SubElement(obj, "bndbox")
            ET.SubElement(box, "xmin").text = str(xmin)
            ET.SubElement(box, "ymin").text = str(ymin)
            ET.SubElement(box, "xmax").text = str(xmax)
            ET.SubElement(box, "ymax").text = str(ymax)

    return ET.ElementTree(root)


for split in SPLITS:
    ids = []
    img_dir = SRC_ROOT / "images" / split
    lbl_dir = SRC_ROOT / "labels" / split

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        shutil.copy2(img_path, VOC_DIR / "JPEGImages" / img_path.name)
        tree = yolo_to_voc(img_path, label_path)
        tree.write(VOC_DIR / "Annotations" / f"{img_path.stem}.xml")
        ids.append(img_path.stem)

    with open(VOC_DIR / "ImageSets" / "Main" / f"{split}.txt", "w") as f:
        for i in ids:
            f.write(i + "\n")

print("Dataset export completed: YOLO / COCO / VOC.")
