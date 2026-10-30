"""
==============================================================================
PPE Detection Project - Dataset Conversion & Preprocessing
==============================================================================
This script converts Pascal VOC (XML) annotations to YOLOv8 format (TXT).
It splits the dataset into:
  - 80% Training (used by YOLO to learn)
  - 20% Validation (used to test accuracy / mAP)

Output Structure:
  dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
==============================================================================
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Mapping dataset classes to numeric IDs for YOLO
# 0 = helmet (Worker wearing safety helmet - Compliant)
# 1 = head   (Worker bare head - Non-compliant violation)
# 2 = person (Worker full body)
CLASS_MAPPING = {
    "helmet": 0,
    "head": 1,
    "person": 2
}


def convert_coordinates(size, box):
    """
    Converts Pascal VOC bounding box (xmin, ymin, xmax, ymax)
    into normalized YOLO format (x_center, y_center, width, height).

    Parameters:
        size (tuple): (image_width, image_height)
        box (tuple): (xmin, ymin, xmax, ymax) in pixels

    Returns:
        tuple: (x_center, y_center, width, height) all normalized [0.0 - 1.0]
    """
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box

    # Safety check: avoid division by zero if image dimensions are missing
    if img_w <= 0 or img_h <= 0:
        return 0, 0, 0, 0

    # Calculate box width and height relative to image size
    box_w = (xmax - xmin) / img_w
    box_h = (ymax - ymin) / img_h

    # Calculate center point coordinates relative to image size
    x_center = (xmin + xmax) / (2.0 * img_w)
    y_center = (ymin + ymax) / (2.0 * img_h)

    # Clip values to ensure strictly between 0.0 and 1.0
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    box_w = max(0.0, min(1.0, box_w))
    box_h = max(0.0, min(1.0, box_h))

    return round(x_center, 6), round(y_center, 6), round(box_w, 6), round(box_h, 6)


def parse_xml_annotation(xml_path):
    """
    Parses a single VOC XML file and extracts image size and all bounding boxes.

    Returns:
        tuple: (image_filename, list of [class_id, x, y, w, h])
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract image filename and dimensions
        filename_elem = root.find("filename")
        filename = filename_elem.text if filename_elem is not None else ""

        size_elem = root.find("size")
        if size_elem is None:
            return None, []

        width = float(size_elem.find("width").text)
        height = float(size_elem.find("height").text)

        yolo_annotations = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip().lower()
            if class_name not in CLASS_MAPPING:
                continue

            class_id = CLASS_MAPPING[class_name]
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Skip degenerate/invalid boxes
            if xmax <= xmin or ymax <= ymin:
                continue

            x, y, w, h = convert_coordinates((width, height), (xmin, ymin, xmax, ymax))
            if w > 0 and h > 0:
                yolo_annotations.append((class_id, x, y, w, h))

        return filename, yolo_annotations
    except Exception as e:
        print(f"[Warning] Could not parse {xml_path}: {e}")
        return None, []


def prepare_yolo_dataset(
    raw_images_dir="DataSet/images",
    raw_annotations_dir="DataSet/annotations",
    target_dataset_dir="dataset",
    train_ratio=0.8,
    seed=42
):
    """
    Processes all raw XML annotations, pairs them with images,
    splits them into train/val sets, and writes YOLO .txt label files.
    """
    random.seed(seed)

    raw_img_path = Path(raw_images_dir)
    raw_ann_path = Path(raw_annotations_dir)
    target_path = Path(target_dataset_dir)

    # 1. Target directory layout
    img_train = target_path / "images" / "train"
    img_val = target_path / "images" / "val"
    lbl_train = target_path / "labels" / "train"
    lbl_val = target_path / "labels" / "val"

    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Reading annotations and matching images...")
    print("=" * 60)

    xml_files = list(raw_ann_path.glob("*.xml"))
    print(f"Found {len(xml_files)} total XML annotation files.")

    valid_pairs = []
    class_counts = {name: 0 for name in CLASS_MAPPING}

    for xml_file in xml_files:
        base_stem = xml_file.stem
        # Check matching image (e.g. .png or .jpg)
        img_candidate = raw_img_path / f"{base_stem}.png"
        if not img_candidate.exists():
            img_candidate = raw_img_path / f"{base_stem}.jpg"
        if not img_candidate.exists():
            continue

        filename_in_xml, boxes = parse_xml_annotation(xml_file)
        if not boxes:
            # Skip images with no relevant objects
            continue

        for cls_id, _, _, _, _ in boxes:
            for cls_name, mapped_id in CLASS_MAPPING.items():
                if mapped_id == cls_id:
                    class_counts[cls_name] += 1

        valid_pairs.append((img_candidate, base_stem, boxes))

    print(f"Matched {len(valid_pairs)} valid image-label pairs.")
    print("Total object instances detected:")
    for name, count in class_counts.items():
        print(f"  - {name:8s}: {count:,} boxes")

    # 2. Shuffle and Split 80% Train / 20% Val
    random.shuffle(valid_pairs)
    split_index = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_index]
    val_pairs = valid_pairs[split_index:]

    print("\n" + "=" * 60)
    print(f"STEP 2: Splitting dataset ({int(train_ratio*100)}% Train / {int((1-train_ratio)*100)}% Val)...")
    print(f"  Training samples:   {len(train_pairs):,}")
    print(f"  Validation samples: {len(val_pairs):,}")
    print("=" * 60)

    def write_partition(pairs, dest_img_dir, dest_lbl_dir, partition_name):
        print(f"Writing {partition_name} files...")
        for img_src, stem, boxes in pairs:
            # 1. Move/Copy image to partition folder
            dest_img_file = dest_img_dir / img_src.name
            if not dest_img_file.exists():
                shutil.copy2(img_src, dest_img_file)

            # 2. Write YOLO .txt label file
            dest_lbl_file = dest_lbl_dir / f"{stem}.txt"
            with open(dest_lbl_file, "w") as f:
                for cls_id, x, y, w, h in boxes:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    write_partition(train_pairs, img_train, lbl_train, "TRAIN")
    write_partition(val_pairs, img_val, lbl_val, "VALIDATION")

    # 3. Create data.yaml
    yaml_content = f"""# YOLOv8 PPE Detection Dataset Configuration
# Automatically generated by utils/converter.py

path: {target_path.resolve().as_posix()}
train: images/train
val: images/val

# Class Names
names:
  0: helmet
  1: head
  2: person
"""

    yaml_file = Path("data.yaml")
    with open(yaml_file, "w") as f:
        f.write(yaml_content)

    print("\n" + "=" * 60)
    print(f"SUCCESS: data.yaml created successfully at: {yaml_file.resolve()}")
    print("Dataset conversion and partitioning complete!")
    print("=" * 60)


if __name__ == "__main__":
    prepare_yolo_dataset()
