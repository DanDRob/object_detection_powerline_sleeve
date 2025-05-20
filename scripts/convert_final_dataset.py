#!/usr/bin/env python3

import os
import json
import shutil
import random
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Define paths
DATA_DIR = Path("data/powerlines")
SOURCE_DIR = DATA_DIR / "final_dataset/images_labels"
TARGET_DIR = DATA_DIR / "final_dataset_yolo"

# Splits
TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.15

# Class map
CLASS_MAP = {"powerline": 0}


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(value, max_val))


def convert_coordinates(
    points: List[List[float]], img_width: int, img_height: int
) -> Optional[Tuple[float, float, float, float]]:
    if len(points) != 2 or len(points[0]) != 2 or len(points[1]) != 2:
        logging.warning(
            f"Unexpected shape points format: {points}. Skipping shape.")
        return None
    x1, y1 = points[0]
    x2, y2 = points[1]
    orig_coords = (x1, y1, x2, y2)
    x1 = clamp(x1, 0.0, float(img_width))
    y1 = clamp(y1, 0.0, float(img_height))
    x2 = clamp(x2, 0.0, float(img_width))
    y2 = clamp(y2, 0.0, float(img_height))
    if (x1, y1, x2, y2) != orig_coords:
        logging.warning(
            f"Clamped coordinates outside bounds: {orig_coords} to {(x1, y1, x2, y2)} "
            f"for image size ({img_width}x{img_height})"
        )
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    x_center = ((x1 + x2) / 2.0) * dw
    y_center = ((y1 + y2) / 2.0) * dh
    width = abs(x2 - x1) * dw
    height = abs(y2 - y1) * dh
    x_center_norm = clamp(x_center)
    y_center_norm = clamp(y_center)
    width_norm = clamp(width)
    height_norm = clamp(height)
    if (x_center, y_center, width, height) != (x_center_norm, y_center_norm, width_norm, height_norm):
        logging.warning(
            f"Clamped normalized YOLO values for points {orig_coords}. "
            f"Original: ({x_center:.6f}, {y_center:.6f}, {width:.6f}, {height:.6f}), "
            f"Clamped: ({x_center_norm:.6f}, {y_center_norm:.6f}, {width_norm:.6f}, {height_norm:.6f})"
        )
    return x_center_norm, y_center_norm, width_norm, height_norm


def convert_labelme_to_yolo(
    data: Dict[str, Any], img_width: int, img_height: int, class_map: Dict[str, int], json_path: Path
) -> List[str]:
    shapes = data.get("shapes", [])
    yolo_lines: List[str] = []
    if not shapes:
        logging.warning(
            f"No shapes found in {json_path.name}. Creating empty TXT file.")
        return yolo_lines
    for shape in shapes:
        label = shape.get("label")
        points = shape.get("points")
        if not label or not points:
            logging.warning(
                f"Skipping invalid shape in {json_path.name}: {shape}")
            continue
        class_id = class_map.get(label)
        if class_id is None:
            logging.warning(
                f"Label '{label}' in {json_path.name} not found in class_map. Skipping shape.")
            continue
        yolo_coords = convert_coordinates(points, img_width, img_height)
        if yolo_coords:
            x_center, y_center, width, height = yolo_coords
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_lines


def create_class_files():
    with open(TARGET_DIR / "classes.txt", 'w') as f:
        for class_name in CLASS_MAP.keys():
            f.write(f"{class_name}\n")
    yaml_content = f"""
path: {os.path.abspath(TARGET_DIR)}
train: train/images
val: val/images

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.keys())}
"""
    with open(TARGET_DIR / "dataset.yaml", 'w') as f:
        f.write(yaml_content)


def process_file(json_file: Path, images_dir: Path, labels_dir: Path, class_map: Dict[str, int]):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        image_path = json_file.with_suffix('.jpg')
        img_width = data.get("imageWidth", 640)
        img_height = data.get("imageHeight", 640)
        if not image_path.exists():
            logging.warning(f"Image file not found: {image_path}")
            return
        yolo_lines = convert_labelme_to_yolo(
            data, img_width, img_height, class_map, json_file)
        dest_image = images_dir / image_path.name
        shutil.copy2(image_path, dest_image)
        txt_path = labels_dir / image_path.with_suffix('.txt').name
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON file: {json_file}. Skipping.")
    except IOError as e:
        logging.error(f"File I/O error with {json_file}: {e}. Skipping.")
    except Exception as e:
        logging.error(
            f"Unexpected error processing {json_file}: {e}. Skipping.")


def main():
    train_images_dir = TARGET_DIR / "train/images"
    train_labels_dir = TARGET_DIR / "train/labels"
    val_images_dir = TARGET_DIR / "val/images"
    val_labels_dir = TARGET_DIR / "val/labels"
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    create_class_files()
    json_files = list(SOURCE_DIR.glob("*.json"))
    logging.info(f"Found {len(json_files)} JSON files")
    random.shuffle(json_files)
    train_size = int(len(json_files) * TRAIN_SPLIT)
    train_files = json_files[:train_size]
    val_files = json_files[train_size:]
    logging.info(
        f"Split into {len(train_files)} training and {len(val_files)} validation files")
    for json_file in train_files:
        process_file(json_file, train_images_dir, train_labels_dir, CLASS_MAP)
    for json_file in val_files:
        process_file(json_file, val_images_dir, val_labels_dir, CLASS_MAP)
    logging.info("Dataset processing complete")


if __name__ == "__main__":
    main()
