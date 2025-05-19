#!/usr/bin/env python3

import os
import json
import shutil
import random
from pathlib import Path
import logging

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


def clamp(value, min_val, max_val):
    """Clamp value between min_val and max_val."""
    return max(min_val, min(max_val, value))


def convert_labelme_to_yolo(json_file, img_width, img_height):
    """Convert labelme format to YOLO format."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    yolo_lines = []

    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = shape.get("points")

        if not label or not points:
            continue

        # Get class index
        try:
            class_idx = CLASS_MAP.get(label, None)
            if class_idx is None:
                logging.warning(
                    f"Unknown class '{label}' in {json_file}, skipping")
                continue
        except (ValueError, KeyError):
            logging.warning(
                f"Unknown class '{label}' in {json_file}, skipping")
            continue

        # Convert to YOLO format
        if shape.get("shape_type") == "rectangle" and len(points) == 2:
            # Rectangle with 2 points [top-left, bottom-right]
            x1, y1 = points[0]
            x2, y2 = points[1]
        else:
            # For other shapes, find bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)

        # Clamp coordinates to image boundaries
        x1 = clamp(x1, 0, img_width)
        x2 = clamp(x2, 0, img_width)
        y1 = clamp(y1, 0, img_height)
        y2 = clamp(y2, 0, img_height)

        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        # Add to lines
        yolo_lines.append(
            f"{class_idx} {x_center} {y_center} {width} {height}")

    return yolo_lines


def create_class_files():
    """Create classes.txt and yaml file."""
    # Create classes.txt
    with open(TARGET_DIR / "classes.txt", 'w') as f:
        for class_name in CLASS_MAP.keys():
            f.write(f"{class_name}\n")

    # Create yaml file
    yaml_content = f"""
path: {os.path.abspath(TARGET_DIR)}
train: train/images
val: val/images

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.keys())}
"""

    with open(TARGET_DIR / "dataset.yaml", 'w') as f:
        f.write(yaml_content)


def main():
    """Main function to process the dataset."""
    # Create directories
    train_images_dir = TARGET_DIR / "train/images"
    train_labels_dir = TARGET_DIR / "train/labels"
    val_images_dir = TARGET_DIR / "val/images"
    val_labels_dir = TARGET_DIR / "val/labels"

    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Create class files
    create_class_files()

    # Get all JSON files
    json_files = list(SOURCE_DIR.glob("*.json"))
    logging.info(f"Found {len(json_files)} JSON files")

    # Randomize and split
    random.shuffle(json_files)
    train_size = int(len(json_files) * TRAIN_SPLIT)

    train_files = json_files[:train_size]
    val_files = json_files[train_size:]

    logging.info(
        f"Split into {len(train_files)} training and {len(val_files)} validation files")

    # Process training files
    for json_file in train_files:
        process_file(json_file, train_images_dir, train_labels_dir)

    # Process validation files
    for json_file in val_files:
        process_file(json_file, val_images_dir, val_labels_dir)

    logging.info("Dataset processing complete")


def process_file(json_file, images_dir, labels_dir):
    """Process a single file and copy to the right location."""
    try:
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Get image info
        image_path = json_file.with_suffix('.jpg')
        img_width = data.get("imageWidth", 640)
        img_height = data.get("imageHeight", 640)

        # Check if image exists
        if not image_path.exists():
            logging.warning(f"Image file not found: {image_path}")
            return

        # Convert to YOLO format
        yolo_lines = convert_labelme_to_yolo(json_file, img_width, img_height)

        if not yolo_lines:
            logging.warning(f"No valid annotations in {json_file}")
            return

        # Copy image
        dest_image = images_dir / image_path.name
        shutil.copy2(image_path, dest_image)

        # Save YOLO labels
        txt_path = labels_dir / image_path.with_suffix('.txt').name
        with open(txt_path, 'w') as f:
            for line in yolo_lines:
                f.write(f"{line}\n")

    except Exception as e:
        logging.error(f"Error processing {json_file}: {e}")


if __name__ == "__main__":
    main()
