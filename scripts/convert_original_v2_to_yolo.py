#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converts the labelme JSON annotations in the original_v2 folder to YOLO txt format.
Creates a structure that matches the original folder with test/train/val containing 
images/ and labels/ subdirectories.
"""

import os
import json
import shutil
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
SOURCE_DIR = DATA_DIR / "original_v2"
TARGET_DIR = DATA_DIR / "original_v2_yolo"

# Class map (same as the original)
CLASS_MAP = {"powerline": 0}


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamps a value within a specified range."""
    return max(min_val, min(value, max_val))


def convert_coordinates(points, img_width, img_height):
    """Converts LabelMe bounding box points to normalized YOLO format."""
    if len(points) != 2 or len(points[0]) != 2 or len(points[1]) != 2:
        logging.warning(
            f"Unexpected shape points format: {points}. Skipping shape.")
        return None

    x1, y1 = points[0]
    x2, y2 = points[1]

    # Ensure coordinates are within image bounds
    x1 = clamp(x1, 0.0, float(img_width))
    y1 = clamp(y1, 0.0, float(img_height))
    x2 = clamp(x2, 0.0, float(img_width))
    y2 = clamp(y2, 0.0, float(img_height))

    # Calculate center, width, height (normalized)
    x_center = ((x1 + x2) / 2.0) / img_width
    y_center = ((y1 + y2) / 2.0) / img_height
    width = abs(x2 - x1) / img_width
    height = abs(y2 - y1) / img_height

    # Ensure normalized values are within [0, 1]
    x_center = clamp(x_center)
    y_center = clamp(y_center)
    width = clamp(width)
    height = clamp(height)

    return x_center, y_center, width, height


def process_json_file(json_path, image_output_dir, labels_output_dir):
    """Process a single JSON file to extract the image and create YOLO annotation."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Get image filename from JSON
        image_filename = data.get("imagePath")
        if not image_filename:
            logging.warning(f"No image path in {json_path}")
            return False

        image_name = os.path.splitext(image_filename)[0]
        img_width = data.get("imageWidth", 640)
        img_height = data.get("imageHeight", 640)

        # Look for the image file in the same directory
        json_dir = json_path.parent
        image_path = json_dir / image_filename

        if not image_path.exists():
            # Try to look for jpg
            jpg_path = json_dir / f"{image_name}.jpg"
            if jpg_path.exists():
                image_path = jpg_path
            else:
                logging.warning(f"Image file not found for {json_path}")
                return False

        # Copy the image file to the output directory
        output_image_path = image_output_dir / image_filename
        if not output_image_path.exists():
            shutil.copy2(image_path, output_image_path)
            logging.debug(f"Copied image to {output_image_path}")

        # Create YOLO annotation file
        output_label_path = labels_output_dir / f"{image_name}.txt"

        yolo_lines = []

        # Process each shape (annotation)
        for shape in data.get("shapes", []):
            # Get label and points
            label = shape.get("label")
            points = shape.get("points")

            # Skip if label or points are missing
            if not label or not points:
                continue

            # Get class index
            class_idx = CLASS_MAP.get(label)
            if class_idx is None:
                logging.warning(
                    f"Unknown class '{label}' in {json_path}, skipping")
                continue

            # Convert to YOLO format
            yolo_coords = convert_coordinates(points, img_width, img_height)
            if yolo_coords:
                x_center, y_center, width, height = yolo_coords
                yolo_lines.append(
                    f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Write to file in YOLO format
        with open(output_label_path, 'w') as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:  # Add newline at the end if file is not empty
                f.write("\n")

        logging.debug(f"Created YOLO annotation at {output_label_path}")
        return True

    except Exception as e:
        logging.error(f"Error processing {json_path}: {e}")
        return False


def process_split(split_name):
    """Process a single data split (train, val, or test)."""
    split_dir = SOURCE_DIR / split_name

    if not split_dir.exists():
        logging.warning(f"Split directory not found: {split_dir}")
        return 0

    # Create output directories
    target_split_dir = TARGET_DIR / split_name
    images_dir = target_split_dir / "images"
    labels_dir = target_split_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_labels_dir = split_dir / "images_labels"
    if not images_labels_dir.exists():
        logging.warning(
            f"images_labels directory not found: {images_labels_dir}")
        return 0

    json_files = list(images_labels_dir.glob("*.json"))
    logging.info(f"Found {len(json_files)} JSON files in {images_labels_dir}")

    processed_count = 0
    for json_path in json_files:
        if process_json_file(json_path, images_dir, labels_dir):
            processed_count += 1

    return processed_count


def main():
    """Main function to orchestrate the conversion."""
    logging.info(f"Starting conversion from {SOURCE_DIR} to {TARGET_DIR}")

    # Create target directory
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Process each data split
    total_processed = 0
    for split_name in ["train", "val", "test"]:
        processed = process_split(split_name)
        logging.info(f"Processed {processed} files in {split_name} split")
        total_processed += processed

    # Copy the classes.txt and yaml files from the original folder
    original_dir = DATA_DIR / "original"
    if original_dir.exists():
        # Copy classes.txt
        classes_file = original_dir / "classes.txt"
        if classes_file.exists():
            shutil.copy2(classes_file, TARGET_DIR / "classes.txt")
            logging.info(f"Copied classes.txt from {original_dir}")

        # Copy and rename yaml file
        yaml_files = list(original_dir.glob("*.yaml"))
        if yaml_files:
            yaml_file = yaml_files[0]
            shutil.copy2(yaml_file, TARGET_DIR / "original_v2.yaml")
            logging.info(f"Copied and renamed {yaml_file} to original_v2.yaml")

    logging.info(f"Conversion completed. Processed {total_processed} files.")


if __name__ == "__main__":
    main()
