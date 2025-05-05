#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converts a dataset annotated in LabelMe JSON format to YOLO TXT format,
restructures directories, and removes original JSON files.
"""

import argparse
import json
import logging
import os
import shutil  # Import shutil for directory copying
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
        # Optionally add logging.FileHandler("conversion.log")
    ],
)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to YOLO TXT format."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Path to the root directory containing train/, val/, test/ folders.",
    )
    parser.add_argument(
        "--image_size",
        type=str,
        required=True,
        help="Comma-separated width,height of images (e.g., '640,640').",
    )
    parser.add_argument(
        "--class_map",
        type=str,
        default='{"powerline": 0}',
        help='JSON string mapping label names to class IDs (e.g., \'{"person": 0, "car": 1}\'). Default: \'{"powerline": 0}\'',
    )
    parser.add_argument(
        "--duplicate_original",
        action='store_true',  # Makes it a flag, default is False
        help="If set, duplicates the original 'images' directory (with JSONs) to 'images_labels' before conversion.",
    )
    return parser.parse_args()


def parse_image_size(size_str: str) -> Optional[Tuple[int, int]]:
    """Parses the image_size string into (width, height)."""
    try:
        width, height = map(int, size_str.split(','))
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive.")
        return width, height
    except ValueError as e:
        logging.error(f"Invalid image_size format: '{size_str}'. Error: {e}")
        return None


def parse_class_map(map_str: str) -> Optional[Dict[str, int]]:
    """Parses the class_map JSON string into a dictionary."""
    try:
        class_map = json.loads(map_str)
        if not isinstance(class_map, dict):
            raise ValueError("Class map must be a JSON object.")
        for key, value in class_map.items():
            if not isinstance(key, str) or not isinstance(value, int):
                raise ValueError("Class map must map strings to integers.")

        # Ensure powerline class is in the map
        if "powerline" not in class_map:
            logging.warning(
                "'powerline' not found in class_map. Adding with ID 0.")
            class_map["powerline"] = 0

        return class_map
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Invalid class_map format: '{map_str}'. Error: {e}")
        logging.info("Using default class map: {'powerline': 0}")
        return {"powerline": 0}


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamps a value within a specified range."""
    return max(min_val, min(value, max_val))


def convert_coordinates(
    points: List[List[float]], img_width: int, img_height: int
) -> Optional[Tuple[float, float, float, float]]:
    """
    Converts LabelMe bounding box points to normalized YOLO format.
    Handles coordinate clamping.
    """
    if len(points) != 2 or len(points[0]) != 2 or len(points[1]) != 2:
        logging.warning(
            f"Unexpected shape points format: {points}. Skipping shape.")
        return None

    x1, y1 = points[0]
    x2, y2 = points[1]

    # Ensure coordinates are within image bounds before normalization
    # Handle edge case: Coordinates outside image bounds
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

    # Calculate center, width, height
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    x_center = ((x1 + x2) / 2.0) * dw
    y_center = ((y1 + y2) / 2.0) * dh
    width = abs(x2 - x1) * dw
    height = abs(y2 - y1) * dh

    # Handle edge case: Ensure normalized values are strictly within [0, 1]
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


def process_split(
    split_dir: Path,
    labels_dir: Path,
    img_width: int,
    img_height: int,
    class_map: Dict[str, int],
) -> None:
    """Processes a single data split (train, val, or test)."""
    images_dir = split_dir / "images"
    if not images_dir.is_dir():
        logging.warning(
            f"Images directory not found for split: {split_dir.name}. Skipping.")
        return

    # Idempotency: Create labels directory if it doesn't exist
    labels_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured labels directory exists: {labels_dir}")

    json_files = list(images_dir.glob("*.json"))
    logging.info(f"Found {len(json_files)} JSON files in {images_dir}")

    processed_count = 0
    skipped_count = 0
    for json_path in json_files:
        txt_filename = json_path.stem + ".txt"
        txt_path = labels_dir / txt_filename

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle edge case: Check if image dimensions match expected
            json_img_width = data.get("imageWidth")
            json_img_height = data.get("imageHeight")
            if json_img_width != img_width or json_img_height != img_height:
                logging.warning(
                    f"Image size mismatch in {json_path.name}: "
                    f"Expected ({img_width}x{img_height}), "
                    f"Found ({json_img_width}x{json_img_height}). Using found dimensions for normalization."
                )
                # Use actual dimensions from JSON if they differ but are valid
                current_img_width = json_img_width if json_img_width and json_img_width > 0 else img_width
                current_img_height = json_img_height if json_img_height and json_img_height > 0 else img_height
            else:
                current_img_width = img_width
                current_img_height = img_height

            shapes = data.get("shapes", [])
            yolo_lines: List[str] = []

            # Handle edge case: Empty shapes list
            if not shapes:
                logging.warning(
                    f"No shapes found in {json_path.name}. Creating empty TXT file.")
                # Create empty file
                with open(txt_path, 'w', encoding='utf-8') as f:
                    pass  # Write nothing
            else:
                # Handle edge case: Multi-shapes per JSON
                for shape in shapes:
                    label = shape.get("label")
                    points = shape.get("points")

                    if not label or not points:
                        logging.warning(
                            f"Skipping invalid shape in {json_path.name}: {shape}")
                        continue

                    # Handle edge case: Label not in class map
                    class_id = class_map.get(label)
                    if class_id is None:
                        logging.warning(
                            f"Label '{label}' in {json_path.name} not found in class_map. Skipping shape."
                        )
                        continue

                    # Convert coordinates
                    yolo_coords = convert_coordinates(
                        points, current_img_width, current_img_height)

                    if yolo_coords:
                        x_center, y_center, width, height = yolo_coords
                        yolo_lines.append(
                            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )

                # Write the TXT file
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_lines))
                # Add newline at the end if file is not empty
                if yolo_lines:
                    with open(txt_path, 'a', encoding='utf-8') as f:
                        f.write("\n")

            # Idempotency: Remove JSON file after successful processing
            try:
                json_path.unlink()
                logging.debug(f"Removed JSON file: {json_path}")
                processed_count += 1
            except OSError as e:
                logging.error(f"Failed to remove JSON file {json_path}: {e}")
                skipped_count += 1

        except json.JSONDecodeError:
            logging.error(
                f"Failed to decode JSON file: {json_path}. Skipping.")
            skipped_count += 1
        except IOError as e:
            logging.error(f"File I/O error with {json_path}: {e}. Skipping.")
            skipped_count += 1
        except Exception as e:
            logging.error(
                f"Unexpected error processing {json_path}: {e}. Skipping.")
            skipped_count += 1

    logging.info(f"Finished processing split: {split_dir.name}. "
                 f"Processed: {processed_count}, Skipped/Errors: {skipped_count}")


def main() -> None:
    """Main function to orchestrate the conversion."""
    args = parse_arguments()

    input_root = Path(args.input_root)
    if not input_root.is_dir():
        logging.error(f"Input root directory not found: {input_root}")
        return

    img_size = parse_image_size(args.image_size)
    if img_size is None:
        return
    img_width, img_height = img_size

    class_map = parse_class_map(args.class_map)
    if class_map is None:
        return

    logging.info(f"Starting conversion process for root: {input_root}")
    logging.info(f"Target image size: {img_width}x{img_height}")
    logging.info(f"Class map: {class_map}")
    logging.info(f"Duplicate original directories: {args.duplicate_original}")

    splits = ["train", "val", "test"]

    # --- Duplication Step (if requested) ---
    if args.duplicate_original:
        logging.info("Duplication requested. Copying original directories...")
        for split_name in splits:
            split_dir = input_root / split_name
            source_images_dir = split_dir / "images"
            dest_images_labels_dir = split_dir / "images_labels"

            if not source_images_dir.is_dir():
                logging.warning(
                    f"Source 'images' directory not found for duplication in split: {split_name}. Skipping duplication for this split.")
                continue

            try:
                # Remove destination if it exists to ensure a fresh copy
                if dest_images_labels_dir.exists():
                    logging.warning(
                        f"Destination '{dest_images_labels_dir}' already exists. Removing it before duplication.")
                    shutil.rmtree(dest_images_labels_dir)

                shutil.copytree(source_images_dir, dest_images_labels_dir)
                logging.info(
                    f"Successfully copied '{source_images_dir}' to '{dest_images_labels_dir}'")
            except OSError as e:
                logging.error(
                    f"Error duplicating directory for split {split_name}: {e}. Stopping duplication.")
                # Decide if you want to stop the whole script or just duplication
                # return # Uncomment to stop entire script on duplication error
                continue  # Continue to next split's duplication
        logging.info("Duplication process finished.")
    # --- End Duplication Step ---

    for split_name in splits:
        split_dir = input_root / split_name
        # Define labels_dir here, as it's needed regardless of duplication
        labels_dir = split_dir / "labels"

        if not split_dir.is_dir():
            logging.warning(
                f"Split directory not found: {split_dir}. Skipping processing.")
            continue

        logging.info(f"Processing split: {split_name}")
        # Pass only necessary paths to process_split
        process_split(split_dir, labels_dir, img_width, img_height, class_map)

    logging.info("Dataset conversion completed.")


if __name__ == "__main__":
    main()
