#!/usr/bin/env python3

import os
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def create_empty_json_template(image_path: Path, img_width: int = 640, img_height: int = 640) -> dict:
    """Create an empty labelme JSON template for an image."""
    return {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Get image dimensions. Returns default 640x640 if unable to read."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except ImportError:
        logging.warning("PIL not available, using default dimensions 640x640")
        return 640, 640
    except Exception as e:
        logging.warning(
            f"Could not read image dimensions for {image_path}: {e}. Using default 640x640")
        return 640, 640


def main():
    test_dir = Path("data/sleeves/sleeves_v4")

    if not test_dir.exists():
        logging.error(f"Directory {test_dir} does not exist")
        return

    # Find all image files (JPEG and PNG)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        image_files.extend(test_dir.glob(ext))

    logging.info(f"Found {len(image_files)} image files")

    # Check which ones don't have corresponding JSON files
    missing_json_count = 0

    for image_file in image_files:
        json_file = image_file.with_suffix('.json')

        if not json_file.exists():
            logging.info(f"Creating missing JSON file: {json_file.name}")

            # Get image dimensions
            width, height = get_image_dimensions(image_file)

            # Create empty JSON template
            empty_json = create_empty_json_template(image_file, width, height)

            # Write JSON file
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(empty_json, f, indent=2, ensure_ascii=False)
                missing_json_count += 1
            except Exception as e:
                logging.error(f"Failed to create JSON file {json_file}: {e}")
        else:
            logging.debug(f"JSON file already exists: {json_file.name}")

    logging.info(f"Created {missing_json_count} missing JSON label files")

    # Verify all image files now have corresponding JSON files
    all_have_json = True
    for image_file in image_files:
        json_file = image_file.with_suffix('.json')
        if not json_file.exists():
            logging.error(f"Image file still missing JSON: {image_file.name}")
            all_have_json = False

    if all_have_json:
        logging.info(
            "✓ All image files now have corresponding JSON label files")
    else:
        logging.error("✗ Some image files still missing JSON label files")


if __name__ == "__main__":
    main()
