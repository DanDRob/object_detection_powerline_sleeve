import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
import shutil

TARGET_SIZE = 640
OUTPUT_DIR = Path('data/sleeves/add_train_sleeves_640')
INPUT_DIR = Path('data/sleeves/add_train_sleeves')


def get_enclosing_bbox(shapes: list) -> tuple:
    """Calculates the bounding box that encloses all label shapes."""
    if not shapes:
        return None

    all_points = []
    for shape in shapes:
        if shape.get('shape_type') == 'rectangle':
            all_points.extend(shape['points'])

    if not all_points:
        return None

    all_points = np.array(all_points)
    min_x = np.min(all_points[:, 0])
    min_y = np.min(all_points[:, 1])
    max_x = np.max(all_points[:, 0])
    max_y = np.max(all_points[:, 1])

    return min_x, min_y, max_x, max_y


def process_and_save(json_path: Path):
    """Processes a single image and its corresponding LabelMe JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {json_path}: {e}")
        return

    # --- 1. Update Labels ---
    for shape in data['shapes']:
        if shape['label'].lower() == 'powerline':
            shape['label'] = 'sleeve'

    # --- 2. Open Image ---
    image_path = INPUT_DIR / data['imagePath']
    if not image_path.exists():
        print(f"Warning: Image not found for {json_path.name}, skipping.")
        return

    with Image.open(image_path) as img:
        original_width, original_height = img.size

        bbox = get_enclosing_bbox(data['shapes'])

        # --- 3. Handle Images Without Labels ---
        if bbox is None:
            # Center crop to a square and resize
            side = min(original_width, original_height)
            crop_box = (
                (original_width - side) / 2,
                (original_height - side) / 2,
                (original_width + side) / 2,
                (original_height + side) / 2
            )
            cropped_img = img.crop(crop_box)
            final_img = cropped_img.resize(
                (TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

            # No need to update coordinates
            data['shapes'] = []

        else:
            # --- 4. Dynamic Cropping/Resizing Logic ---
            min_x, min_y, max_x, max_y = bbox
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y

            # Scenario A: Smart 640x640 Crop
            if bbox_width <= TARGET_SIZE and bbox_height <= TARGET_SIZE:
                center_x = min_x + bbox_width / 2
                center_y = min_y + bbox_height / 2

                crop_x1 = center_x - TARGET_SIZE / 2
                crop_y1 = center_y - TARGET_SIZE / 2

                # Prevent cropping outside the image
                if crop_x1 < 0:
                    crop_x1 = 0
                if crop_y1 < 0:
                    crop_y1 = 0
                if crop_x1 + TARGET_SIZE > original_width:
                    crop_x1 = original_width - TARGET_SIZE
                if crop_y1 + TARGET_SIZE > original_height:
                    crop_y1 = original_height - TARGET_SIZE

                crop_box = (crop_x1, crop_y1, crop_x1 +
                            TARGET_SIZE, crop_y1 + TARGET_SIZE)
                final_img = img.crop(crop_box)

                # Update coordinates
                for shape in data['shapes']:
                    points = np.array(shape['points'])
                    points[:, 0] -= crop_x1
                    points[:, 1] -= crop_y1
                    shape['points'] = points.tolist()

            # Scenario B: Crop to Content and Resize
            else:
                padding = 20
                crop_x1 = max(0, min_x - padding)
                crop_y1 = max(0, min_y - padding)
                crop_x2 = min(original_width, max_x + padding)
                crop_y2 = min(original_height, max_y + padding)

                crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
                cropped_img = img.crop(crop_box)

                scale_x = TARGET_SIZE / cropped_img.width
                scale_y = TARGET_SIZE / cropped_img.height

                final_img = cropped_img.resize(
                    (TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

                # Update coordinates
                for shape in data['shapes']:
                    points = np.array(shape['points'])
                    points[:, 0] = (points[:, 0] - crop_x1) * scale_x
                    points[:, 1] = (points[:, 1] - crop_y1) * scale_y
                    shape['points'] = points.tolist()

        # --- 5. Save Results ---
        # Save new image
        output_image_path = OUTPUT_DIR / image_path.name
        final_img.convert("RGB").save(output_image_path)

        # Update and save new JSON
        data['imagePath'] = image_path.name
        data['imageWidth'] = TARGET_SIZE
        data['imageHeight'] = TARGET_SIZE
        data['imageData'] = None

        output_json_path = OUTPUT_DIR / json_path.name
        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=2)


def process_image_only(image_path: Path):
    """Center-crops an image to a square and resizes to TARGET_SIZE, then saves it.

    This is used when no corresponding LabelMe JSON exists.
    """
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size

            # Center crop to square
            side = min(original_width, original_height)
            crop_box = (
                (original_width - side) / 2,
                (original_height - side) / 2,
                (original_width + side) / 2,
                (original_height + side) / 2
            )
            cropped_img = img.crop(crop_box)
            final_img = cropped_img.resize(
                (TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

            output_image_path = OUTPUT_DIR / image_path.name
            final_img.convert("RGB").save(output_image_path)
    except FileNotFoundError:
        print(f"Warning: Image not found '{image_path}', skipping.")
    except OSError as e:
        print(f"Error processing image '{image_path}': {e}")


def main():
    """Main function to run the processing script."""
    if not INPUT_DIR.exists():
        print(f"Error: Input directory not found at '{INPUT_DIR}'")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(INPUT_DIR.glob('*.json'))

    # Gather images in the input directory
    image_extensions = {'.jpg', '.jpeg', '.png',
                        '.bmp', '.tif', '.tiff', '.webp'}
    image_files = [
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if json_files:
        print(
            f"Found {len(json_files)} JSON files. Starting labeled processing...")
    else:
        print(f"No JSON files found in '{INPUT_DIR}'. Resizing images only...")

    # Process labeled images first
    for json_path in json_files:
        process_and_save(json_path)

    # Process images without JSON
    json_stems = {p.stem for p in json_files}
    unlabeled_images = [p for p in image_files if p.stem not in json_stems]
    if unlabeled_images:
        print(f"Processing {len(unlabeled_images)} images without JSON...")
        for image_path in unlabeled_images:
            process_image_only(image_path)

    print(f"\nProcessing complete. Output is in '{OUTPUT_DIR}'")


if __name__ == '__main__':
    main()
