import os
import json
from pathlib import Path
from typing import List, Dict, Any
import shutil
from PIL import Image


def yolo_to_labelme(yolo_line: str, image_width: int, image_height: int) -> Dict[str, Any]:
    class_id, x_center, y_center, width, height = map(
        float, yolo_line.strip().split())

    # Convert YOLO coordinates to pixel coordinates
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    # Calculate bounding box coordinates
    x1 = x_center - width/2
    y1 = y_center - height/2
    x2 = x_center + width/2
    y2 = y_center + height/2

    return {
        "label": "powerline",
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "shape_type": "rectangle",
        "flags": {}
    }


def convert_file(txt_path: Path, image_path: Path) -> None:
    # Read image to get dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size
        
    # Read YOLO labels
    with open(txt_path, 'r') as f:
        yolo_lines = f.readlines()

    # Create LabelMe JSON structure
    labelme_data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }

    # Convert each YOLO line to LabelMe format
    for line in yolo_lines:
        if line.strip():
            shape = yolo_to_labelme(
                line, labelme_data["imageWidth"], labelme_data["imageHeight"])
            labelme_data["shapes"].append(shape)

    # Write JSON file in the images directory
    json_path = image_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(labelme_data, f, indent=2)


def process_directory(base_dir: str, output_dir: str) -> None:
    base_path = Path(base_dir)
    output_path = Path(output_dir)

    labels_dir = base_path / 'label'
    images_dir = base_path / 'img'

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if not labels_dir.exists() or not images_dir.exists():
        print(f"Labels or images directory not found in {base_dir}")
        return

    for txt_file in labels_dir.glob('*.txt'):
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_image_file = images_dir / txt_file.with_suffix(ext).name
            if potential_image_file.exists():
                image_file = potential_image_file
                break
        
        if not image_file:
            print(f"No corresponding image found for {txt_file.name}")
            continue
            
        dest_image_path = output_path / image_file.name
        shutil.copy(image_file, dest_image_path)
        
        convert_file(txt_file, dest_image_path)


if __name__ == '__main__':
    base_dir = 'data/sleeves/original_yolo'
    output_dir = 'data/sleeves/original_labelme'
    process_directory(base_dir, output_dir)
    print(f"Conversion complete. LabelMe files are in {output_dir}")
