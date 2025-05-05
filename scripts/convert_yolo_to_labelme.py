import os
import json
from pathlib import Path
from typing import List, Dict, Any


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
        "imageHeight": 640,  # Updated to correct Google Street View dimensions
        "imageWidth": 640    # Updated to correct Google Street View dimensions
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


def process_directory(base_dir: str) -> None:
    base_path = Path(base_dir)

    # Process train, val, and test directories
    for split in ['train', 'val', 'test']:
        labels_dir = base_path / split / 'labels'
        images_dir = base_path / split / 'images'

        if not labels_dir.exists() or not images_dir.exists():
            continue

        for txt_file in labels_dir.glob('*.txt'):
            image_file = images_dir / txt_file.with_suffix('.jpg').name
            if image_file.exists():
                convert_file(txt_file, image_file)


if __name__ == '__main__':
    base_dir = 'data/powerlines/original'
    process_directory(base_dir)
