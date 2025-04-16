"""
Utilities for working with LabelMe annotations and converting them to YOLO format.
"""

import os
import json
import glob
import shutil
from pathlib import Path


def convert_labelme_to_yolo(config, input_dir=None, output_dir=None, class_names=None):
    """
    Convert LabelMe JSON annotations to YOLO format.
    
    Args:
        config: Configuration dictionary
        input_dir: Directory containing LabelMe JSON annotations
                  If None, uses the default directory from config
        output_dir: Directory to save YOLO annotations
                   If None, uses the default directory from config
        class_names: List of class names
                    If None, uses the classes from config
    
    Returns:
        int: Number of annotations converted
    """
    # Set default directories if not provided
    if input_dir is None:
        input_dir = os.path.join(config["paths"]["labeled_images"], "labelme")
    
    if output_dir is None:
        output_dir = config["paths"]["labeled_images"]
    
    # Set default class names if not provided
    if class_names is None:
        class_names = config["labeling"]["classes"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"Found {len(json_files)} LabelMe annotation files")
    
    # Process each JSON file
    converted_count = 0
    for json_file in json_files:
        success = convert_single_annotation(json_file, output_dir, class_names)
        if success:
            converted_count += 1
    
    print(f"Converted {converted_count} LabelMe annotations to YOLO format")
    return converted_count


def convert_single_annotation(json_file, output_dir, class_names):
    """
    Convert a single LabelMe JSON annotation to YOLO format.
    
    Args:
        json_file: Path to the JSON file
        output_dir: Directory to save YOLO annotations
        class_names: List of class names
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load JSON annotation
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract image information
        image_filename = data.get("imagePath")
        if not image_filename:
            print(f"No image path in {json_file}")
            return False
        
        image_name = os.path.splitext(image_filename)[0]
        img_width = data.get("imageWidth")
        img_height = data.get("imageHeight")
        
        if not all([img_width, img_height]):
            print(f"Missing image dimensions in {json_file}")
            return False
        
        # Check if image file exists next to JSON
        json_dir = os.path.dirname(json_file)
        image_path = os.path.join(json_dir, image_filename)
        if not os.path.exists(image_path):
            # Try to find the image in the same directory as the JSON
            alt_image_files = [
                os.path.join(json_dir, image_filename),
                os.path.join(json_dir, f"{image_name}.jpg"),
                os.path.join(json_dir, f"{image_name}.jpeg"),
                os.path.join(json_dir, f"{image_name}.png")
            ]
            
            found = False
            for alt_path in alt_image_files:
                if os.path.exists(alt_path):
                    image_path = alt_path
                    found = True
                    break
            
            if not found:
                print(f"Image file not found for {json_file}")
                return False
        
        # Copy the image file to the output directory
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        if not os.path.exists(output_image_path):
            shutil.copy2(image_path, output_image_path)
        
        # Create YOLO annotation file
        output_label_path = os.path.join(output_dir, f"{image_name}.txt")
        
        with open(output_label_path, 'w') as f:
            # Process each shape (annotation)
            for shape in data.get("shapes", []):
                # Get label and points
                label = shape.get("label")
                points = shape.get("points")
                
                # Skip if label or points are missing
                if not label or not points:
                    continue
                
                # Get class index
                try:
                    class_idx = class_names.index(label)
                except ValueError:
                    print(f"Unknown class '{label}' in {json_file}, skipping")
                    continue
                
                # Convert to YOLO format
                # LabelMe saves polygons or rectangles with multiple points
                # YOLO uses normalized center x, center y, width, height
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
                
                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Write to file in YOLO format
                f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")
        
        return True
    
    except Exception as e:
        print(f"Error converting {json_file}: {e}")
        return False


def create_labelme_guide(output_path="LABELME_GUIDE.md"):
    """
    Create a markdown guide for using LabelMe to annotate images.
    
    Args:
        output_path: Path to save the guide
        
    Returns:
        bool: True if successful, False otherwise
    """
    guide_content = """# LabelMe Guide for Powerline Sleeve Detection

This guide will help you annotate images using LabelMe for powerline sleeve detection.

## Installation

1. Install LabelMe:
   ```
   pip install labelme
   ```

2. Launch LabelMe:
   ```
   labelme
   ```

## Annotation Guidelines

1. **Open Image**: Click 'Open' and select an image to annotate.

2. **Create Annotations**:
   - Click 'Create Rectangle' to create a bounding box.
   - Draw a tight rectangle around the object.
   - Enter the appropriate label:
     - Use 'powerline' for power lines.
     - Use 'sleeve' for sleeves on power lines.

3. **Save Annotations**:
   - Click 'Save' to save your annotations.
   - Save the JSON file with the same name as the image in the same directory.

4. **Tips**:
   - Be consistent with your annotations.
   - Annotate all instances of objects in each image.
   - Make bounding boxes as tight as possible.
   - If an object is partially occluded, annotate only the visible part.

## Converting Annotations

After annotating images with LabelMe, you can convert the JSON annotations to YOLO format using the provided utility:

```python
from src.labeling import convert_labelme_to_yolo
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Convert annotations
convert_labelme_to_yolo(
    config, 
    input_dir="path/to/labelme/annotations",
    output_dir="path/to/output/directory"
)
```

## Best Practices

1. **Consistency**: Maintain consistent criteria for what constitutes each object class.
2. **Completeness**: Annotate all instances of target objects in each image.
3. **Precision**: Create tight bounding boxes around objects.
4. **Regular Backup**: Regularly backup your annotation files.
"""
    
    try:
        with open(output_path, 'w') as f:
            f.write(guide_content)
        return True
    except Exception as e:
        print(f"Error creating LabelMe guide: {e}")
        return False


if __name__ == "__main__":
    # Example usage when run as a script
    import yaml
    
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create a LabelMe guide
    create_labelme_guide("../../LABELME_GUIDE.md")
    
    # Convert example annotations
    convert_labelme_to_yolo(config) 