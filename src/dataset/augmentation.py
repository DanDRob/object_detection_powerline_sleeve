"""
Data augmentation for powerline and sleeve detection datasets.
"""

import os
import glob
import shutil
import cv2
import numpy as np
from pathlib import Path
import albumentations as A


def augment_dataset_images(config, input_dir, output_dir, num_augmentations=3):
    """
    Augment images in a dataset with various transformations.
    
    Args:
        config: Configuration dictionary
        input_dir: Directory containing images and labels to augment
        output_dir: Directory to save augmented images and labels
        num_augmentations: Number of augmented versions to create per image
        
    Returns:
        int: Number of augmented images created
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    # Get augmentation parameters from config
    aug_config = config["dataset"]["augmentation"]
    
    # Create augmentation pipeline
    transform = create_augmentation_pipeline(aug_config)
    
    # Get all image files
    image_dir = os.path.join(input_dir, "images")
    label_dir = os.path.join(input_dir, "labels")
    
    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                 glob.glob(os.path.join(image_dir, "*.jpeg")) + \
                 glob.glob(os.path.join(image_dir, "*.png"))
    
    print(f"Found {len(image_files)} images to augment")
    
    # Copy original files to output directory
    for img_file in image_files:
        img_filename = os.path.basename(img_file)
        img_name = os.path.splitext(img_filename)[0]
        label_file = os.path.join(label_dir, f"{img_name}.txt")
        
        # Copy original image
        shutil.copy2(
            img_file,
            os.path.join(output_dir, "images", img_filename)
        )
        
        # Copy original label if exists
        if os.path.exists(label_file):
            shutil.copy2(
                label_file,
                os.path.join(output_dir, "labels", f"{img_name}.txt")
            )
    
    # Augment each image
    augmented_count = 0
    for img_file in image_files:
        img_filename = os.path.basename(img_file)
        img_name = os.path.splitext(img_filename)[0]
        
        # Load image and bounding boxes
        image = cv2.imread(img_file)
        if image is None:
            print(f"Warning: Could not read image {img_file}")
            continue
        
        # Convert BGR to RGB for albumentations
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding label file
        label_file = os.path.join(label_dir, f"{img_name}.txt")
        if not os.path.exists(label_file):
            print(f"Warning: No label file found for {img_file}")
            continue
        
        # Parse YOLO format annotations
        bboxes, class_ids = parse_yolo_annotations(label_file, image.shape)
        
        # Skip if there are no valid bounding boxes
        if not bboxes:
            print(f"Warning: No valid bounding boxes in {label_file}")
            continue
        
        # Generate augmentations
        for i in range(num_augmentations):
            try:
                # Apply augmentation
                augmented = transform(
                    image=image,
                    bboxes=bboxes,
                    class_ids=class_ids
                )
                
                # Get augmented data
                aug_image = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_class_ids = augmented["class_ids"]
                
                # Skip if no valid bounding boxes remain after augmentation
                if not aug_bboxes:
                    print(f"Warning: No valid bounding boxes after augmentation for {img_file} (aug {i+1})")
                    continue
                
                # Validate bounding boxes to ensure they are within valid ranges
                valid_bboxes = []
                valid_class_ids = []
                
                for bbox, class_id in zip(aug_bboxes, aug_class_ids):
                    x_center, y_center, width, height = bbox
                    
                    # Ensure values are strictly within [0, 1] and dimensions are positive
                    if (0 < x_center < 1 and 0 < y_center < 1 and 
                        0 < width < 1 and 0 < height < 1 and
                        width > 0.001 and height > 0.001):  # Minimum size threshold
                        valid_bboxes.append(bbox)
                        valid_class_ids.append(class_id)
                
                # Skip if no valid bounding boxes remain after validation
                if not valid_bboxes:
                    print(f"Warning: No valid bounding boxes after validation for {img_file} (aug {i+1})")
                    continue
                
                # Save augmented image with _aug_N suffix
                aug_img_filename = f"{img_name}_aug_{i+1}.jpg"
                aug_image_path = os.path.join(output_dir, "images", aug_img_filename)
                
                # Convert RGB back to BGR for OpenCV
                aug_image = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(aug_image_path, aug_image)
                
                # Save augmented annotations with _aug_N suffix
                aug_label_filename = f"{img_name}_aug_{i+1}.txt"
                aug_label_path = os.path.join(output_dir, "labels", aug_label_filename)
                
                # Write YOLO format annotations
                write_yolo_annotations(
                    aug_label_path,
                    valid_bboxes,
                    valid_class_ids,
                    aug_image.shape
                )
                
                augmented_count += 1
            
            except Exception as e:
                print(f"Error augmenting {img_file} (aug {i+1}): {e}")
    
    print(f"Created {augmented_count} augmented images")
    return augmented_count


def create_augmentation_pipeline(aug_config):
    """
    Create an augmentation pipeline based on configuration.
    
    Args:
        aug_config: Augmentation configuration dictionary
        
    Returns:
        A.Compose: Albumentation composition of transforms
    """
    transforms = []
    
    # Add transforms based on config
    if "rotation_range" in aug_config and aug_config["rotation_range"] > 0:
        transforms.append(
            A.Rotate(
                limit=aug_config["rotation_range"],
                p=0.7
            )
        )
    
    if "brightness_range" in aug_config:
        min_val, max_val = aug_config["brightness_range"]
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=(min_val - 1, max_val - 1),
                contrast_limit=0.1,
                p=0.5
            )
        )
    
    if "contrast_range" in aug_config:
        min_val, max_val = aug_config["contrast_range"]
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=(min_val - 1, max_val - 1),
                p=0.5
            )
        )
    
    # Add horizontal and vertical flips if enabled
    if aug_config.get("horizontal_flip", False):
        transforms.append(A.HorizontalFlip(p=0.5))
    
    if aug_config.get("vertical_flip", False):
        transforms.append(A.VerticalFlip(p=0.5))
    
    # Add some standard transforms if the list is empty
    if not transforms:
        transforms = [
            A.RandomBrightnessContrast(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.2)
        ]
    
    # Create the composition with bounding box compatible format
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_ids'],
            min_visibility=0.3,  # Skip boxes with low visibility after transform
            min_area=0.0001  # Skip very small boxes
        )
    )


def parse_yolo_annotations(label_file, img_shape):
    """
    Parse YOLO format annotation file into bounding boxes and class IDs.
    
    Args:
        label_file: Path to YOLO annotation file
        img_shape: Image shape (height, width, channels)
        
    Returns:
        tuple: (bboxes, class_ids) where bboxes is a list of [x_center, y_center, width, height]
               in normalized coordinates and class_ids is a list of integer class IDs
    """
    img_height, img_width = img_shape[:2]
    
    bboxes = []
    class_ids = []
    
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Skip invalid boxes
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 < width <= 1 and 0 < height <= 1):
                        continue
                    
                    # YOLO format is already normalized
                    bboxes.append([x_center, y_center, width, height])
                    class_ids.append(class_id)
                except (ValueError, IndexError):
                    print(f"Warning: Invalid bounding box format in {label_file}")
    
    return bboxes, class_ids


def write_yolo_annotations(label_file, bboxes, class_ids, img_shape):
    """
    Write YOLO format annotation file.
    
    Args:
        label_file: Path to save YOLO annotation file
        bboxes: List of bounding boxes [x_center, y_center, width, height] in normalized coordinates
        class_ids: List of integer class IDs
        img_shape: Image shape (height, width, channels)
    """
    with open(label_file, 'w') as f:
        for bbox, class_id in zip(bboxes, class_ids):
            x_center, y_center, width, height = bbox
            
            # Ensure values are within [0, 1]
            x_center = max(0.001, min(0.999, x_center))
            y_center = max(0.001, min(0.999, y_center))
            width = max(0.001, min(0.999, width))
            height = max(0.001, min(0.999, height))
            
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":
    # Example usage when run as a script
    import yaml
    
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Augment a dataset for powerlines
    print("Augmenting powerline dataset...")
    augment_dataset_images(
        config,
        "../../data/powerlines/original",
        "../../data/powerlines/augmented",
        num_augmentations=3
    )
    
    # Later, augment a dataset for sleeves (when available)
    # print("Augmenting sleeve dataset...")
    # augment_dataset_images(
    #     config,
    #     "../../data/sleeves/original",
    #     "../../data/sleeves/augmented",
    #     num_augmentations=3
    # ) 