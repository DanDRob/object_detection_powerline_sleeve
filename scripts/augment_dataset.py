#!/usr/bin/env python3
"""
Script to augment powerline and sleeve datasets independently.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import glob
import random
import numpy as np
import cv2
import shutil
from tqdm import tqdm
import albumentations as A

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))
from src.dataset.augmentation import augment_dataset_images


def load_yolo_bbox(label_path, image_shape):
    """
    Load YOLO format bounding boxes from a label file
    
    Args:
        label_path: Path to the label file
        image_shape: Tuple (height, width) of the corresponding image
        
    Returns:
        List of dictionaries with bbox information in albumentations format
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    
    height, width = image_shape
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    # Convert YOLO format (center, width, height) to albumentations format (x_min, y_min, width, height)
                    x_min = (x_center - w/2)
                    y_min = (y_center - h/2)
                    
                    bboxes.append({
                        'bbox': [x_min, y_min, w, h],
                        'class_id': class_id
                    })
                except (ValueError, IndexError):
                    print(f"Warning: Invalid bounding box format in {label_path}")
    
    return bboxes


def save_yolo_bbox(bboxes, output_path):
    """
    Save bounding boxes in YOLO format
    
    Args:
        bboxes: List of bounding boxes in albumentations format
        output_path: Path to save the label file
    """
    with open(output_path, 'w') as f:
        for bbox_data in bboxes:
            bbox = bbox_data['bbox']
            class_id = bbox_data['class_id']
            
            # Convert albumentations format (x_min, y_min, width, height) back to YOLO (center, width, height)
            x_center = bbox[0] + bbox[2]/2
            y_center = bbox[1] + bbox[3]/2
            width = bbox[2]
            height = bbox[3]
            
            # Write YOLO format line
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def create_augmentations():
    """
    Create a single augmentation pipeline with safe transformations for bounding boxes
    
    Returns:
        Augmentation transform object
    """
    # Only using augmentations that are safe for bounding boxes
    safe_augs = [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ]
    
    transform = A.Compose(
        safe_augs,
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_id'])
    )
    
    return transform


def augment_dataset(source_dir, target_dir, samples_per_image=2, seed=42, preserve_orig=True):
    """
    Augment dataset with multiple augmentations per image
    
    Args:
        source_dir: Source directory (should contain train/val/test splits)
        target_dir: Target directory for augmented dataset
        samples_per_image: Number of augmented samples to generate per original image
        seed: Random seed for reproducibility
        preserve_orig: Whether to include original images in the augmented dataset
        
    Returns:
        Dictionary with statistics about the augmentation process
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Create augmentation pipeline
    transform = create_augmentations()
    
    # Determine the structure of the source dataset
    splits = ['train', 'val', 'test']
    stats = {
        'total_source_images': 0,
        'total_augmented_images': 0,
        'augmented_with_objects': 0,
        'augmented_without_objects': 0,
        'skipped_images': 0,
        'splits': {}
    }
    
    for split in splits:
        split_source_dir = os.path.join(source_dir, split)
        split_target_dir = os.path.join(target_dir, split)
        
        if not os.path.exists(split_source_dir):
            print(f"Warning: Source split directory {split_source_dir} does not exist. Skipping.")
            continue
        
        # Create split directories
        os.makedirs(split_target_dir, exist_ok=True)
        os.makedirs(os.path.join(split_target_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_target_dir, 'labels'), exist_ok=True)
        
        # Initialize stats for this split
        stats['splits'][split] = {
            'source_images': 0,
            'augmented_images': 0,
            'with_objects': 0,
            'without_objects': 0
        }
        
        # Get all image files in the split
        image_dir = os.path.join(split_source_dir, 'images')
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(glob.glob(os.path.join(image_dir, f'*.{ext}')))
        
        stats['splits'][split]['source_images'] = len(image_files)
        stats['total_source_images'] += len(image_files)
        
        print(f"Processing {split} split: {len(image_files)} images")
        
        # Process each image in the split
        for img_path in tqdm(image_files, desc=f"Augmenting {split}"):
            # Get image name and corresponding label path
            img_filename = os.path.basename(img_path)
            img_name = os.path.splitext(img_filename)[0]
            label_path = os.path.join(split_source_dir, 'labels', f"{img_name}.txt")
            
            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                stats['skipped_images'] += 1
                continue
                
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Load bounding boxes
            bboxes_data = load_yolo_bbox(label_path, (height, width))
            bboxes = [data['bbox'] for data in bboxes_data]
            class_ids = [data['class_id'] for data in bboxes_data]
            has_objects = len(bboxes) > 0
            
            # Count if this image has objects
            if has_objects:
                stats['splits'][split]['with_objects'] += 1
            else:
                stats['splits'][split]['without_objects'] += 1
            
            # Copy original images if required
            if preserve_orig:
                target_img_path = os.path.join(split_target_dir, 'images', img_filename)
                target_label_path = os.path.join(split_target_dir, 'labels', f"{img_name}.txt")
                
                # Copy image
                shutil.copy2(img_path, target_img_path)
                
                # Copy label if it exists
                if os.path.exists(label_path):
                    shutil.copy2(label_path, target_label_path)
                
                stats['splits'][split]['augmented_images'] += 1
                stats['total_augmented_images'] += 1
                
                if has_objects:
                    stats['augmented_with_objects'] += 1
                else:
                    stats['augmented_without_objects'] += 1
            
            # Generate augmented samples
            aug_count = 0
            max_attempts = samples_per_image * 3  # Allow for some failures
            
            for attempt in range(max_attempts):
                if aug_count >= samples_per_image:
                    break
                
                try:
                    # Apply augmentation
                    if bboxes:
                        augmented = transform(image=img, bboxes=bboxes, class_id=class_ids)
                        aug_img = augmented['image']
                        aug_bboxes = augmented['bboxes']
                        aug_class_ids = augmented['class_id']
                        
                        # Skip if no valid bounding boxes after augmentation
                        if not aug_bboxes:
                            continue
                    else:
                        augmented = transform(image=img)
                        aug_img = augmented['image']
                        aug_bboxes = []
                        aug_class_ids = []
                    
                    # Create augmented file names
                    aug_img_name = f"{img_name}_aug_{aug_count+1}"
                    aug_img_filename = f"{aug_img_name}.jpg"
                    
                    # Save augmented image
                    aug_img_path = os.path.join(split_target_dir, 'images', aug_img_filename)
                    cv2.imwrite(aug_img_path, aug_img)
                    
                    # Save augmented labels
                    if bboxes:
                        aug_label_path = os.path.join(split_target_dir, 'labels', f"{aug_img_name}.txt")
                        aug_bbox_data = [
                            {'bbox': bbox, 'class_id': class_id}
                            for bbox, class_id in zip(aug_bboxes, aug_class_ids)
                        ]
                        save_yolo_bbox(aug_bbox_data, aug_label_path)
                    
                    aug_count += 1
                    stats['splits'][split]['augmented_images'] += 1
                    stats['total_augmented_images'] += 1
                    
                    if has_objects:
                        stats['augmented_with_objects'] += 1
                    else:
                        stats['augmented_without_objects'] += 1
                    
                except Exception as e:
                    print(f"Warning: Augmentation failed for {img_path}: {e}")
    
    # Copy classes.txt if it exists
    classes_file = os.path.join(source_dir, 'classes.txt')
    if os.path.exists(classes_file):
        shutil.copy2(classes_file, os.path.join(target_dir, 'classes.txt'))
    
    return stats


def print_stats(stats):
    """Print statistics about the augmentation process"""
    print("\nAugmentation Statistics:")
    print(f"Total source images: {stats['total_source_images']}")
    print(f"Total augmented images: {stats['total_augmented_images']}")
    print(f"Images with objects: {stats['augmented_with_objects']}")
    print(f"Images without objects: {stats['augmented_without_objects']}")
    
    if stats['skipped_images'] > 0:
        print(f"Skipped images: {stats['skipped_images']}")
    
    print("\nSplit statistics:")
    for split, split_stats in stats['splits'].items():
        print(f"  {split}:")
        print(f"    Source: {split_stats['source_images']} images")
        print(f"    Augmented: {split_stats['augmented_images']} images")
        print(f"    With objects: {split_stats['with_objects']} original images")
        print(f"    Without objects: {split_stats['without_objects']} original images")


def main():
    parser = argparse.ArgumentParser(description="Augment datasets for powerline and sleeve detection")
    parser.add_argument("--dataset", type=str, required=True, choices=["powerlines", "sleeves"],
                        help="Dataset to augment: 'powerlines' or 'sleeves'")
    parser.add_argument("--num-augmentations", type=int, default=3,
                        help="Number of augmented versions to create per image")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Set input and output directories based on dataset choice
    data_dir = Path(config["paths"]["data_dir"])
    input_dir = data_dir / args.dataset / "original"
    output_dir = data_dir / args.dataset / "augmented"
    
    # Create output directories for splits
    for split in ['train', 'val', 'test']:
        os.makedirs(output_dir / split / "images", exist_ok=True)
        os.makedirs(output_dir / split / "labels", exist_ok=True)
    
    # Verify input directory has images in at least one split
    total_image_count = 0
    for split in ['train', 'val', 'test']:
        split_dir = input_dir / split
        if not split_dir.exists():
            print(f"Warning: Split directory {split_dir} does not exist.")
            continue
            
        split_image_count = len(list((split_dir / "images").glob("*.jpg"))) + \
                          len(list((split_dir / "images").glob("*.jpeg"))) + \
                          len(list((split_dir / "images").glob("*.png")))
        total_image_count += split_image_count
        print(f"Found {split_image_count} images in {split} split")
    
    if total_image_count == 0:
        print(f"Error: No images found in any split directory under {input_dir}")
        print("Please ensure you have images in at least one of train/val/test splits")
        sys.exit(1)
    
    # Run augmentation
    print(f"Augmenting {args.dataset} dataset with {args.num_augmentations} versions per image")
    augmented_count = augment_dataset(
        source_dir=str(input_dir),
        target_dir=str(output_dir),
        samples_per_image=args.num_augmentations,
        seed=42,
        preserve_orig=True
    )
    
    print(f"Dataset augmentation complete. Created {augmented_count['total_augmented_images']} augmented images.")


if __name__ == "__main__":
    main() 