"""
Dataset manager for preparing powerline sleeve detection datasets.
"""

import os
import glob
import shutil
import random
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from .augmentation import augment_dataset_images


def prepare_dataset(config, input_dir=None, output_dir=None, augment=None):
    """
    Prepare a dataset for training by splitting it into train/val/test sets.
    
    Args:
        config: Configuration dictionary
        input_dir: Directory containing labeled images and annotations
                  If None, uses the labeled images directory from config
        output_dir: Directory to save the processed dataset
                   If None, uses the processed dataset directory from config
        augment: Whether to augment the training dataset
                If None, uses the augmentation setting from config
    
    Returns:
        dict: Dictionary with paths to the train, validation, and test datasets
    """
    # Set default directories if not provided
    if input_dir is None:
        input_dir = config["paths"]["labeled_images"]
    
    if output_dir is None:
        output_dir = config["paths"]["processed_dataset"]
    
    # Set default augmentation setting if not provided
    if augment is None:
        augment = config["dataset"]["augmentation"]["enabled"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = [".jpg", ".jpeg", ".png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    # Get corresponding annotation files
    annotation_files = []
    for img_file in image_files:
        img_name = os.path.splitext(os.path.basename(img_file))[0]
        ann_file = os.path.join(input_dir, f"{img_name}.txt")
        
        if os.path.exists(ann_file):
            annotation_files.append(ann_file)
        else:
            print(f"Warning: No annotation file found for {img_file}")
            image_files.remove(img_file)
    
    print(f"Found {len(image_files)} images with annotations")
    
    # Split the dataset
    dataset_paths = split_dataset(
        config,
        image_files,
        annotation_files,
        output_dir
    )
    
    # Create YOLO dataset.yaml file
    yaml_path = create_yolo_yaml(
        config,
        dataset_paths["train"],
        dataset_paths["val"],
        dataset_paths["test"],
        output_dir
    )
    
    print(f"Created YOLO dataset configuration: {yaml_path}")
    
    # Augment the training dataset if enabled
    if augment:
        print("Augmenting training dataset...")
        augment_dataset_images(
            config,
            dataset_paths["train"],
            os.path.join(output_dir, "train_augmented")
        )
        
        # Update the path to include augmented data
        dataset_paths["train_augmented"] = os.path.join(output_dir, "train_augmented")
        
        # Create a new YAML file with augmented training data
        yaml_path_aug = create_yolo_yaml(
            config,
            dataset_paths["train_augmented"],
            dataset_paths["val"],
            dataset_paths["test"],
            output_dir,
            suffix="_augmented"
        )
        
        print(f"Created augmented YOLO dataset configuration: {yaml_path_aug}")
    
    return dataset_paths


def split_dataset(config, image_files, annotation_files, output_dir):
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        config: Configuration dictionary
        image_files: List of image file paths
        annotation_files: List of annotation file paths
        output_dir: Directory to save the split datasets
        
    Returns:
        dict: Dictionary with paths to the train, validation, and test datasets
    """
    # Get split ratios from config
    train_ratio = config["dataset"]["train_ratio"]
    val_ratio = config["dataset"]["val_ratio"]
    test_ratio = config["dataset"]["test_ratio"]
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        print("Warning: Split ratios do not sum to 1.0. Normalizing...")
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    
    # First split: train vs (val+test)
    train_images, temp_images, train_annotations, temp_annotations = train_test_split(
        image_files, annotation_files, train_size=train_ratio, random_state=42
    )
    
    # Second split: val vs test from the remaining data
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_images, test_images, val_annotations, test_annotations = train_test_split(
        temp_images, temp_annotations, train_size=val_ratio_adjusted, random_state=42
    )
    
    # Create directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
        os.makedirs(os.path.join(directory, "images"), exist_ok=True)
        os.makedirs(os.path.join(directory, "labels"), exist_ok=True)
    
    # Copy files to the appropriate directories
    copy_files_to_directory(train_images, train_annotations, train_dir)
    copy_files_to_directory(val_images, val_annotations, val_dir)
    copy_files_to_directory(test_images, test_annotations, test_dir)
    
    print(f"Dataset split: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test")
    
    return {
        "train": train_dir,
        "val": val_dir,
        "test": test_dir
    }


def copy_files_to_directory(image_files, annotation_files, output_dir):
    """
    Copy image and annotation files to the output directory.
    
    Args:
        image_files: List of image file paths
        annotation_files: List of annotation file paths
        output_dir: Directory to save the files
    """
    for img_file, ann_file in zip(image_files, annotation_files):
        # Get filenames
        img_filename = os.path.basename(img_file)
        ann_filename = os.path.basename(ann_file)
        
        # Copy image
        shutil.copy2(
            img_file,
            os.path.join(output_dir, "images", img_filename)
        )
        
        # Copy annotation
        shutil.copy2(
            ann_file,
            os.path.join(output_dir, "labels", ann_filename)
        )


def create_yolo_yaml(config, train_dir, val_dir, test_dir, output_dir, suffix=""):
    """
    Create a YOLO dataset.yaml file.
    
    Args:
        config: Configuration dictionary
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        test_dir: Directory containing test data
        output_dir: Directory to save the YAML file
        suffix: Optional suffix to add to the filename
        
    Returns:
        str: Path to the created YAML file
    """
    # Get class names
    class_names = config["labeling"]["classes"]
    
    # Create YAML content
    yaml_content = {
        "path": os.path.abspath(output_dir),
        "train": os.path.join("train", "images"),
        "val": os.path.join("val", "images"),
        "test": os.path.join("test", "images"),
        "nc": len(class_names),
        "names": class_names
    }
    
    # Save YAML file
    yaml_path = os.path.join(output_dir, f"dataset{suffix}.yaml")
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    return yaml_path


if __name__ == "__main__":
    # Example usage when run as a script
    import yaml
    
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    prepare_dataset(config) 