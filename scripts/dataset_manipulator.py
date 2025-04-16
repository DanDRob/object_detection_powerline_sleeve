#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path
import yaml

def dataset_manipulator(source_dataset_path, output_dir=None, create_balanced=True, create_powerline_only=True):
    """
    Create modified versions of the dataset:
    1. Balanced dataset - equal number of images with and without powerlines
    2. Powerline-only dataset - only images with powerlines
    
    Args:
        source_dataset_path (str): Path to the original dataset directory
        output_dir (str, optional): Path to store new datasets. If None, will use parent of source.
        create_balanced (bool): Whether to create the balanced dataset
        create_powerline_only (bool): Whether to create the powerline-only dataset
    
    Returns:
        dict: Paths to the created datasets
    """
    source_path = Path(source_dataset_path)
    
    if output_dir is None:
        output_dir = source_path.parent
    
    # Define output directories
    output_paths = {
        'balanced': Path(output_dir) / 'original_balanced',
        'powerline_only': Path(output_dir) / 'original_powerline_only'
    }
    
    # Make sure source exists
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset path not found: {source_path}")
    
    results = {}
    
    # Process each split (train, val, test) if they exist
    for split in ['train', 'val', 'test']:
        split_path = source_path / split
        if not split_path.exists():
            continue
        
        # Get labels path
        labels_path = split_path / 'labels'
        images_path = split_path / 'images'
        
        if not labels_path.exists() or not images_path.exists():
            print(f"Skipping {split} split, missing labels or images directory")
            continue
        
        # Get list of all label files
        label_files = list(labels_path.glob('*.txt'))
        
        # Separate labeled and unlabeled files
        labeled_files = []
        unlabeled_files = []
        
        for label_file in label_files:
            # Empty files are considered unlabeled (no powerlines)
            if os.path.getsize(label_file) == 0:
                unlabeled_files.append(label_file.name)
            else:
                labeled_files.append(label_file.name)
        
        print(f"Split {split}: Found {len(labeled_files)} images with powerlines and {len(unlabeled_files)} without")
        
        # Create balanced dataset if requested
        if create_balanced:
            balanced_path = output_paths['balanced'] / split
            os.makedirs(balanced_path / 'labels', exist_ok=True)
            os.makedirs(balanced_path / 'images', exist_ok=True)
            
            # Keep all labeled files, randomly sample the same number from unlabeled
            sampled_unlabeled = random.sample(unlabeled_files, len(labeled_files)) if len(unlabeled_files) > len(labeled_files) else unlabeled_files
            
            # Copy files
            for label_file_name in labeled_files + sampled_unlabeled:
                # Copy label file
                shutil.copy2(
                    labels_path / label_file_name,
                    balanced_path / 'labels' / label_file_name
                )
                
                # Copy corresponding image file
                image_name = label_file_name.replace('.txt', '.jpg')
                if not (images_path / image_name).exists():
                    # Try other extensions
                    for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                        alt_name = label_file_name.replace('.txt', ext)
                        if (images_path / alt_name).exists():
                            image_name = alt_name
                            break
                
                shutil.copy2(
                    images_path / image_name,
                    balanced_path / 'images' / image_name
                )
            
            print(f"Created balanced dataset for {split} with {len(labeled_files)} images of each class")
            results['balanced'] = str(output_paths['balanced'])
        
        # Create powerline-only dataset if requested
        if create_powerline_only:
            powerline_only_path = output_paths['powerline_only'] / split
            os.makedirs(powerline_only_path / 'labels', exist_ok=True)
            os.makedirs(powerline_only_path / 'images', exist_ok=True)
            
            # Only copy labeled files
            for label_file_name in labeled_files:
                # Copy label file
                shutil.copy2(
                    labels_path / label_file_name,
                    powerline_only_path / 'labels' / label_file_name
                )
                
                # Copy corresponding image file
                image_name = label_file_name.replace('.txt', '.jpg')
                if not (images_path / image_name).exists():
                    # Try other extensions
                    for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                        alt_name = label_file_name.replace('.txt', ext)
                        if (images_path / alt_name).exists():
                            image_name = alt_name
                            break
                
                shutil.copy2(
                    images_path / image_name,
                    powerline_only_path / 'images' / image_name
                )
            
            print(f"Created powerline-only dataset for {split} with {len(labeled_files)} images")
            results['powerline_only'] = str(output_paths['powerline_only'])
    
    # Copy classes.txt file to new datasets
    classes_file = source_path / 'classes.txt'
    if classes_file.exists():
        if create_balanced and 'balanced' in results:
            shutil.copy2(classes_file, output_paths['balanced'] / 'classes.txt')
        
        if create_powerline_only and 'powerline_only' in results:
            shutil.copy2(classes_file, output_paths['powerline_only'] / 'classes.txt')
    
    # Create YAML files for new datasets
    if create_balanced and 'balanced' in results:
        create_dataset_yaml(output_paths['balanced'], 'original_balanced')
    
    if create_powerline_only and 'powerline_only' in results:
        create_dataset_yaml(output_paths['powerline_only'], 'original_powerline_only')
    
    return results

def create_dataset_yaml(dataset_path, dataset_name):
    """Create a YAML file for the dataset to use with YOLOv8"""
    yaml_content = {
        'path': str(dataset_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'powerline'
        }
    }
    
    # Write YAML file
    yaml_path = dataset_path / f"{dataset_name}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created dataset YAML file: {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create balanced and powerline-only datasets')
    parser.add_argument('--source', type=str, required=True, help='Path to source dataset')
    parser.add_argument('--output', type=str, default=None, help='Output directory for new datasets')
    parser.add_argument('--no-balanced', action='store_true', help='Skip creating balanced dataset')
    parser.add_argument('--no-powerline-only', action='store_true', help='Skip creating powerline-only dataset')
    
    args = parser.parse_args()
    
    dataset_manipulator(
        args.source, 
        args.output,
        create_balanced=not args.no_balanced,
        create_powerline_only=not args.no_powerline_only
    ) 