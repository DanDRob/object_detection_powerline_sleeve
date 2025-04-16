#!/usr/bin/env python3

import os
import shutil
import glob
import argparse
from pathlib import Path


def count_files(directory, extension=None):
    """Count files in a directory with optional extension filter"""
    if not os.path.exists(directory):
        return 0
    
    if extension:
        return len(glob.glob(os.path.join(directory, f"*.{extension}")))
    else:
        return sum(len(files) for _, _, files in os.walk(directory))


def get_split_structure(source_dir):
    """Analyze the directory structure to identify train/val/test splits"""
    structure = {}
    
    # Check for top-level splits
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(source_dir, split)
        if os.path.isdir(split_dir):
            structure[split] = {
                'path': split_dir,
                'images_dir': None,
                'labels_dir': None,
                'image_count': 0,
                'label_count': 0
            }
            
            # Check for images/labels subdirectories
            img_dir = os.path.join(split_dir, 'images')
            if os.path.isdir(img_dir):
                structure[split]['images_dir'] = img_dir
                structure[split]['image_count'] = count_files(img_dir, 'jpg') + \
                                                 count_files(img_dir, 'jpeg') + \
                                                 count_files(img_dir, 'png')
            
            label_dir = os.path.join(split_dir, 'labels')
            if os.path.isdir(label_dir):
                structure[split]['labels_dir'] = label_dir
                structure[split]['label_count'] = count_files(label_dir, 'txt')
    
    # Check if there's a flat structure with images/labels at top level
    if not structure:
        img_dir = os.path.join(source_dir, 'images')
        label_dir = os.path.join(source_dir, 'labels')
        
        if os.path.isdir(img_dir) and os.path.isdir(label_dir):
            structure['flat'] = {
                'path': source_dir,
                'images_dir': img_dir,
                'labels_dir': label_dir,
                'image_count': count_files(img_dir, 'jpg') + \
                               count_files(img_dir, 'jpeg') + \
                               count_files(img_dir, 'png'),
                'label_count': count_files(label_dir, 'txt')
            }
    
    return structure


def fix_dataset_structure(source_dir, target_dir, dry_run=False):
    """
    Fix the dataset structure by properly copying files from source to target
    while maintaining the train/val/test splits.
    """
    # Ensure source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist")
        return False
    
    # Create target directory if it doesn't exist
    if not dry_run and not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    # Analyze source structure
    source_structure = get_split_structure(source_dir)
    if not source_structure:
        print(f"Error: Could not identify dataset structure in {source_dir}")
        return False
    
    print(f"\nSource structure ({source_dir}):")
    total_source_images = 0
    total_source_labels = 0
    
    for split_name, split_info in source_structure.items():
        print(f"  {split_name}: {split_info['image_count']} images, {split_info['label_count']} labels")
        total_source_images += split_info['image_count']
        total_source_labels += split_info['label_count']
    
    print(f"  Total: {total_source_images} images, {total_source_labels} labels")
    
    # Analyze target structure (if it exists)
    target_structure = get_split_structure(target_dir) if os.path.exists(target_dir) else {}
    
    if target_structure:
        print(f"\nExisting target structure ({target_dir}):")
        total_target_images = 0
        total_target_labels = 0
        
        for split_name, split_info in target_structure.items():
            print(f"  {split_name}: {split_info['image_count']} images, {split_info['label_count']} labels")
            total_target_images += split_info['image_count']
            total_target_labels += split_info['label_count']
        
        print(f"  Total: {total_target_images} images, {total_target_labels} labels")
    
    # If dry run, just report statistics
    if dry_run:
        print("\nDRY RUN: No files will be copied")
        return True
    
    # Clear target directory if it exists
    if os.path.exists(target_dir):
        print(f"\nClearing existing files in {target_dir}...")
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    
    # Copy files maintaining the original structure
    print(f"\nCopying files from {source_dir} to {target_dir}...")
    
    copy_count = {
        'images': 0,
        'labels': 0
    }
    
    for split_name, split_info in source_structure.items():
        # Handle flat structure differently
        if split_name == 'flat':
            # Create images and labels directories
            target_images_dir = os.path.join(target_dir, 'images')
            target_labels_dir = os.path.join(target_dir, 'labels')
            
            os.makedirs(target_images_dir, exist_ok=True)
            os.makedirs(target_labels_dir, exist_ok=True)
            
            # Copy images
            for ext in ['jpg', 'jpeg', 'png']:
                for img_file in glob.glob(os.path.join(split_info['images_dir'], f'*.{ext}')):
                    img_name = os.path.basename(img_file)
                    target_img_path = os.path.join(target_images_dir, img_name)
                    shutil.copy2(img_file, target_img_path)
                    copy_count['images'] += 1
            
            # Copy labels
            for label_file in glob.glob(os.path.join(split_info['labels_dir'], '*.txt')):
                label_name = os.path.basename(label_file)
                target_label_path = os.path.join(target_labels_dir, label_name)
                shutil.copy2(label_file, target_label_path)
                copy_count['labels'] += 1
        else:
            # Create split directory structure
            target_split_dir = os.path.join(target_dir, split_name)
            target_images_dir = os.path.join(target_split_dir, 'images')
            target_labels_dir = os.path.join(target_split_dir, 'labels')
            
            os.makedirs(target_split_dir, exist_ok=True)
            os.makedirs(target_images_dir, exist_ok=True)
            os.makedirs(target_labels_dir, exist_ok=True)
            
            # Copy images
            if split_info['images_dir']:
                for ext in ['jpg', 'jpeg', 'png']:
                    for img_file in glob.glob(os.path.join(split_info['images_dir'], f'*.{ext}')):
                        img_name = os.path.basename(img_file)
                        target_img_path = os.path.join(target_images_dir, img_name)
                        shutil.copy2(img_file, target_img_path)
                        copy_count['images'] += 1
            
            # Copy labels
            if split_info['labels_dir']:
                for label_file in glob.glob(os.path.join(split_info['labels_dir'], '*.txt')):
                    label_name = os.path.basename(label_file)
                    target_label_path = os.path.join(target_labels_dir, label_name)
                    shutil.copy2(label_file, target_label_path)
                    copy_count['labels'] += 1
    
    # Copy classes.txt if it exists
    classes_file = os.path.join(source_dir, 'classes.txt')
    if os.path.exists(classes_file):
        shutil.copy2(classes_file, os.path.join(target_dir, 'classes.txt'))
    
    # Verify copy operation
    target_structure_after = get_split_structure(target_dir)
    
    print(f"\nVerification - target structure after copy ({target_dir}):")
    total_target_images_after = 0
    total_target_labels_after = 0
    
    for split_name, split_info in target_structure_after.items():
        print(f"  {split_name}: {split_info['image_count']} images, {split_info['label_count']} labels")
        total_target_images_after += split_info['image_count']
        total_target_labels_after += split_info['label_count']
    
    print(f"  Total: {total_target_images_after} images, {total_target_labels_after} labels")
    
    # Report results
    print("\nCopy operation complete:")
    print(f"  Copied {copy_count['images']} images, {copy_count['labels']} labels")
    
    if total_source_images != total_target_images_after or total_source_labels != total_target_labels_after:
        print("\nWARNING: File count mismatch after copy operation!")
        print(f"  Source: {total_source_images} images, {total_source_labels} labels")
        print(f"  Target: {total_target_images_after} images, {total_target_labels_after} labels")
        print(f"  Difference: {total_source_images - total_target_images_after} images, "
              f"{total_source_labels - total_target_labels_after} labels")
        return False
    else:
        print("\nSuccessfully copied all files with matching counts")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix dataset structure by properly copying files')
    parser.add_argument('--source', type=str, default='data/datasets_original',
                        help='Source directory path (default: data/datasets_original)')
    parser.add_argument('--target', type=str, default='data/powerlines/original',
                        help='Target directory path (default: data/powerlines/original)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Perform a dry run without copying files')
    
    args = parser.parse_args()
    
    # Make paths absolute if they're relative
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.isabs(args.source):
        args.source = os.path.join(root_dir, args.source)
    
    if not os.path.isabs(args.target):
        args.target = os.path.join(root_dir, args.target)
    
    print(f"Source directory: {args.source}")
    print(f"Target directory: {args.target}")
    
    success = fix_dataset_structure(args.source, args.target, args.dry_run)
    
    if success:
        exit(0)
    else:
        exit(1) 