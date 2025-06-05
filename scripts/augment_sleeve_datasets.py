import os
import json
import random
import shutil
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = Path('data/powerlines/final_dataset/images_labels/')
POWERLINES_DEST_DIR = Path('data/sleeves/sleeves_powerlines/')
ALL_DEST_DIR = Path('data/sleeves/sleeves_all/')
NUM_IMAGES_TO_ADD = 350

def categorize_images(source_dir: Path) -> (list, list):
    """Categorizes images into those with and without labels."""
    with_labels = []
    without_labels = []
    
    for json_path in source_dir.glob('*.json'):
        image_path = json_path.with_suffix('.jpg')
        if not image_path.exists():
            image_path = json_path.with_suffix('.png') # Check for .png as well
            if not image_path.exists():
                print(f"Warning: No corresponding image for {json_path.name}")
                continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # An image has labels if 'shapes' is not empty
            if data.get('shapes'):
                with_labels.append((image_path, json_path))
            else:
                without_labels.append((image_path, json_path))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {json_path}: {e}")
            
    return with_labels, without_labels

def copy_files(file_pairs: list, dest_dir: Path, clear_labels: bool):
    """Copies a list of image/json pairs to a destination directory."""
    if not dest_dir.exists():
        print(f"Destination {dest_dir} not found. It was expected to exist.")
        return
        
    for img_path, json_path in file_pairs:
        # Copy the image file directly
        shutil.copy(img_path, dest_dir)
        
        # If clearing labels, read, modify, and write the JSON
        if clear_labels:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                data['shapes'] = [] # Remove all annotations
                data['imageData'] = None # Set imageData to null
                
                dest_json_path = dest_dir / json_path.name
                with open(dest_json_path, 'w') as f:
                    json.dump(data, f, indent=2)

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Could not process or clear labels for {json_path.name}: {e}")
        else:
            # Otherwise, just copy the original JSON
            shutil.copy(json_path, dest_dir)

def main():
    """Main function to run the augmentation script."""
    print("Step 1: Categorizing images from source directory...")
    with_powerlines, without_powerlines = categorize_images(SOURCE_DIR)
    
    print(f"Found {len(with_powerlines)} images with powerlines.")
    print(f"Found {len(without_powerlines)} images without powerlines.")

    # --- Augment sleeves_powerlines dataset ---
    print(f"\nStep 2: Augmenting '{POWERLINES_DEST_DIR}'...")
    if len(with_powerlines) < NUM_IMAGES_TO_ADD:
        print(f"Warning: Not enough images with powerlines ({len(with_powerlines)}) to meet the requirement of {NUM_IMAGES_TO_ADD}.")
        powerlines_to_copy = with_powerlines
    else:
        powerlines_to_copy = random.sample(with_powerlines, NUM_IMAGES_TO_ADD)
    
    copy_files(powerlines_to_copy, POWERLINES_DEST_DIR, clear_labels=True)
    print(f"Added {len(powerlines_to_copy)} images with powerlines (labels removed).")

    # --- Augment sleeves_all dataset ---
    print(f"\nStep 3: Augmenting '{ALL_DEST_DIR}'...")
    num_with = NUM_IMAGES_TO_ADD // 2
    num_without = NUM_IMAGES_TO_ADD - num_with

    if len(with_powerlines) < num_with:
        print(f"Warning: Not enough images with powerlines for the 'all' mix.")
        num_with = len(with_powerlines)
    
    if len(without_powerlines) < num_without:
         print(f"Warning: Not enough images without powerlines for the 'all' mix.")
         num_without = len(without_powerlines)

    all_mix_to_copy = []
    if with_powerlines:
        all_mix_to_copy.extend(random.sample(with_powerlines, num_with))
    if without_powerlines:
        all_mix_to_copy.extend(random.sample(without_powerlines, num_without))
    
    copy_files(all_mix_to_copy, ALL_DEST_DIR, clear_labels=True)
    print(f"Added {len(all_mix_to_copy)} total images ({num_with} with, {num_without} without powerlines) (labels removed).")
    
    print("\nAugmentation complete.")


if __name__ == '__main__':
    main() 