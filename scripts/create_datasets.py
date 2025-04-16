#!/usr/bin/env python3
from dataset_manipulator import dataset_manipulator
import os
from pathlib import Path

# Path to the original dataset
SOURCE_DATASET = os.path.join('data', 'powerlines', 'original')

def main():
    """
    Create balanced and powerline-only datasets from the original dataset
    """
    # Make sure SOURCE_DATASET is a full path
    source_path = Path(SOURCE_DATASET)
    if not source_path.is_absolute():
        # Get the project root directory
        project_root = Path(__file__).resolve().parent.parent
        source_path = project_root / source_path
    
    print(f"Creating datasets from: {source_path}")
    
    # Create the datasets
    result = dataset_manipulator(
        source_dataset_path=source_path,
        output_dir=None,  # Use default (parent of source)
        create_balanced=True,
        create_powerline_only=True
    )
    
    print("\nDataset creation complete!")
    if 'balanced' in result:
        print(f"Balanced dataset created at: {result['balanced']}")
    
    if 'powerline_only' in result:
        print(f"Powerline-only dataset created at: {result['powerline_only']}")
    
    print("\nTo use these datasets with YOLOv8, use the following YAML files:")
    
    if 'balanced' in result:
        yaml_path = os.path.join(result['balanced'], 'original_balanced.yaml')
        print(f"Balanced dataset: {yaml_path}")
    
    if 'powerline_only' in result:
        yaml_path = os.path.join(result['powerline_only'], 'original_powerline_only.yaml')
        print(f"Powerline-only dataset: {yaml_path}")

if __name__ == "__main__":
    main() 