#!/usr/bin/env python
"""
Main entry point for the powerline sleeve detection pipeline.
This script runs the entire pipeline from image acquisition to detection.
"""

import os
import yaml
import argparse
import time
from pathlib import Path

# Import all pipeline components
from src.utils.config import load_config
from src.acquisition.route_processor import acquire_images
from src.labeling.auto_labeler import auto_label_images
from src.dataset.dataset_manager import prepare_dataset
from src.training.powerline.trainer import train_powerline_model
from src.training.sleeve.trainer import train_sleeve_model
from src.detection.detector import run_detection
from src.visualization.visualizer import visualize_results


def setup_directories(config):
    """Create necessary directories if they don't exist."""
    directories = [
        config["paths"]["data_dir"],
        config["paths"]["raw_images"],
        config["paths"]["labeled_images"],
        config["paths"]["processed_dataset"],
        config["paths"]["models_dir"],
        config["paths"]["results_dir"],
        config["acquisition"]["cache_dir"],
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main(config_path, stages=None):
    """
    Run the complete powerline sleeve detection pipeline.
    
    Args:
        config_path: Path to the configuration file
        stages: List of stages to run, or None to run all stages
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup directories
    setup_directories(config)
    
    # Get stages to run
    all_stages = [
        "acquisition", 
        "labeling", 
        "dataset", 
        "train_powerline", 
        "train_sleeve", 
        "detection", 
        "visualization"
    ]
    
    stages_to_run = stages if stages else all_stages
    print(f"Running stages: {', '.join(stages_to_run)}")
    
    start_time = time.time()
    
    # Run each stage if specified
    if "acquisition" in stages_to_run:
        print("\n--- Running Image Acquisition ---")
        acquire_images(config)
    
    if "labeling" in stages_to_run:
        print("\n--- Running Auto-Labeling ---")
        auto_label_images(config)
    
    if "dataset" in stages_to_run:
        print("\n--- Preparing Dataset ---")
        prepare_dataset(config)
    
    if "train_powerline" in stages_to_run:
        print("\n--- Training Powerline Detection Model ---")
        train_powerline_model(config)
    
    if "train_sleeve" in stages_to_run:
        print("\n--- Training Sleeve Detection Model ---")
        train_sleeve_model(config)
    
    if "detection" in stages_to_run:
        print("\n--- Running Detection ---")
        run_detection(config)
    
    if "visualization" in stages_to_run:
        print("\n--- Visualizing Results ---")
        visualize_results(config)
    
    elapsed_time = time.time() - start_time
    print(f"\nPipeline completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run powerline sleeve detection pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--stages", nargs="+", choices=[
        "acquisition", "labeling", "dataset", "train_powerline", 
        "train_sleeve", "detection", "visualization"
    ], help="Specific stages to run")
    
    args = parser.parse_args()
    main(args.config, args.stages) 