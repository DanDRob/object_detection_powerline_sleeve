"""
Dataset visualization utilities for powerline and sleeve detection datasets.
"""

import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path
import yaml
import argparse
from collections import Counter, defaultdict


def visualize_dataset(dataset_dir, output_dir=None, sample_size=16, grid_size=(4, 4), 
                      color=(0, 0, 255), thickness=2, random_seed=42, split=None):
    """
    Visualize a sample of images from a dataset with their bounding boxes.
    
    Args:
        dataset_dir: Root directory containing split subdirectories with 'images' and 'labels'
        output_dir: Directory to save visualization outputs. If None, uses dataset_dir
        sample_size: Number of images to sample from the dataset
        grid_size: Tuple (rows, cols) for the visualization grid
        color: Bounding box color in BGR format
        thickness: Bounding box line thickness
        random_seed: Seed for random sampling
        split: Specific split to visualize ('train', 'val', 'test'). If None, samples from all splits.
        
    Returns:
        str: Path to the saved visualization image file
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Set default output directory
    if output_dir is None:
        output_dir = dataset_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which splits to process
    splits = ['train', 'val', 'test'] if split is None else [split]
    valid_splits = []
    
    valid_image_files = []
    
    # Process each split
    for current_split in splits:
        split_dir = os.path.join(dataset_dir, current_split)
        
        # Skip if the split directory doesn't exist
        if not os.path.exists(split_dir):
            continue
            
        valid_splits.append(current_split)
        
        # Get paths to images and labels for this split
        image_dir = os.path.join(split_dir, "images")
        label_dir = os.path.join(split_dir, "labels")
        
        # Skip if the image or label directory doesn't exist
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            continue
        
        # Get all image files for this split
        image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                      glob.glob(os.path.join(image_dir, "*.jpeg")) + \
                      glob.glob(os.path.join(image_dir, "*.png"))
        
        print(f"Found {len(image_files)} images in {current_split} split")
        
        # Only keep images that have corresponding label files
        for img_file in image_files:
            img_filename = os.path.basename(img_file)
            img_name = os.path.splitext(img_filename)[0]
            label_file = os.path.join(label_dir, f"{img_name}.txt")
            
            if os.path.exists(label_file):
                valid_image_files.append((img_file, label_file, current_split))
    
    if not valid_splits:
        raise ValueError(f"No valid splits found in {dataset_dir}")
    
    print(f"Found {len(valid_image_files)} images with valid labels across {', '.join(valid_splits)} splits")
    
    # Sample images if needed
    if len(valid_image_files) > sample_size:
        sampled_files = random.sample(valid_image_files, sample_size)
    else:
        sampled_files = valid_image_files
        
    # Prepare the grid
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    
    # Flatten axes for easier indexing if grid_size > 1x1
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Hide all axes initially
    for ax in axes:
        ax.axis('off')
    
    # Load and visualize each sampled image
    for i, (img_file, label_file, current_split) in enumerate(sampled_files):
        if i >= len(axes):
            break
            
        # Load image
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: Could not read image {img_file}")
            continue
            
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Load and parse YOLO labels
        bboxes, class_ids = parse_yolo_annotations(label_file)
        
        # Display image
        axes[i].imshow(img_rgb)
        
        # Draw bounding boxes
        for bbox, class_id in zip(bboxes, class_ids):
            x_center, y_center, w, h = bbox
            
            # Convert normalized coordinates to pixel coordinates
            x = int((x_center - w/2) * width)
            y = int((y_center - h/2) * height)
            w = int(w * width)
            h = int(h * height)
            
            # Create rectangle patch
            rect = Rectangle((x, y), w, h, linewidth=thickness, 
                            edgecolor=[c/255 for c in reversed(color)], 
                            facecolor='none')
            axes[i].add_patch(rect)
        
        # Add title with filename, split, and box count
        img_name = os.path.basename(img_file)
        axes[i].set_title(f"{img_name} ({current_split})\n{len(bboxes)} boxes", fontsize=9)
    
    # Set the overall title
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    split_info = f" ({', '.join(valid_splits)} splits)" if split is None else f" ({split} split)"
    fig.suptitle(f"{dataset_name} Dataset Sample{split_info}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    split_suffix = f"_{split}" if split else "_all_splits"
    output_path = os.path.join(output_dir, f"{dataset_name}_sample{split_suffix}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Visualization saved to {output_path}")
    return output_path


def parse_yolo_annotations(label_file):
    """
    Parse YOLO format annotation file into bounding boxes and class IDs.
    
    Args:
        label_file: Path to YOLO annotation file
        
    Returns:
        tuple: (bboxes, class_ids) where bboxes is a list of [x_center, y_center, width, height]
               in normalized coordinates and class_ids is a list of integer class IDs
    """
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
                    
                    bboxes.append([x_center, y_center, width, height])
                    class_ids.append(class_id)
                except (ValueError, IndexError):
                    print(f"Warning: Invalid bounding box format in {label_file}")
    
    return bboxes, class_ids


def analyze_dataset(dataset_dir):
    """
    Analyze dataset statistics to determine image counts and distributions.
    
    Args:
        dataset_dir: Root directory containing split subdirectories with 'images' and 'labels'
        
    Returns:
        dict: Dictionary containing dataset statistics
    """
    # Initialize statistics dictionary
    stats = {
        "total_images": 0,
        "images_with_labels": 0,
        "images_without_labels": 0,
        "images_with_objects": 0,
        "images_without_objects": 0,
        "total_objects": 0,
        "objects_per_class": defaultdict(int),
        "split_distribution": defaultdict(int),
    }
    
    # Determine which splits to process
    splits = ['train', 'val', 'test']
    
    # Process each split
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        
        # Skip if the split directory doesn't exist
        if not os.path.exists(split_dir):
            continue
            
        image_dir = os.path.join(split_dir, "images")
        label_dir = os.path.join(split_dir, "labels")
        
        # Skip if the image or label directory doesn't exist
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            continue
        
        # Get all image files for this split
        image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                     glob.glob(os.path.join(image_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(image_dir, "*.png"))
        
        # Update split distribution
        stats["split_distribution"][split] = len(image_files)
        stats["total_images"] += len(image_files)
        
        # Process each image
        for img_file in image_files:
            img_filename = os.path.basename(img_file)
            img_name = os.path.splitext(img_filename)[0]
            label_file = os.path.join(label_dir, f"{img_name}.txt")
            
            # Check if image has a corresponding label file
            if os.path.exists(label_file):
                stats["images_with_labels"] += 1
                
                # Parse the label file
                bboxes, class_ids = parse_yolo_annotations(label_file)
                
                # Update statistics
                stats["total_objects"] += len(bboxes)
                
                if len(bboxes) > 0:
                    stats["images_with_objects"] += 1
                    
                    # Count objects per class
                    for class_id in class_ids:
                        stats["objects_per_class"][class_id] += 1
                else:
                    stats["images_without_objects"] += 1
            else:
                stats["images_without_labels"] += 1
                
    return stats


def visualize_dataset_stats(datasets, output_dir=None, title="Dataset Manipulation Results"):
    """
    Create statistical visualizations for datasets, including pie charts and bar charts.
    
    Args:
        datasets: Dictionary mapping dataset names to their statistics
        output_dir: Directory to save visualization outputs
        title: Overall title for the visualization
        
    Returns:
        str: Path to the saved visualization image file
    """
    if not datasets:
        print("No valid datasets to visualize")
        return None
    
    # Ensure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Generate plots
    ax1 = fig.add_subplot(gs[0, 0])  # Original dataset composition
    ax2 = fig.add_subplot(gs[0, 1])  # Balanced dataset composition
    ax3 = fig.add_subplot(gs[1, 0])  # Image distribution by split
    ax4 = fig.add_subplot(gs[1, 1])  # Total images in each dataset
    
    # Prepare dataset names and image counts for bar chart
    dataset_names = list(datasets.keys())
    image_counts = [stats["total_images"] for stats in datasets.values() if stats]
    
    # Prepare data for pie charts (assuming first dataset is original, second is balanced)
    if "original" in datasets and datasets["original"]:
        orig_stats = datasets["original"]
        orig_with_objects = orig_stats["images_with_objects"]
        orig_without_objects = orig_stats["total_images"] - orig_with_objects
        
        # Plot original dataset pie chart
        ax1.pie([orig_with_objects, orig_without_objects], 
                labels=["With Powerlines", "Without Powerlines"],
                autopct='%1.1f%%', 
                colors=['#77c9aa', '#ff8a65'],
                wedgeprops={'width': 0.5, 'edgecolor': 'w'},
                textprops={'fontsize': 10})
        ax1.set_title("Original Dataset Composition", fontsize=12)
    
    if "balanced" in datasets and datasets["balanced"]:
        bal_stats = datasets["balanced"]
        bal_with_objects = bal_stats["images_with_objects"]
        bal_without_objects = bal_stats["total_images"] - bal_with_objects
        
        # Plot balanced dataset pie chart
        ax2.pie([bal_with_objects, bal_without_objects], 
                labels=["With Powerlines", "Without Powerlines"],
                autopct='%1.1f%%', 
                colors=['#77c9aa', '#ff8a65'],
                wedgeprops={'width': 0.5, 'edgecolor': 'w'},
                textprops={'fontsize': 10})
        ax2.set_title("Balanced Dataset Composition", fontsize=12)
    
    # Plot split distribution if available
    all_splits = set()
    for stats in datasets.values():
        if stats and "split_distribution" in stats:
            all_splits.update(stats["split_distribution"].keys())
    
    all_splits = sorted(list(all_splits))
    if all_splits:
        bar_width = 0.8 / len(datasets)
        bar_positions = np.arange(len(all_splits))
        
        for i, (dataset_name, stats) in enumerate(datasets.items()):
            if not stats or "split_distribution" not in stats:
                continue
                
            split_counts = [stats["split_distribution"].get(split, 0) for split in all_splits]
            
            # Determine color based on dataset name
            if "original" in dataset_name.lower() and "with" in dataset_name.lower():
                color = '#77c9aa'  # teal for original with powerlines
            elif "original" in dataset_name.lower():
                color = '#5c6bc0'  # blue for original without powerlines
            elif "balanced" in dataset_name.lower() and "with" in dataset_name.lower():
                color = '#ff8a65'  # orange for balanced with powerlines
            elif "balanced" in dataset_name.lower():
                color = '#9575cd'  # purple for balanced without powerlines
            else:
                color = '#81c784'  # green for others
            
            ax3.bar(bar_positions + i*bar_width - (len(datasets)-1)*bar_width/2, 
                    split_counts, width=bar_width, label=dataset_name, color=color)
        
        ax3.set_xticks(bar_positions)
        ax3.set_xticklabels(all_splits)
        ax3.set_xlabel("Split")
        ax3.set_ylabel("Number of Images")
        ax3.set_title("Image Distribution by Split")
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        ax3.legend()
    
    # Plot total images bar chart
    colors = ['#5c6bc0', '#77c9aa', '#81c784']  # blue, teal, green
    bars = ax4.bar(dataset_names, image_counts, color=colors[:len(dataset_names)])
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    ax4.set_xlabel("Dataset")
    ax4.set_ylabel("Number of Images")
    ax4.set_title("Total Images in Each Dataset")
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set the overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    if output_dir:
        output_path = os.path.join(output_dir, "dataset_stats.png")
    else:
        output_path = "dataset_stats.png"
        
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Dataset statistics visualization saved to {output_path}")
    return output_path


def visualize_powerline_datasets(config=None, output_dir=None, visualize_samples=True, visualize_stats=True):
    """
    Visualize both original and augmented powerline datasets.
    
    Args:
        config: Configuration dictionary. If None, loads from default config path
        output_dir: Directory to save visualization outputs. If None, uses dataset directories
        visualize_samples: Whether to visualize dataset samples (grid of images)
        visualize_stats: Whether to visualize dataset statistics (pie charts, bar charts)
        
    Returns:
        dict: Dictionary of paths to visualization output files
    """
    # Load config if not provided
    if config is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  "config.yaml")
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = {}
    
    # Get dataset paths
    data_dir = config.get("paths", {}).get("data_dir", "data")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), data_dir)
    
    # Define paths for powerline datasets
    powerline_original = os.path.join(data_dir, "powerlines", "original")
    powerline_augmented = os.path.join(data_dir, "powerlines", "augmented")
    
    # Create output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    outputs = {}
    datasets_stats = {}
    
    # Process original dataset if it exists
    if os.path.exists(powerline_original):
        print(f"Processing original powerline dataset: {powerline_original}")
        
        # Visualize sample images (combined across splits)
        if visualize_samples:
            try:
                output_path = visualize_dataset(
                    powerline_original, 
                    output_dir=output_dir if output_dir else powerline_original,
                    sample_size=16,
                    grid_size=(4, 4)
                )
                outputs["original_samples"] = output_path
                
                # Also visualize each split separately
                for split in ['train', 'val', 'test']:
                    split_path = os.path.join(powerline_original, split)
                    if os.path.exists(split_path):
                        try:
                            output_path = visualize_dataset(
                                powerline_original, 
                                output_dir=output_dir if output_dir else powerline_original,
                                sample_size=16,
                                grid_size=(4, 4),
                                split=split
                            )
                            outputs[f"original_{split}_samples"] = output_path
                        except Exception as e:
                            print(f"Error visualizing original {split} split: {e}")
            except Exception as e:
                print(f"Error visualizing original dataset samples: {e}")
        
        # Analyze dataset statistics
        if visualize_stats:
            try:
                stats = analyze_dataset(powerline_original)
                if stats:
                    datasets_stats["original"] = stats
            except Exception as e:
                print(f"Error analyzing original dataset: {e}")
    else:
        print(f"Original powerline dataset not found at {powerline_original}")
    
    # Process augmented dataset if it exists
    if os.path.exists(powerline_augmented):
        print(f"Processing augmented powerline dataset: {powerline_augmented}")
        
        # Visualize sample images (combined across splits)
        if visualize_samples:
            try:
                output_path = visualize_dataset(
                    powerline_augmented, 
                    output_dir=output_dir if output_dir else powerline_augmented,
                    sample_size=16,
                    grid_size=(4, 4)
                )
                outputs["augmented_samples"] = output_path
                
                # Also visualize each split separately
                for split in ['train', 'val', 'test']:
                    split_path = os.path.join(powerline_augmented, split)
                    if os.path.exists(split_path):
                        try:
                            output_path = visualize_dataset(
                                powerline_augmented, 
                                output_dir=output_dir if output_dir else powerline_augmented,
                                sample_size=16,
                                grid_size=(4, 4),
                                split=split
                            )
                            outputs[f"augmented_{split}_samples"] = output_path
                        except Exception as e:
                            print(f"Error visualizing augmented {split} split: {e}")
            except Exception as e:
                print(f"Error visualizing augmented dataset samples: {e}")
        
        # Analyze dataset statistics
        if visualize_stats:
            try:
                stats = analyze_dataset(powerline_augmented)
                if stats:
                    datasets_stats["balanced"] = stats
            except Exception as e:
                print(f"Error analyzing augmented dataset: {e}")
    else:
        print(f"Augmented powerline dataset not found at {powerline_augmented}")
    
    # Generate combined statistics visualization
    if visualize_stats and datasets_stats:
        try:
            stats_path = visualize_dataset_stats(
                datasets_stats,
                output_dir=output_dir if output_dir else data_dir
            )
            if stats_path:
                outputs["statistics"] = stats_path
        except Exception as e:
            print(f"Error generating statistics visualization: {e}")
    
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset visualization tool')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, help='Directory to save visualization outputs')
    parser.add_argument('--samples-only', action='store_true', help='Only visualize sample images')
    parser.add_argument('--stats-only', action='store_true', help='Only visualize dataset statistics')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], 
                        help='Specific split to visualize')
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    
    # Determine what to visualize
    visualize_samples = not args.stats_only
    visualize_stats = not args.samples_only
    
    if args.samples_only and args.stats_only:
        print("Warning: Both --samples-only and --stats-only specified. Visualizing both.")
        visualize_samples = visualize_stats = True
    
    # Visualize powerline datasets
    output_paths = visualize_powerline_datasets(
        config, 
        args.output_dir,
        visualize_samples=visualize_samples,
        visualize_stats=visualize_stats
    )
    
    # Print summary
    print("\nVisualization Summary:")
    for output_type, path in output_paths.items():
        print(f"- {output_type}: {path}") 