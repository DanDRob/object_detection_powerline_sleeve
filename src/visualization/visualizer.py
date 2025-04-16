"""
Visualizer for powerline sleeve detection results.
"""

import os
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import folium
from pathlib import Path


def visualize_results(config, results_dir=None, output_dir=None):
    """
    Visualize detection results from a directory.
    
    Args:
        config: Configuration dictionary
        results_dir: Directory containing detection results
                    If None, uses the results directory from config
        output_dir: Directory to save visualization outputs
                   If None, uses the same directory as results_dir
                   
    Returns:
        dict: Dictionary of visualization outputs
    """
    # Set default directories if not provided
    if results_dir is None:
        results_dir = config["paths"]["results_dir"]
    
    if output_dir is None:
        output_dir = results_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON detection results
    json_files = glob.glob(os.path.join(results_dir, "*_detections.json"))
    
    if not json_files:
        print(f"No detection results found in {results_dir}")
        return {}
    
    print(f"Found {len(json_files)} detection result files")
    
    # Load all detections
    all_detections = {}
    for json_file in json_files:
        img_name = os.path.basename(json_file).replace("_detections.json", "")
        
        try:
            with open(json_file, 'r') as f:
                detections = json.load(f)
            
            all_detections[img_name] = detections
        except Exception as e:
            print(f"Error loading detections from {json_file}: {e}")
    
    # Create outputs
    outputs = {}
    
    # Generate summary visualizations
    summary_fig = create_detection_summary(all_detections)
    if summary_fig:
        summary_path = os.path.join(output_dir, "detection_summary.png")
        summary_fig.savefig(summary_path)
        plt.close(summary_fig)
        outputs["summary"] = summary_path
    
    # Generate map if enabled and location data is available
    if config["visualization"]["map"]["enabled"]:
        map_path = create_detection_map(
            config, all_detections, results_dir, output_dir
        )
        if map_path:
            outputs["map"] = map_path
    
    print(f"Visualization completed. Outputs saved to {output_dir}")
    return outputs


def create_detection_summary(all_detections):
    """
    Create a summary visualization of detection results.
    
    Args:
        all_detections: Dictionary mapping image names to detections
        
    Returns:
        matplotlib.figure.Figure: Figure object or None if failed
    """
    try:
        # Collect statistics
        total_images = len(all_detections)
        images_with_detections = sum(1 for dets in all_detections.values() if dets)
        total_detections = sum(len(dets) for dets in all_detections.values())
        
        # Get confidence distribution
        all_confidences = [
            det["confidence"] 
            for img_dets in all_detections.values() 
            for det in img_dets
        ]
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Detection counts
        detection_data = [
            images_with_detections,
            total_images - images_with_detections
        ]
        labels = [
            f"Images with detections ({images_with_detections})",
            f"Images without detections ({total_images - images_with_detections})"
        ]
        
        ax1.pie(
            detection_data, 
            labels=labels, 
            autopct='%1.1f%%',
            colors=['#66b3ff', '#d9d9d9']
        )
        ax1.set_title(f"Detection Coverage ({total_images} images)")
        
        # Confidence histogram
        if all_confidences:
            ax2.hist(
                all_confidences, 
                bins=10, 
                range=(0, 1), 
                alpha=0.7, 
                color='#66b3ff'
            )
            ax2.set_title(f"Confidence Distribution ({total_detections} detections)")
            ax2.set_xlabel("Confidence")
            ax2.set_ylabel("Count")
            ax2.grid(alpha=0.3)
        else:
            ax2.text(
                0.5, 0.5, 
                "No detections found", 
                ha='center', va='center',
                fontsize=12
            )
        
        fig.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error creating detection summary: {e}")
        return None


def create_detection_map(config, all_detections, results_dir, output_dir):
    """
    Create a map visualization of detection results if GPS data is available.
    
    Args:
        config: Configuration dictionary
        all_detections: Dictionary mapping image names to detections
        results_dir: Directory containing detection results
        output_dir: Directory to save visualization outputs
        
    Returns:
        str: Path to the created map HTML file or None if failed
    """
    try:
        # Check if we have location data
        metadata_file = os.path.join(results_dir, "metadata.json")
        if not os.path.exists(metadata_file):
            print("No metadata file with GPS coordinates found")
            return None
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check if metadata contains location data
        if not isinstance(metadata, dict) or not all(
            img in metadata and "latitude" in metadata[img] and "longitude" in metadata[img]
            for img in all_detections
        ):
            print("Metadata doesn't contain location information for all images")
            return None
        
        # Get map settings
        zoom_level = config["visualization"]["map"]["zoom_level"]
        marker_size = config["visualization"]["map"]["marker_size"]
        
        # Create a map centered on the average location
        valid_locations = [
            (metadata[img]["latitude"], metadata[img]["longitude"])
            for img in all_detections
            if img in metadata and "latitude" in metadata[img] and "longitude" in metadata[img]
        ]
        
        if not valid_locations:
            print("No valid location data found in metadata")
            return None
        
        avg_lat = sum(lat for lat, _ in valid_locations) / len(valid_locations)
        avg_lng = sum(lng for _, lng in valid_locations) / len(valid_locations)
        
        # Create map
        detection_map = folium.Map(
            location=[avg_lat, avg_lng],
            zoom_start=zoom_level,
            tiles="OpenStreetMap"
        )
        
        # Add markers for each image with detections
        for img_name, detections in all_detections.items():
            if img_name not in metadata:
                continue
            
            img_meta = metadata[img_name]
            if "latitude" not in img_meta or "longitude" not in img_meta:
                continue
            
            lat = img_meta["latitude"]
            lng = img_meta["longitude"]
            
            # Create marker with popup showing detection count
            detection_count = len(detections)
            
            if detection_count > 0:
                # Red marker for images with detections
                folium.CircleMarker(
                    location=[lat, lng],
                    radius=marker_size,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.7,
                    popup=f"{img_name}: {detection_count} sleeves detected"
                ).add_to(detection_map)
            else:
                # Blue marker for images without detections
                folium.CircleMarker(
                    location=[lat, lng],
                    radius=marker_size / 1.5,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.5,
                    popup=f"{img_name}: No sleeves detected"
                ).add_to(detection_map)
        
        # Save the map
        map_path = os.path.join(output_dir, "detection_map.html")
        detection_map.save(map_path)
        
        print(f"Detection map saved to {map_path}")
        return map_path
    
    except Exception as e:
        print(f"Error creating detection map: {e}")
        return None


def visualize_training_results(config, run_dir=None, output_dir=None):
    """
    Visualize training results from a training run.
    
    Args:
        config: Configuration dictionary
        run_dir: Directory containing training results
                If None, uses the most recent training run
        output_dir: Directory to save visualization outputs
                   If None, uses the same directory as run_dir
    
    Returns:
        dict: Dictionary of visualization outputs
    """
    # Load training results if available
    try:
        # Find the results.csv file from the training run
        if run_dir is None:
            # Try to find the most recent training run
            sleeve_dir = os.path.join(config["paths"]["models_dir"], "sleeve", "train")
            powerline_dir = os.path.join(config["paths"]["models_dir"], "powerline", "train")
            
            candidates = []
            for directory in [sleeve_dir, powerline_dir]:
                if os.path.exists(directory):
                    candidates.append(directory)
            
            if not candidates:
                print("No training run directories found")
                return {}
            
            # Select the most recently modified directory
            run_dir = max(candidates, key=os.path.getmtime)
        
        if output_dir is None:
            output_dir = run_dir
        
        # Check for results.csv file
        results_file = os.path.join(run_dir, "results.csv")
        if not os.path.exists(results_file):
            print(f"No results.csv file found in {run_dir}")
            return {}
        
        # Load results
        results = np.genfromtxt(results_file, delimiter=",", names=True)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        metrics_fig = plot_training_metrics(results)
        metrics_path = os.path.join(output_dir, "training_metrics.png")
        metrics_fig.savefig(metrics_path)
        plt.close(metrics_fig)
        
        return {"metrics": metrics_path}
    
    except Exception as e:
        print(f"Error visualizing training results: {e}")
        return {}


def plot_training_metrics(results):
    """
    Plot training metrics from results.csv.
    
    Args:
        results: NumPy structured array containing training metrics
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot metrics
    epochs = results["epoch"]
    
    # Box loss
    if "box_loss" in results.dtype.names:
        ax = axes[0, 0]
        train_box_loss = results["box_loss"]
        val_box_loss = results["val_box_loss"] if "val_box_loss" in results.dtype.names else None
        
        ax.plot(epochs, train_box_loss, 'b-', label='Train')
        if val_box_loss is not None:
            ax.plot(epochs, val_box_loss, 'r-', label='Validation')
        
        ax.set_title("Box Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()
    
    # Class loss (if available)
    if "cls_loss" in results.dtype.names:
        ax = axes[0, 1]
        train_cls_loss = results["cls_loss"]
        val_cls_loss = results["val_cls_loss"] if "val_cls_loss" in results.dtype.names else None
        
        ax.plot(epochs, train_cls_loss, 'b-', label='Train')
        if val_cls_loss is not None:
            ax.plot(epochs, val_cls_loss, 'r-', label='Validation')
        
        ax.set_title("Class Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()
    
    # mAP50
    if "metrics/mAP50(B)" in results.dtype.names:
        ax = axes[1, 0]
        map50 = results["metrics/mAP50(B)"]
        
        ax.plot(epochs, map50, 'g-')
        ax.set_title("mAP50")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP50")
        ax.grid(alpha=0.3)
    
    # mAP50-95
    if "metrics/mAP50-95(B)" in results.dtype.names:
        ax = axes[1, 1]
        map = results["metrics/mAP50-95(B)"]
        
        ax.plot(epochs, map, 'g-')
        ax.set_title("mAP50-95")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP50-95")
        ax.grid(alpha=0.3)
    
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage when run as a script
    import yaml
    
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    visualize_results(config) 