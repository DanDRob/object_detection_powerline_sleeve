"""
Trainer for sleeve detection models.
"""

import os
import yaml
import glob
import cv2
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def train_sleeve_model(config, dataset_path=None, model_type=None, epochs=None, 
                      powerline_model_path=None, use_two_stage=None):
    """
    Train a YOLOv8 model for sleeve detection.
    
    Args:
        config: Configuration dictionary
        dataset_path: Path to the dataset YAML file
                     If None, uses the default from the processed dataset directory
        model_type: Type of YOLO model to use (e.g., 'yolov8n', 'yolov8s')
                   If None, uses the model_type from config
        epochs: Number of training epochs
               If None, uses the epochs from config
        powerline_model_path: Path to the powerline detection model
                            If None, attempts to find it automatically
        use_two_stage: Whether to use two-stage detection (powerline first, then sleeve)
                      If None, uses the two_stage setting from config
    
    Returns:
        str: Path to the trained model weights
    """
    # Set defaults from config if not provided
    if model_type is None:
        model_type = config["training"]["sleeve"]["model_type"]
    
    if epochs is None:
        epochs = config["training"]["epochs"]
    
    if use_two_stage is None:
        use_two_stage = config["training"]["sleeve"]["two_stage"]
    
    # Get paths and create directories
    sleeve_output_dir = os.path.join(config["paths"]["models_dir"], "sleeve")
    os.makedirs(sleeve_output_dir, exist_ok=True)
    
    if dataset_path is None:
        # Check if augmented dataset exists
        aug_dataset_path = os.path.join(
            config["paths"]["processed_dataset"],
            "dataset_augmented.yaml"
        )
        
        if os.path.exists(aug_dataset_path):
            dataset_path = aug_dataset_path
            print(f"Using augmented dataset: {dataset_path}")
        else:
            # Use standard dataset
            dataset_path = os.path.join(
                config["paths"]["processed_dataset"],
                "dataset.yaml"
            )
            print(f"Using standard dataset: {dataset_path}")
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # For two-stage detection, we need a pre-trained powerline model
    if use_two_stage:
        if powerline_model_path is None:
            # Try to find a powerline model
            powerline_model_path = os.path.join(
                config["paths"]["models_dir"],
                "powerline",
                "best.pt"
            )
            
            if not os.path.exists(powerline_model_path):
                print("No powerline model found for two-stage detection")
                print("Training a powerline model first...")
                
                from ..powerline.trainer import train_powerline_model
                powerline_model_path = train_powerline_model(config, dataset_path)
                
                if powerline_model_path is None or not os.path.exists(powerline_model_path):
                    print("Failed to train powerline model. Falling back to direct sleeve detection.")
                    use_two_stage = False
        
        if use_two_stage and os.path.exists(powerline_model_path):
            print(f"Using two-stage detection with powerline model: {powerline_model_path}")
            
            # Create a sleeve-focused dataset by cropping around powerlines
            cropped_dataset_dir = os.path.join(
                config["paths"]["processed_dataset"],
                "sleeve_focused"
            )
            
            # Extract focused dataset
            create_sleeve_focused_dataset(
                config,
                powerline_model_path,
                dataset_path,
                cropped_dataset_dir
            )
            
            # Create YOLO dataset YAML for the focused dataset
            sleeve_dataset_path = os.path.join(
                cropped_dataset_dir,
                "sleeve_dataset.yaml"
            )
            
            # Use the sleeve-focused dataset for training
            dataset_path = sleeve_dataset_path
    
    # Load dataset YAML to adjust for sleeve-only training if needed
    with open(dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Modify dataset for sleeve detection only if not already using a sleeve-focused dataset
    if not use_two_stage or not dataset_path.endswith("sleeve_dataset.yaml"):
        class_names = config["training"]["sleeve"]["classes"]
        
        # Create a new YAML for sleeve detection
        sleeve_dataset_path = os.path.join(
            os.path.dirname(dataset_path),
            "sleeve_dataset.yaml"
        )
        
        # Update dataset configuration
        dataset_config["names"] = class_names
        dataset_config["nc"] = len(class_names)
        
        # Save modified dataset YAML
        with open(sleeve_dataset_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        dataset_path = sleeve_dataset_path
    
    # Initialize model
    model = YOLO(model_type)
    
    # Set training parameters
    params = {
        "data": dataset_path,
        "epochs": epochs,
        "imgsz": config["training"]["image_size"],
        "batch": config["training"]["batch_size"],
        "patience": config["training"]["patience"],
        "device": config["training"]["device"],
        "project": sleeve_output_dir,
        "name": "train",
        "exist_ok": True,
        "pretrained": config["training"]["sleeve"]["pretrained"],
        "optimizer": config["training"]["optimizer"],
        "lr0": config["training"]["learning_rate"]
    }
    
    print(f"Training sleeve detection model with {model_type} for {epochs} epochs")
    print(f"Using dataset: {dataset_path}")
    
    # Train the model
    try:
        results = model.train(**params)
        print("Training completed successfully")
        
        # Copy best model to standard location
        run_dir = os.path.join(sleeve_output_dir, "train")
        best_model = os.path.join(run_dir, "weights", "best.pt")
        
        if os.path.exists(best_model):
            # Copy to output directory root for easier access
            shutil.copy2(best_model, os.path.join(sleeve_output_dir, "best.pt"))
            
            # Create a metadata file indicating if this is a two-stage model
            metadata = {
                "two_stage": use_two_stage,
                "powerline_model": powerline_model_path if use_two_stage else None
            }
            
            with open(os.path.join(sleeve_output_dir, "metadata.yaml"), 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            print(f"Best model saved to: {os.path.join(sleeve_output_dir, 'best.pt')}")
            
            # Validate the model
            validate_model(model, dataset_path, sleeve_output_dir)
            
            return os.path.join(sleeve_output_dir, "best.pt")
        else:
            print("Warning: Best model not found after training")
            return None
    
    except Exception as e:
        print(f"Error during training: {e}")
        return None


def create_sleeve_focused_dataset(config, powerline_model_path, source_dataset_path, 
                                 output_dir, expansion_factor=1.5):
    """
    Create a sleeve-focused dataset by cropping around detected powerlines.
    
    Args:
        config: Configuration dictionary
        powerline_model_path: Path to the powerline detection model
        source_dataset_path: Path to the source dataset YAML
        output_dir: Directory to save the focused dataset
        expansion_factor: Factor to expand the powerline bounding box by
        
    Returns:
        str: Path to the created dataset YAML
    """
    # Load source dataset config
    with open(source_dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    base_path = dataset_config["path"]
    
    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)
    
    # Load powerline model
    powerline_model = YOLO(powerline_model_path)
    
    # Process each split
    for split in ["train", "val", "test"]:
        split_images_dir = os.path.join(base_path, split, "images")
        split_labels_dir = os.path.join(base_path, split, "labels")
        
        # Get all image files
        image_files = glob.glob(os.path.join(split_images_dir, "*.jpg")) + \
                     glob.glob(os.path.join(split_images_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(split_images_dir, "*.png"))
        
        print(f"Processing {len(image_files)} images for {split} split")
        
        # Process each image
        for img_file in image_files:
            img_filename = os.path.basename(img_file)
            img_name = os.path.splitext(img_filename)[0]
            label_file = os.path.join(split_labels_dir, f"{img_name}.txt")
            
            # Skip if no label file exists
            if not os.path.exists(label_file):
                continue
            
            # Detect powerlines in the image
            results = powerline_model(img_file, conf=0.4)[0]
            
            # Load the image
            image = cv2.imread(img_file)
            if image is None:
                continue
            
            img_height, img_width = image.shape[:2]
            
            # Get powerline detections
            powerline_boxes = []
            if len(results.boxes) > 0:
                for box in results.boxes:
                    # Only consider powerline class (class 0)
                    if int(box.cls.item()) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        powerline_boxes.append((x1, y1, x2, y2))
            
            # If no powerlines detected, use an alternative method
            if not powerline_boxes:
                # Try to use the ground truth labels
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5 and int(parts[0]) == 0:  # Class 0 (powerline)
                            # Convert YOLO format to pixel coordinates
                            x_center = float(parts[1]) * img_width
                            y_center = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height
                            
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            
                            powerline_boxes.append((x1, y1, x2, y2))
            
            if not powerline_boxes:
                # Still no powerlines, skip this image
                continue
            
            # Process each powerline region
            for i, (x1, y1, x2, y2) in enumerate(powerline_boxes):
                # Expand box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                box_width = (x2 - x1)
                box_height = (y2 - y1)
                
                # Expand by the expansion factor
                expanded_width = box_width * expansion_factor
                expanded_height = box_height * expansion_factor
                
                # Calculate new coordinates
                new_x1 = max(0, center_x - expanded_width / 2)
                new_y1 = max(0, center_y - expanded_height / 2)
                new_x2 = min(img_width, center_x + expanded_width / 2)
                new_y2 = min(img_height, center_y + expanded_height / 2)
                
                # Crop image
                crop_img = image[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]
                
                # Skip if crop is too small
                if crop_img.shape[0] < 10 or crop_img.shape[1] < 10:
                    continue
                
                # Save cropped image
                crop_filename = f"{img_name}_powerline_{i}.jpg"
                crop_path = os.path.join(output_dir, split, "images", crop_filename)
                cv2.imwrite(crop_path, crop_img)
                
                # Adjust annotations for the crop
                crop_labels = []
                
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        
                        cls_id = int(parts[0])
                        
                        # For the sleeve-focused dataset, we only care about sleeves (class 1)
                        if cls_id != 1:
                            continue
                        
                        # Convert YOLO format to pixel coordinates
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height
                        
                        # Calculate box coordinates
                        box_x1 = x_center - width / 2
                        box_y1 = y_center - height / 2
                        box_x2 = x_center + width / 2
                        box_y2 = y_center + height / 2
                        
                        # Check if the box intersects with the crop
                        if (box_x2 < new_x1 or box_x1 > new_x2 or
                            box_y2 < new_y1 or box_y1 > new_y2):
                            continue
                        
                        # Adjust coordinates to the crop
                        crop_box_x1 = max(0, box_x1 - new_x1)
                        crop_box_y1 = max(0, box_y1 - new_y1)
                        crop_box_x2 = min(new_x2 - new_x1, box_x2 - new_x1)
                        crop_box_y2 = min(new_y2 - new_y1, box_y2 - new_y1)
                        
                        # Convert back to YOLO format
                        crop_width = new_x2 - new_x1
                        crop_height = new_y2 - new_y1
                        
                        crop_x_center = (crop_box_x1 + crop_box_x2) / 2 / crop_width
                        crop_y_center = (crop_box_y1 + crop_box_y2) / 2 / crop_height
                        crop_width = (crop_box_x2 - crop_box_x1) / crop_width
                        crop_height = (crop_box_y2 - crop_box_y1) / crop_height
                        
                        # Add adjusted label (as class 0 since we only care about sleeves)
                        crop_labels.append(f"0 {crop_x_center} {crop_y_center} {crop_width} {crop_height}")
                
                # Save annotations if any boxes intersect with the crop
                if crop_labels:
                    crop_label_path = os.path.join(
                        output_dir, split, "labels", 
                        f"{img_name}_powerline_{i}.txt"
                    )
                    
                    with open(crop_label_path, 'w') as f:
                        f.write("\n".join(crop_labels))
                else:
                    # Remove the image if no relevant annotations
                    if os.path.exists(crop_path):
                        os.remove(crop_path)
    
    # Create YOLO dataset YAML
    output_yaml_path = os.path.join(output_dir, "sleeve_dataset.yaml")
    
    yaml_content = {
        "path": os.path.abspath(output_dir),
        "train": os.path.join("train", "images"),
        "val": os.path.join("val", "images"),
        "test": os.path.join("test", "images"),
        "nc": 1,
        "names": ["sleeve"]
    }
    
    with open(output_yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    return output_yaml_path


def validate_model(model, dataset_path, output_dir):
    """
    Validate a trained model on the validation set.
    
    Args:
        model: YOLO model instance
        dataset_path: Path to the dataset YAML file
        output_dir: Directory to save validation results
    """
    try:
        print("Validating model on validation dataset...")
        
        # Set validation parameters
        val_params = {
            "data": dataset_path,
            "split": "val",
            "project": output_dir,
            "name": "validate",
            "exist_ok": True
        }
        
        # Run validation
        metrics = model.val(**val_params)
        
        # Print validation metrics
        if hasattr(metrics, "box"):
            print(f"Validation metrics:")
            print(f"  mAP50: {metrics.box.map50:.4f}")
            print(f"  mAP50-95: {metrics.box.map:.4f}")
            print(f"  Precision: {metrics.box.p:.4f}")
            print(f"  Recall: {metrics.box.r:.4f}")
    
    except Exception as e:
        print(f"Error during validation: {e}")


if __name__ == "__main__":
    # Example usage when run as a script
    import yaml
    
    with open("../../../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    train_sleeve_model(config) 