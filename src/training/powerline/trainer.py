"""
Trainer for powerline detection models.
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO


def train_powerline_model(config, dataset_path=None, model_type=None, epochs=None):
    """
    Train a YOLOv8 model for powerline detection.
    
    Args:
        config: Configuration dictionary
        dataset_path: Path to the dataset YAML file
                     If None, uses the default from the processed dataset directory
        model_type: Type of YOLO model to use (e.g., 'yolov8n', 'yolov8s')
                   If None, uses the model_type from config
        epochs: Number of training epochs
               If None, uses the epochs from config
    
    Returns:
        str: Path to the trained model weights
    """
    # Set defaults from config if not provided
    if model_type is None:
        model_type = config["training"]["powerline"]["model_type"]
    
    if epochs is None:
        epochs = config["training"]["epochs"]
    
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
    
    # Load dataset YAML to adjust for powerline-only training
    with open(dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Modify dataset for powerline detection only
    class_names = config["training"]["powerline"]["classes"]
    
    # Create a new YAML for powerline detection
    powerline_dataset_path = os.path.join(
        os.path.dirname(dataset_path),
        "powerline_dataset.yaml"
    )
    
    # Update dataset configuration
    dataset_config["names"] = class_names
    dataset_config["nc"] = len(class_names)
    
    # Save modified dataset YAML
    with open(powerline_dataset_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    # Create output directory
    output_dir = os.path.join(config["paths"]["models_dir"], "powerline")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = YOLO(model_type)
    
    # Set training parameters
    params = {
        "data": powerline_dataset_path,
        "epochs": epochs,
        "imgsz": config["training"]["image_size"],
        "batch": config["training"]["batch_size"],
        "patience": config["training"]["patience"],
        "device": config["training"]["device"],
        "project": output_dir,
        "name": "train",
        "exist_ok": True,
        "pretrained": config["training"]["powerline"]["pretrained"],
        "optimizer": config["training"]["optimizer"],
        "lr0": config["training"]["learning_rate"]
    }
    
    print(f"Training powerline detection model with {model_type} for {epochs} epochs")
    print(f"Using dataset: {powerline_dataset_path}")
    
    # Train the model
    try:
        results = model.train(**params)
        print("Training completed successfully")
        
        # Copy best model to standard location
        run_dir = os.path.join(output_dir, "train")
        best_model = os.path.join(run_dir, "weights", "best.pt")
        
        if os.path.exists(best_model):
            # Copy to output directory root for easier access
            shutil.copy2(best_model, os.path.join(output_dir, "best.pt"))
            print(f"Best model saved to: {os.path.join(output_dir, 'best.pt')}")
            
            # Validate the model
            validate_model(model, powerline_dataset_path, output_dir)
            
            return os.path.join(output_dir, "best.pt")
        else:
            print("Warning: Best model not found after training")
            return None
    
    except Exception as e:
        print(f"Error during training: {e}")
        return None


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
    
    train_powerline_model(config) 