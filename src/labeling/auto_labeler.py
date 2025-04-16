"""
Auto-labeling functionality for powerline sleeve detection.
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def auto_label_images(config, model_path=None, input_dir=None, output_dir=None):
    """
    Automatically label images using a pre-trained YOLOv8 model.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the pre-trained model to use for labeling
                   If None, uses a default model from config
        input_dir: Directory containing images to label
                  If None, uses the raw images directory from config
        output_dir: Directory to save labeled images and annotations
                   If None, uses the labeled images directory from config
    
    Returns:
        int: Number of images labeled
    """
    # Set default directories if not provided
    if input_dir is None:
        input_dir = config["paths"]["raw_images"]
    
    if output_dir is None:
        output_dir = config["paths"]["labeled_images"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the model
    model = load_model(config, model_path)
    
    if model is None:
        print("Failed to load model for auto-labeling")
        return 0
    
    # Get confidence threshold from config
    conf_threshold = config["labeling"]["auto_labeling"]["confidence_threshold"]
    
    # Get list of images to process
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                 glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                 glob.glob(os.path.join(input_dir, "*.png"))
    
    print(f"Found {len(image_paths)} images to auto-label")
    
    # Process each image
    labeled_count = 0
    for img_path in image_paths:
        success = process_image(model, img_path, output_dir, conf_threshold, config["labeling"]["classes"])
        if success:
            labeled_count += 1
    
    print(f"Auto-labeled {labeled_count} images")
    return labeled_count


def load_model(config, model_path=None):
    """
    Load a YOLOv8 model for auto-labeling.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the model to load, or None to use default
        
    Returns:
        YOLO model object or None if failed
    """
    try:
        # If model_path is not provided, try to use a previously trained model
        if model_path is None:
            # Try sleeve model first, then powerline model
            models_dir = config["paths"]["models_dir"]
            
            # Check if sleeve model exists
            sleeve_model = os.path.join(models_dir, "sleeve", "best.pt")
            if os.path.exists(sleeve_model):
                model_path = sleeve_model
                print(f"Using trained sleeve model: {model_path}")
            else:
                # Check if powerline model exists
                powerline_model = os.path.join(models_dir, "powerline", "best.pt")
                if os.path.exists(powerline_model):
                    model_path = powerline_model
                    print(f"Using trained powerline model: {model_path}")
                else:
                    # Use pretrained YOLOv8 model if no trained models exist
                    model_type = config["training"]["powerline"]["model_type"]
                    print(f"No trained models found, using pretrained {model_type}")
                    return YOLO(model_type)
        
        # Load the model
        model = YOLO(model_path)
        return model
    
    except Exception as e:
        print(f"Error loading model for auto-labeling: {e}")
        return None


def process_image(model, image_path, output_dir, conf_threshold, class_names):
    """
    Process a single image with the model and save the results.
    
    Args:
        model: YOLO model
        image_path: Path to the image
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        class_names: List of class names
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Run inference
        results = model(image_path, conf=conf_threshold)[0]
        
        # Get image filename without extension
        image_filename = os.path.basename(image_path)
        image_name = os.path.splitext(image_filename)[0]
        
        # Copy image to output directory
        image = cv2.imread(image_path)
        output_image_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(output_image_path, image)
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Create YOLO format annotation file
        output_label_path = os.path.join(output_dir, f"{image_name}.txt")
        
        with open(output_label_path, 'w') as f:
            # Process each detection
            for box in results.boxes:
                # Get class ID, coordinates, and confidence
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                
                # Skip if below confidence threshold
                if conf < conf_threshold:
                    continue
                
                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Write to file in YOLO format
                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
        
        return True
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False


def semi_supervised_labeling(config, initial_model_path, iterations=3):
    """
    Perform semi-supervised labeling using an iterative approach.
    1. Label images with the initial model
    2. Train a new model on the labeled images
    3. Use the new model to label more images
    4. Repeat
    
    Args:
        config: Configuration dictionary
        initial_model_path: Path to the initial model
        iterations: Number of iterations
        
    Returns:
        str: Path to the final model
    """
    print("Starting semi-supervised labeling")
    
    raw_images_dir = config["paths"]["raw_images"]
    labeled_dir = config["paths"]["labeled_images"]
    models_dir = config["paths"]["models_dir"]
    
    # Create temporary directories for each iteration
    temp_base_dir = os.path.join(labeled_dir, "semi_supervised")
    os.makedirs(temp_base_dir, exist_ok=True)
    
    current_model_path = initial_model_path
    
    for iteration in range(iterations):
        print(f"\nIteration {iteration+1}/{iterations}")
        
        # Create directories for this iteration
        iter_dir = os.path.join(temp_base_dir, f"iter_{iteration+1}")
        iter_images_dir = os.path.join(iter_dir, "images")
        iter_model_dir = os.path.join(iter_dir, "model")
        
        os.makedirs(iter_images_dir, exist_ok=True)
        os.makedirs(iter_model_dir, exist_ok=True)
        
        # Auto-label images using the current model
        print(f"Labeling images with model: {current_model_path}")
        auto_label_images(config, model_path=current_model_path, 
                         input_dir=raw_images_dir, output_dir=iter_images_dir)
        
        # Train a new model on the labeled images
        print("Training new model on labeled images")
        # In a real implementation, you would call the training module here
        # For now, we'll just use the same model for the next iteration
        
        # Update for next iteration
        current_model_path = os.path.join(iter_model_dir, "best.pt")
    
    # Copy final labeled images to the main labeled directory
    final_iter_dir = os.path.join(temp_base_dir, f"iter_{iterations}")
    final_iter_images_dir = os.path.join(final_iter_dir, "images")
    
    # In a real implementation, you would copy the final labeled images
    
    return current_model_path


if __name__ == "__main__":
    # Example usage when run as a script
    import yaml
    
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    auto_label_images(config)