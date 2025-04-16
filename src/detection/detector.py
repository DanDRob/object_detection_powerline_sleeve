"""
Detector for powerline sleeves.
"""

import os
import yaml
import glob
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class SleeveDetector:
    """Detector for powerline sleeves, with support for two-stage detection."""
    
    def __init__(self, config, powerline_model_path=None, sleeve_model_path=None, 
                use_two_stage=None):
        """
        Initialize the sleeve detector.
        
        Args:
            config: Configuration dictionary
            powerline_model_path: Path to the powerline detection model
                                 If None, attempts to find it automatically
            sleeve_model_path: Path to the sleeve detection model
                              If None, attempts to find it automatically
            use_two_stage: Whether to use two-stage detection (powerline first, then sleeve)
                          If None, tries to determine from the sleeve model's metadata
        """
        self.config = config
        self.powerline_model = None
        self.sleeve_model = None
        self.use_two_stage = use_two_stage
        
        # Initialize models
        self._initialize_models(powerline_model_path, sleeve_model_path)
    
    def _initialize_models(self, powerline_model_path=None, sleeve_model_path=None):
        """
        Initialize detection models.
        
        Args:
            powerline_model_path: Path to the powerline detection model
            sleeve_model_path: Path to the sleeve detection model
        """
        # Find best models if paths not provided
        models_dir = self.config["paths"]["models_dir"]
        
        # Find sleeve model
        if sleeve_model_path is None:
            default_sleeve_model = os.path.join(models_dir, "sleeve", "best.pt")
            if os.path.exists(default_sleeve_model):
                sleeve_model_path = default_sleeve_model
            else:
                raise FileNotFoundError("No sleeve detection model found")
        
        # Load sleeve model and check for metadata
        self.sleeve_model = YOLO(sleeve_model_path)
        
        # Determine if two-stage detection should be used
        if self.use_two_stage is None:
            # Check for metadata
            metadata_path = os.path.join(os.path.dirname(sleeve_model_path), "metadata.yaml")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                self.use_two_stage = metadata.get("two_stage", False)
                stored_powerline_model = metadata.get("powerline_model")
                
                if self.use_two_stage and stored_powerline_model and os.path.exists(stored_powerline_model):
                    powerline_model_path = stored_powerline_model
            else:
                # No metadata, default to false
                self.use_two_stage = False
        
        # Load powerline model if two-stage detection is enabled
        if self.use_two_stage:
            if powerline_model_path is None:
                default_powerline_model = os.path.join(models_dir, "powerline", "best.pt")
                if os.path.exists(default_powerline_model):
                    powerline_model_path = default_powerline_model
                else:
                    raise FileNotFoundError("No powerline detection model found for two-stage detection")
            
            self.powerline_model = YOLO(powerline_model_path)
        
        print(f"Initialized {'two-stage' if self.use_two_stage else 'single-stage'} detector")
        if self.use_two_stage:
            print(f"  Powerline model: {powerline_model_path}")
        
        print(f"  Sleeve model: {sleeve_model_path}")
    
    def detect(self, image, conf_threshold=None, iou_threshold=None):
        """
        Detect sleeves in an image.
        
        Args:
            image: Image array (BGR format from OpenCV) or path to image file
            conf_threshold: Confidence threshold for detections
                          If None, uses the threshold from config
            iou_threshold: IoU threshold for non-maximum suppression
                         If None, uses the threshold from config
        
        Returns:
            list: List of detection dictionaries with format:
                 [{"class": class_id, "label": "sleeve", 
                   "box": [x1, y1, x2, y2], "confidence": conf}, ...]
        """
        # Set default thresholds from config if not provided
        if conf_threshold is None:
            conf_threshold = self.config["detection"]["confidence_threshold"]
        
        if iou_threshold is None:
            iou_threshold = self.config["detection"]["iou_threshold"]
        
        # Check if the image is a path or an array
        if isinstance(image, str):
            # Image path
            img_path = image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
        else:
            # Image array
            img = image
        
        # Get image dimensions
        img_height, img_width = img.shape[:2]
        
        # Two-stage detection
        if self.use_two_stage:
            return self._detect_two_stage(img, conf_threshold, iou_threshold)
        
        # Single-stage detection
        return self._detect_single_stage(img, conf_threshold, iou_threshold)
    
    def _detect_single_stage(self, image, conf_threshold, iou_threshold):
        """
        Perform single-stage sleeve detection.
        
        Args:
            image: Image array (BGR format from OpenCV)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
        
        Returns:
            list: List of detection dictionaries
        """
        # Run inference
        results = self.sleeve_model(
            image, 
            conf=conf_threshold, 
            iou=iou_threshold,
            verbose=False
        )[0]
        
        # Process results
        return self._process_detections(results)
    
    def _detect_two_stage(self, image, conf_threshold, iou_threshold):
        """
        Perform two-stage detection (powerline first, then sleeve).
        
        Args:
            image: Image array (BGR format from OpenCV)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
        
        Returns:
            list: List of detection dictionaries
        """
        # Stage 1: Detect powerlines
        powerline_results = self.powerline_model(
            image, 
            conf=conf_threshold, 
            iou=iou_threshold,
            verbose=False
        )[0]
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # If no powerlines detected, return empty list
        if len(powerline_results.boxes) == 0:
            return []
        
        # Collect all sleeve detections
        all_sleeve_detections = []
        
        # For each powerline, crop and detect sleeves
        for box in powerline_results.boxes:
            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Expand box for context
            expansion_factor = 1.5
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
            
            # Stage 2: Detect sleeves in cropped image
            sleeve_results = self.sleeve_model(
                crop_img, 
                conf=conf_threshold, 
                iou=iou_threshold,
                verbose=False
            )[0]
            
            # Process sleeve detections
            for sleeve_box in sleeve_results.boxes:
                # Get bounding box in crop coordinates
                crop_x1, crop_y1, crop_x2, crop_y2 = sleeve_box.xyxy[0].tolist()
                
                # Convert to original image coordinates
                orig_x1 = crop_x1 + new_x1
                orig_y1 = crop_y1 + new_y1
                orig_x2 = crop_x2 + new_x1
                orig_y2 = crop_y2 + new_y1
                
                # Add detection
                all_sleeve_detections.append({
                    "class": 0,  # Sleeve class
                    "label": "sleeve",
                    "box": [orig_x1, orig_y1, orig_x2, orig_y2],
                    "confidence": sleeve_box.conf.item()
                })
        
        return all_sleeve_detections
    
    def _process_detections(self, results):
        """
        Process YOLO detection results into a standardized format.
        
        Args:
            results: YOLO detection results
            
        Returns:
            list: List of detection dictionaries
        """
        detections = []
        
        # Process each detection
        for box in results.boxes:
            # Get class ID, coordinates, and confidence
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf.item()
            
            # Get label
            label = results.names[cls_id] if cls_id in results.names else f"class_{cls_id}"
            
            # Add detection
            detections.append({
                "class": cls_id,
                "label": label,
                "box": [x1, y1, x2, y2],
                "confidence": conf
            })
        
        return detections


def run_detection(config, input_dir=None, output_dir=None, model_path=None):
    """
    Run detection on a directory of images.
    
    Args:
        config: Configuration dictionary
        input_dir: Directory containing images to process
                  If None, uses the raw images directory from config
        output_dir: Directory to save detection results
                   If None, uses the results directory from config
        model_path: Path to the model to use for detection
                   If None, uses the default models from config
    
    Returns:
        dict: Dictionary mapping image paths to detection results
    """
    # Set default directories if not provided
    if input_dir is None:
        input_dir = config["paths"]["raw_images"]
    
    if output_dir is None:
        output_dir = config["paths"]["results_dir"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the detector
    detector = SleeveDetector(config, sleeve_model_path=model_path)
    
    # Get all image files
    image_extensions = [".jpg", ".jpeg", ".png"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = {}
    for i, img_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(img_file)}")
        
        try:
            # Load image
            image = cv2.imread(img_file)
            if image is None:
                print(f"Warning: Could not read image {img_file}")
                continue
            
            # Run detection
            detections = detector.detect(image)
            
            # Save results
            results[img_file] = detections
            
            # Visualize and save results if configured
            if config["visualization"]["save_images"]:
                output_image = visualize_detections(
                    image, 
                    detections, 
                    line_thickness=config["visualization"]["line_thickness"],
                    text_size=config["visualization"]["text_size"]
                )
                
                # Save visualized image
                img_filename = os.path.basename(img_file)
                output_img_path = os.path.join(output_dir, f"detected_{img_filename}")
                cv2.imwrite(output_img_path, output_image)
            
            # Save JSON results
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            json_path = os.path.join(output_dir, f"{img_name}_detections.json")
            
            with open(json_path, 'w') as f:
                json.dump(detections, f, indent=2)
        
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "detection_summary.json")
    
    summary = {
        "total_images": len(image_files),
        "processed_images": len(results),
        "total_detections": sum(len(dets) for dets in results.values())
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Detection completed. Processed {len(results)} images with {summary['total_detections']} detections.")
    return results


def visualize_detections(image, detections, line_thickness=2, text_size=1.0):
    """
    Visualize detection results on an image.
    
    Args:
        image: Image array (BGR format from OpenCV)
        detections: List of detection dictionaries
        line_thickness: Thickness of bounding box lines
        text_size: Size of text labels
        
    Returns:
        image: Image with visualized detections
    """
    # Make a copy of the image
    output_image = image.copy()
    
    # Define colors for different classes (BGR format)
    colors = {
        0: (0, 0, 255),    # Red for sleeves
        1: (0, 255, 0),    # Green for other objects
    }
    
    # Draw each detection
    for det in detections:
        # Get box coordinates
        x1, y1, x2, y2 = [int(coord) for coord in det["box"]]
        
        # Get label and confidence
        label = det["label"]
        confidence = det["confidence"]
        class_id = det["class"]
        
        # Get color
        color = colors.get(class_id, (255, 0, 0))  # Default to blue
        
        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label background
        text = f"{label} {confidence:.2f}"
        text_size_px = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_size, 1)[0]
        
        cv2.rectangle(
            output_image,
            (x1, y1 - text_size_px[1] - 5),
            (x1 + text_size_px[0], y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            output_image,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return output_image


if __name__ == "__main__":
    # Example usage when run as a script
    import yaml
    
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    run_detection(config) 