import os
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional
import logging
import io

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Extract any relevant config later if needed for processing
        self.output_dir = config.get("paths", {}).get("raw_images", "data/raw")
        os.makedirs(self.output_dir, exist_ok=True)

    def save_image(self, image_data: bytes, point_index: int, road_bearing: float, camera_params: Dict[str, Any]) -> Optional[str]:
        """
        Saves the fetched image data to disk with a descriptive filename.

        Args:
            image_data: The raw image bytes.
            point_index: The index of the route point this image corresponds to.
            road_bearing: The bearing of the road at the route point.
            camera_params: Dict containing 'heading', 'pitch', 'fov'.

        Returns:
            The full path to the saved image file, or None if saving failed.
        """
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error(f"Failed to open image data for saving: {e}")
            return None

        heading = camera_params.get("heading", 0)
        pitch = camera_params.get("pitch", 0)
        fov = camera_params.get("fov", 90)

        # Determine side based on relative angle
        relative_heading = (heading - road_bearing + 360) % 360
        # Define threshold ranges for side classification (adjust as needed)
        if 1 <= relative_heading <= 179:
            side = "right"
        elif 181 <= relative_heading <= 359:
            side = "left"
        else:  # Exactly forward or backward
            side = "front" if pitch < 45 else "rear"  # Simplistic distinction

        # Create filename (similar to acquisition_v1 format)
        # Example: pl_r{route_id}_p{point_index}_{side}_h{heading}_p{pitch}_fov{fov}.jpg
        # We need route_id here - it should be passed down or included in params
        # Get route_id if available
        route_id = camera_params.get("route_id", "000")

        filename = f"pl_r{route_id}_p{point_index}_{side}_h{int(heading)}_p{int(pitch)}_fov{int(fov)}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        # Save image
        try:
            # Make sure directory exists (though __init__ should handle the base)
            # os.makedirs(os.path.dirname(filepath), exist_ok=True) # Redundant if output_dir is flat
            image.save(filepath, format="JPEG")
            logger.debug(f"Saved image: {filepath}")
            return filepath
        except IOError as e:
            logger.error(f"Failed to save image {filepath}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error saving image {filepath}: {e}")
            return None
