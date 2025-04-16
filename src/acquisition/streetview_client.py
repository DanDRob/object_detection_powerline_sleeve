"""
Google Street View client for fetching images.
"""

import os
import requests
import urllib.parse
from PIL import Image
from io import BytesIO


class StreetViewClient:
    """Client for interacting with Google Street View Static API."""
    
    def __init__(self, api_key, cache_manager=None, image_size=(640, 640)):
        """
        Initialize the Street View client.
        
        Args:
            api_key: Google API key
            cache_manager: Optional cache manager
            image_size: Image dimensions (width, height)
        """
        self.api_key = api_key
        self.cache_manager = cache_manager
        self.image_size = image_size
        self.base_url = "https://maps.googleapis.com/maps/api/streetview"
    
    def get_image(self, latitude, longitude, output_path, heading=0, pitch=0, fov=90):
        """
        Get a Street View image for the given location.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            output_path: Path to save the image
            heading: Camera heading in degrees (0=north, 90=east)
            pitch: Camera pitch in degrees (90=up, -90=down)
            fov: Field of view in degrees (max 120)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check cache first if available
        if self.cache_manager and self.cache_manager.is_cached(latitude, longitude, heading, pitch):
            return self.cache_manager.retrieve_image(latitude, longitude, heading, pitch, output_path)
        
        # Build request URL
        params = {
            "size": f"{self.image_size[0]}x{self.image_size[1]}",
            "location": f"{latitude},{longitude}",
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "key": self.api_key
        }
        
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save the image
            img = Image.open(BytesIO(response.content))
            img.save(output_path)
            
            # Cache the image if cache manager is available
            if self.cache_manager:
                self.cache_manager.cache_image(latitude, longitude, heading, pitch, response.content)
            
            return True
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Street View image: {e}")
            return False
        
        except Exception as e:
            print(f"Unexpected error processing Street View image: {e}")
            return False
    
    def get_metadata(self, latitude, longitude):
        """
        Get metadata about a Street View location.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            dict: Metadata dictionary or None if failed
        """
        metadata_url = f"{self.base_url}/metadata"
        
        params = {
            "location": f"{latitude},{longitude}",
            "key": self.api_key
        }
        
        url = f"{metadata_url}?{urllib.parse.urlencode(params)}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            print(f"Error fetching Street View metadata: {e}")
            return None 