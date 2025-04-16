"""
Cache manager for Google Street View images.
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from PIL import Image
from io import BytesIO


class CacheManager:
    """Manages caching of Street View images to avoid redundant API calls."""
    
    def __init__(self, cache_dir, enabled=True):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for caching images
            enabled: Whether caching is enabled
        """
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.index_file = os.path.join(cache_dir, "cache_index.json")
        self.cache_index = {}
        
        if enabled:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_index()
    
    def _load_index(self):
        """Load the cache index from disk."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                print(f"Error loading cache index: {e}")
                self.cache_index = {}
    
    def _save_index(self):
        """Save the cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            print(f"Error saving cache index: {e}")
    
    def _get_cache_key(self, latitude, longitude, heading, pitch):
        """
        Generate a unique cache key for the given parameters.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            heading: Camera heading
            pitch: Camera pitch
            
        Returns:
            str: Cache key
        """
        # Create a string representation
        key_str = f"{latitude:.6f}_{longitude:.6f}_{heading}_{pitch}"
        
        # Create a hash (MD5 is fine for this non-security use)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def is_cached(self, latitude, longitude, heading, pitch):
        """
        Check if an image is in the cache.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            heading: Camera heading
            pitch: Camera pitch
            
        Returns:
            bool: True if in cache, False otherwise
        """
        if not self.enabled:
            return False
        
        cache_key = self._get_cache_key(latitude, longitude, heading, pitch)
        
        # Check if the key exists in the index and the file exists
        if cache_key in self.cache_index:
            cache_path = self.cache_index[cache_key]
            return os.path.exists(cache_path)
        
        return False
    
    def cache_image(self, latitude, longitude, heading, pitch, image_data):
        """
        Cache an image.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            heading: Camera heading
            pitch: Camera pitch
            image_data: Image data (bytes)
            
        Returns:
            str: Cache file path or None if failed
        """
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(latitude, longitude, heading, pitch)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.jpg")
        
        try:
            # Save the image to the cache
            with open(cache_path, 'wb') as f:
                f.write(image_data)
            
            # Update the index
            self.cache_index[cache_key] = cache_path
            self._save_index()
            
            return cache_path
        
        except Exception as e:
            print(f"Error caching image: {e}")
            return None
    
    def retrieve_image(self, latitude, longitude, heading, pitch, output_path):
        """
        Retrieve an image from the cache.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            heading: Camera heading
            pitch: Camera pitch
            output_path: Path to save the retrieved image
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        cache_key = self._get_cache_key(latitude, longitude, heading, pitch)
        
        if cache_key in self.cache_index:
            cache_path = self.cache_index[cache_key]
            
            if os.path.exists(cache_path):
                try:
                    # Copy the cached image to the output path
                    shutil.copy2(cache_path, output_path)
                    return True
                except Exception as e:
                    print(f"Error retrieving image from cache: {e}")
        
        return False
    
    def clear_cache(self):
        """Clear the entire cache."""
        if not self.enabled:
            return
        
        # Delete all files in the cache directory
        for file_path in Path(self.cache_dir).glob("*.jpg"):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing cache file {file_path}: {e}")
        
        # Clear the index
        self.cache_index = {}
        self._save_index()
        
        print(f"Cache cleared: {self.cache_dir}")
    
    def get_cache_size(self):
        """
        Get the total size of the cache in bytes.
        
        Returns:
            int: Cache size in bytes
        """
        if not self.enabled:
            return 0
        
        total_size = 0
        for file_path in Path(self.cache_dir).glob("*.jpg"):
            total_size += os.path.getsize(file_path)
        
        return total_size 