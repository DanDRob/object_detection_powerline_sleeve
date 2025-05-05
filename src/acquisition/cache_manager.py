"""
Cache manager for Google Street View images.
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


class CacheManager:
    """Manages caching of Street View images to avoid redundant API calls."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cache manager.

        Args:
            config: Configuration dictionary
        """
        self.enabled: bool = config.get(
            "acquisition", {}).get("cache_enabled", False)
        self.cache_dir: Optional[str] = config.get(
            "acquisition", {}).get("cache_dir")
        self.index_file: Optional[str] = None
        self.cache_index: Dict[str, str] = {}

        if self.enabled:
            if not self.cache_dir:
                # Consider logging a warning or raising an error if cache is enabled but no dir is specified
                self.enabled = False
                # Or raise ValueError("Cache directory must be specified when cache is enabled")
                return

            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            self.index_file = os.path.join(self.cache_dir, "cache_index.json")
            self._load_index()

    def _load_index(self) -> None:
        """Load the cache index from disk."""
        if not self.enabled or not self.index_file or not os.path.exists(self.index_file):
            return
        try:
            with open(self.index_file, 'r') as f:
                self.cache_index = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # Log error: print(f"Error loading cache index: {e}")
            self.cache_index = {}  # Reset index on error

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        if not self.enabled or not self.index_file:
            return
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except IOError as e:
            # Log error: print(f"Error saving cache index: {e}")
            pass

    def _get_cache_key(self, lat: float, lon: float, heading: float, pitch: float, fov: float) -> str:
        """
        Generate a unique cache key for the given parameters.

        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            heading: Camera heading
            pitch: Camera pitch
            fov: Field of view

        Returns:
            str: Cache key
        """
        # Using a simpler, more readable key format before hashing
        key_str = f"loc_{lat:.6f}_{lon:.6f}_h{heading:.1f}_p{pitch:.1f}_f{fov:.1f}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def is_cached(self, lat: float, lon: float, heading: float, pitch: float, fov: float) -> bool:
        """
        Check if an image is in the cache.

        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            heading: Camera heading
            pitch: Camera pitch
            fov: Field of view

        Returns:
            bool: True if in cache, False otherwise
        """
        if not self.enabled or not self.cache_dir:
            return False

        cache_key = self._get_cache_key(lat, lon, heading, pitch, fov)
        if cache_key in self.cache_index:
            cache_path = os.path.join(
                self.cache_dir, self.cache_index[cache_key])
            return os.path.exists(cache_path)
        return False

    def cache_image(self, lat: float, lon: float, heading: float, pitch: float, fov: float, image_data: bytes) -> Optional[str]:
        """
        Cache an image.

        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            heading: Camera heading
            pitch: Camera pitch
            fov: Field of view
            image_data: Image data (bytes)

        Returns:
            str: Cache file path or None if failed
        """
        if not self.enabled or not self.cache_dir:
            return None

        cache_key = self._get_cache_key(lat, lon, heading, pitch, fov)
        # Use only the hash as filename, store relative path in index
        filename = f"{cache_key}.jpg"
        cache_path = os.path.join(self.cache_dir, filename)

        try:
            with open(cache_path, 'wb') as f:
                f.write(image_data)
            self.cache_index[cache_key] = filename  # Store relative path
            self._save_index()
            return cache_path
        except IOError as e:
            # Log error: print(f"Error caching image: {e}")
            # Clean up potentially corrupted file
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if cache_key in self.cache_index:
                del self.cache_index[cache_key]
                self._save_index()
            return None

    def retrieve_image(self, lat: float, lon: float, heading: float, pitch: float, fov: float, output_path: str) -> bool:
        """
        Retrieve an image from the cache.

        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            heading: Camera heading
            pitch: Camera pitch
            fov: Field of view
            output_path: Path to save the retrieved image

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled or not self.cache_dir:
            return False

        cache_key = self._get_cache_key(lat, lon, heading, pitch, fov)
        if cache_key in self.cache_index:
            filename = self.cache_index[cache_key]
            cache_path = os.path.join(self.cache_dir, filename)
            if os.path.exists(cache_path):
                try:
                    shutil.copy2(cache_path, output_path)
                    return True
                except IOError as e:
                    # Log error: print(f"Error retrieving image from cache: {e}")
                    return False
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
