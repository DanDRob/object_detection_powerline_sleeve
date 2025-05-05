"""
Google Street View client for fetching images.
"""

import os
import requests
import urllib.parse
import time
from typing import Dict, Any, Tuple
import logging

# Assuming CacheManager is in the same directory or accessible via PYTHONPATH
from cache_manager import CacheManager

logger = logging.getLogger(__name__)


class StreetViewClient:
    """Client for interacting with Google Street View Static API."""

    def __init__(self, config: Dict[str, Any], cache_manager: CacheManager):
        self.api_key: str = config.get("api", {}).get("key")
        if not self.api_key:
            raise ValueError("API key not found in configuration (api.key)")

        self.image_width: int = config.get("api", {}).get("image_width", 640)
        self.image_height: int = config.get("api", {}).get("image_height", 640)
        self.min_request_interval: float = config.get(
            "api", {}).get("min_request_interval", 0.1)
        self.max_retries: int = config.get("api", {}).get("max_retries", 3)
        self.retry_delay: float = config.get("api", {}).get("retry_delay", 1.0)

        self.cache_manager = cache_manager
        self.base_url: str = "https://maps.googleapis.com/maps/api/streetview"
        self.last_request_time: float = 0

    def _apply_rate_limit(self) -> None:
        """Ensure minimum interval between requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            delay = self.min_request_interval - elapsed
            logger.debug(f"Rate limiting: waiting {delay:.2f}s")
            time.sleep(delay)
        # Update last request time only after a successful request or failed attempt

    def get_image(self, lat: float, lon: float, output_path: str, heading: float, pitch: float, fov: float) -> bool:
        """
        Get a Street View image, checking cache first, then fetching from API.

        Args:
            lat: Latitude coordinate.
            lon: Longitude coordinate.
            output_path: Path to save the image.
            heading: Camera heading in degrees (0=north, 90=east).
            pitch: Camera pitch in degrees (0=level, 90=up, -90=down).
            fov: Field of view in degrees (max 120 recommended).

        Returns:
            bool: True if image was successfully obtained (cached or fetched), False otherwise.
        """
        # Check cache first
        if self.cache_manager.retrieve_image(lat, lon, heading, pitch, fov, output_path):
            logger.debug(
                f"Cache hit for image at {lat},{lon}, h={heading}, p={pitch}, f={fov}")
            return True

        logger.debug(
            f"Cache miss for image at {lat},{lon}, h={heading}, p={pitch}, f={fov}. Fetching from API.")

        # Apply rate limiting before making the request
        self._apply_rate_limit()

        params = {
            "size": f"{self.image_width}x{self.image_height}",
            "location": f"{lat},{lon}",
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "key": self.api_key,
            "source": "outdoor"  # Prefer outdoor images
        }
        request_url = f"{self.base_url}?{urllib.parse.urlencode(params)}"

        image_data: bytes | None = None
        success = False

        for attempt in range(self.max_retries):
            try:
                response = requests.get(request_url, timeout=15)
                self.last_request_time = time.time()  # Update time after each attempt

                if response.status_code == 200:
                    # Check if response content is non-empty and appears to be an image
                    # Basic check for empty/error response
                    if response.content and len(response.content) > 100:
                        image_data = response.content
                        success = True
                        logger.debug(
                            f"Successfully fetched image ({len(image_data)} bytes) attempt {attempt + 1}")
                        break  # Exit retry loop on success
                    else:
                        logger.warning(
                            f"API returned status 200 but empty/small content for {lat},{lon}, h={heading}, p={pitch}. Attempt {attempt + 1}/{self.max_retries}")
                        # Treat as failure and retry

                elif response.status_code == 404:
                    logger.warning(
                        f"API returned 404 (Not Found) for {lat},{lon}, h={heading}, p={pitch}. No image likely exists. Stopping retries.")
                    success = False
                    break  # Don't retry on 404
                # Retry on these errors
                elif response.status_code in [403, 429, 500, 503]:
                    logger.warning(
                        f"API request failed with status {response.status_code} for {lat},{lon}, h={heading}, p={pitch}. Attempt {attempt + 1}/{self.max_retries}")
                else:
                    logger.error(
                        f"API request failed with unhandled status {response.status_code} for {lat},{lon}, h={heading}, p={pitch}: {response.text}")
                    success = False
                    break  # Don't retry unhandled codes

            except requests.exceptions.Timeout:
                self.last_request_time = time.time()
                logger.warning(
                    f"API request timed out for {lat},{lon}, h={heading}, p={pitch}. Attempt {attempt + 1}/{self.max_retries}")
            except requests.exceptions.RequestException as e:
                self.last_request_time = time.time()
                logger.error(
                    f"API request failed for {lat},{lon}, h={heading}, p={pitch}: {e}. Attempt {attempt + 1}/{self.max_retries}")
                # Break on critical network errors usually
                success = False
                break

            # Wait before retrying if not the last attempt
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                logger.info(f"Waiting {wait_time:.2f}s before retrying...")
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Failed to fetch image for {lat},{lon}, h={heading}, p={pitch} after {self.max_retries} attempts.")

        # Process successful fetch
        if success and image_data:
            try:
                # Save the fetched image
                with open(output_path, 'wb') as f:
                    f.write(image_data)

                # Cache the fetched image
                self.cache_manager.cache_image(
                    lat, lon, heading, pitch, fov, image_data)
                return True
            except IOError as e:
                logger.error(
                    f"Failed to save or cache fetched image {output_path}: {e}")
                return False  # Failed to save/cache

        return False  # Fetching failed

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
