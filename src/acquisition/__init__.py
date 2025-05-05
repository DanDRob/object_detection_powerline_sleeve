"""
Image acquisition module for the powerline sleeve detection project.
This module handles fetching images from Google Street View based on routes.
"""

from .image_processor import ImageProcessor
from .cache_manager import CacheManager
from .streetview_client import StreetViewClient
from .route_processor import RoutePlanner
from .acquisition_run import run_acquisition
import logging

# Configure logging for the acquisition package
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())

# Expose main function and potentially classes

__all__ = [
    "run_acquisition",
    "RoutePlanner",
    "StreetViewClient",
    "CacheManager",
    "ImageProcessor"
]
