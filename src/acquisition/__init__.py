"""
Image acquisition module for the powerline sleeve detection project.
This module handles fetching images from Google Street View based on routes.
"""

from .route_processor import acquire_images

__all__ = ["acquire_images"] 