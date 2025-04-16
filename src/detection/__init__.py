"""
Detection module for powerline sleeve detection.
This module handles inference using trained models.
"""

from .detector import run_detection, SleeveDetector

__all__ = ["run_detection", "SleeveDetector"] 