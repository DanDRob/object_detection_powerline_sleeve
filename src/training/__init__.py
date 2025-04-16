"""
Training module for powerline sleeve detection.
This module handles model training for both powerline and sleeve detection.
"""

from .powerline.trainer import train_powerline_model
from .sleeve.trainer import train_sleeve_model

__all__ = ["train_powerline_model", "train_sleeve_model"] 