"""
Configuration utilities for the powerline sleeve detection project.
"""

import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_path):
    """
    Load and validate the configuration file.
    Supports environment variable interpolation using ${VAR_NAME} syntax.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    # Load environment variables from .env file
    load_dotenv()
    
    try:
        # Read the config file as string
        with open(config_path, 'r') as file:
            config_str = file.read()
        
        # Replace environment variables
        pattern = r'\${([^}^{]+)}'
        def replace_env_vars(match):
            env_var = match.group(1)
            value = os.getenv(env_var)
            if value is None:
                raise ValueError(f"Environment variable {env_var} not set")
            return value
        
        config_str = re.sub(pattern, replace_env_vars, config_str)
        
        # Parse YAML
        config = yaml.safe_load(config_str)
        
        validate_config(config)
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        raise


def validate_config(config):
    """
    Validate the configuration structure and required fields.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = [
        "paths", "acquisition", "labeling", 
        "dataset", "training", "detection", "visualization"
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Check for critical settings
    if not config["paths"]["data_dir"]:
        raise ValueError("Data directory path is required")
    
    # Convert relative paths to absolute if needed
    config["paths"]["data_dir"] = os.path.abspath(config["paths"]["data_dir"])
    
    # Additional validation could be added here for specific parameters 