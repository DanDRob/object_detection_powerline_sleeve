"""
Route processor for acquiring images from Google Street View.
"""

import os
import pandas as pd
import time
from pathlib import Path
from .streetview_client import StreetViewClient
from .cache_manager import CacheManager


def acquire_images(config, route_file=None):
    """
    Acquire images from Google Street View based on routes.
    
    Args:
        config: Configuration dictionary
        route_file: Optional path to a CSV file with routes.
                   If None, uses the default from config
    
    Returns:
        list: Paths to acquired images
    """
    # Initialize clients
    cache_manager = CacheManager(
        cache_dir=config["acquisition"]["cache_dir"],
        enabled=config["acquisition"]["cache_enabled"]
    )
    
    streetview_client = StreetViewClient(
        api_key=config["acquisition"]["api_key"],
        cache_manager=cache_manager,
        image_size=config["acquisition"]["image_size"]
    )
    
    # Determine route file
    if route_file is None:
        route_file = os.path.join(config["paths"]["data_dir"], "routes.csv")
    
    if not os.path.exists(route_file):
        raise FileNotFoundError(f"Route file not found: {route_file}")
    
    # Load routes
    routes_df = pd.read_csv(route_file)
    print(f"Loaded {len(routes_df)} routes from {route_file}")
    
    # Create output directory
    output_dir = config["paths"]["raw_images"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each route
    acquired_images = []
    for idx, route in routes_df.iterrows():
        print(f"Processing route {idx+1}/{len(routes_df)}: {route.get('name', f'Route {idx+1}')}")
        route_images = process_route(
            streetview_client,
            route,
            output_dir,
            distance_interval=config["acquisition"]["distance_interval"]
        )
        acquired_images.extend(route_images)
        
    print(f"Acquired {len(acquired_images)} images from {len(routes_df)} routes")
    return acquired_images


def process_route(streetview_client, route, output_dir, distance_interval):
    """
    Process a single route and acquire images.
    
    Args:
        streetview_client: StreetViewClient instance
        route: Route information (pandas Series or dict)
        output_dir: Output directory for images
        distance_interval: Distance between consecutive images (meters)
    
    Returns:
        list: Paths to acquired images for this route
    """
    # Extract route information
    start_lat = route.get("start_lat", 0)
    start_lng = route.get("start_lng", 0)
    end_lat = route.get("end_lat", 0)
    end_lng = route.get("end_lng", 0)
    
    if not all([start_lat, start_lng, end_lat, end_lng]):
        print(f"Skipping route with incomplete coordinates: {route}")
        return []
    
    # Generate points along the route
    points = generate_route_points(
        start_lat, start_lng, end_lat, end_lng, 
        interval=distance_interval
    )
    
    # Acquire images for each point
    images = []
    for i, (lat, lng) in enumerate(points):
        image_path = os.path.join(
            output_dir, 
            f"route_{route.get('name', 'route')}_{i:04d}.jpg"
        )
        
        success = streetview_client.get_image(lat, lng, image_path)
        if success:
            images.append(image_path)
        
        # Respect API rate limits
        time.sleep(0.1)
    
    return images


def generate_route_points(start_lat, start_lng, end_lat, end_lng, interval=10):
    """
    Generate points along a route at specified intervals.
    
    In a real implementation, this would use Google Directions API or similar
    to get the actual route between points. This is a simplified version that
    just generates points along a straight line.
    
    Args:
        start_lat: Starting latitude
        start_lng: Starting longitude
        end_lat: Ending latitude
        end_lng: Ending longitude
        interval: Distance between points in meters
    
    Returns:
        list: List of (lat, lng) tuples
    """
    # Simple linear interpolation (in a real implementation, use routing API)
    # Calculate distance and number of points
    from math import sqrt, sin, cos, atan2, radians, degrees
    
    # Calculate distance in kilometers
    R = 6371  # Earth radius in km
    
    lat1, lng1 = radians(start_lat), radians(start_lng)
    lat2, lng2 = radians(end_lat), radians(end_lng)
    
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    # Number of points
    num_points = max(2, int(distance * 1000 / interval))
    
    # Generate points
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        lat = start_lat + t * (end_lat - start_lat)
        lng = start_lng + t * (end_lng - start_lng)
        points.append((lat, lng))
    
    return points


if __name__ == "__main__":
    # Example usage when run as a script
    import yaml
    
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    acquire_images(config) 