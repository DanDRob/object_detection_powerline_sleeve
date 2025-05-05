"""
Route processor for acquiring images from Google Street View.
"""

import os
import requests
import polyline  # Need to add 'polyline' to requirements
import math
import time
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import logging
import geopy.distance  # Need to add 'geopy' to requirements
from pyproj import Geod  # Need to add 'pyproj' to requirements

from streetview_client import StreetViewClient
from cache_manager import CacheManager

logger = logging.getLogger(__name__)


class RoutePlanner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key: str = config.get("api", {}).get("key")
        self.max_retries: int = config.get("api", {}).get("max_retries", 3)
        self.retry_delay: float = config.get("api", {}).get("retry_delay", 1.0)
        self.base_interval: float = config.get(
            "sampling", {}).get("base_interval", 20.0)
        self.powerline_offset: float = config.get(
            "powerline", {}).get("offset_distance", 10.0)
        self.powerline_side: str = config.get(
            "powerline", {}).get("side", "both")

        if not self.api_key:
            raise ValueError("API key not found in configuration (api.key)")

        self.geod = Geod(ellps='WGS84')

    def _get_route_coordinates_from_api(self, start_location: str, end_location: str) -> Optional[List[Tuple[float, float]]]:
        base_url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            'origin': start_location,
            'destination': end_location,
            'key': self.api_key,
        }
        logger.info(
            f"Fetching route from Directions API: {start_location} to {end_location}")

        for attempt in range(self.max_retries):
            try:
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()

                if data.get('status') != 'OK' or not data.get('routes'):
                    logger.error(
                        f"Directions API error: {data.get('status', 'Unknown Status')}. Response: {data.get('error_message', '')}")
                    return None

                coordinates = []
                for leg in data['routes'][0]['legs']:
                    for step in leg['steps']:
                        points_str = step['polyline']['points']
                        decoded_points = polyline.decode(points_str)
                        coordinates.extend(decoded_points)

                logger.info(
                    f"Successfully retrieved route with {len(coordinates)} points from API")
                return coordinates

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Directions API request timed out. Attempt {attempt + 1}/{self.max_retries}")
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Directions API request failed: {e}. Attempt {attempt + 1}/{self.max_retries}")

            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                logger.info(
                    f"Waiting {wait_time:.2f}s before retrying Directions API...")
                time.sleep(wait_time)
            else:
                logger.error(
                    "Failed to get route from Directions API after multiple attempts")
                return None
        return None  # Should not be reached, but added for clarity

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        az12, _, _ = self.geod.inv(lon1, lat1, lon2, lat2)
        return (az12 + 360) % 360  # Normalize to 0-360

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        # Use geopy for distance calculation as it's simpler than geod.inv for just distance
        return geopy.distance.distance((lat1, lon1), (lat2, lon2)).meters

    def _interpolate_route(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not points or len(points) < 2:
            logger.warning("Not enough points to interpolate route")
            return points

        interpolated_points = [points[0]]
        cumulative_distance = 0.0

        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            lat1, lon1 = p1
            lat2, lon2 = p2

            segment_distance = self._calculate_distance(lat1, lon1, lat2, lon2)
            if segment_distance <= 1e-6:  # Avoid division by zero or issues with identical points
                continue

            segment_bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)

            # Calculate how many intervals fit entirely within the remaining part of the segment
            while cumulative_distance + segment_distance >= self.base_interval:
                needed_distance = self.base_interval - cumulative_distance
                fraction = needed_distance / segment_distance

                # Use geopy's destination method for interpolation
                intermediate_point = geopy.distance.distance(
                    meters=needed_distance).destination(p1, segment_bearing)
                interpolated_points.append(
                    (intermediate_point.latitude, intermediate_point.longitude))

                # Update p1 and remaining distances for the next interpolation point in the *same* segment
                p1 = (intermediate_point.latitude,
                      intermediate_point.longitude)
                segment_distance -= needed_distance
                cumulative_distance = 0.0  # Reset cumulative distance for the new interval

            # Add remaining distance of this segment to cumulative distance
            cumulative_distance += segment_distance

        # Always add the very last point of the original route
        if interpolated_points[-1] != points[-1]:
            interpolated_points.append(points[-1])

        logger.info(
            f"Interpolated route to {len(interpolated_points)} points at ~{self.base_interval}m intervals")
        return interpolated_points

    def _calculate_route_bearings(self, points: List[Tuple[float, float]]) -> List[float]:
        if len(points) < 2:
            # Return default bearing if not enough points
            return [0.0] * len(points)

        bearings = []
        for i in range(len(points) - 1):
            lat1, lon1 = points[i]
            lat2, lon2 = points[i + 1]
            bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)
            bearings.append(bearing)

        # For the last point, use the bearing of the previous segment
        if bearings:
            bearings.append(bearings[-1])
        else:  # Handle case with only one point
            bearings.append(0.0)

        return bearings

    def _calculate_powerline_points(
        self,
        route_points: List[Tuple[float, float]],
        bearings: List[float]
    ) -> Dict[str, List[Tuple[float, float]]]:

        if len(route_points) != len(bearings):
            logger.error(
                f"Mismatch in route points ({len(route_points)}) and bearings ({len(bearings)}) for powerline calculation.")
            # Attempt to fix if bearings list is one short (common case)
            if len(bearings) == len(route_points) - 1 and bearings:
                bearings.append(bearings[-1])
            else:
                return {"left": [], "right": []}

        powerline_points: Dict[str, List[Tuple[float, float]]] = {
            "left": [], "right": []}
        sides_to_process: List[str] = []

        if self.powerline_side == "both":
            sides_to_process = ["right", "left"]
        elif self.powerline_side in ["right", "left"]:
            sides_to_process = [self.powerline_side]
        else:
            logger.warning(
                f"Invalid powerline side '{self.powerline_side}' in config. Defaulting to 'both'.")
            sides_to_process = ["right", "left"]

        for side in sides_to_process:
            offset_angle = 90 if side == "right" else -90  # 270 simplifies to -90
            side_points = []
            for i, (lat, lon) in enumerate(route_points):
                road_bearing = bearings[i]
                powerline_bearing = (road_bearing + offset_angle + 360) % 360

                # Calculate offset point using pyproj Geod fwd method
                # geod.fwd expects lon, lat, bearing, distance
                lon_pl, lat_pl, _ = self.geod.fwd(
                    lon, lat, powerline_bearing, self.powerline_offset)
                side_points.append((lat_pl, lon_pl))
            powerline_points[side] = side_points

        total_pl_points = len(
            powerline_points["left"]) + len(powerline_points["right"])
        logger.info(
            f"Generated {total_pl_points} potential powerline points (Left: {len(powerline_points['left'])}, Right: {len(powerline_points['right'])}) at {self.powerline_offset}m offset.")
        return powerline_points

    def plan_route(self, start_location: str, end_location: str) -> Optional[Dict[str, Any]]:
        """
        Plans a single route, fetching coordinates, interpolating, and calculating powerline points.

        Args:
            start_location: Starting address or lat,lon string.
            end_location: Ending address or lat,lon string.

        Returns:
            A dictionary containing planned route details or None if planning fails.
            Keys: 'route_points', 'route_bearings', 'powerline_points_left', 'powerline_points_right'
        """
        route_coords = self._get_route_coordinates_from_api(
            start_location, end_location)
        if not route_coords:
            logger.error(
                f"Could not retrieve route coordinates for {start_location} -> {end_location}")
            return None

        interpolated_route = self._interpolate_route(route_coords)
        if not interpolated_route:
            logger.error(
                f"Could not interpolate route for {start_location} -> {end_location}")
            return None

        bearings = self._calculate_route_bearings(interpolated_route)

        powerline_points = self._calculate_powerline_points(
            interpolated_route, bearings)

        return {
            "route_points": interpolated_route,
            "route_bearings": bearings,
            "powerline_points_left": powerline_points.get("left", []),
            "powerline_points_right": powerline_points.get("right", [])
        }

# --- Main acquisition logic (previously in acquire_images) should be moved to acquisition_run.py --- #
# --- Keeping the example __main__ block for potential testing --- #


if __name__ == "__main__":
    # Example usage when run as a script
    import yaml
    logging.basicConfig(level=logging.INFO)

    # Load config relative to this script's location if run directly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(
        script_dir, '../../config.yaml')  # Adjust path as needed

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        exit(1)

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Example route
    # Replace with actual start/end locations or load from a file
    start = "40.7128,-74.0060"  # New York City Hall
    end = "40.7580,-73.9855"   # Times Square

    planner = RoutePlanner(config_data)
    planned_route = planner.plan_route(start, end)

    if planned_route:
        print(f"Route planned successfully!")
        print(f"  Route Points: {len(planned_route['route_points'])} points")
        print(
            f"  Powerline Points Left: {len(planned_route['powerline_points_left'])} points")
        print(
            f"  Powerline Points Right: {len(planned_route['powerline_points_right'])} points")
        # Example: print first 5 points
        # print("  First 5 Route Points:", planned_route['route_points'][:5])
        # print("  First 5 Route Bearings:", planned_route['route_bearings'][:5])
    else:
        print("Route planning failed.")
