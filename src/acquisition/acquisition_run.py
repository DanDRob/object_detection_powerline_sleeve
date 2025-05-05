import yaml
import os
import logging
import time
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv

# Assuming components are in the same directory
from cache_manager import CacheManager

from route_processor import RoutePlanner
from streetview_client import StreetViewClient
from image_processor import ImageProcessor

# --- Configuration --- #
CONFIG_PATH = "config.yaml"  # Relative to project root
ROUTES_CSV = "data/routes.csv"  # Default relative to project root

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions --- #


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")

        # --- Environment Variable Substitution --- #
        api_key_placeholder = "${GOOGLE_API_KEY}"
        config_api_key = config.get("api", {}).get("key")

        if config_api_key == api_key_placeholder:
            actual_api_key = os.getenv("GOOGLE_API_KEY")
            if actual_api_key:
                logger.info(
                    "Substituting GOOGLE_API_KEY environment variable for api.key")
                # Ensure 'api' key exists before trying to set 'key'
                if "api" not in config:
                    config["api"] = {}
                config["api"]["key"] = actual_api_key
            else:
                logger.error(
                    f"Configuration expects {api_key_placeholder} but GOOGLE_API_KEY environment variable is not set.")
                # Decide how to handle: raise error, or let it fail later?
                # Raising an error here is clearer.
                raise ValueError(
                    "GOOGLE_API_KEY environment variable not set, but required by config.")
        elif not config_api_key:
            logger.warning("API key is not defined in config file (api.key)")
            # Depending on requirements, might want to raise ValueError here too

        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration {config_path}: {e}")
        raise


def load_routes(routes_csv_path: str, route_id_filter: Optional[str] = None) -> pd.DataFrame:
    """Loads routes from a CSV file."""
    if not os.path.exists(routes_csv_path):
        logger.error(f"Routes CSV file not found: {routes_csv_path}")
        return pd.DataFrame()  # Return empty dataframe

    try:
        routes_df = pd.read_csv(routes_csv_path, dtype={'route_id': str})
        logger.info(f"Loaded {len(routes_df)} routes from {routes_csv_path}")

        if route_id_filter:
            route_id_filter = route_id_filter.zfill(3)  # Assuming 3-digit IDs
            routes_df = routes_df[routes_df['route_id'] == route_id_filter]
            if routes_df.empty:
                logger.warning(f"No route found with ID: {route_id_filter}")
            else:
                logger.info(f"Filtered to route ID: {route_id_filter}")

        # Validate required columns
        required_cols = ['route_id', 'start_location', 'end_location']
        if not all(col in routes_df.columns for col in required_cols):
            logger.error(
                f"Routes CSV missing required columns: {required_cols}")
            return pd.DataFrame()

        return routes_df

    except pd.errors.EmptyDataError:
        logger.error(f"Routes CSV file is empty: {routes_csv_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading routes CSV {routes_csv_path}: {e}")
        return pd.DataFrame()


def calculate_camera_views(road_bearing: float, camera_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Calculates required heading/pitch/fov for a point based on config."""
    views = []
    pitch_values = camera_config.get("pitch_values", [0])
    fov_default = camera_config.get("fov_default", [90])
    fov_special = camera_config.get("fov_special", fov_default)
    angles_right = camera_config.get("relative_angles_right", [90])
    angles_left = camera_config.get("relative_angles_left", [270])
    special_angles = [45, 135, 230, 310]  # Angles requiring special FOV

    # Right side
    for angle in angles_right:
        fovs = fov_special if angle in special_angles else fov_default
        heading = (road_bearing + angle + 360) % 360
        for pitch in pitch_values:
            for fov in fovs:
                views.append({"heading": heading, "pitch": pitch, "fov": fov})

    # Left side
    for angle in angles_left:
        fovs = fov_special if angle in special_angles else fov_default
        heading = (road_bearing + angle + 360) % 360
        for pitch in pitch_values:
            for fov in fovs:
                views.append({"heading": heading, "pitch": pitch, "fov": fov})

    return views

# --- Main Execution --- #


def run_acquisition(config_path: str = CONFIG_PATH, route_id: Optional[str] = None):
    """Main function to run the image acquisition process."""
    # --- Load .env file first --- #
    load_dotenv()  # Load variables from .env into environment
    logger.info(
        "Attempted to load environment variables from .env file (if present).")
    # --------------------------- #

    start_time = time.time()
    logger.info("Starting image acquisition process...")

    try:
        config = load_config(config_path)
    except Exception:
        return  # Error logged in load_config

    # Initialize components
    try:
        cache_mgr = CacheManager(config)
        route_planner = RoutePlanner(config)
        streetview_cli = StreetViewClient(config, cache_mgr)
        image_proc = ImageProcessor(config)
    except ValueError as e:
        logger.error(f"Failed to initialize components: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error during component initialization: {e}")
        return

    # Determine routes file path
    routes_file = os.path.join(config.get(
        "paths", {}).get("data_dir", "data"), "routes.csv")
    logger.info(f"Using routes file: {routes_file}")

    # Load routes
    routes = load_routes(routes_file, route_id_filter=route_id)
    if routes.empty:
        logger.error("No routes to process. Exiting.")
        return

    total_images_attempted = 0
    total_images_saved = 0
    processed_route_count = 0

    # Process each route
    for index, route_info in routes.iterrows():
        r_id = route_info['route_id']
        start_loc = route_info['start_location']
        end_loc = route_info['end_location']
        processed_route_count += 1
        logger.info(
            f"--- Processing Route {r_id} ({processed_route_count}/{len(routes)}): {start_loc} -> {end_loc} ---")

        planned_route = route_planner.plan_route(start_loc, end_loc)

        if not planned_route:
            logger.error(f"Failed to plan route {r_id}. Skipping.")
            continue

        route_points = planned_route['route_points']
        route_bearings = planned_route['route_bearings']
        camera_config = config.get("camera", {})
        raw_images_dir = config.get("paths", {}).get("raw_images", "data/raw")

        images_saved_this_route = 0
        # Iterate through planned points
        for i, (lat, lon) in enumerate(route_points):
            point_start_time = time.time()
            road_bearing = route_bearings[i]
            logger.debug(
                f"Processing Point {i+1}/{len(route_points)} ({lat:.6f}, {lon:.6f}), Road Bearing: {road_bearing:.1f}")

            required_views = calculate_camera_views(
                road_bearing, camera_config)
            logger.debug(
                f"  - Calculated {len(required_views)} camera views for this point.")

            point_images_saved = 0
            # Fetch images for all required views at this point
            for view_params in required_views:
                total_images_attempted += 1
                # Construct output path using ImageProcessor's logic (or similar)
                # This avoids fetching data if we only need the path for saving
                heading = view_params["heading"]
                pitch = view_params["pitch"]
                fov = view_params["fov"]

                # Use ImageProcessor to generate the intended filename/path
                # Example filename generation (mirrors image_processor.save_image logic)
                relative_heading = (heading - road_bearing + 360) % 360
                if 1 <= relative_heading <= 179:
                    side = "right"
                elif 181 <= relative_heading <= 359:
                    side = "left"
                else:
                    side = "front" if pitch < 45 else "rear"
                filename = f"pl_r{r_id}_p{i}_{side}_h{int(heading)}_p{int(pitch)}_fov{int(fov)}.jpg"
                output_image_path = os.path.join(raw_images_dir, filename)

                # Pass camera params needed for saving/logging
                cam_params_for_save = {**view_params, "route_id": r_id}

                # Get image (checks cache, fetches if needed)
                # The StreetViewClient now handles saving and caching directly if fetched
                success = streetview_cli.get_image(
                    lat, lon, output_image_path, **view_params)

                if success:
                    # StreetViewClient handles saving now, just log success
                    logger.debug(
                        f"Successfully obtained/saved image: {os.path.basename(output_image_path)}")
                    point_images_saved += 1
                else:
                    logger.warning(
                        f"Failed to obtain image for view: h={heading}, p={pitch}, f={fov}")

            images_saved_this_route += point_images_saved
            point_elapsed = time.time() - point_start_time
            logger.debug(
                f"  - Point {i+1} processed in {point_elapsed:.2f}s, saved {point_images_saved}/{len(required_views)} images.")

        logger.info(
            f"--- Route {r_id} finished. Saved {images_saved_this_route} images for this route. ---")
        total_images_saved += images_saved_this_route

    # End of processing
    end_time = time.time()
    total_duration = end_time - start_time
    logger.info("Image acquisition process finished.")
    logger.info(f"Summary:")
    logger.info(
        f"  - Total routes processed: {processed_route_count}/{len(routes)}")
    logger.info(f"  - Total images attempted: {total_images_attempted}")
    logger.info(f"  - Total images saved: {total_images_saved}")
    logger.info(f"  - Total duration: {total_duration:.2f} seconds")


if __name__ == "__main__":
    # adjust with ids you want to run
    run_acquisition(route_id="004")
