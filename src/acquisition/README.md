# Image Acquisition Module (`src/acquisition`)

## Overview

This module is responsible for acquiring street-level images along specified routes using the Google Street View Static API. It plans routes, fetches images based on configured camera parameters, manages caching to avoid redundant API calls, and saves the raw images for further processing.

## Components

- `__init__.py`: Initializes the package and exposes the main components.
- `acquisition_run.py`: The main executable script that orchestrates the image acquisition workflow. It loads configuration, reads routes, initializes components, and manages the overall process.
- `route_processor.py`: Handles route planning. It takes start and end locations, fetches detailed route polylines from the Google Directions API, interpolates points along the route at a configured interval, calculates road bearings, and determines potential powerline locations based on offsets.
- `streetview_client.py`: Interacts with the Google Street View Static API. It fetches images for specific locations, headings, pitches, and fields of view (FOV). It includes logic for rate limiting, retries with exponential backoff, and integrates with the `CacheManager`.
- `cache_manager.py`: Manages a local cache for fetched Street View images. This helps reduce API costs and speeds up subsequent runs by storing previously downloaded images based on location and camera parameters.
- `image_processor.py`: Currently handles the saving of fetched images to disk using a standardized naming convention derived from route ID, point index, camera side, heading, pitch, and FOV. (Future enhancements could include image preprocessing or augmentation).

## Configuration

This module relies on settings defined in the main `config.yaml` file located in the project root. Key sections include:

- `api`:
  - `key`: Your Google Cloud API Key (expected as `${GOOGLE_API_KEY}` environment variable).
  - `min_request_interval`: Minimum time (seconds) between API calls to avoid rate limiting.
  - `max_retries`: Number of times to retry failed API requests.
  - `retry_delay`: Base delay (seconds) for exponential backoff on retries.
  - `image_width`, `image_height`: Dimensions of the requested Street View images.
- `sampling`:
  - `base_interval`: Desired distance (meters) between interpolated points along the route.
- `camera`:
  - `pitch_values`: List of camera pitch angles (degrees) to capture at each point.
  - `fov_default`, `fov_special`: Default and special Field of View (FOV) values (degrees).
  - `relative_angles_right`, `relative_angles_left`: Lists of camera heading angles (degrees) relative to the road bearing for capturing images on the right and left sides.
- `powerline`:
  - `offset_distance`: Estimated distance (meters) from the road centerline to the powerlines.
  - `side`: Which side(s) relative to the road direction to calculate powerline points for (`right`, `left`, or `both`).
- `paths`:
  - `data_dir`: Base directory for data (used to find `routes.csv`).
  - `raw_images`: Output directory where fetched images will be saved.
- `acquisition`:
  - `cache_enabled`: Boolean flag to enable/disable image caching.
  - `cache_dir`: Directory to store cached images and the index file.

**Important:** Ensure the `GOOGLE_API_KEY` environment variable is set before running the acquisition.

## Usage

1.  **Prepare Routes:** Create a `routes.csv` file inside the directory specified by `paths.data_dir` (default: `data/`). This file must contain columns: `route_id`, `start_location`, `end_location`.
    - `route_id`: A unique identifier for the route (e.g., `001`).
    - `start_location`, `end_location`: Addresses or latitude,longitude pairs understood by the Google Directions API.
2.  **Set API Key:** Export your Google API key as an environment variable:
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    # On Windows (Command Prompt)
    # set GOOGLE_API_KEY="YOUR_API_KEY"
    # On Windows (PowerShell)
    # $env:GOOGLE_API_KEY="YOUR_API_KEY"
    ```
3.  **Run Acquisition:** Execute the main script from the project root directory:
    ```bash
    python -m src.acquisition.acquisition_run
    ```
    - _Optional:_ To run for a specific route ID, you can modify the script `acquisition_run.py`.

## Dependencies

This module requires the following Python libraries:

- `requests`: For making HTTP API calls.
- `PyYAML`: For loading the `config.yaml` file.
- `polyline`: For decoding Google Directions API polylines.
- `geopy`: For geodetic distance calculations and point interpolation.
- `pyproj`: For accurate geodetic bearing and offset calculations.
- `pandas`: For reading the `routes.csv` file.
- `Pillow`: For opening and saving images.

Ensure these are installed, preferably via a `requirements.txt` file:

```
requests
PyYAML
polyline
geopy
pyproj
pandas
Pillow
```

## Output

The script will create the directory specified in `config.yaml` under `paths.raw_images` (default: `data/raw/`) if it doesn't exist.

Successfully fetched images will be saved in this directory with filenames following the pattern:
`pl_r<route_id>_p<point_index>_<side>_h<heading>_p<pitch>_fov<fov>.jpg`

- `<route_id>`: The ID from the `routes.csv`.
- `<point_index>`: The zero-based index of the point along the interpolated route.
- `<side>`: The calculated side relative to the road bearing (`left`, `right`, `front`, `rear`).
- `<heading>`, `<pitch>`, `<fov>`: The integer values of the camera parameters used for the specific image.

Log messages indicating progress, API calls, cache hits/misses, and errors will be printed to the console.
