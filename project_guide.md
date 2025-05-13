# Powerline Sleeve Detection Project Guide
## Data Collection Pipeline Documentation

### Table of Contents
1. [Project Overview](#project-overview)
2. [System Requirements](#system-requirements)
3. [Project Structure](#project-structure)
4. [Data Collection Pipeline](#data-collection-pipeline)
5. [Configuration Guide](#configuration-guide)
6. [Step-by-Step Guide](#step-by-step-guide)
7. [Step by Step Process](#step-by-step-process)
8. [Troubleshooting](#troubleshooting)

## 1. Project Overview

This project aims to detect powerline sleeves using computer vision techniques. The data collection pipeline is designed to systematically gather street-level images of powerlines using the Google Street View API. The system follows these main steps:

1. Route Planning
2. Image Acquisition
3. Data Organization
4. Image Processing
5. Dataset Preparation

## 2. System Requirements

### Software Dependencies
- Python 3.x
- Required Python packages (install via `pip install -r requirements.txt`):
  - requests
  - PyYAML
  - polyline
  - geopy
  - pyproj
  - pandas
  - Pillow
  - ultralytics (for YOLOv8)

### API Requirements
- Google Cloud Platform account
- Enabled APIs:
  - Google Street View Static API
  - Google Directions API
- Valid API key with billing enabled

## 3. Project Structure

```
object_detection_powerline_sleeve/
├── data/
│   ├── cache/              # Cached Street View images
│   ├── powerlines/         # Organized dataset
│   │   ├── augmented/      # Augmented training data
│   │   ├── original/       # Original collected data
│   │   └── original_balanced/ # Balanced dataset
│   ├── raw/                # Raw collected images
│   └── sleeves/           # Sleeve-specific data
├── scripts/               # Utility scripts
├── src/
│   ├── acquisition/       # Data collection code
│   ├── dataset/          # Dataset management
│   ├── detection/        # Object detection
│   ├── labeling/         # Image labeling
│   └── training/         # Model training
└── config.yaml           # Configuration file
```

## 4. Data Collection Pipeline

### 4.1 Route Planning (`route_processor.py`)
- **Purpose**: Plans routes for image collection
- **Key Features**:
  - Fetches route coordinates from Google Directions API
  - Interpolates points along route at specified intervals
  - Calculates road bearings and powerline offsets
  - Supports both left and right side powerline detection

#### 4.1.1 Technical Implementation Details
- **RoutePlanner Class**: The central component that handles all route planning operations
  - **Initialization**: Takes configuration parameters from `config.yaml` such as API key, base interval, powerline offset, and powerline side (left, right, or both)
  - **Google Directions API Integration**: Uses the API to fetch detailed polylines for routes between specified start and end locations
    ```python
    # Example API request
    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        'origin': start_location,
        'destination': end_location,
        'key': self.api_key,
    }
    ```
  - **Polyline Decoding**: Processes encoded polylines from the API response using the `polyline` library to extract coordinate sequences
    ```python
    # Decoding polylines to coordinates
    points_str = step['polyline']['points']
    decoded_points = polyline.decode(points_str)
    ```

- **Route Interpolation**: Ensures consistent sampling of points along the route
  - **Adaptive Algorithm**: Interpolates points at the specified interval (default 20 meters) to create evenly spaced points along the route
  - **Geodetic Calculations**: Uses `geopy.distance` and `pyproj.Geod` for accurate distance and bearing calculations on the Earth's surface
  - **Point Generation**: Creates new coordinate points that maintain the exact specified interval
    ```python
    # Interpolation example
    intermediate_point = geopy.distance.distance(
        meters=needed_distance).destination(p1, segment_bearing)
    ```

- **Bearing Calculation**: Determines road direction at each point
  - **Geodetic Azimuth**: Calculates true bearing using pyproj's geodetic functions rather than simple Euclidean calculations
  - **Forward Looking**: Computes bearing based on the direction to the next point
  - **Normalization**: Ensures bearings are within 0-360 degrees

- **Powerline Point Generation**: Calculates potential powerline locations based on road positions
  - **Offset Calculation**: Uses perpendicular offsets from the road (default 10 meters) to estimate powerline locations
  - **Side Selection**: Supports left side, right side, or both sides of the road
  - **GeoSpatial Transformation**: Uses Geod.fwd method to calculate offset coordinates
    ```python
    # Calculate offset point (lon, lat is reversed in pyproj)
    lon_pl, lat_pl, _ = self.geod.fwd(
        lon, lat, powerline_bearing, self.powerline_offset)
    ```

#### 4.1.2 Error Handling and Robustness
- **API Error Recovery**: Implements exponential backoff with retries for API failures
- **Input Validation**: Checks for valid configuration and API responses
- **Edge Cases**: Handles special cases like identical consecutive points and insufficient points

#### 4.1.3 Output Format
The `plan_route` method returns a dictionary containing:
- `route_points`: List of (latitude, longitude) tuples for the interpolated route
- `route_bearings`: List of bearings in degrees for each route point
- `powerline_points_left`: List of potential powerline locations on the left side
- `powerline_points_right`: List of potential powerline locations on the right side

### 4.2 Image Acquisition (`streetview_client.py`)
- **Purpose**: Manages Street View image collection
- **Key Features**:
  - Handles API requests with rate limiting
  - Implements exponential backoff for failed requests
  - Caches images to reduce API costs
  - Supports multiple camera angles and pitches

#### 4.2.1 Technical Implementation Details
- **StreetViewClient Class**: Core component responsible for fetching images
  - **Initialization**: Configures with API key, request intervals, retry parameters, and image dimensions from `config.yaml`
  - **Cache Integration**: Connects with `CacheManager` to optimize API usage

- **API Request Construction**:
  - **URL Formation**: Builds Google Street View Static API URLs with precise parameters
    ```python
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        'size': f"{self.image_width}x{self.image_height}",
        'location': f"{lat},{lon}",
        'heading': heading,
        'pitch': pitch,
        'fov': fov,
        'key': self.api_key
    }
    ```
  - **Parameter Validation**: Ensures all required parameters are within valid ranges

- **Rate Limiting Implementation**:
  - **Time-based Throttling**: Enforces minimum wait times between requests (configurable, default 0.1 seconds)
  - **Last Request Tracking**: Maintains timestamp of most recent request to calculate appropriate wait time
  - **Adaptive Delays**: Dynamically adjusts wait times based on API response codes

- **Exponential Backoff for Failures**:
  - **Progressive Retry Logic**: Increases wait time exponentially after failed requests
  - **Maximum Retry Configuration**: Limits retry attempts based on configuration
  - **Error Classification**: Different handling for server errors (5xx), client errors (4xx), and network issues

- **Response Handling**:
  - **Status Validation**: Checks HTTP status codes and handles different error scenarios
  - **Binary Data Processing**: Processes image binary data from responses
  - **Metadata Extraction**: Captures relevant headers and metadata from the API response

#### 4.2.2 Image Retrieval and Storage
- **Cache-First Strategy**:
  - **Check Cache**: Verifies if requested image exists in cache before making API call
  - **Cache Key Generation**: Creates unique keys based on location and camera parameters
  - **Update Cache**: Stores newly fetched images in cache with proper metadata

- **Direct Image Saving**:
  - **Path Construction**: Creates appropriate file paths based on route ID and camera parameters
  - **Directory Creation**: Ensures target directories exist before saving
  - **Image Format**: Saves images as high-quality JPEGs

#### 4.2.3 Error Handling and Logging
- **Comprehensive Error Capture**: Catches and logs different types of errors at multiple levels
- **Detailed Logging**: Records request details, cache hits/misses, and error information
- **Recovery Mechanisms**: Attempts to recover from temporary API issues
- **Fallback Strategies**: When images cannot be obtained, logs detailed reason

#### 4.2.4 Performance Optimization
- **Session Reuse**: Maintains HTTP session to reduce connection overhead
- **Request Batching**: Groups related requests where possible
- **Memory Management**: Properly handles large binary data to prevent memory issues

### 4.3 Cache Management (`cache_manager.py`)
- **Purpose**: Optimizes API usage through caching
- **Key Features**:
  - Stores previously downloaded images
  - Indexes images by location and camera parameters
  - Reduces API costs and speeds up subsequent runs

#### 4.3.1 Technical Implementation Details
- **CacheManager Class**: Handles all caching operations
  - **Initialization**: Sets up cache directory, index file path, and cache configuration from `config.yaml`
  - **Cache Structure**: Creates and maintains a file-based cache system with index

- **Cache Index Implementation**:
  - **Index Format**: Maintains a JSON-based index mapping cache keys to file paths
    ```python
    # Example index structure
    {
        "40.7128,-74.0060_heading90_pitch0_fov55": "cache/img_001.jpg",
        "40.7128,-74.0060_heading270_pitch30_fov55": "cache/img_002.jpg"
    }
    ```
  - **Persistence**: Saves and loads index to disk to maintain state between runs
  - **Synchronization**: Ensures index file is properly updated after modifications

- **Cache Key Generation**:
  - **Parameter Normalization**: Standardizes location coordinates and camera parameters
  - **Hash Formation**: Creates unique hash strings that identify specific image requests
    ```python
    # Key generation example
    cache_key = f"{lat},{lon}_heading{int(heading)}_pitch{int(pitch)}_fov{int(fov)}"
    ```
  - **Collision Handling**: Ensures different requests generate different keys

- **Cache Storage System**:
  - **File Organization**: Maintains organized file structure for cached images
  - **Clean Directory Structure**: Prevents cluttering by using subdirectories
  - **Atomic Operations**: Ensures file writes are completed properly to prevent corruption

#### 4.3.2 Cache Operations
- **Cache Retrieval**:
  - **Key Lookup**: Efficiently searches index for matching cache keys
  - **File Access**: Reads cached images from disk when found
  - **Validation**: Verifies file integrity before returning cached content

- **Cache Storage**:
  - **Write Operations**: Saves new images to cache with proper metadata
  - **Index Update**: Updates index with new entries
  - **Duplicate Handling**: Prevents storing duplicate images

- **Cache Management**:
  - **Size Monitoring**: Tracks cache size to prevent excessive disk usage
  - **Clearing Mechanism**: Provides methods to clear cache if needed
  - **Cache Invalidation**: Logic to handle outdated cache entries

#### 4.3.3 Performance Considerations
- **In-Memory Index**: Maintains index in memory for fast lookups
- **Lazy Loading**: Loads cache data only when needed
- **Disk I/O Optimization**: Minimizes file operations

#### 4.3.4 Configurability
- **Enable/Disable Option**: Can turn caching on/off via configuration
- **Cache Location**: Configurable cache directory path
- **Retention Policy**: Framework for future implementation of cache expiration

### 4.4 Image Processing (`image_processor.py`)
- **Purpose**: Handles image storage and naming
- **Key Features**:
  - Standardized naming convention
  - Organizes images by route and capture parameters
  - Supports future preprocessing capabilities

#### 4.4.1 Technical Implementation Details
- **ImageProcessor Class**: Manages image saving and organization
  - **Initialization**: Configures with output paths and image parameters from `config.yaml`
  - **Directory Management**: Ensures output directories exist

- **Filename Generation**:
  - **Naming Convention**: Creates standardized filenames based on capture parameters
    ```python
    # Example filename format
    filename = f"pl_r{route_id}_p{point_index}_{side}_h{int(heading)}_p{int(pitch)}_fov{int(fov)}.jpg"
    ```
  - **Components**:
    - `pl_r`: Prefix indicating powerline image
    - `route_id`: Identifier for the specific route (e.g., "001")
    - `point_index`: Zero-based index of the point along the interpolated route
    - `side`: Direction relative to road bearing ("left", "right", "front", "rear")
    - `heading`: Camera heading in degrees (0-359)
    - `pitch`: Camera pitch in degrees (typically 0, 30, 60)
    - `fov`: Field of view in degrees (typically 40 or 55)

- **Side Determination Algorithm**:
  - **Relative Heading Calculation**: Computes camera orientation relative to road bearing
    ```python
    relative_heading = (heading - road_bearing + 360) % 360
    ```
  - **Classification Rules**:
    - 1° to 179°: "right" side
    - 181° to 359°: "left" side
    - 0° or 180°: "front" (if pitch < 45°) or "rear" (if pitch ≥ 45°)

#### 4.4.2 Image Handling
- **Save Operations**:
  - **File Writing**: Saves images with appropriate quality and format
  - **Directory Creation**: Creates nested directories as needed
  - **Overwrite Protection**: Option to prevent accidental overwriting

- **Image Metadata**:
  - **EXIF Data**: Preserves or adds relevant metadata to saved images
  - **Geolocation**: Maintains geographic coordinates in image metadata
  - **Camera Parameters**: Records capture settings in metadata

#### 4.4.3 Future Extensions
- **Preprocessing Pipeline**: Framework for adding image preprocessing steps
  - **Noise Reduction**: Placeholders for implementing noise filtering
  - **Color Correction**: Infrastructure for color normalization
  - **Resolution Standardization**: Utilities for consistent image sizing

- **Quality Control**:
  - **Image Validation**: Checks for corrupt or unusable images
  - **Blur Detection**: Framework for identifying and flagging blurry images
  - **Exposure Assessment**: Structure for detecting over/under-exposed images

#### 4.4.4 Integration Points
- **Acquisition Workflow**: Seamlessly integrates with the image acquisition process
- **Dataset Preparation**: Provides organized images ready for labeling
- **Extensibility**: Modular design allows for adding new processing steps

## 5. Configuration Guide

The system is configured through `config.yaml`. Key sections include:

### 5.1 API Configuration
```yaml
api:
  key: ${GOOGLE_API_KEY}
  min_request_interval: 0.1
  max_retries: 3
  retry_delay: 1.0
  image_width: 640
  image_height: 640
```

### 5.2 Sampling Parameters
```yaml
sampling:
  base_interval: 20.0  # meters between points
  max_extra_points: 5
```

### 5.3 Camera Settings
```yaml
camera:
  altitude: 2.5  # meters
  pitch_values: [0, 30, 60]
  fov_default: [55]
  fov_special: [40]
  relative_angles_right: [90, 45, 135]
  relative_angles_left: [270, 310, 230]
```

### 5.4 Powerline Parameters
```yaml
powerline:
  offset_distance: 10.0  # meters from road
  side: "both"  # right, left, or both
```

## 6. Step-by-Step Guide

### 6.1 Initial Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Google Cloud Platform:
   - Create a project
   - Enable required APIs
   - Generate API key
   - Set environment variable:
     ```bash
     export GOOGLE_API_KEY="your-api-key"
     ```

### 6.2 Route Preparation
1. Create a `routes.csv` file in the `data` directory with columns:
   - `route_id`: Unique identifier (e.g., "001")
   - `start_location`: Starting point (address or lat,lon)
   - `end_location`: Ending point (address or lat,lon)

### 6.3 Running Data Collection
1. Verify configuration in `config.yaml`
2. Run the acquisition script:
   ```bash
   python -m src.acquisition.acquisition_run
   ```
   - Optionally specify route ID in the script

### 6.4 Monitoring Progress
- Check console output for:
  - Route processing status
  - Image acquisition progress
  - API request statistics
  - Error messages
- Monitor the `data/raw` directory for collected images

## 7. Step by Step Process

This section provides a detailed technical walkthrough of the exact sequence of operations that occur when running the data collection pipeline, from loading routes to saving images.

### 7.1 Program Execution Flow

1. **Entry Point**: `acquisition_run.py` - `run_acquisition()` function
   ```python
   python -m src.acquisition.acquisition_run
   ```

2. **Environment Setup**:
   - Load environment variables (including API key)
   - Start timing the process
   - Initialize logging

3. **Configuration Loading**:
   ```python
   config = load_config(config_path)
   ```
   - Reads `config.yaml`
   - Substitutes environment variables (e.g., `${GOOGLE_API_KEY}`)
   - Validates essential configuration parameters

4. **Component Initialization**:
   ```python
   cache_mgr = CacheManager(config)
   route_planner = RoutePlanner(config)
   streetview_cli = StreetViewClient(config, cache_mgr)
   image_proc = ImageProcessor(config)
   ```
   - Each component reads its specific configuration sections
   - Cache manager initializes cache directory and loads index
   - Route planner initializes geodetic tools
   - Street View client configures request parameters

5. **Route Data Loading**:
   ```python
   routes_file = os.path.join(config.get("paths", {}).get("data_dir", "data"), "routes.csv")
   routes = load_routes(routes_file, route_id_filter=route_id)
   ```
   - Locates routes.csv file in the configured data directory
   - Loads routes into a pandas DataFrame
   - Optionally filters by specific route ID if provided

### 7.2 Processing Each Route

For each route in the routes DataFrame, the following steps are executed:

1. **Route Planning**:
   ```python
   planned_route = route_planner.plan_route(start_loc, end_loc)
   ```
   - **Direction API Call**: Fetches route data from Google Directions API
     ```python
     route_coords = self._get_route_coordinates_from_api(start_location, end_location)
     ```
   - **Route Interpolation**: Generates evenly spaced points along the route
     ```python
     interpolated_route = self._interpolate_route(route_coords)
     ```
   - **Bearing Calculation**: Determines road direction at each point
     ```python
     bearings = self._calculate_route_bearings(interpolated_route)
     ```
   - **Powerline Locations**: Calculates potential powerline points with offsets
     ```python
     powerline_points = self._calculate_powerline_points(interpolated_route, bearings)
     ```
   - Returns a dictionary with route points, bearings, and powerline points

2. **Route Data Extraction**:
   ```python
   route_points = planned_route['route_points']
   route_bearings = planned_route['route_bearings']
   ```
   - Extracts the interpolated route points (latitude, longitude pairs)
   - Extracts corresponding bearing values for each point

### 7.3 Processing Each Point on Route

For each point along the interpolated route:

1. **Camera View Calculation**:
   ```python
   required_views = calculate_camera_views(road_bearing, camera_config)
   ```
   - Determines all required camera views based on configuration
   - Combines road bearing with relative angles (left/right sides)
   - Applies configured pitch values
   - Applies configured field of view (FOV) values
   - Returns a list of dictionaries with heading, pitch, and FOV parameters

2. **Processing Each Camera View**:
   For each required view at a given location point:

   a. **Filename Preparation**:
      ```python
      # Calculate side (right, left, front, rear) based on relative heading
      relative_heading = (heading - road_bearing + 360) % 360
      if 1 <= relative_heading <= 179:
          side = "right"
      elif 181 <= relative_heading <= 359:
          side = "left"
      else:
          side = "front" if pitch < 45 else "rear"
      
      # Generate standardized filename
      filename = f"pl_r{r_id}_p{i}_{side}_h{int(heading)}_p{int(pitch)}_fov{int(fov)}.jpg"
      output_image_path = os.path.join(raw_images_dir, filename)
      ```

   b. **Image Retrieval**:
      ```python
      success = streetview_cli.get_image(lat, lon, output_image_path, **view_params)
      ```
      
      i. **Cache Check**: Verifies if image exists in cache
         ```python
         cache_key = self._generate_cache_key(lat, lon, heading, pitch, fov)
         cached_img_path = self.cache_mgr.get_from_cache(cache_key)
         ```
         - If found in cache, loads image from cache
         - If not found, proceeds to API call

      ii. **API Request**: If not in cache, requests image from API
          ```python
          response = self._make_api_request(lat, lon, heading, pitch, fov)
          ```
          - Applies rate limiting to avoid exceeding API quota
          - Handles retries with exponential backoff for errors
          - Validates response status and content
          
      iii. **Image Saving**: Saves the retrieved image
           ```python
           # If from cache, copy file from cache to destination
           shutil.copy2(cached_img_path, output_path)
           
           # If from API, save response content
           with open(output_path, 'wb') as f:
               f.write(response.content)
           ```
           
      iv. **Cache Update**: If from API, adds to cache
          ```python
          self.cache_mgr.add_to_cache(cache_key, output_path, image_data)
          ```
          - Saves image file to cache directory
          - Updates cache index

### 7.4 Sequence Diagram

```
┌─────────────────┐     ┌───────────────┐    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ acquisition_run │     │ RoutePlanner  │    │StreetViewClient│    │ CacheManager  │    │ImageProcessor │
└────────┬────────┘     └───────┬───────┘    └───────┬───────┬┘    └───────┬───────┘    └───────┬───────┘
         │                      │                    │       │              │                    │
         │ 1. Initialize        │                    │       │              │                    │
         │─────────────────────>│                    │       │              │                    │
         │                      │                    │       │              │                    │
         │ 2. Initialize        │                    │       │              │                    │
         │───────────────────────────────────────────>       │              │                    │
         │                      │                    │       │              │                    │
         │ 3. Initialize        │                    │       │              │                    │
         │─────────────────────────────────────────────────────────────────>│                    │
         │                      │                    │       │              │                    │
         │ 4. Initialize        │                    │       │              │                    │
         │─────────────────────────────────────────────────────────────────────────────────────>│
         │                      │                    │       │              │                    │
         │ 5. Plan Route        │                    │       │              │                    │
         │─────────────────────>│                    │       │              │                    │
         │                      │                    │       │              │                    │
         │<─────────────────────│                    │       │              │                    │
         │ route data           │                    │       │              │                    │
         │                      │                    │       │              │                    │
         │                      │                    │       │              │                    │
         │ 6. For each point:   │                    │       │              │                    │
         │ Calculate views      │                    │       │              │                    │
         │                      │                    │       │              │                    │
         │ 7. For each view:    │                    │       │              │                    │
         │ Get image            │                    │       │              │                    │
         │───────────────────────────────────────────>       │              │                    │
         │                      │                    │       │              │                    │
         │                      │                    │ 8. Check cache       │                    │
         │                      │                    │─────────────────────>│                    │
         │                      │                    │       │              │                    │
         │                      │                    │<─────────────────────│                    │
         │                      │                    │ cache result         │                    │
         │                      │                    │       │              │                    │
         │                      │                    │ 9. If not in cache:  │                    │
         │                      │                    │ API request          │                    │
         │                      │                    │       │              │                    │
         │                      │                    │       │ 10. Save to cache                 │
         │                      │                    │       │─────────────>│                    │
         │                      │                    │       │              │                    │
         │<───────────────────────────────────────────────────────────────────────────────────────
         │ success/failure      │                    │       │              │                    │
         │                      │                    │       │              │                    │
```

### 7.5 Output Verification

To verify successful execution, you can check:

1. **Console Output**: Look for messages indicating successful route processing, image acquisition, and summary statistics.
   ```
   --- Route 001 finished. Saved 450 images for this route. ---
   
   Image acquisition process finished.
   Summary:
     - Total routes processed: 1/1
     - Total images attempted: 450
     - Total images saved: 450
     - Total duration: 380.25 seconds
   ```

2. **Raw Images Directory**:
   ```bash
   ls -la data/raw/
   ```
   - Should contain images named according to the pattern:
     `pl_r<route_id>_p<point_index>_<side>_h<heading>_p<pitch>_fov<fov>.jpg`
   - Example: `pl_r001_p0_right_h90_p0_fov55.jpg`

3. **Cache Directory** (if enabled):
   ```bash
   ls -la data/cache/
   ```
   - Should contain cached images and a cache index file
   - Cache index.json should map location/parameter keys to image file paths

## 8. Troubleshooting

### 8.1 Common Issues
1. **API Key Problems**
   - Ensure `GOOGLE_API_KEY` is set
   - Verify API billing is enabled
   - Check API quotas and limits

2. **Route Planning Failures**
   - Validate `routes.csv` format
   - Check address/coordinate formats
   - Ensure route is accessible via Street View

3. **Image Acquisition Issues**
   - Check network connectivity
   - Verify API rate limits
   - Ensure sufficient disk space

### 8.2 Error Messages
- "Configuration file not found": Check `config.yaml` path
- "API key not found": Set `GOOGLE_API_KEY` environment variable
- "Routes CSV file not found": Create/check `routes.csv`
- "Failed to get route from Directions API": Validate route endpoints

### 8.3 Support Resources
- Check project documentation
- Review Google API documentation
- Contact project maintainers
- Submit issues on project repository

---

*Note: This guide is maintained by the project team. For updates or corrections, please contact the project maintainers.* 