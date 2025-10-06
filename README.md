# Powerline Sleeve Detection

This project detects powerline sleeves from Google Street View images using YOLOv12.
The Final Project Report can be viewed via Google Drive here: https://docs.google.com/document/d/1RsHusC6kWwPUfkBeQyUASQZ29fyRe4P1Mfr5pA0K1-s/edit?tab=t.0#heading=h.yml1oshqlpep

## Project Structure

```
.
├── README.md                  # This file
├── run.py                     # Main entry point to run the full pipeline
├── config.yaml                # Configuration parameters
├── .env                       # Environment variables (e.g., GOOGLE_API_KEY)
├── requirements.txt           # Python dependencies
├── data/
│   ├── routes.csv             # CSV file defining routes for acquisition
│   ├── powerlines/             # Powerline images
│   ├── sleeves/                # Sleeve images
│   ├── raw/                   # Raw images downloaded by acquisition
|   ├── tests/                 # Test images
├── models/                    # Trained models
├── results/                   # Detection results
├── scripts/                   # Support functions for dataset val. & conv. (to be removed)
└── src/
    ├── __init__.py
    ├── acquisition/           # Image acquisition from Street View
    │   ├── __init__.py
    │   ├── acquisition_run.py # Main script for the acquisition module
    │   ├── route_processor.py # Plans routes, interpolates points (will be renamed ideally)
    │   ├── streetview_client.py # Interacts with Google Street View API
    │   ├── cache_manager.py   # Caches downloaded images
    │   └── image_processor.py # Saves images (can be extended)
    ├── labeling/
    ├── dataset/
    ├── training/
    │   ├── powerline/
    │   └── sleeve/
    ├── detection/
    ├── visualization/
    └── utils/
```

## Modules

1.  **Acquisition**: Fetch images from Google Street View based on routes defined in `data/routes.csv`.
2.  **Labeling**: Tools for manual and automated labeling of images.
3.  **Dataset**: Prepare datasets for training (splitting, augmentation).
4.  **Training**: Train YOLOv12 models for powerline and sleeve detection.
5.  **Detection**: Run inference on new images.
6.  **Visualization**: Visualize detection results.

## Setup

1.  **Clone the repository.**
2.  **Create Environment:** Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **API Key:** Create a `.env` file in the project root directory and add your Google Cloud API Key:
    ```
    GOOGLE_API_KEY=YOUR_ACTUAL_API_KEY
    ```
    Ensure this key has permissions for the **Google Directions API** and the **Street View Static API**.
5.  **Define Routes:** Create or modify the `data/routes.csv` file with columns `route_id`, `start_location`, `end_location`.

## Usage

Modules can often be run independently, or the full pipeline via `run.py` (if implemented).

````bash
# Example: Run the image acquisition module from the project root
python -m src.acquisition.acquisition_run


```python
# Example: Programmatic usage (if needed)
from src.acquisition import run_acquisition

# Ensure .env is loaded or GOOGLE_API_KEY is set in the environment
run_acquisition(route_id="004")
````

## Requirements

See `requirements.txt`. Key dependencies include:

- Python 3.8+
- PyTorch
- Ultralytics YOLOv12
- OpenCV-Python
- Pandas
- Requests
- PyYAML
- polyline
- geopy
- pyproj
- Pillow
- python-dotenv
