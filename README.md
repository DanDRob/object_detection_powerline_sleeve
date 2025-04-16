# Powerline Sleeve Detection

This project detects powerline sleeves from Google Street View images using YOLOv8.

## Project Structure

```
.
├── README.md                  # This file
├── run.py                     # Main entry point to run the full pipeline
├── config.yaml                # Configuration parameters
├── data/                      # Directory for storing datasets
│   ├── raw/                   # Raw images
│   ├── labeled/               # Manually labeled images
│   └── processed/             # Processed datasets (train/val/test splits)
├── models/                    # Trained models
├── results/                   # Detection results
└── src/                       # Source code
    ├── acquisition/           # Image acquisition from Street View
    ├── labeling/              # Tools for manual and auto-labeling
    ├── dataset/               # Dataset preparation and splitting
    ├── training/              # Model training and evaluation
    │   ├── powerline/         # Powerline detection
    │   └── sleeve/            # Sleeve detection
    ├── detection/             # Inference code
    ├── visualization/         # Result visualization
    └── utils/                 # Utility functions
```

## Modules

1. **Acquisition**: Fetch images from Google Street View based on routes
2. **Labeling**: Tools for manual and automated labeling of images
3. **Dataset**: Prepare datasets for training (splitting, augmentation)
4. **Training**: Train YOLOv8 models for powerline and sleeve detection
5. **Detection**: Run inference on new images
6. **Visualization**: Visualize detection results

## Usage

Each module can be run independently:

```python
# Acquire images
from src.acquisition import acquire_images
acquire_images(route_file="routes.csv")

# Train powerline detection model
from src.training.powerline import train_powerline_model
train_powerline_model()

# Or run the entire pipeline
python run.py
```

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Pandas 