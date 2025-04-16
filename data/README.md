# Data Directory

This directory contains datasets for powerline and sleeve detection, organized in a modular way to keep each component independent.

## Directory Structure

```
data/
│
├── powerlines/             # Powerline detection dataset
│   ├── original/           # Original powerline images and labels
│   │   ├── images/         # Original images
│   │   └── labels/         # Original labels in YOLO format
│   │
│   └── augmented/          # Augmented powerline images and labels
│       ├── images/         # Augmented images with _aug_N suffix
│       └── labels/         # Augmented labels with _aug_N suffix
│
├── sleeves/                # Sleeve detection dataset
│   ├── original/           # Original sleeve images and labels
│   │   ├── images/         # Original images
│   │   └── labels/         # Original labels in YOLO format
│   │
│   └── augmented/          # Augmented sleeve images and labels
│       ├── images/         # Augmented images with _aug_N suffix
│       └── labels/         # Augmented labels with _aug_N suffix
│
├── datasets_original/      # Original datasets (for reference)
├── raw/                    # Raw (unprocessed) images
├── labeled/                # Manually labeled images
├── processed/              # Processed datasets
└── cache/                  # Cache directory for image acquisition
```

## Usage

### Dataset Organization

1. Place original powerline images in `powerlines/original/images/`
2. Place original powerline labels in `powerlines/original/labels/`
3. Place original sleeve images in `sleeves/original/images/`
4. Place original sleeve labels in `sleeves/original/labels/`

### Data Augmentation

Use the `scripts/augment_dataset.py` script to augment the datasets:

```bash
# Augment powerlines dataset
python scripts/augment_dataset.py --dataset powerlines --num-augmentations 3

# Augment sleeves dataset
python scripts/augment_dataset.py --dataset sleeves --num-augmentations 3
```

This will create augmented versions of the images and labels with suffixes like `_aug_1`, `_aug_2`, etc.

## Naming Convention

- Original images and labels use their original filenames
- Augmented images and labels have the format: `original_filename_aug_N.ext` where N is the augmentation number 