#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check dataset statistics: counts images, labels, and empty labels."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset root directory containing train/, val/, test/ folders.",
    )
    return parser.parse_args()


def count_files(directory: Path, extension: str) -> int:
    """Count files with specific extension in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(f"*.{extension}")))


def analyze_labels(labels_dir: Path) -> Tuple[int, int, int]:
    """Count total, empty, and non-empty label files in a directory."""
    if not labels_dir.exists():
        return 0, 0, 0

    label_files = list(labels_dir.glob("*.txt"))
    total_labels = len(label_files)

    empty_labels = 0
    for label_file in label_files:
        if label_file.stat().st_size == 0:
            empty_labels += 1

    non_empty_labels = total_labels - empty_labels
    return total_labels, empty_labels, non_empty_labels


def check_split(split_dir: Path) -> Dict:
    """Analyze a dataset split (train, val, or test)."""
    if not split_dir.exists():
        logging.warning(f"Split directory not found: {split_dir}")
        return {
            "images": 0,
            "labels": 0,
            "empty_labels": 0,
            "non_empty_labels": 0,
        }

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    # Count images (jpg, jpeg, png)
    jpg_count = count_files(images_dir, "jpg")
    jpeg_count = count_files(images_dir, "jpeg")
    png_count = count_files(images_dir, "png")
    total_images = jpg_count + jpeg_count + png_count

    # Count JSON files (original LabelMe annotations)
    json_count = count_files(images_dir, "json")

    # Analyze labels
    total_labels, empty_labels, non_empty_labels = analyze_labels(labels_dir)

    return {
        "images": total_images,
        "json_files": json_count,
        "labels": total_labels,
        "empty_labels": empty_labels,
        "non_empty_labels": non_empty_labels,
    }


def main() -> None:
    args = parse_arguments()
    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        logging.error(f"Dataset path not found: {dataset_path}")
        return

    # Analyze each split
    splits = ["train", "val", "test"]
    results = {}

    print("\n=== Dataset Analysis ===")
    print(f"Dataset path: {dataset_path}")
    print("-" * 40)

    for split in splits:
        split_dir = dataset_path / split
        results[split] = check_split(split_dir)

        print(f"\n{split.upper()} Split:")
        print(f"  Images:          {results[split]['images']}")
        print(f"  JSON files:      {results[split]['json_files']}")
        print(f"  Labels:          {results[split]['labels']}")
        print(f"  Empty labels:    {results[split]['empty_labels']}")
        print(f"  Non-empty labels: {results[split]['non_empty_labels']}")

    # Calculate totals
    total_images = sum(results[split]["images"] for split in splits)
    total_json = sum(results[split]["json_files"] for split in splits)
    total_labels = sum(results[split]["labels"] for split in splits)
    total_empty_labels = sum(
        results[split]["empty_labels"] for split in splits)
    total_non_empty_labels = sum(
        results[split]["non_empty_labels"] for split in splits)

    print("\n=== SUMMARY ===")
    print(f"Total images:           {total_images}")
    print(f"Total JSON files:       {total_json}")
    print(f"Total labels:           {total_labels}")
    print(f"Total empty labels:     {total_empty_labels}")
    print(f"Total non-empty labels: {total_non_empty_labels}")

    # Check for potential issues
    if total_images != total_labels:
        print("\n⚠️ Warning: Number of images does not match number of labels!")

    if total_empty_labels > 0:
        print(f"\n⚠️ Warning: Found {total_empty_labels} empty label files!")


if __name__ == "__main__":
    main()
