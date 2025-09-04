
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


# Configure dataset paths here. You can list multiple datasets.
# Example:
# DATASET_PATHS = [
#     Path("data/sleeves/sleeves_v3_yolo"),
#     Path("data/sleeves/sleeves_v4_yolo"),
# ]
DATASET_PATHS: List[Path] = [
    Path("data/sleeves/sleeves_v4_org_yolo"),
    Path("data/powerlines/powerlines_yolo")

]


def count_files(directory: Path, extension: str) -> int:
    """Count files with specific extension in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(f"*.{extension}")))


def analyze_labels(labels_dir: Path) -> Tuple[int, int, int, int, int, Set[str], Set[str]]:
    """Analyze YOLO label files.

    - Empty label files are zero-byte or whitespace-only.
    - Malformed files contain any non-blank line that doesn't have exactly 5 tokens.
    - Instances are the count of valid lines across all label files.

    Returns:
        total_label_files, empty_label_files, non_empty_label_files,
        malformed_label_files, total_instances, non_empty_label_stems,
        all_label_stems
    """
    if not labels_dir.exists():
        return 0, 0, 0, 0, 0, set(), set()

    label_paths = list(labels_dir.glob("*.txt"))
    total_label_files = len(label_paths)

    empty_label_files = 0
    malformed_label_files = 0
    total_instances = 0
    non_empty_label_stems: Set[str] = set()
    all_label_stems: Set[str] = set()

    for label_path in label_paths:
        all_label_stems.add(label_path.stem)

        try:
            if label_path.stat().st_size == 0:
                empty_label_files += 1
                continue

            content = label_path.read_text(encoding="utf-8", errors="ignore")
            lines = [ln.strip() for ln in content.splitlines()]
            lines = [ln for ln in lines if ln != ""]

            if len(lines) == 0:
                empty_label_files += 1
                continue

            file_has_malformed = False
            valid_count = 0
            for ln in lines:
                parts = ln.split()
                if len(parts) == 5:
                    valid_count += 1
                else:
                    file_has_malformed = True

            if file_has_malformed:
                malformed_label_files += 1

            if valid_count > 0:
                non_empty_label_stems.add(label_path.stem)
                total_instances += valid_count
            else:
                # Only malformed lines → treat effectively as empty for image-level stats
                empty_label_files += 1
        except Exception as exc:
            logging.warning(f"Failed to read label file {label_path}: {exc}")

    non_empty_label_files = total_label_files - empty_label_files
    return (
        total_label_files,
        empty_label_files,
        non_empty_label_files,
        malformed_label_files,
        total_instances,
        non_empty_label_stems,
        all_label_stems,
    )


def check_split(split_dir: Path) -> Dict:
    """Analyze a dataset split (train, val, or test)."""
    if not split_dir.exists():
        logging.warning(f"Split directory not found: {split_dir}")
        return {
            "images": 0,
            "images_with_labels": 0,
            "images_without_labels": 0,
            "instances": 0,
            "label_files": 0,
            "empty_label_files": 0,
            "non_empty_label_files": 0,
            "malformed_labels": 0,
            "labels_without_images": 0,
            "json_files": 0,
        }

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    # Collect image files and stems
    image_files: List[Path] = []
    for ext in ("jpg", "jpeg", "png"):
        image_files.extend(images_dir.glob(f"*.{ext}"))
    total_images = len(image_files)
    image_stems: Set[str] = {p.stem for p in image_files}

    # Count JSON files (original LabelMe annotations)
    json_count = count_files(images_dir, "json")

    # Analyze labels
    (
        total_label_files,
        empty_label_files,
        non_empty_label_files,
        malformed_label_files,
        total_instances,
        non_empty_label_stems,
        all_label_stems,
    ) = analyze_labels(labels_dir)

    images_with_labels = len(image_stems.intersection(non_empty_label_stems))
    images_without_labels = max(total_images - images_with_labels, 0)
    labels_without_images = len(all_label_stems - image_stems)

    return {
        "images": total_images,
        "json_files": json_count,
        "images_with_labels": images_with_labels,
        "images_without_labels": images_without_labels,
        "instances": total_instances,
        "label_files": total_label_files,
        "empty_label_files": empty_label_files,
        "non_empty_label_files": non_empty_label_files,
        "malformed_labels": malformed_label_files,
        "labels_without_images": labels_without_images,
    }


def main() -> None:
    if not DATASET_PATHS:
        logging.error("DATASET_PATHS is empty. Please configure one or more dataset directories.")
        return

    for dataset_path in DATASET_PATHS:
        if not dataset_path.exists():
            logging.error(f"Dataset path not found: {dataset_path}")
            continue

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
            print(f"  Images:                {results[split]['images']}")
            print(f"  Images with labels:    {results[split]['images_with_labels']}")
            print(f"  Images without labels: {results[split]['images_without_labels']}")
            print(f"  Instances:             {results[split]['instances']}")
            print(f"  Label files:           {results[split]['label_files']}")
            print(f"    - empty:             {results[split]['empty_label_files']}")
            print(f"    - non-empty:         {results[split]['non_empty_label_files']}")
            print(f"    - malformed:         {results[split]['malformed_labels']}")
            print(f"  Labels without images: {results[split]['labels_without_images']}")

        # Calculate totals
        total_images = sum(results[split]["images"] for split in splits)
        total_json = sum(results[split]["json_files"] for split in splits)
        total_images_with_labels = sum(
            results[split]["images_with_labels"] for split in splits
        )
        total_images_without_labels = sum(
            results[split]["images_without_labels"] for split in splits
        )
        total_instances = sum(results[split]["instances"] for split in splits)
        total_label_files = sum(results[split]["label_files"] for split in splits)
        total_empty_label_files = sum(
            results[split]["empty_label_files"] for split in splits
        )
        total_non_empty_label_files = sum(
            results[split]["non_empty_label_files"] for split in splits
        )
        total_malformed_labels = sum(
            results[split]["malformed_labels"] for split in splits
        )
        total_labels_without_images = sum(
            results[split]["labels_without_images"] for split in splits
        )

        print("\n=== SUMMARY ===")
        print(f"Total images:             {total_images}")
        print(f"Total JSON files:         {total_json}")
        print(f"Total images w/ labels:   {total_images_with_labels}")
        print(f"Total images w/o labels:  {total_images_without_labels}")
        print(f"Total instances:          {total_instances}")
        print(f"Total label files:        {total_label_files}")
        print(f"  - empty:                {total_empty_label_files}")
        print(f"  - non-empty:            {total_non_empty_label_files}")
        print(f"Total malformed labels:   {total_malformed_labels}")
        print(f"Labels without images:    {total_labels_without_images}")

        # Check for potential issues
        if total_empty_label_files > 0:
            print(f"\n⚠️ Warning: Found {total_empty_label_files} empty label files!")
        if total_malformed_labels > 0:
            print(f"\n⚠️ Warning: Found {total_malformed_labels} malformed label files!")
        if total_labels_without_images > 0:
            print(
                f"\n⚠️ Warning: Found {total_labels_without_images} label files without matching images!"
            )


if __name__ == "__main__":
    main()
