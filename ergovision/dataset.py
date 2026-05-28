"""
Dataset inspection and loading module.

Walks a directory tree to find image files without assuming internal folder
structure, making it compatible with Kaggle and local datasets.
"""

import os
import random
from pathlib import Path
from .config import IMAGE_EXTENSIONS


def find_images(root_dir, shuffle=True, seed=42):
    """
    Walk the dataset directory and collect all image file paths.

    Args:
        root_dir: Path to the dataset directory.
        shuffle: Whether to randomize the result order.
        seed: Random seed for reproducibility.

    Returns:
        List of Path objects for every image found.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    images = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(Path(dirpath) / fname)

    if not images:
        raise FileNotFoundError(f"No image files found in {root}")

    if shuffle:
        random.seed(seed)
        random.shuffle(images)

    return images


def select_subset(image_paths, subset_size=50):
    """
    Select a subset of images for initial testing.

    Args:
        image_paths: Full list of image paths.
        subset_size: Maximum number of images to include.

    Returns:
        First *subset_size* paths from the list (or all if fewer exist).
    """
    return image_paths[:min(subset_size, len(image_paths))]
