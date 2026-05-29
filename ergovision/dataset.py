"""
Dataset-agnostic file discovery and management.

Walks directory trees to find image / video files without assuming any
internal folder structure, making ErgoVision compatible with arbitrary
datasets on local machines and Kaggle.
"""

import os
import random
from pathlib import Path
from dataclasses import dataclass, field
from .config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    """Summary of a dataset directory."""

    name: str
    """Folder name of the dataset."""

    path: Path
    """Absolute path to the dataset root."""

    n_videos: int = 0
    n_images: int = 0
    video_files: list = field(default_factory=list)
    image_files: list = field(default_factory=list)

    @property
    def total_files(self):
        return self.n_videos + self.n_images

    @property
    def has_videos(self):
        return self.n_videos > 0

    @property
    def has_images(self):
        return self.n_images > 0

    def short_summary(self):
        parts = []
        if self.n_videos:
            parts.append(f"{self.n_videos} video(s)")
        if self.n_images:
            parts.append(f"{self.n_images} image(s)")
        return f"{self.name}: {', '.join(parts)}"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def find_media(root_dir):
    """Walk a directory and return separate lists of image and video files.

    Parameters
    ----------
    root_dir : str or Path
        Root directory to scan recursively.

    Returns
    -------
    images : list[Path]
        All image files found.
    videos : list[Path]
        All video files found.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    images, videos = [], []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(Path(dirpath) / fname)
            elif ext in VIDEO_EXTENSIONS:
                videos.append(Path(dirpath) / fname)

    return images, videos


def inspect_dataset(dataset_path):
    """Scan a dataset directory and return structured *DatasetInfo*.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the dataset folder.

    Returns
    -------
    DatasetInfo
    """
    path = Path(dataset_path)
    images, videos = find_media(path)
    return DatasetInfo(
        name=path.name,
        path=path.resolve(),
        n_videos=len(videos),
        n_images=len(images),
        video_files=videos,
        image_files=images,
    )


# ---------------------------------------------------------------------------
# Backward-compatible legacy functions
# ---------------------------------------------------------------------------

def find_images(root_dir, shuffle=True, seed=42):
    """Walk the dataset directory and collect all image file paths.

    Parameters
    ----------
    root_dir : str or Path
        Path to the dataset directory.
    shuffle : bool
        Whether to randomise the result order.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[Path]
        Every image found under *root_dir*.
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
    """Select a subset of images for initial testing.

    Parameters
    ----------
    image_paths : list[Path]
        Full list of image paths.
    subset_size : int
        Maximum number of images to include.

    Returns
    -------
    list[Path]
        First *subset_size* paths (or all if fewer exist).
    """
    return image_paths[:min(subset_size, len(image_paths))]
