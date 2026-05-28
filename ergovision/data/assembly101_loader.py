"""
Assembly101 dataset loader.

Assembly101 is a multi-view procedural activity dataset capturing people
assembling and disassembling objects.  This loader discovers video files,
extracts metadata, and supports configurable subset selection.

Official website: https://assembly-101.github.io/
Download scripts: https://github.com/Assembly101-2022/assembly101-download

This loader does NOT download the dataset.  The user must obtain it manually
or via the official download scripts.
"""

import re
from pathlib import Path

import cv2

from ..config import ASSEMBLY101_VIDEO_EXTENSIONS


class Assembly101Video:
    """Lightweight metadata container for a single Assembly101 video."""

    def __init__(self, path):
        self.path = Path(path)
        self.video_name = self.path.stem          # filename without extension
        self._metadata = None                     # lazy-loaded

    def __repr__(self):
        return f"Assembly101Video('{self.video_name}')"

    def load_metadata(self):
        """Open the video file and extract stream metadata.  Returns self."""
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            self._metadata = {
                'fps': 0.0,
                'frame_count': 0,
                'duration_seconds': 0.0,
                'width': 0,
                'height': 0,
                'valid': False,
            }
            cap.release()
            return self

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0.0

        cap.release()

        self._metadata = {
            'fps': round(fps, 2),
            'frame_count': frame_count,
            'duration_seconds': round(duration, 2),
            'width': width,
            'height': height,
            'valid': fps > 0 and frame_count > 0,
        }
        return self

    @property
    def metadata(self):
        if self._metadata is None:
            self.load_metadata()
        return self._metadata

    @property
    def is_valid(self):
        return self.metadata['valid']

    def guess_camera_view(self):
        """
        Attempt to extract a camera/view label from the video filename.

        Assembly101 filenames sometimes contain patterns like ``_view_1``,
        ``_cam_2``, ``_C1``, or similar.  Returns ``None`` if no pattern
        is detected.
        """
        patterns = [
            r'_view[_-]?(\d+)',
            r'_cam[_-]?(\d+)',
            r'_c[_-]?(\d+)',
            r'cam[_-]?(\d+)',
            r'view[_-]?(\d+)',
        ]
        for p in patterns:
            match = re.search(p, self.video_name, re.IGNORECASE)
            if match:
                return f"view_{match.group(1)}"
        return None


class Assembly101Loader:
    """
    Discover and select Assembly101 video files.

    Parameters
    ----------
    video_folder : str or Path
        Root directory containing Assembly101 video files (searched recursively).
    max_videos : int or None
        Maximum number of videos to return from ``discover_videos``.  ``None``
        means no limit.
    extensions : set of str
        Video file extensions to search for.
    """

    def __init__(self, video_folder, max_videos=None,
                 extensions=None):
        self.video_folder = Path(video_folder)
        self.max_videos = max_videos
        self.extensions = extensions or ASSEMBLY101_VIDEO_EXTENSIONS

    def discover_videos(self):
        """
        Walk *video_folder* recursively and return a list of
        ``Assembly101Video`` objects.

        The list is sorted alphabetically for reproducibility.
        """
        if not self.video_folder.exists():
            raise FileNotFoundError(
                f"Assembly101 video folder not found: {self.video_folder}"
            )

        videos = []
        for ext in self.extensions:
            for fpath in sorted(self.video_folder.rglob(f'*{ext}')):
                if fpath.is_file():
                    videos.append(Assembly101Video(fpath))

        if not videos:
            raise FileNotFoundError(
                f"No video files ({', '.join(self.extensions)}) found in "
                f"{self.video_folder}"
            )

        # Apply limit
        if self.max_videos is not None and self.max_videos > 0:
            videos = videos[:self.max_videos]

        return videos

    def load_selected(self, video_paths):
        """
        Load metadata for a list of video paths in one call.

        Returns a list of ``Assembly101Video`` objects with metadata populated.
        """
        videos = []
        for path in video_paths:
            v = Assembly101Video(path)
            v.load_metadata()
            videos.append(v)
        return videos
