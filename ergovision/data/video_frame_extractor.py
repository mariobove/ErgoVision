"""
Video frame extraction with quality filtering.

Extracts frames from video files at a configurable sampling rate, applies
optional blur filtering, and saves frames to disk with associated metadata.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

# Default paths (overridable via constructor)
_DEFAULT_FRAMES_DIR = 'outputs/extracted_frames'
_DEFAULT_BLUR_THRESHOLD = 0  # disabled by default


def _laplacian_variance(image):
    """
    Compute the variance of the Laplacian as a blur metric.

    Returns
    -------
    float
        A lower value indicates a blurrier image.  A common heuristic:
        ``< 100`` → blurry, ``>= 100`` → sharp.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


class ExtractionResult:
    """Metadata collected during a single video extraction pass."""

    def __init__(self, video_name):
        self.video_name = video_name
        self.total_frames_in_video = 0
        self.frames_extracted = 0
        self.frames_skipped_blurry = 0
        self.frames_skipped_unreadable = 0
        self.extracted_frame_paths = []   # list of Path objects

    @property
    def kept_ratio(self):
        if self.frames_extracted == 0:
            return 0.0
        return self.frames_extracted / max(self.total_frames_in_video, 1)

    def __repr__(self):
        return (f"ExtractionResult({self.video_name}: "
                f"{self.frames_extracted} kept, "
                f"{self.frames_skipped_blurry} blurry, "
                f"{self.frames_skipped_unreadable} unreadable)")


class VideoFrameExtractor:
    """
    Extract frames from videos at a configurable sampling rate.

    Parameters
    ----------
    output_dir : str or Path
        Root directory for extracted frames (default: ``outputs/extracted_frames/``).
    sampling_rate : float
        Target frames per second to extract (default: 1.0).
    max_frames_per_video : int
        Maximum frames to keep from a single video (default: 100).
    blur_threshold : float
        Laplacian variance below which a frame is considered blurry and
        skipped.  Set to ``0`` to disable blur filtering.
    use_filename_as_id : bool
        If True, use the full video filename (stem + extension) as the
        subdirectory name, preventing collisions when videos share the same
        stem but differ in extension or path (default: True).
    """

    def __init__(self, output_dir=None, sampling_rate=1.0,
                 max_frames_per_video=100, blur_threshold=None,
                 use_filename_as_id=True):
        self.output_dir = Path(output_dir or _DEFAULT_FRAMES_DIR)
        self.sampling_rate = sampling_rate
        self.max_frames_per_video = max_frames_per_video
        self.blur_threshold = blur_threshold if blur_threshold is not None \
            else _DEFAULT_BLUR_THRESHOLD
        self.use_filename_as_id = use_filename_as_id

    def extract(self, video_path, video_name=None):
        """
        Extract frames from a single video file.

        Processing steps
        -----------------
        1. Open the video with OpenCV ``VideoCapture``.
        2. Compute the frame interval based on video FPS and target sampling rate.
        3. Iterate through frames, keeping every N-th frame.
        4. For each kept frame:
           a. Optionally skip if Laplacian variance < threshold.
           b. Save as JPEG to ``output_dir / video_name / frame_XXXXX.jpg``.
           c. Record metadata.
        5. Stop after *max_frames_per_video* kept frames.

        Parameters
        ----------
        video_path : str or Path
            Path to the video file.
        video_name : str or None
            Name used for the output subdirectory.  Defaults to the file stem.

        Returns
        -------
        ExtractionResult
            Summary statistics and list of saved frame paths.
        """
        video_path = Path(video_path)
        # Use full filename (stem + suffix) as the unique subdirectory name
        # to avoid collisions when videos share the same stem but differ in
        # extension or are located in different directories.
        vname = video_name or (video_path.name if self.use_filename_as_id else video_path.stem)
        result = ExtractionResult(vname)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            result.frames_skipped_unreadable = 1
            return result

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result.total_frames_in_video = total_frames

        # Compute frame sampling interval
        if fps > 0 and self.sampling_rate > 0:
            frame_interval = max(1, round(fps / self.sampling_rate))
        else:
            frame_interval = 1  # fallback: every frame

        # Output subdirectory for this video
        video_out_dir = self.output_dir / vname
        video_out_dir.mkdir(parents=True, exist_ok=True)

        # Track metadata for CSV
        frame_metadata = []

        frame_idx = 0
        kept = 0

        while kept < self.max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample every N-th frame
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / fps if fps > 0 else 0.0

            # Quality: skip blurry frames
            if self.blur_threshold > 0:
                blur_score = _laplacian_variance(frame)
                if blur_score < self.blur_threshold:
                    result.frames_skipped_blurry += 1
                    frame_idx += 1
                    continue

            # Save frame
            out_name = f"frame_{frame_idx:06d}.jpg"
            out_path = video_out_dir / out_name
            cv2.imwrite(str(out_path), frame)

            kept += 1
            result.extracted_frame_paths.append(out_path)
            frame_metadata.append({
                'video_name': vname,
                'frame_index': frame_idx,
                'timestamp_seconds': round(timestamp, 3),
                'output_path': str(out_path),
            })

            frame_idx += 1

        cap.release()

        result.frames_extracted = kept

        # Write per-video metadata CSV
        if frame_metadata:
            meta_csv = video_out_dir / '_metadata.csv'
            with open(meta_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'video_name', 'frame_index', 'timestamp_seconds',
                    'output_path',
                ])
                writer.writeheader()
                writer.writerows(frame_metadata)

        return result
