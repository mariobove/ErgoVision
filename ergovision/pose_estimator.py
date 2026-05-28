"""
Pose estimation module using YOLOv8-pose.

Loads a pretrained model and runs inference.  No training or fine-tuning.
"""

import numpy as np
from ultralytics import YOLO
from .config import POSE_MODEL_NAME


class PoseEstimator:
    """Load a pretrained YOLOv8-pose model and run inference on images."""

    def __init__(self, model_name=None):
        model_name = model_name or POSE_MODEL_NAME
        self.model = YOLO(model_name)

    def estimate(self, image_path):
        """
        Run pose estimation on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            dict with:
              - image_path : source path
              - detections : list of detected persons, each containing
                  - keypoints  : (17, 2) array of (x, y) coordinates
                  - confidence : (17,) array of keypoint confidences
                  - bbox       : [x1, y1, x2, y2] bounding box (if available)
        """
        results = self.model(str(image_path), verbose=False)
        result = results[0]

        detections = []
        if result.keypoints is not None:
            kps = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes = result.boxes

            for i in range(kps.shape[0]):
                xy = kps[i, :, :2]
                conf = kps[i, :, 2]

                person = {
                    'keypoints': xy,
                    'confidence': conf,
                }

                if boxes is not None:
                    person['bbox'] = boxes.xyxy[i].cpu().numpy().tolist()

                detections.append(person)

        return {
            'image_path': str(image_path),
            'detections': detections,
        }

    def estimate_batch(self, image_paths, verbose=False):
        """
        Run pose estimation on multiple images.

        Args:
            image_paths: List of image paths.
            verbose: Whether to print progress per image.

        Returns:
            List of detection dicts (one per image).
        """
        results = []
        for path in image_paths:
            if verbose:
                print(f"  Processing: {path.name}")
            results.append(self.estimate(path))
        return results
