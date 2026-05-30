"""
Pose estimation module for ErgoVision.

Supports two modes:
1. **Full-image** — run YOLOv8/YOLO11-pose on the entire frame (legacy).
2. **Crop-based** — run pose on a cropped person region, then remap
   keypoints back to original frame coordinates.  Used in the two-stage
   pipeline for higher precision.

No training or fine-tuning.
"""

import numpy as np
from ultralytics import YOLO
from .config import POSE_MODEL_NAME, CROP_POSE_MODEL


# ===================================================================
# Legacy full-image pose estimator
# ===================================================================

class PoseEstimator:
    """Load a pretrained YOLOv8-pose model and run inference on images."""

    def __init__(self, model_name=None, confidence_threshold=0.5):
        model_name = model_name or POSE_MODEL_NAME
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def estimate(self, image_path, conf_threshold=None):
        """
        Run pose estimation on a single image (full-frame).

        Parameters
        ----------
        image_path : str or Path or np.ndarray
        conf_threshold : float or None

        Returns
        -------
        dict with:
          - image_path : source path
          - detections : list of detected persons
        """
        conf = conf_threshold if conf_threshold is not None else self.confidence_threshold
        results = self.model(str(image_path) if not isinstance(image_path, np.ndarray) else image_path,
                             verbose=False, conf=conf)
        result = results[0]

        detections = self._parse_result(result)
        return {
            'image_path': str(image_path) if not isinstance(image_path, np.ndarray) else 'array',
            'detections': detections,
        }

    def estimate_batch(self, image_paths, verbose=False, conf_threshold=None):
        """Run pose estimation on multiple images."""
        results = []
        for path in image_paths:
            if verbose:
                print(f"  Processing: {path.name if hasattr(path, 'name') else path}")
            results.append(self.estimate(path, conf_threshold=conf_threshold))
        return results

    @staticmethod
    def _parse_result(result):
        """Extract person detections from a YOLO result object."""
        detections = []
        if result.keypoints is not None:
            kps = result.keypoints.data.cpu().numpy()
            boxes = result.boxes

            for i in range(kps.shape[0]):
                xy = kps[i, :, :2]
                conf_arr = kps[i, :, 2]
                person = {
                    'keypoints': xy,
                    'confidence': conf_arr,
                    'confidence_mean': float(np.mean(conf_arr[conf_arr > 0])) if np.any(conf_arr > 0) else 0.0,
                }
                if boxes is not None:
                    person['bbox'] = boxes.xyxy[i].cpu().numpy().tolist()
                    if boxes.conf is not None:
                        person['detection_conf'] = float(boxes.conf[i].cpu().numpy())
                detections.append(person)
        return detections


# ===================================================================
# Crop-based pose estimator (two-stage pipeline)
# ===================================================================

class CropPoseEstimator:
    """Run pose estimation on cropped person bounding boxes.

    Each person is cropped from the frame (with configurable padding),
    the crop is passed to a YOLO-pose model, and keypoints are remapped
    back to the original image coordinates so scoring remains independent
    of the cropping.

    Parameters
    ----------
    model_name : str
        YOLO pose model for the crop stage (e.g. ``yolov8m-pose.pt``).
    """

    def __init__(self, model_name=None):
        self.model_name = model_name or CROP_POSE_MODEL
        self.model = YOLO(self.model_name)

    def estimate_on_crop(self, image, bbox, padding=0.15):
        """Run pose on a cropped person region, return keypoints in frame coords.

        Parameters
        ----------
        image : np.ndarray
            Full BGR frame.
        bbox : list[float]
            [x1, y1, x2, y2] in frame coordinates.
        padding : float
            Fraction of bbox size to add on each side (reduces edge effects).

        Returns
        -------
        dict with:
          - keypoints        : (17, 3) array — (x, y, conf) in *frame* coords
          - confidence       : (17,) keypoint confidence
          - confidence_mean  : float
          - bbox             : [x1, y1, x2, y2] of the crop (padded)
          - crop_valid       : bool
        """
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)

        pad_x = int(bw * padding)
        pad_y = int(bh * padding)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w_img, x2 + pad_x)
        crop_y2 = min(h_img, y2 + pad_y)

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return {
                'keypoints': np.zeros((17, 3)),
                'confidence': np.zeros(17),
                'confidence_mean': 0.0,
                'bbox': [crop_x1, crop_y1, crop_x2, crop_y2],
                'crop_valid': False,
            }

        results = self.model(crop, verbose=False)
        result = results[0]

        if result.keypoints is None or len(result.keypoints.data) == 0:
            return {
                'keypoints': np.zeros((17, 3)),
                'confidence': np.zeros(17),
                'confidence_mean': 0.0,
                'bbox': [crop_x1, crop_y1, crop_x2, crop_y2],
                'crop_valid': False,
            }

        # Take the highest-confidence person in the crop
        kps_data = result.keypoints.data.cpu().numpy()
        best = 0
        if len(kps_data) > 1:
            # Pick person with highest mean keypoint confidence
            means = [np.mean(k[i, 2]) for i, k in enumerate(kps_data)]
            best = int(np.argmax(means))

        kp_crop = kps_data[best]  # (17, 3) in crop coordinates

        # Remap to frame coordinates
        kp_frame = kp_crop.copy()
        kp_frame[:, 0] += crop_x1  # x
        kp_frame[:, 1] += crop_y1  # y

        conf_arr = kp_frame[:, 2]
        mean_conf = float(np.mean(conf_arr[conf_arr > 0])) if np.any(conf_arr > 0) else 0.0

        return {
            'keypoints': kp_frame[:, :2],
            'confidence': conf_arr,
            'confidence_mean': mean_conf,
            'bbox': [crop_x1, crop_y1, crop_x2, crop_y2],
            'crop_valid': True,
        }

    def estimate_crops(self, image, detections, padding=0.15):
        """Run crop-pose for every detection in a single frame.

        Parameters
        ----------
        image : np.ndarray
        detections : list[dict]
            Each must have ``bbox``.
        padding : float

        Returns
        -------
        list[dict] — one pose result per input detection.
        """
        results = []
        for det in detections:
            bbox = det['bbox']
            pose_result = self.estimate_on_crop(image, bbox, padding=padding)
            pose_result['detection_confidence'] = det.get('confidence', 0.0)
            results.append(pose_result)
        return results
