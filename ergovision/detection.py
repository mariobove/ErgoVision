"""
Human detection module for the two-stage ErgoVision pipeline.

Uses YOLO detection models (not pose) to find persons reliably before
running pose estimation on each crop.  Significantly reduces false
positives compared with the one-stage YOLOv8-pose approach.
"""

import numpy as np
from ultralytics import YOLO
from .config import HUMAN_DETECTOR_MODEL, COCO_PERSON_CLASS_ID


class HumanDetector:
    """YOLO-based person detector.

    Parameters
    ----------
    model_name : str
        YOLO detection model path (e.g. ``yolov8m.pt``).
    confidence : float
        Detection confidence threshold.
    """

    def __init__(self, model_name=None, confidence=0.5):
        self.model_name = model_name or HUMAN_DETECTOR_MODEL
        self.model = YOLO(self.model_name)
        self.confidence = confidence

    def detect(self, image, conf_threshold=None, return_raw=False):
        """Run person detection on a single image.

        Parameters
        ----------
        image : np.ndarray or str
            BGR image array or path to image.
        conf_threshold : float or None
            Overrides instance default.
        return_raw : bool
            If True, return all detections before filtering.

        Returns
        -------
        list[dict] — each with keys:
          - bbox       : [x1, y1, x2, y2] in pixel coordinates
          - confidence : float
          - raw        : (optional) full detection dict
        """
        conf = conf_threshold if conf_threshold is not None else self.confidence
        results = self.model(image, conf=conf, verbose=False)
        result = results[0]

        detections = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            if int(classes[i]) != COCO_PERSON_CLASS_ID:
                continue

            det = {
                'bbox': boxes[i].tolist(),
                'confidence': float(confs[i]),
            }
            if return_raw:
                det['raw'] = {
                    'class_id': int(classes[i]),
                    'class_name': 'person',
                }
            detections.append(det)

        return detections

    def detect_batch(self, images, conf_threshold=None, verbose=False):
        """Run detection on multiple images.

        Parameters
        ----------
        images : list[np.ndarray or str]
        conf_threshold : float or None
        verbose : bool

        Returns
        -------
        list[list[dict]] — one list of detections per image.
        """
        return [
            self.detect(img, conf_threshold=conf_threshold)
            for img in images
        ]

    @staticmethod
    def filter_detections(detections, image_shape, cfg):
        """Apply spatial filters to remove implausible person detections.

        Parameters
        ----------
        detections : list[dict]
            Output from ``detect()``.
        image_shape : tuple
            (height, width) of the source image.
        cfg : ExperimentConfig
            Holds filtering thresholds.

        Returns
        -------
        valid : list[dict]
            Detections that passed all filters.
        discarded : list[tuple[dict, str]]
            (detection, reason) for each filtered-out detection.
        """
        h_img, w_img = image_shape[:2]
        valid, discarded = [], []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            aspect = bw / bh

            reason = None
            if area < cfg.min_bbox_area:
                reason = f'bbox too small ({area:.0f} < {cfg.min_bbox_area} px²)'
            elif bh < cfg.min_bbox_height:
                reason = f'bbox too short ({bh:.0f} < {cfg.min_bbox_height} px)'
            elif aspect < cfg.min_bbox_aspect:
                reason = f'aspect ratio too narrow ({aspect:.2f})'
            elif aspect > cfg.max_bbox_aspect:
                reason = f'aspect ratio too wide ({aspect:.2f})'
            elif det['confidence'] < cfg.min_person_confidence:
                reason = f'low confidence ({det["confidence"]:.2f})'

            if reason:
                discarded.append((det, reason))
            else:
                valid.append(det)

        return valid, discarded

    @staticmethod
    def debug_visualization(image, valid, discarded, filtered_out,
                            output_path=None):
        """Draw debug overlay showing which detections were accepted/rejected.

        Colors
        ------
        - Green  = accepted (valid)
        - Yellow = passed filters but uncertain
        - Red    = discarded
        """
        import cv2
        vis = image.copy()

        for det in valid:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"P:{det['confidence']:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        for det, reason in discarded:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(vis, reason[:30], (x1, y2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if output_path:
            from .visualization import save_prediction
            save_prediction(vis, output_path)

        return vis
