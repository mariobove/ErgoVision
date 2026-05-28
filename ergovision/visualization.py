"""
Visualisation utilities for ErgoVision.

Draws keypoints, skeleton connections, and ergonomic risk overlays on images.
"""

from pathlib import Path

import cv2
import numpy as np

from .config import SKELETON

# BGR colours
KEYPOINT_COLOR = (0, 255, 0)        # green
SKELETON_COLOR = (255, 255, 0)      # cyan
TEXT_COLOR = (255, 255, 255)        # white
TEXT_BG_COLOR = (0, 0, 0)           # black
RISK_COLORS = {
    'Low Risk': (0, 255, 0),        # green
    'Medium Risk': (0, 255, 255),   # yellow
    'High Risk': (0, 0, 255),       # red
}


def draw_skeleton(image, keypoints, confidence=None, conf_threshold=0.3):
    """
    Draw keypoint dots and skeleton lines on a copy of *image*.

    Parameters
    ----------
    image : np.ndarray  (H, W, 3)  BGR.
    keypoints : np.ndarray  (17, 2)  (x, y).
    confidence : np.ndarray or None  (17,).
    conf_threshold : float  — minimum confidence to draw a keypoint.

    Returns
    -------
    Annotated image copy.
    """
    img = image.copy()
    visible = set()

    for i in range(17):
        x, y = float(keypoints[i, 0]), float(keypoints[i, 1])
        if confidence is not None and float(confidence[i]) < conf_threshold:
            continue
        if x == 0 and y == 0:
            continue
        visible.add(i)
        cv2.circle(img, (int(x), int(y)), 4, KEYPOINT_COLOR, -1)

    for i, j in SKELETON:
        if i in visible and j in visible:
            p1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            p2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            cv2.line(img, p1, p2, SKELETON_COLOR, 2)

    return img


def draw_risk_info(image, risk_class, final_score, explanation):
    """
    Overlay risk badge (top-left) and explanation text (bottom).

    Parameters
    ----------
    image : np.ndarray  (H, W, 3)  BGR.
    risk_class : str  ``'Low Risk'`` | ``'Medium Risk'`` | ``'High Risk'``.
    final_score : int  1, 2, or 3 (the maximum partial score).
    explanation : str  Single human-readable explanation string.

    Returns
    -------
    Annotated image copy.
    """
    img = image.copy()
    color = RISK_COLORS.get(risk_class, (255, 255, 255))
    label = f"{risk_class}  (score: {final_score})"

    # Top-left badge
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (5, 5), (15 + tw, 10 + th + 10), color, -1)
    cv2.putText(img, label, (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Bottom explanation — word-wrap at ~65 chars
    h, w = img.shape[:2]
    words = explanation.split()
    lines = []
    current = ''
    for word in words:
        test = current + ' ' + word if current else word
        if len(test) <= 65:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    if lines:
        for i, line in enumerate(lines):
            if i >= 10:   # don't overflow the image
                break
            (tw2, th2), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            y = h - 20 * (len(lines) - i) - 5
            cv2.rectangle(img, (5, y - th2 - 4),
                          (10 + tw2, y + 4), TEXT_BG_COLOR, -1)
            cv2.putText(img, line, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)

    return img


def save_prediction(image, output_path):
    """Write an annotated BGR image to *output_path* (creates parent dirs)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
