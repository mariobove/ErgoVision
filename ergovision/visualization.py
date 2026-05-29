"""
Visualisation utilities for ErgoVision.

Draws keypoints, skeleton connections, bounding boxes, and ergonomic risk
overlays on images.  Supports annotated frame saving, risk-level example
extraction, and video compilation.
"""

from pathlib import Path

import cv2
import numpy as np

from .config import SKELETON, RISK_CLASSES

# BGR colours
KEYPOINT_COLOR = (0, 255, 0)          # green
SKELETON_COLOR = (255, 255, 0)        # cyan
TEXT_COLOR = (255, 255, 255)          # white
TEXT_BG_COLOR = (0, 0, 0)            # black
BBOX_COLOR = (0, 255, 255)            # yellow
RISK_COLORS = {
    'Low Risk': (0, 255, 0),          # green
    'Medium Risk': (0, 255, 255),     # yellow
    'High Risk': (0, 0, 255),         # red
}
RISK_COLORS_SHORT = {
    'LOW': (0, 255, 0),
    'MEDIUM': (0, 255, 255),
    'HIGH': (0, 0, 255),
}


def draw_skeleton(image, keypoints, confidence=None, conf_threshold=0.3):
    """
    Draw keypoint dots and skeleton lines on a copy of *image*.
    """
    img = image.copy()
    visible = set()

    for i in range(min(17, keypoints.shape[0])):
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


def draw_bbox(image, bbox):
    """Draw a bounding box on the image.  *bbox* is [x1, y1, x2, y2]."""
    img = image.copy()
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)
    return img


def draw_risk_info(image, risk_class, risk_score, partial_scores, explanation):
    """
    Overlay risk badge (top-left) and explanation lines (bottom).
    """
    img = image.copy()
    color = RISK_COLORS.get(risk_class, (255, 255, 255))
    label = f"{risk_class}  (score: {risk_score})"

    # Top-left badge
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (5, 5), (15 + tw, 10 + th + 10), color, -1)
    cv2.putText(img, label, (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Bottom: explanation + per-feature summary
    lines = [explanation]
    for name, ps in partial_scores.items():
        val = ps.get('value')
        score = ps.get('score')
        if val is not None and score is not None:
            lines.append(f"{name}: {val} (score {score})")
        else:
            lines.append(f"{name}: N/A")

    h, w = img.shape[:2]
    for i, text in enumerate(lines):
        text = text if len(text) <= 65 else text[:62] + "..."
        (tw2, th2), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.45, 1)
        y = h - 20 * (len(lines) - i) - 5
        cv2.rectangle(img, (5, y - th2 - 4),
                      (10 + tw2, y + 4), TEXT_BG_COLOR, -1)
        cv2.putText(img, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)

    return img


def annotate_frame(image, bbox, keypoints, confidence, risk_class, risk_score,
                   partial_scores, explanation, person_id=0,
                   keypoint_conf_thresh=0.3, detection_confidence=None):
    """Full annotation: bbox + skeleton + risk overlay.

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    bbox : list[float] or None
        [x1, y1, x2, y2].
    keypoints : np.ndarray
        (17, 2) or (17, 3) keypoints.
    confidence : np.ndarray
        (17,) keypoint confidence.
    risk_class : str
        "Low Risk", "Medium Risk", or "High Risk".
    risk_score : int
        1, 2, or 3.
    partial_scores : dict
        From ErgonomicScorer.score().
    explanation : str
        Text explanation.
    person_id : int
        Person index in the frame (shown on the bbox).
    keypoint_conf_thresh : float
        Minimum keypoint confidence to draw.
    detection_confidence : float or None
        Detection model confidence for this person (shown on bbox).

    Returns
    -------
    np.ndarray
    """
    img = image.copy()

    # Bounding box with detection confidence
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = map(int, bbox[:4])
        color = RISK_COLORS.get(risk_class, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Person ID label above bbox
        if detection_confidence is not None:
            pid_label = f"P{person_id} ({detection_confidence:.2f})"
        else:
            pid_label = f"P{person_id}"
        (pw, ph), _ = cv2.getTextSize(pid_label, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.5, 1)
        cv2.rectangle(img, (x1, y1 - ph - 6), (x1 + pw + 6, y1),
                      color, -1)
        cv2.putText(img, pid_label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Skeleton
    img = draw_skeleton(img, keypoints, confidence, keypoint_conf_thresh)

    # Risk overlay (compact)
    color = RISK_COLORS.get(risk_class, (255, 255, 255))
    short_label = f"P{person_id}: {risk_class} ({risk_score})"
    (tw, th), _ = cv2.getTextSize(short_label, cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, 1)
    x_pos = 10 + person_id * 220
    cv2.rectangle(img, (x_pos, 5), (x_pos + tw + 10, 10 + th + 4),
                  color, -1)
    cv2.putText(img, short_label, (x_pos + 5, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


# ---------------------------------------------------------------------------
# Risk example extraction
# ---------------------------------------------------------------------------

def extract_risk_examples(all_results, images_dir, output_dir, max_per_level=10):
    """Copy annotated frames to risk-level subdirectories for visual inspection.

    Parameters
    ----------
    all_results : list[dict]
        Frame-person results rows (with 'frame_path', 'risk_level', etc.).
    images_dir : Path
        Directory containing the source annotated frames.
    output_dir : Path
        Root output directory for examples.
    max_per_level : int
        Maximum examples per risk level.
    """
    output_dir = Path(output_dir)
    images_dir = Path(images_dir)

    for level in ['LOW', 'MEDIUM', 'HIGH']:
        level_dir = output_dir / level.lower()
        level_dir.mkdir(parents=True, exist_ok=True)

    # Group by risk level and collect unique frame paths
    frames_by_level = {level: [] for level in ['LOW', 'MEDIUM', 'HIGH']}
    seen_frames = set()

    for r in all_results:
        level = r.get('risk_level', '')
        frame_path = r.get('annotated_frame_path', '')
        if level in frames_by_level and frame_path and frame_path not in seen_frames:
            frames_by_level[level].append(frame_path)
            seen_frames.add(frame_path)

    for level, frame_paths in frames_by_level.items():
        level_dir = output_dir / level.lower()
        for fp in frame_paths[:max_per_level]:
            src = Path(fp)
            if src.exists():
                dst = level_dir / src.name
                import shutil
                shutil.copy2(str(src), str(dst))


def save_failure_cases(discarded_rows, annotated_dir, output_dir,
                       max_examples=20):
    """Copy annotated frames of discarded postures to a failure directory.

    Parameters
    ----------
    discarded_rows : list[dict]
        Frame-person results rows where ``discarded == True``.
    annotated_dir : Path
        Directory with all annotated frames.
    output_dir : Path
        Root examples directory (``examples/failure/`` will be created).
    max_examples : int
        Maximum number of failure examples to save.
    """
    output_dir = Path(output_dir) / 'failure'
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    seen = set()
    for r in discarded_rows:
        if copied >= max_examples:
            break
        fp = r.get('annotated_frame_path', '')
        if fp and fp not in seen:
            src = Path(fp)
            if src.exists():
                import shutil
                dst = output_dir / src.name
                shutil.copy2(str(src), str(dst))
                copied += 1
                seen.add(fp)


# ---------------------------------------------------------------------------
# Video annotation
# ---------------------------------------------------------------------------

def create_annotated_video(frames_dir, annotated_frames_dir, output_path, fps=10):
    """Compile annotated frames into a video.

    Parameters
    ----------
    frames_dir : Path
        Directory with original extracted frames (to get total count).
    annotated_frames_dir : Path
        Directory with corresponding annotated PNG/JPG frames.
    output_path : Path
        Output video path (e.g. ``.mp4``).
    fps : int
        Output video frame rate.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect annotated frame files, sorted
    annotated_files = sorted(annotated_frames_dir.glob('*'))
    if not annotated_files:
        return

    # Read first frame to get dimensions
    first = cv2.imread(str(annotated_files[0]))
    if first is None:
        return
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for fp in annotated_files:
        frame = cv2.imread(str(fp))
        if frame is not None:
            writer.write(frame)

    writer.release()


def save_prediction(image, output_path):
    """Write an annotated BGR image to *output_path* (creates parent dirs)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
