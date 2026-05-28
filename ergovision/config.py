"""
Configuration constants for ErgoVision.
"""

from pathlib import Path

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# YOLOv8-pose model
POSE_MODEL_NAME = 'yolov8n-pose.pt'

# COCO keypoint indices used by YOLOv8-pose
KEYPOINT_INDICES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

# Skeleton connections (index pairs)
SKELETON = [
    (5, 6),   # shoulders
    (5, 7),   # left upper arm
    (7, 9),   # left forearm
    (6, 8),   # right upper arm
    (8, 10),  # right forearm
    (5, 11),  # left torso
    (6, 12),  # right torso
    (11, 12), # hips
    (11, 13), # left thigh
    (13, 15), # left shin
    (12, 14), # right thigh
    (14, 16), # right shin
]

# Ergonomic scoring thresholds (degrees)
TORSO_THRESHOLDS = {
    'low': (0, 20),
    'medium': (20, 60),
    'high': (60, 180),
}

NECK_THRESHOLDS = {
    'low': (0, 10),
    'medium': (10, 30),
    'high': (30, 180),
}

KNEE_THRESHOLDS = {
    'low': (0, 30),
    'medium': (30, 60),
    'high': (60, 180),
}

SHOULDER_ASYMMETRY_THRESHOLDS = {
    'low': (0, 10),
    'medium': (10, 30),
    'high': (30, float('inf')),
}

INCLINATION_THRESHOLDS = {
    'low': (0, 10),
    'medium': (10, 30),
    'high': (30, float('inf')),
}

# Risk class labels (index 0 -> score 1 = Low Risk, index 1 -> score 2 = Medium Risk,
# index 2 -> score 3 = High Risk)
RISK_CLASSES = ['Low Risk', 'Medium Risk', 'High Risk']

# Output directories
OUTPUT_DIR = Path('outputs')
VISUALIZATION_DIR = OUTPUT_DIR / 'visualizations'
CSV_OUTPUT = OUTPUT_DIR / 'ergonomic_assessment.csv'
JSON_OUTPUT = OUTPUT_DIR / 'ergonomic_assessment.json'

# ---------------------------------------------------------------------------
# Assembly101 configuration
# ---------------------------------------------------------------------------

# Supported video file extensions
ASSEMBLY101_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

# Assembly101 output subdirectories
ASSEMBLY101_OUTPUT_DIR = OUTPUT_DIR / 'assembly101'
ASSEMBLY101_FRAMES_DIR = ASSEMBLY101_OUTPUT_DIR / 'extracted_frames'
ASSEMBLY101_PREDICTIONS_DIR = ASSEMBLY101_OUTPUT_DIR / 'predictions'
ASSEMBLY101_FIGURES_DIR = ASSEMBLY101_OUTPUT_DIR / 'figures'
ASSEMBLY101_METRICS_DIR = ASSEMBLY101_OUTPUT_DIR / 'metrics'
ASSEMBLY101_FAILURE_CASES_DIR = ASSEMBLY101_OUTPUT_DIR / 'failure_cases'

# Frame extraction defaults
ASSEMBLY101_DEFAULT_SAMPLING_RATE = 1.0       # frames per second
ASSEMBLY101_DEFAULT_MAX_VIDEOS = 5             # max videos to process
ASSEMBLY101_DEFAULT_MAX_FRAMES_PER_VIDEO = 100 # max frames per video

# Frame quality filtering
ASSEMBLY101_BLUR_THRESHOLD = 100.0             # Laplacian variance threshold (lower = blurrier)
