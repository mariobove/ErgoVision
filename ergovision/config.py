"""
Configuration constants and runtime experiment settings for ErgoVision.

Static thresholds and geometry data live at module level.
Per-run settings are bundled in ExperimentConfig so multiple experiment
configurations can coexist without touching the source.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Static constants
# ---------------------------------------------------------------------------

# Supported file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

# YOLOv8-pose model (used in legacy one-stage mode)
POSE_MODEL_NAME = 'yolov8n-pose.pt'

# Two-stage pipeline default models
HUMAN_DETECTOR_MODEL = 'yolov8m.pt'
CROP_POSE_MODEL = 'yolov8m-pose.pt'

# COCO dataset class IDs
COCO_PERSON_CLASS_ID = 0

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

# ===================================================================
# Angle severity thresholds for conservative rule-based scoring
# ===================================================================
#
# Thresholds define the boundaries between risk levels for each feature.
# Values are in degrees unless otherwise noted.
#
# Calibrated on RULA / REBA / OWAS cut-points from the literature:
#   McAtamney & Corlett (1993) — RULA
#   Hignett & McAtamney (2000) — REBA
#   Karhu et al. (1977) — OWAS
#
# Each feature has three thresholds:
#   low_max      — maximum angle considered neutral / low risk
#   medium_max   — maximum angle considered moderate (monitoring zone)
#   extreme_min  — minimum angle for "extreme" classification
#                  (used for continuous severity scaling)
# Between low_max and medium_max the risk is MEDIUM.
# Above medium_max the risk is HIGH.

TRUNK_ANGLE_THRESHOLDS = {
    'low_max': 20,
    'medium_max': 60,
    'extreme_min': 80,
}

NECK_ANGLE_THRESHOLDS = {
    'low_max': 15,
    'medium_max': 45,
    'extreme_min': 60,
}

ARM_ANGLE_THRESHOLDS = {
    'low_max': 20,
    'medium_max': 60,
    'extreme_min': 90,
}

FOREARM_DEVIATION_THRESHOLDS = {
    'low_max': 30,
    'medium_max': 90,
    'extreme_min': 120,
}

KNEE_BEND_THRESHOLDS = {
    'low_max': 15,
    'medium_max': 45,
    'extreme_min': 60,
}

SHOULDER_ASYMMETRY_THRESHOLDS = {
    'low_max': 10,
    'medium_max': 25,
    'extreme_min': 40,
}

BODY_INCLINATION_THRESHOLDS = {
    'low_max': 10,
    'medium_max': 25,
    'extreme_min': 40,
}


# Legacy thresholds (kept for backward-compatible partial scores)
TORSO_THRESHOLDS = {
    'low': (0, 20), 'medium': (20, 45), 'high': (45, 180),
}
NECK_THRESHOLDS = {
    'low': (0, 10), 'medium': (10, 35), 'high': (35, 180),
}
KNEE_THRESHOLDS = {
    'low': (0, 20), 'medium': (20, 60), 'high': (60, 180),
}
SHOULDER_ASYMMETRY_THRESHOLDS = {
    'low': (0, 10), 'medium': (10, 25), 'high': (25, float('inf')),
}
INCLINATION_THRESHOLDS = {
    'low': (0, 10), 'medium': (10, 25), 'high': (25, float('inf')),
}
LEGACY_UPPER_ARM_THRESHOLDS = {
    'low': (0, 20), 'medium': (20, 60), 'high': (60, 180),
}
FOREARM_THRESHOLDS = {
    'low': (0, 30), 'medium': (30, 90), 'high': (90, 180),
}

# Risk class labels (score 1 -> Low Risk, 2 -> Medium Risk, 3 -> High Risk)
# Risk class labels (RULA-inspired Action Levels)
#   score 1 = AL1 (Low Risk)    — acceptable, no action needed
#   score 2 = AL2 (Medium Risk)  — further investigation recommended
#   score 3 = AL3+ (High Risk)   — investigation and changes required
RISK_CLASSES = ['AL1 (Low Risk)', 'AL2 (Medium Risk)', 'AL3+ (High Risk)']
RISK_LEVEL_SHORT = {1: 'AL1 (LOW)', 2: 'AL2 (MEDIUM)', 3: 'AL3+ (HIGH)'}

# Central colour mapping (single source of truth for overlays)
RISK_COLOUR_MAP = {
    1: (0, 255, 0),     # GREEN
    2: (0, 255, 255),   # YELLOW
    3: (0, 0, 255),     # RED
}

# Feature names used in score output
ALL_ANGLE_FEATURES = [
    'trunk_angle',
    'neck_angle',
    'upper_arm_angle_left',
    'upper_arm_angle_right',
    'forearm_angle_left',
    'forearm_angle_right',
    'knee_angle_left',
    'knee_angle_right',
    'shoulder_asymmetry',
    'body_inclination',
]

# ---------------------------------------------------------------------------
# Legacy static output paths (kept for backward compatibility)
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path('outputs')
VISUALIZATION_DIR = OUTPUT_DIR / 'visualizations'
CSV_OUTPUT = OUTPUT_DIR / 'ergonomic_assessment.csv'
JSON_OUTPUT = OUTPUT_DIR / 'ergonomic_assessment.json'


# ---------------------------------------------------------------------------
# Runtime experiment configuration
# ---------------------------------------------------------------------------

class ExperimentConfig:
    """Mutable, per-run configuration for an experimental pipeline execution.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset folder under ``data/``.  Used to derive default
        input and output paths.
    """

    def __init__(self, dataset_name="default"):
        self.dataset_name = dataset_name

        # Derived paths (overridable after construction)
        self.input_path = Path('data') / dataset_name
        self.output_path = Path('outputs') / dataset_name

        # Frame extraction
        self.frame_sampling_fps = 1.0
        self.max_frames_per_video = 200

        # --- Two-stage pipeline ---
        self.use_two_stage = True
        self.human_detector_model = 'yolov8l.pt'
        self.crop_pose_model = 'yolov8l-pose.pt'

        # Detection thresholds (high confidence to avoid false positives)
        self.detection_confidence = 0.5
        self.min_person_confidence = 0.5
        self.min_bbox_area = 1000
        self.min_bbox_height = 50
        self.min_bbox_aspect = 0.15
        self.max_bbox_aspect = 3.0

        # Crop-pose parameters
        self.bbox_padding = 0.15

        # Keypoint quality
        self.keypoint_confidence = 0.3
        self.min_valid_keypoints = 8

        # Temporal smoothing (asymmetric EMA)
        self.temporal_smoothing = True
        self.ema_alpha = 0.25       # slower rise
        self.ema_decay_alpha = 0.50 # faster decay when raw drops
        self.smooth_window = 5  # frames

        # Postural risk thresholds (0-100 severity → class)
        self.severity_low_max = 35
        self.severity_medium_max = 65

        # Manual context for semi-automatic mode (optional, user-provided)
        self.manual_context = {
            'load_force': None,
            'muscle_use': None,
            'repetition': None,
            'static_posture': None,
        }

        # Debug
        self.debug_detection = False

        # Output flags
        self.save_annotated_frames = True
        self.save_annotated_video = True
        self.save_failure_cases = True
        self.save_csv = True
        self.save_plots = True

    # -- output sub-directories (computed each call) ------------------------

    @property
    def output_frames_dir(self):
        return self.output_path / 'frames'

    @property
    def output_annotated_frames_dir(self):
        return self.output_path / 'annotated_frames'

    @property
    def output_annotated_videos_dir(self):
        return self.output_path / 'annotated_videos'

    @property
    def output_csv_dir(self):
        return self.output_path / 'csv'

    @property
    def output_plots_dir(self):
        return self.output_path / 'plots'

    @property
    def output_examples_dir(self):
        return self.output_path / 'examples'

    @property
    def output_reports_dir(self):
        return self.output_path / 'reports'

    def mkdirs(self):
        """Create all output sub-directories."""
        for d in [self.output_frames_dir, self.output_annotated_frames_dir,
                  self.output_annotated_videos_dir, self.output_csv_dir,
                  self.output_plots_dir, self.output_examples_dir,
                  self.output_reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # -- CSV paths ----------------------------------------------------------

    @property
    def csv_frame_person_path(self):
        return self.output_csv_dir / 'frame_person_results.csv'

    @property
    def csv_video_summary_path(self):
        return self.output_csv_dir / 'video_summary.csv'

    def __repr__(self):
        return (
            f"ExperimentConfig(dataset_name='{self.dataset_name}', "
            f"fps={self.frame_sampling_fps}, "
            f"max_frames={self.max_frames_per_video})"
        )
