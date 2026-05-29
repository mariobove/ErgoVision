"""
ErgoVision — Vision-based ergonomic risk assessment.

Lightweight, RULA-inspired, training-free pipeline for industrial posture
risk analysis using YOLO detection + pose estimation.
"""

from .pose_estimator import PoseEstimator, CropPoseEstimator
from .ergonomic_scoring import ErgonomicScorer
from .pipeline import ErgoPipeline
from .detection import HumanDetector
from .dataset import find_images, select_subset, inspect_dataset, DatasetInfo
from .config import ExperimentConfig

__version__ = "0.4.0"
