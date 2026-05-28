"""
ErgoVision — Vision-based ergonomic risk assessment.

A research prototype for RULA-inspired lightweight ergonomic risk estimation
using pretrained pose estimation (YOLOv8-pose).
"""

from .pose_estimator import PoseEstimator
from .ergonomic_scoring import ErgonomicScorer
from .pipeline import ErgoPipeline
from .dataset import find_images, select_subset
from .data.assembly101_loader import Assembly101Loader
from .data.video_frame_extractor import VideoFrameExtractor
from .evaluation.robustness_metrics import RobustnessMetrics
from .evaluation.experimental_reports import ExperimentalReports

__version__ = "0.1.0"
