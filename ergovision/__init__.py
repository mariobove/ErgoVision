"""
ErgoVision — Vision-based ergonomic risk assessment.
"""

from .pose_estimator import PoseEstimator
from .ergonomic_scoring import ErgonomicScorer
from .pipeline import ErgoPipeline
from .dataset import find_images, select_subset

__version__ = "0.1.0"
