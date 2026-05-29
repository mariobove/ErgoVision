"""
Experiment 1: Baseline posture dataset evaluation.

Runs the standard ErgoVision pipeline on the Kaggle posture-keypoints-detection
image dataset.  Produces a predictions CSV, risk distribution figure, and a
set of sample visualisations.

This experiment establishes a baseline for posture-based risk scoring on a
controlled image dataset before moving to the more challenging Assembly101
video data.
"""

from pathlib import Path

from ..pipeline import ErgoPipeline
from ..config import OUTPUT_DIR


def run_experiment(dataset_path, subset_size=50, output_dir=None, verbose=True):
    """
    Run Experiment 1.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the root of the Kaggle posture-keypoints-detection dataset.
    subset_size : int
        Number of images to process (default 50).
    output_dir : str or Path or None
        Root output directory (defaults to ``outputs/``).
    verbose : bool

    Returns
    -------
    list[dict]  — pipeline results (one entry per image).
    """
    output_dir = Path(output_dir or OUTPUT_DIR)

    if verbose:
        print("\n" + "=" * 60)
        print("Experiment 1: Baseline Posture Dataset")
        print("=" * 60)

    pipeline = ErgoPipeline()
    results = pipeline.run(
        dataset_path=dataset_path,
        subset_size=subset_size,
        save_visualizations=True,
        verbose=verbose,
    )

    if verbose:
        print("\nExperiment 1 complete.")
        print(f"  Results saved to {output_dir.resolve()}")

    return results
