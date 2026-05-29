"""
Experiment 3: Qualitative explainability examples.

Selects representative frames from the Assembly101 robustness experiment
and saves annotated images for each risk class and common failure modes.

Outputs are saved to ``outputs/assembly101/figures/`` and
``outputs/assembly101/failure_cases/``.
"""

from pathlib import Path

import cv2
import numpy as np

from ..pose_estimator import PoseEstimator
from ..ergonomic_scoring import ErgonomicScorer
from ..visualization import draw_skeleton, draw_risk_info, save_prediction
from ..config import (
    ASSEMBLY101_OUTPUT_DIR,
    ASSEMBLY101_FRAMES_DIR,
    ASSEMBLY101_PREDICTIONS_DIR,
    ASSEMBLY101_METRICS_DIR,
    ASSEMBLY101_FIGURES_DIR,
    ASSEMBLY101_FAILURE_CASES_DIR,
)


def _find_frames_by_risk(predictions_csv, frames_dir, risk_class, max_count=3):
    """
    Search the predictions CSV for frames matching a risk class.

    Returns a list of (frame_path, explanation, score) tuples.
    """
    import csv
    results = []
    pred_path = Path(predictions_csv)
    if not pred_path.exists():
        return results

    with open(pred_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('final_risk_class', '') == risk_class:
                frame_path = Path(row.get('frame_path', ''))
                if not frame_path.exists():
                    # Try to reconstruct from frames_dir + video_name + frame_index
                    alt = frames_dir / row.get('video_name', '') / f"{Path(row.get('frame_id', '')).stem}.jpg"
                    if alt.exists():
                        frame_path = alt
                    else:
                        continue
                results.append((
                    frame_path,
                    row.get('explanation', ''),
                    row.get('final_score', ''),
                ))
                if len(results) >= max_count:
                    break
    return results


def _find_failure_case(predictions_csv, frames_dir, failure_type, max_count=1):
    """
    Search for frames matching a failure type.

    failure_type can be ``'no_person_detected'`` or frames with many
    unavailable features (``missing_keypoints``).
    """
    import csv
    results = []
    pred_path = Path(predictions_csv)
    if not pred_path.exists():
        return results

    with open(pred_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if failure_type == 'no_person_detected':
                if row.get('final_risk_class', '') == 'No Person Detected':
                    frame_path = Path(row.get('frame_path', ''))
                    if not frame_path.exists():
                        alt = frames_dir / row.get('video_name', '') / f"{Path(row.get('frame_id', '')).stem}.jpg"
                        if alt.exists():
                            frame_path = alt
                        else:
                            continue
                    results.append((frame_path, 'No person detected in frame', ''))
                    if len(results) >= max_count:
                        break
            elif failure_type == 'missing_keypoints':
                uf = row.get('unavailable_features', '')
                if uf and len(uf.split(',')) >= 3:
                    frame_path = Path(row.get('frame_path', ''))
                    if not frame_path.exists():
                        alt = frames_dir / row.get('video_name', '') / f"{Path(row.get('frame_id', '')).stem}.jpg"
                        if alt.exists():
                            frame_path = alt
                        else:
                            continue
                    results.append((frame_path, f'Missing keypoints: {uf}', ''))
                    if len(results) >= max_count:
                        break
    return results


def run_experiment(
    predictions_csv=None,
    frames_dir=None,
    figures_dir=None,
    failure_cases_dir=None,
    verbose=True,
):
    """
    Run Experiment 3: Qualitative explainability.

    Parameters
    ----------
    predictions_csv : str or Path or None
        Path to the predictions CSV from Experiment 2.
    frames_dir : str or Path or None
        Path to the extracted frames directory.
    figures_dir : str or Path or None
        Output directory for sample risk figures.
    failure_cases_dir : str or Path or None
        Output directory for failure case images.
    verbose : bool
    """
    predictions_csv = Path(predictions_csv or ASSEMBLY101_PREDICTIONS_DIR / 'assembly101_predictions.csv')
    frames_dir = Path(frames_dir or ASSEMBLY101_FRAMES_DIR)
    figures_dir = Path(figures_dir or ASSEMBLY101_FIGURES_DIR)
    failure_cases_dir = Path(failure_cases_dir or ASSEMBLY101_FAILURE_CASES_DIR)

    figures_dir.mkdir(parents=True, exist_ok=True)
    failure_cases_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 60)
        print("Experiment 3: Qualitative Explainability Examples")
        print("=" * 60)

    # Check that the predictions CSV exists
    if not predictions_csv.exists():
        print(f"  WARNING: Predictions CSV not found at {predictions_csv}")
        print(f"  Run Experiment 2 first to generate predictions.")
        return

    # Load pose estimator and scorer for re-inference on selected frames
    pose_estimator = PoseEstimator()
    scorer = ErgonomicScorer()

    def _annotate_and_save(frame_path, explanation, score, output_path, label):
        """Run pose + scoring on a frame and save annotated image."""
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"  WARNING: Could not read {frame_path}, skipping.")
            return False

        # Run inference and scoring
        pose_result = pose_estimator.estimate(frame_path)
        if pose_result['detections']:
            person = pose_result['detections'][0]
            score_result = scorer.score(person['keypoints'])
            used_explanation = score_result['explanation']
            used_score = score_result['final_score']
            used_class = score_result['final_risk_class']
            # Draw skeleton
            image = draw_skeleton(image, person['keypoints'], person['confidence'])
        else:
            used_explanation = explanation or 'No person detected'
            used_score = score or 'N/A'
            used_class = label

        # Draw risk info
        final_score = used_score if isinstance(used_score, int) else 1
        image = draw_risk_info(image, used_class, final_score, used_explanation)

        # Add a title label
        cv2.putText(image, label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        save_prediction(image, output_path)
        return True

    # ---- Select and annotate samples ----
    n_annotated = 0

    # Low risk sample
    low_candidates = _find_frames_by_risk(predictions_csv, frames_dir, 'Low Risk', max_count=3)
    if low_candidates:
        fp, expl, score = low_candidates[0]
        out = figures_dir / 'sample_low_risk.png'
        if _annotate_and_save(fp, expl, score, out, 'Low Risk'):
            n_annotated += 1
            if verbose:
                print(f"  Saved low risk sample: {out}")

    # Medium risk sample
    med_candidates = _find_frames_by_risk(predictions_csv, frames_dir, 'Medium Risk', max_count=3)
    if med_candidates:
        fp, expl, score = med_candidates[0]
        out = figures_dir / 'sample_medium_risk.png'
        if _annotate_and_save(fp, expl, score, out, 'Medium Risk'):
            n_annotated += 1
            if verbose:
                print(f"  Saved medium risk sample: {out}")

    # High risk sample
    high_candidates = _find_frames_by_risk(predictions_csv, frames_dir, 'High Risk', max_count=3)
    if high_candidates:
        fp, expl, score = high_candidates[0]
        out = figures_dir / 'sample_high_risk.png'
        if _annotate_and_save(fp, expl, score, out, 'High Risk'):
            n_annotated += 1
            if verbose:
                print(f"  Saved high risk sample: {out}")

    # Occlusion failure case (look for no-person-detected frames)
    occ_candidates = _find_failure_case(predictions_csv, frames_dir, 'no_person_detected', max_count=3)
    if occ_candidates:
        fp, expl, score = occ_candidates[0]
        out = failure_cases_dir / 'occlusion_failure_case.png'
        # For no-person frames, just read and annotate without inference
        image = cv2.imread(str(fp))
        if image is not None:
            cv2.putText(image, 'Occlusion / No Person Detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            save_prediction(image, out)
            n_annotated += 1
            if verbose:
                print(f"  Saved occlusion failure case: {out}")

    # Missing keypoints failure case
    mk_candidates = _find_failure_case(predictions_csv, frames_dir, 'missing_keypoints', max_count=3)
    if mk_candidates:
        fp, expl, score = mk_candidates[0]
        out = failure_cases_dir / 'missing_keypoints_failure_case.png'
        image = cv2.imread(str(fp))
        if image is not None:
            pose_result = pose_estimator.estimate(fp)
            if pose_result['detections']:
                person = pose_result['detections'][0]
                score_result = scorer.score(person['keypoints'])
                image = draw_skeleton(image, person['keypoints'], person['confidence'])
                image = draw_risk_info(
                    image,
                    score_result['final_risk_class'],
                    score_result['final_score'],
                    score_result['explanation'],
                )
            cv2.putText(image, 'Missing Keypoints Failure', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            save_prediction(image, out)
            n_annotated += 1
            if verbose:
                print(f"  Saved missing keypoints failure case: {out}")

    if verbose:
        print(f"\nExperiment 3 complete. {n_annotated} sample images saved.")
        if n_annotated < 5:
            print("  (Fewer samples than expected — the predictions CSV may contain")
            print("   limited diversity. Run Experiment 2 with more videos.)")
