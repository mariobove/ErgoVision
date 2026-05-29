"""
Experiment 2: Assembly101 robustness evaluation.

This is the core experimental protocol for the ErgoVision paper.  It:
  1. Discovers Assembly101 video files.
  2. Extracts frames at a configurable sampling rate with quality filtering.
  3. Runs YOLOv8-pose inference (no training, no fine-tuning).
  4. Computes RULA-inspired lightweight ergonomic risk scores.
  5. Aggregates robustness metrics and saves all outputs.

No ground-truth ergonomic labels are used — this is a feasibility and
robustness evaluation, not a benchmark.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from ..data.assembly101_loader import Assembly101Loader
from ..data.video_frame_extractor import VideoFrameExtractor, _laplacian_variance
from ..pose_estimator import PoseEstimator
from ..ergonomic_scoring import ErgonomicScorer
from ..visualization import draw_skeleton, draw_risk_info, save_prediction
from ..evaluation.robustness_metrics import RobustnessMetrics
from ..evaluation.experimental_reports import ExperimentalReports
from ..config import (
    ASSEMBLY101_OUTPUT_DIR,
    ASSEMBLY101_PREDICTIONS_DIR,
    ASSEMBLY101_METRICS_DIR,
    ASSEMBLY101_FIGURES_DIR,
    ASSEMBLY101_FAILURE_CASES_DIR,
    ASSEMBLY101_DEFAULT_SAMPLING_RATE,
    ASSEMBLY101_DEFAULT_MAX_VIDEOS,
    ASSEMBLY101_DEFAULT_MAX_FRAMES_PER_VIDEO,
    ASSEMBLY101_BLUR_THRESHOLD,
)

import csv


def run_experiment(
    video_folder,
    max_videos=None,
    sampling_rate=None,
    max_frames_per_video=None,
    blur_threshold=None,
    output_dir=None,
    verbose=True,
):
    """
    Run Experiment 2: Assembly101 robustness evaluation.

    Parameters
    ----------
    video_folder : str or Path
        Path to the directory containing Assembly101 video files.
    max_videos : int or None
        Maximum number of videos to process.  ``None`` uses the config default.
    sampling_rate : float or None
        Target frames per second for extraction.  ``None`` uses the config default.
    max_frames_per_video : int or None
        Maximum frames to keep per video.  ``None`` uses the config default.
    blur_threshold : float or None
        Laplacian variance threshold for blur filtering. ``None`` uses the config default.
    output_dir : str or Path or None
        Root output directory (defaults to ``outputs/assembly101/``).
    verbose : bool

    Returns
    -------
    dict  — aggregated robustness metrics.
    """
    # Resolve config
    max_videos = max_videos or ASSEMBLY101_DEFAULT_MAX_VIDEOS
    sampling_rate = sampling_rate or ASSEMBLY101_DEFAULT_SAMPLING_RATE
    max_frames_per_video = max_frames_per_video or ASSEMBLY101_DEFAULT_MAX_FRAMES_PER_VIDEO
    blur_threshold = blur_threshold if blur_threshold is not None else ASSEMBLY101_BLUR_THRESHOLD
    output_dir = Path(output_dir or ASSEMBLY101_OUTPUT_DIR)

    pred_dir = output_dir / 'predictions'
    metrics_dir = output_dir / 'metrics'
    figures_dir = output_dir / 'figures'
    failure_dir = output_dir / 'failure_cases'
    for d in [pred_dir, metrics_dir, figures_dir, failure_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 60)
        print("Experiment 2: Assembly101 Robustness Evaluation")
        print("=" * 60)
        print(f"  Video folder        : {video_folder}")
        print(f"  Max videos          : {max_videos}")
        print(f"  Sampling rate       : {sampling_rate} fps")
        print(f"  Max frames/video    : {max_frames_per_video}")
        print(f"  Blur threshold      : {blur_threshold}")
        print()

    # ---- Step 1: Discover videos ----
    if verbose:
        print("[1/6] Discovering videos...")
    loader = Assembly101Loader(
        video_folder=video_folder,
        max_videos=max_videos,
    )
    videos = loader.discover_videos()
    if verbose:
        print(f"  Found {len(videos)} videos")

    # ---- Step 2: Extract frames ----
    if verbose:
        print("[2/6] Extracting frames...")
    extractor = VideoFrameExtractor(
        output_dir=output_dir / 'extracted_frames',
        sampling_rate=sampling_rate,
        max_frames_per_video=max_frames_per_video,
        blur_threshold=blur_threshold,
    )

    all_frame_metadata = []  # list of (video_name, frame_path, timestamp)
    total_frames_extracted = 0
    extraction_stats = []

    for vid in videos:
        if not vid.is_valid:
            if verbose:
                print(f"  Skipping invalid video: {vid.video_name}")
            continue
        result = extractor.extract(vid.path, video_name=vid.video_name)
        extraction_stats.append(result)
        total_frames_extracted += result.frames_extracted
        for fp in result.extracted_frame_paths:
            all_frame_metadata.append((vid.video_name, fp))
        if verbose:
            print(f"  {vid.video_name}: {result.frames_extracted} frames "
                  f"({result.frames_skipped_blurry} blurry skipped)")

    if verbose:
        print(f"  Total extracted: {total_frames_extracted} frames")

    # ---- Step 3: Pose inference + scoring ----
    if verbose:
        print("[3/6] Running pose inference and ergonomic scoring...")

    pose_estimator = PoseEstimator()
    scorer = ErgonomicScorer()
    metrics_aggregator = RobustnessMetrics()

    all_predictions = []   # list of dicts for CSV
    video_summaries = []   # per-video aggregation

    # Group frames by video
    from collections import defaultdict
    frames_by_video = defaultdict(list)
    for vname, fpath in all_frame_metadata:
        frames_by_video[vname].append(fpath)

    for vname, frame_paths in frames_by_video.items():
        if verbose:
            print(f"  Processing video: {vname} ({len(frame_paths)} frames)")

        video_detections = 0
        video_no_person = 0
        video_inference_times = []
        video_risk = {'Low Risk': 0, 'Medium Risk': 0, 'High Risk': 0}

        for fpath in frame_paths:
            frame_id = f"{vname}/{fpath.stem}"

            # Read frame
            image = cv2.imread(str(fpath))
            if image is None:
                metrics_aggregator.update({
                    'frame_id': frame_id,
                    'has_person': False,
                    'inference_time': None,
                    'detections': [],
                })
                continue

            # Pose inference
            t_start = time.perf_counter()
            pose_result = pose_estimator.estimate(fpath)
            t_elapsed = time.perf_counter() - t_start

            has_person = len(pose_result['detections']) > 0
            person_scores = []

            for person in pose_result['detections']:
                score_result = scorer.score(person['keypoints'])
                person_scores.append(score_result)
                video_risk[score_result['final_risk_class']] += 1

            # Frame-level result for metrics
            frame_result = {
                'frame_id': frame_id,
                'has_person': has_person,
                'inference_time': t_elapsed,
                'detections': [
                    {
                        'final_risk_class': ps['final_risk_class'],
                        'final_score': ps['final_score'],
                        'partial_scores': ps['partial_scores'],
                        'unavailable_features': ps['unavailable_features'],
                        'explanation': ps['explanation'],
                    }
                    for ps in person_scores
                ],
            }
            metrics_aggregator.update(frame_result)

            # Track per-video stats
            if has_person:
                video_detections += 1
            else:
                video_no_person += 1
            video_inference_times.append(t_elapsed)

            # Save to predictions list (one row per detected person)
            if person_scores:
                for ps in person_scores:
                    all_predictions.append({
                        'frame_id': frame_id,
                        'video_name': vname,
                        'frame_path': str(fpath),
                        'final_score': ps['final_score'],
                        'final_risk_class': ps['final_risk_class'],
                        'torso_angle': ps['partial_scores'].get('torso_angle', {}).get('value', ''),
                        'torso_score': ps['partial_scores'].get('torso_angle', {}).get('score', ''),
                        'neck_angle': ps['partial_scores'].get('neck_angle', {}).get('value', ''),
                        'neck_score': ps['partial_scores'].get('neck_angle', {}).get('score', ''),
                        'knee_angle': ps['partial_scores'].get('knee_angle', {}).get('value', ''),
                        'knee_score': ps['partial_scores'].get('knee_angle', {}).get('score', ''),
                        'shoulder_asymmetry': ps['partial_scores'].get('shoulder_asymmetry', {}).get('value', ''),
                        'shoulder_score': ps['partial_scores'].get('shoulder_asymmetry', {}).get('score', ''),
                        'body_inclination': ps['partial_scores'].get('body_inclination', {}).get('value', ''),
                        'body_inclination_score': ps['partial_scores'].get('body_inclination', {}).get('score', ''),
                        'explanation': ps['explanation'],
                        'unavailable_features': ', '.join(ps['unavailable_features']),
                    })
            else:
                all_predictions.append({
                    'frame_id': frame_id,
                    'video_name': vname,
                    'frame_path': str(fpath),
                    'final_score': '',
                    'final_risk_class': 'No Person Detected',
                    'torso_angle': '',
                    'torso_score': '',
                    'neck_angle': '',
                    'neck_score': '',
                    'knee_angle': '',
                    'knee_score': '',
                    'shoulder_asymmetry': '',
                    'shoulder_score': '',
                    'body_inclination': '',
                    'body_inclination_score': '',
                    'explanation': 'No person detected in frame.',
                    'unavailable_features': '',
                })

        # Per-video summary
        avg_it = np.mean(video_inference_times) if video_inference_times else 0.0
        video_summaries.append({
            'video_name': vname,
            'frames_processed': len(frame_paths),
            'frames_with_detection': video_detections,
            'frames_no_person': video_no_person,
            'avg_inference_time': avg_it,
            'risk_low': video_risk['Low Risk'],
            'risk_medium': video_risk['Medium Risk'],
            'risk_high': video_risk['High Risk'],
        })

    # ---- Step 4: Aggregate metrics ----
    if verbose:
        print("[4/6] Computing robustness metrics...")

    metrics = metrics_aggregator.summarize()
    metrics['total_videos'] = len(videos)
    metrics['total_frames_extracted'] = total_frames_extracted

    # ---- Step 5: Generate reports ----
    if verbose:
        print("[5/6] Generating reports and figures...")

    # Predictions CSV
    if all_predictions:
        pred_fieldnames = [
            'frame_id', 'video_name', 'frame_path',
            'final_score', 'final_risk_class',
            'torso_angle', 'torso_score',
            'neck_angle', 'neck_score',
            'knee_angle', 'knee_score',
            'shoulder_asymmetry', 'shoulder_score',
            'body_inclination', 'body_inclination_score',
            'explanation', 'unavailable_features',
        ]
        pred_csv = pred_dir / 'assembly101_predictions.csv'
        with open(pred_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=pred_fieldnames)
            writer.writeheader()
            writer.writerows(all_predictions)
        if verbose:
            print(f"  Saved predictions: {pred_csv} ({len(all_predictions)} rows)")

    # Runtime report
    runtime_path = metrics_dir / 'assembly101_runtime_report.txt'
    ExperimentalReports.write_runtime_report(metrics, runtime_path)
    if verbose:
        print(f"  Saved runtime report: {runtime_path}")

    # Robustness per-video CSV
    if video_summaries:
        rob_csv = metrics_dir / 'assembly101_robustness_report.csv'
        ExperimentalReports.write_robustness_csv(video_summaries, rob_csv)
        if verbose:
            print(f"  Saved robustness CSV: {rob_csv} ({len(video_summaries)} videos)")

    # Risk distribution CSV
    risk_csv = metrics_dir / 'assembly101_risk_distribution.csv'
    ExperimentalReports.write_risk_distribution_csv(metrics, risk_csv)
    if verbose:
        print(f"  Saved risk distribution: {risk_csv}")

    # Feature availability CSV
    feat_csv = metrics_dir / 'assembly101_feature_availability.csv'
    ExperimentalReports.write_feature_availability_csv(metrics, feat_csv)
    if verbose:
        print(f"  Saved feature availability: {feat_csv}")

    # Figures
    risk_fig = figures_dir / 'assembly101_risk_distribution.png'
    ExperimentalReports.plot_risk_distribution(metrics, risk_fig)
    if verbose:
        print(f"  Saved risk figure: {risk_fig}")

    feat_fig = figures_dir / 'assembly101_feature_availability.png'
    ExperimentalReports.plot_feature_availability(metrics, feat_fig)
    if verbose:
        print(f"  Saved feature availability figure: {feat_fig}")

    # Paper summary
    paper_path = metrics_dir / 'paper_experimental_summary.txt'
    ExperimentalReports.write_paper_summary(
        metrics, paper_path,
        extra={
            'total_videos': len(videos),
            'total_frames_extracted': total_frames_extracted,
        },
    )
    if verbose:
        print(f"  Saved paper summary: {paper_path}")

    # ---- Step 6: Print final summary ----
    if verbose:
        print("\n[6/6] Experiment 2 complete.")
        print(f"  Videos processed        : {len(videos)}")
        print(f"  Frames extracted        : {total_frames_extracted}")
        print(f"  Frames analysed         : {metrics['total_frames']}")
        print(f"  Pose detection rate     : {metrics['pose_detection_rate']:.2%}")
        print(f"  No-person rate          : {metrics['no_person_detection_rate']:.2%}")
        print(f"  Missing keypoint rate   : {metrics['missing_keypoint_rate']:.2%}")
        print(f"  Avg inference time      : {metrics['avg_inference_time_seconds']:.4f}s")
        print(f"  FPS                     : {metrics['fps']:.2f}")
        print(f"  Total failure cases     : {metrics['total_failure_cases']}")
        print(f"\n  All outputs → {output_dir.resolve()}")

    return metrics
