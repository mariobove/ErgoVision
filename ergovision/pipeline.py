"""
ErgoVision experimental pipeline — orchestrates dataset discovery, frame
extraction, pose estimation, ergonomic scoring, and output generation.

Two execution modes:

* **Experimental mode** (``dataset_name=``): full dataset-agnostic pipeline
  with video support, per-dataset output structure, plots, examples, and
  markdown report.
* **Legacy mode** (``dataset_path=``): backward-compatible single-pass
  image processing (original behaviour).
"""

import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .dataset import find_images, select_subset, inspect_dataset, find_media
from .detection import HumanDetector
from .pose_estimator import PoseEstimator, CropPoseEstimator
from .ergonomic_scoring import ErgonomicScorer, temporal_smooth_severity
from .visualization import (
    annotate_frame,
    save_prediction,
    extract_risk_examples,
    save_failure_cases,
    create_annotated_video,
)
from .config import (
    ExperimentConfig,
    OUTPUT_DIR,
    VISUALIZATION_DIR,
    CSV_OUTPUT,
    JSON_OUTPUT,
    RISK_CLASSES,
    RISK_LEVEL_SHORT,
    ALL_ANGLE_FEATURES,
)


# ===================================================================
# Result data structures
# ===================================================================

class PipelineResult:
    """Holds all outputs from an experimental pipeline run."""

    def __init__(self):
        self.frame_person_rows = []
        self.video_summary_rows = []
        self.stats = {}
        self.config = None
        self.total_time_sec = 0.0

    @property
    def total_postures(self):
        return len(self.frame_person_rows)

    @property
    def valid_postures(self):
        return sum(
            1 for r in self.frame_person_rows
            if not r.get('discarded', False)
        )


# ===================================================================
# Pipeline
# ===================================================================

class ErgoPipeline:
    """End-to-end ErgoVision pipeline.

    Parameters
    ----------
    model_name : str or None
        Path to a YOLOv8-pose model.  Defaults to ``yolov8n-pose.pt``.
    config : ExperimentConfig or None
        Runtime configuration.  A default one is created if omitted.
    """

    def __init__(self, model_name=None, config=None):
        cfg = config or ExperimentConfig()
        self.config = cfg

        # Legacy one-stage pose estimator (yolov8n-pose or custom)
        self.pose_estimator = PoseEstimator(
            model_name=model_name,
            confidence_threshold=cfg.detection_confidence,
        )

        # Two-stage pipeline components
        self.detector = HumanDetector(
            model_name=cfg.human_detector_model,
            confidence=cfg.min_person_confidence,
        )
        self.crop_pose = CropPoseEstimator(
            model_name=cfg.crop_pose_model,
        )

        self.scorer = ErgonomicScorer()
        self.results = []
        self._timing = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, dataset_path=None, subset_size=50,
            save_visualizations=True, verbose=True,
            dataset_name=None):
        """Execute the pipeline.

        Parameters
        ----------
        dataset_path : str or Path, optional
            Path to image folder (legacy mode).
        subset_size : int
            Maximum images to process (legacy mode).
        save_visualizations : bool
            Whether to save annotated images (legacy mode).
        verbose : bool
            Print progress.
        dataset_name : str, optional
            Dataset folder name under ``data/`` (experimental mode).

        Returns
        -------
        PipelineResult or list[dict]
        """
        if dataset_name is not None:
            return self._run_experimental(dataset_name, verbose=verbose)
        else:
            return self._run_legacy(
                dataset_path, subset_size,
                save_visualizations, verbose,
            )

    # ------------------------------------------------------------------
    # Legacy mode
    # ------------------------------------------------------------------

    def _run_legacy(self, dataset_path, subset_size,
                    save_visualizations, verbose):
        """Original single-pass image pipeline (backward compatible)."""
        if verbose:
            print("=" * 60)
            print("ErgoVision — Ergonomic Risk Assessment Pipeline")
            print("=" * 60)
            print(f"\n[1/4] Inspecting dataset: {dataset_path}")

        all_images = find_images(dataset_path)
        images = select_subset(all_images, subset_size)

        if verbose:
            print(f"  Found {len(all_images)} images -> subset of {len(images)}")

        if verbose:
            print(f"\n[2/4] Running YOLOv8-pose on {len(images)} images...")
        detections = self.pose_estimator.estimate_batch(images, verbose=verbose)

        if verbose:
            print(f"\n[3/4] Computing ergonomic risk scores...")
        results = []
        for det in detections:
            entry = {
                'image_path': det['image_path'],
                'num_people': len(det['detections']),
                'people': [],
            }
            for idx, person in enumerate(det['detections']):
                score_result = self.scorer.score(person['keypoints'])
                person_data = {
                    'person': idx,
                    'final_risk_class': score_result['final_risk_class'],
                    'final_score': score_result['final_score'],
                    'partial_scores': score_result['partial_scores'],
                    'explanation': score_result['explanation'],
                    'unavailable_features': score_result['unavailable_features'],
                }
                entry['people'].append(person_data)
                if verbose:
                    name = Path(det['image_path']).name
                    print(f"  {name}  person {idx}: "
                          f"{score_result['final_risk_class']} "
                          f"(score {score_result['final_score']})")
            results.append(entry)
        self.results = results

        if verbose:
            print(f"\n[4/4] Saving outputs...")
        if save_visualizations:
            self._save_visualizations_legacy(detections, results, verbose)
        self._save_csv_legacy(results, verbose)
        self._save_json_legacy(results, verbose)
        if verbose:
            self._print_summary_legacy(results)

        return results

    def _save_visualizations_legacy(self, detections, results, verbose=True):
        for det, res in zip(detections, results):
            image = cv2.imread(det['image_path'])
            if image is None:
                continue
            for person in det['detections']:
                image = self._draw_skeleton_legacy(
                    image, person['keypoints'], person['confidence']
                )
            if res['people']:
                p = res['people'][0]
                image = self._draw_risk_legacy(
                    image, p['final_risk_class'], p['final_score'],
                    p['partial_scores'], p['explanation']
                )
            out_path = VISUALIZATION_DIR / Path(det['image_path']).name
            save_prediction(image, out_path)
            if verbose:
                print(f"  Saved: {out_path}")

    @staticmethod
    def _draw_skeleton_legacy(image, keypoints, confidence):
        from .visualization import draw_skeleton
        return draw_skeleton(image, keypoints, confidence)

    @staticmethod
    def _draw_risk_legacy(image, risk_class, risk_score, partial_scores,
                          explanation):
        from .visualization import draw_risk_info
        return draw_risk_info(image, risk_class, risk_score,
                              partial_scores, explanation)

    def _save_csv_legacy(self, results, verbose=True):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            'image_path', 'person',
            'torso_angle', 'torso_score', 'torso_label',
            'neck_angle', 'neck_score', 'neck_label',
            'knee_angle', 'knee_score', 'knee_label',
            'shoulder_asymmetry', 'shoulder_score', 'shoulder_label',
            'body_inclination', 'body_inclination_score',
            'body_inclination_label',
            'final_score', 'final_risk_class', 'explanation',
            'unavailable_features',
        ]
        with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in results:
                for p in res['people']:
                    ps = p['partial_scores']
                    row = {
                        'image_path': res['image_path'],
                        'person': p['person'],
                        'torso_angle': _val(ps, 'trunk_angle'),
                        'torso_score': _score(ps, 'trunk_angle'),
                        'torso_label': _label(ps, 'trunk_angle'),
                        'neck_angle': _val(ps, 'neck_angle'),
                        'neck_score': _score(ps, 'neck_angle'),
                        'neck_label': _label(ps, 'neck_angle'),
                        'knee_angle': _val(ps, 'knee_angle'),
                        'knee_score': _score(ps, 'knee_angle'),
                        'knee_label': _label(ps, 'knee_angle'),
                        'shoulder_asymmetry': _val(ps, 'shoulder_asymmetry'),
                        'shoulder_score': _score(ps, 'shoulder_asymmetry'),
                        'shoulder_label': _label(ps, 'shoulder_asymmetry'),
                        'body_inclination': _val(ps, 'body_inclination'),
                        'body_inclination_score': _score(ps, 'body_inclination'),
                        'body_inclination_label': _label(ps, 'body_inclination'),
                        'final_score': p['final_score'],
                        'final_risk_class': p['final_risk_class'],
                        'explanation': p['explanation'],
                        'unavailable_features':
                            '; '.join(p['unavailable_features']),
                    }
                    writer.writerow(row)
        if verbose:
            print(f"  Saved CSV: {CSV_OUTPUT}")

    def _save_json_legacy(self, results, verbose=True):
        with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"  Saved JSON: {JSON_OUTPUT}")

    @staticmethod
    def _print_summary_legacy(results):
        total_people = sum(r['num_people'] for r in results)
        counts = {c: 0 for c in RISK_CLASSES}
        for r in results:
            for p in r['people']:
                counts[p['final_risk_class']] = (
                    counts.get(p['final_risk_class'], 0) + 1
                )
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Images processed : {len(results)}")
        print(f"  People detected  : {total_people}")
        print(f"  Risk distribution:")
        for cls in RISK_CLASSES:
            n = counts.get(cls, 0)
            pct = (n / max(total_people, 1)) * 100
            print(f"    {cls:15s}: {n:3d} ({pct:5.1f}%)")
        print(f"\n  Outputs -> {OUTPUT_DIR.resolve()}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Experimental mode
    # ------------------------------------------------------------------

    def _run_experimental(self, dataset_name, verbose=True):
        """Full experimental pipeline."""
        t_start = time.time()

        cfg = self.config
        cfg.dataset_name = dataset_name
        cfg.input_path = Path('data') / dataset_name
        cfg.output_path = Path('outputs') / dataset_name

        result = PipelineResult()
        result.config = cfg

        if verbose:
            _banner(dataset_name)

        # ---- Step 1: Dataset inspection ----
        if verbose:
            print(f"\n[1/6] Inspecting dataset: {cfg.input_path}")
        dataset_info = inspect_dataset(cfg.input_path)
        if verbose:
            print(f"  {dataset_info.short_summary()}")

        if dataset_info.total_files == 0:
            print(f"  WARNING: no media files found in {cfg.input_path}")
            return result

        cfg.mkdirs()

        # ---- Step 2: Frame extraction (if videos) ----
        if dataset_info.has_videos:
            if verbose:
                print(f"\n[2/6] Extracting frames from {dataset_info.n_videos} video(s)...")
            self._extract_frames(dataset_info, cfg, verbose)
            frame_images, _ = find_media(cfg.output_frames_dir)
            if verbose:
                print(f"  Total extracted frames: {len(frame_images)}")
        else:
            frame_images = dataset_info.image_files
            if verbose:
                print(f"\n[2/6] Using {len(frame_images)} direct image(s)...")

        if not frame_images:
            print("  No frames/images to process.")
            return result

        # ---- Step 3: Pose estimation (one-stage or two-stage) ----
        if cfg.use_two_stage:
            if verbose:
                print(f"\n[3/6] Two-stage pipeline: "
                      f"detector={Path(cfg.human_detector_model).name} "
                      f"pose={Path(cfg.crop_pose_model).name} ...")
            detections = self._detect_and_pose(frame_images, cfg, verbose)
        else:
            if verbose:
                print(f"\n[3/6] Single-stage: running YOLOv8-pose on "
                      f"{len(frame_images)} frame(s)...")
            detections = self.pose_estimator.estimate_batch(
                frame_images, verbose=verbose,
                conf_threshold=cfg.detection_confidence,
            )

        # ---- Step 4: Ergonomic scoring ----
        if verbose:
            print(f"\n[4/6] Computing ergonomic risk scores...")
        all_rows = []
        processed_frames = 0
        frames_with_people = 0
        total_person_detections = 0

        video_frame_counts = {}
        video_det_counts = {}

        for det_idx, det in enumerate(detections):
            processed_frames += 1
            det_path = Path(det['image_path'])

            video_id = det_path.parent.name if det_path.parent.name != cfg.output_frames_dir.name else 'images'
            frame_id = det_path.stem
            timestamp_sec = _estimate_timestamp(frame_id, cfg.frame_sampling_fps)

            if video_id not in video_frame_counts:
                video_frame_counts[video_id] = {'processed': 0, 'with_people': 0}
                video_det_counts[video_id] = 0

            video_frame_counts[video_id]['processed'] += 1

            num_people = len(det['detections'])
            total_person_detections += num_people
            video_det_counts[video_id] += num_people

            if num_people > 0:
                frames_with_people += 1
                video_frame_counts[video_id]['with_people'] += 1

            # Per-person processing
            for person_idx, person in enumerate(det['detections']):
                kps = person['keypoints']
                conf = person.get('confidence', np.ones(17))
                bbox = person.get('bbox', None)

                valid_kp_count = int(np.sum((kps[:, 0] != 0) & (kps[:, 1] != 0)))
                mean_kp_conf = float(np.mean(conf[conf > 0])) if np.any(conf > 0) else 0.0

                # --- False-positive filters ---
                fp_reason = self._check_false_positive(bbox, valid_kp_count, cfg)
                if fp_reason is not None:
                    score_result = self.scorer.score(kps)
                    ps = score_result['partial_scores']
                    row = {
                        'dataset_name': dataset_name,
                        'video_id': video_id,
                        'video_path': str(det_path),
                        'frame_id': frame_id,
                        'timestamp_sec': round(timestamp_sec, 3),
                        'person_id': person_idx,
                        'bbox_x1': round(bbox[0], 1) if bbox else '',
                        'bbox_y1': round(bbox[1], 1) if bbox else '',
                        'bbox_x2': round(bbox[2], 1) if bbox else '',
                        'bbox_y2': round(bbox[3], 1) if bbox else '',
                        'detection_confidence': round(float(person.get('confidence_mean', 0)), 3),
                        'valid_keypoints_count': valid_kp_count,
                        'mean_keypoint_confidence': round(mean_kp_conf, 4),
                    }
                    for angle_name in ALL_ANGLE_FEATURES:
                        row[angle_name] = ''
                    row['risk_score'] = 1
                    row['risk_level'] = 'LOW'
                    row['discarded'] = True
                    row['discard_reason'] = fp_reason
                    row['_partial_scores'] = ps
                    row['_explanation'] = fp_reason
                    row['_final_risk_class'] = 'Low Risk'
                    annotated_filename = f"{video_id}_{frame_id}_p{person_idx}.jpg"
                    row['_annotated_frame_path'] = str(
                        cfg.output_annotated_frames_dir / annotated_filename
                    )
                    all_rows.append(row)
                    continue

                # --- Valid posture: score it ---
                score_result = self.scorer.score(kps)
                ps = score_result['partial_scores']

                n_available = sum(
                    1 for v in ps.values() if v.get('score') is not None
                )
                min_features = 3
                discarded = n_available < min_features
                discard_reason = (
                    '' if not discarded
                    else f'Only {n_available}/{len(ps)} features available'
                )

                risk_level_short = RISK_LEVEL_SHORT.get(
                    score_result['final_score'], 'UNKNOWN'
                )

                row = {
                    'dataset_name': dataset_name,
                    'video_id': video_id,
                    'video_path': str(det_path),
                    'frame_id': frame_id,
                    'timestamp_sec': round(timestamp_sec, 3),
                    'person_id': person_idx,
                    'bbox_x1': round(bbox[0], 1) if bbox else '',
                    'bbox_y1': round(bbox[1], 1) if bbox else '',
                    'bbox_x2': round(bbox[2], 1) if bbox else '',
                    'bbox_y2': round(bbox[3], 1) if bbox else '',
                    'detection_confidence': round(float(person.get('confidence_mean', 0)), 3),
                    'valid_keypoints_count': valid_kp_count,
                    'mean_keypoint_confidence': round(mean_kp_conf, 4),
                }

                for angle_name in ALL_ANGLE_FEATURES:
                    v = ps.get(angle_name, {}).get('value')
                    row[angle_name] = v if v is not None else ''

                row['risk_score'] = score_result['final_score']
                row['risk_level'] = risk_level_short
                row['action_level'] = score_result.get('action_level', '')
                row['discarded'] = discarded
                row['discard_reason'] = discard_reason
                row['postural_severity'] = score_result.get('continuous_severity', '')
                row['smoothed_severity'] = score_result.get('smoothed_severity',
                                           score_result.get('continuous_severity', ''))
                row['pose_confidence'] = score_result.get('confidence', '')

                # Segment scores
                row['trunk_score'] = score_result.get('trunk_score', '')
                row['neck_score'] = score_result.get('neck_score', '')
                row['upper_arm_score'] = score_result.get('upper_arm_score', '')
                row['forearm_score'] = score_result.get('forearm_score', '')
                row['leg_score'] = score_result.get('leg_score', '')

                # Debug fields
                row['action_level'] = score_result.get('action_level', '')
                row['action_level_reason'] = score_result.get('action_level_reason', '')
                row['approximate_action_level'] = score_result.get('approximate_action_level', '')
                row['neutral_gate_applied'] = score_result.get('neutral_gate_applied', False)
                row['neck_capped'] = score_result.get('neck_capped', False)
                row['primary_risk_drivers'] = '; '.join(score_result.get('primary_risk_drivers', []))
                row['secondary_risk_drivers'] = '; '.join(score_result.get('secondary_risk_drivers', []))
                row['mapping_consistent'] = score_result.get('mapping_consistent', True)

                row['_partial_scores'] = ps
                row['_explanation'] = score_result['explanation']
                row['_final_risk_class'] = score_result['final_risk_class']

                annotated_filename = f"{video_id}_{frame_id}_p{person_idx}.jpg"
                row['_annotated_frame_path'] = str(
                    cfg.output_annotated_frames_dir / annotated_filename
                )

                all_rows.append(row)

                if verbose and (det_idx % max(1, len(detections) // 20) == 0):
                    print(f"  Frame {det_idx + 1}/{len(detections)} "
                          f"({num_people} people)")

        result.frame_person_rows = all_rows

        # ---- Temporal smoothing (post-processing) ----
        if cfg.temporal_smoothing and len(all_rows) > 1:
            if verbose:
                print(f"\n  Applying temporal EMA smoothing "
                      f"(alpha={cfg.ema_alpha}, window={cfg.smooth_window})...")
            all_rows = temporal_smooth_severity(
                all_rows,
                alpha=cfg.ema_alpha,
                decay_alpha=cfg.ema_decay_alpha,
                severity_low_max=cfg.severity_low_max,
                severity_medium_max=cfg.severity_medium_max,
            )
            result.frame_person_rows = all_rows

        # ---- Step 5: Save CSV ----
        if cfg.save_csv:
            if verbose:
                print(f"\n[5/6] Saving CSV outputs...")
            self._save_frame_person_csv(all_rows, cfg.csv_frame_person_path, verbose)
            video_summary_rows = self._compute_video_summary(
                all_rows, video_frame_counts, video_det_counts, t_start, cfg
            )
            result.video_summary_rows = video_summary_rows
            self._save_video_summary_csv(video_summary_rows,
                                         cfg.csv_video_summary_path, verbose)

        # ---- Step 6: Visual outputs ----
        if verbose:
            print(f"\n[6/6] Generating visual outputs...")

        if cfg.save_annotated_frames:
            self._save_annotated_frames(all_rows, detections, cfg, verbose)
            for row in all_rows:
                row['annotated_frame_path'] = row.get('_annotated_frame_path', '')

        if cfg.save_annotated_video and dataset_info.has_videos:
            self._save_annotated_video(cfg, dataset_info, verbose)

        discarded_rows = [r for r in all_rows if r.get('discarded')]
        if cfg.save_failure_cases and discarded_rows:
            save_failure_cases(discarded_rows, cfg.output_annotated_frames_dir,
                               cfg.output_examples_dir)
            if verbose:
                print(f"  Saved {min(len(discarded_rows), 20)} failure case(s)")

        if cfg.save_annotated_frames:
            extract_risk_examples(all_rows, cfg.output_annotated_frames_dir,
                                  cfg.output_examples_dir)
            if verbose:
                print("  Extracted risk-level examples")

        self._generate_plots_and_report(all_rows, result, cfg, verbose)

        t_elapsed = time.time() - t_start
        result.total_time_sec = t_elapsed
        total_valid = result.valid_postures
        fps_eff = total_valid / t_elapsed if t_elapsed > 0 else 0

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"EXPERIMENT COMPLETE — {dataset_name}")
            print(f"{'=' * 60}")
            print(f"  Frames processed    : {processed_frames}")
            print(f"  Frames with people  : {frames_with_people}")
            print(f"  Total detections    : {total_person_detections}")
            print(f"  Valid postures      : {total_valid}")
            print(f"  Discarded postures  : {len(discarded_rows)}")
            print(f"  Elapsed time        : {t_elapsed:.1f}s")
            print(f"  Effective FPS       : {fps_eff:.2f}")
            print(f"  Outputs -> {cfg.output_path.resolve()}")
            print(f"{'=' * 60}")

        self.results = all_rows
        return result

    # ------------------------------------------------------------------
    # False-positive filter
    # ------------------------------------------------------------------

    @staticmethod
    def _check_false_positive(bbox, valid_kp_count, cfg):
        """Return a reason string if detection is likely false positive, else None."""
        # Bounding-box sanity
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            bw = x2 - x1
            bh = y2 - y1
            if bw > 0 and bh > 0:
                aspect = bw / bh
                area = bw * bh
                if aspect < 0.15 or aspect > 3.0:
                    return f'bad aspect ratio {aspect:.2f} (w/h)'
                if area < 400:
                    return f'bbox too small ({area:.0f} px)'

        # Minimum valid keypoints
        if valid_kp_count < cfg.min_valid_keypoints:
            return (
                f'False positive: only {valid_kp_count}/{17} valid keypoints '
                f'(min required: {cfg.min_valid_keypoints})'
            )

        return None

    # ------------------------------------------------------------------
    # Experimental pipeline helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_frames(dataset_info, cfg, verbose):
        from .data.video_frame_extractor import VideoFrameExtractor
        extractor = VideoFrameExtractor(
            output_dir=cfg.output_frames_dir,
            sampling_rate=cfg.frame_sampling_fps,
            max_frames_per_video=cfg.max_frames_per_video,
            blur_threshold=0,
        )
        for video_path in dataset_info.video_files:
            ex_result = extractor.extract(video_path)
            if verbose:
                print(f"  {video_path.name}: {ex_result.frames_extracted} frames")

    @staticmethod
    def _save_frame_person_csv(rows, output_path, verbose=True):
        fieldnames = [
            'dataset_name', 'video_id', 'video_path', 'frame_id',
            'timestamp_sec', 'person_id',
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
            'detection_confidence', 'valid_keypoints_count',
            'mean_keypoint_confidence',
            'trunk_angle', 'neck_angle',
            'upper_arm_angle_left', 'upper_arm_angle_right',
            'forearm_angle_left', 'forearm_angle_right',
            'knee_angle_left', 'knee_angle_right',
            'shoulder_asymmetry', 'body_inclination',
            'trunk_score', 'neck_score', 'upper_arm_score',
            'forearm_score', 'leg_score',
            'postural_severity', 'smoothed_severity', 'pose_confidence',
            'risk_score', 'risk_level', 'action_level',
            'action_level_reason', 'approximate_action_level',
            'neutral_gate_applied', 'neck_capped',
            'primary_risk_drivers', 'secondary_risk_drivers',
            'mapping_consistent',
            'discarded', 'discard_reason',
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        if verbose:
            print(f"  Saved: {output_path} ({len(rows)} rows)")

    @staticmethod
    def _compute_video_summary(all_rows, video_frame_counts, video_det_counts,
                               t_start, cfg):
        from collections import Counter
        video_risk = {}
        for r in all_rows:
            vid = r.get('video_id', 'unknown')
            if vid not in video_risk:
                video_risk[vid] = Counter()
            video_risk[vid][r.get('risk_level', 'UNKNOWN')] += 1

        summary = []
        for vid in sorted(video_frame_counts.keys()):
            fc = video_frame_counts.get(vid, {'processed': 0, 'with_people': 0})
            det_count = video_det_counts.get(vid, 0)
            risk_c = video_risk.get(vid, Counter())

            vid_rows = [r for r in all_rows if r.get('video_id') == vid]
            valid = sum(1 for r in vid_rows if not r.get('discarded'))
            discarded = sum(1 for r in vid_rows if r.get('discarded'))

            scores = []
            for r in vid_rows:
                s = r.get('risk_score')
                if s is not None and s != '' and s != 'None':
                    try:
                        scores.append(float(s))
                    except (ValueError, TypeError):
                        pass
            mean_risk = float(np.mean(scores)) if scores else 0.0

            summary.append({
                'dataset_name': cfg.dataset_name,
                'video_id': vid,
                'processed_frames': fc['processed'],
                'frames_with_people': fc['with_people'],
                'total_person_detections': det_count,
                'valid_postures': valid,
                'discarded_postures': discarded,
                'low_risk_count': risk_c.get('LOW', 0),
                'medium_risk_count': risk_c.get('MEDIUM', 0),
                'high_risk_count': risk_c.get('HIGH', 0),
                'low_risk_percentage': round(
                    (risk_c.get('LOW', 0) / max(valid + discarded, 1)) * 100, 1
                ),
                'medium_risk_percentage': round(
                    (risk_c.get('MEDIUM', 0) / max(valid + discarded, 1)) * 100, 1
                ),
                'high_risk_percentage': round(
                    (risk_c.get('HIGH', 0) / max(valid + discarded, 1)) * 100, 1
                ),
                'mean_risk_score': round(mean_risk, 2),
                'max_risk_score': max(scores) if scores else 0,
                'mean_trunk_angle': round(_mean_angle(vid_rows, 'trunk_angle'), 1),
                'mean_neck_angle': round(_mean_angle(vid_rows, 'neck_angle'), 1),
                'mean_upper_arm_angle': round(
                    _mean_angle(vid_rows, 'upper_arm_angle_left', 'upper_arm_angle_right'), 1
                ),
                'mean_keypoint_confidence': round(
                    _mean_angle(vid_rows, 'mean_keypoint_confidence'), 4
                ),
                'processing_time_sec': round(time.time() - t_start, 1),
                'fps_effective': round(
                    det_count / max(time.time() - t_start, 0.001), 2
                ),
            })
        return summary

    def _save_video_summary_csv(self, rows, output_path, verbose=True):
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if verbose:
            print(f"  Saved: {output_path} ({len(rows)} videos)")

    def _save_annotated_frames(self, all_rows, detections, cfg, verbose):
        from collections import defaultdict
        frame_to_people = defaultdict(list)
        for r in all_rows:
            key = (r['video_id'], r['frame_id'])
            frame_to_people[key].append(r)

        saved = 0
        for det_idx, det in enumerate(detections):
            det_path = Path(det['image_path'])
            video_id = det_path.parent.name
            frame_id = det_path.stem
            key = (video_id, frame_id)

            image = cv2.imread(det['image_path'])
            if image is None:
                continue

            people_data = frame_to_people.get(key, [])
            for p_idx, person in enumerate(det['detections']):
                if p_idx >= len(people_data):
                    continue
                pd = people_data[p_idx]
                ps = pd.get('_partial_scores', {})
                explanation = pd.get('_explanation', '')
                risk_class = pd.get('_final_risk_class', 'Low Risk')
                risk_score = pd.get('risk_score', 1)

                image = annotate_frame(
                    image,
                    person.get('bbox'),
                    person['keypoints'],
                    person.get('confidence', np.ones(17)),
                    risk_class, risk_score,
                    ps, explanation,
                    person_id=p_idx,
                    detection_confidence=person.get('detection_conf'),
                )

            out_path = cfg.output_annotated_frames_dir / f"{video_id}_{frame_id}.jpg"
            save_prediction(image, out_path)
            saved += 1

            for pd in people_data:
                pd['_annotated_frame_path'] = str(out_path)

        if verbose:
            print(f"  Saved {saved} annotated frame(s)")

    def _save_annotated_video(self, cfg, dataset_info, verbose):
        video_frames = {}
        for af in cfg.output_annotated_frames_dir.glob('*'):
            parts = af.stem.split('_', 1)
            if len(parts) >= 2:
                vid = parts[0]
                video_frames.setdefault(vid, []).append(af)

        for vid, frames in video_frames.items():
            out_path = cfg.output_annotated_videos_dir / f"{vid}_annotated.mp4"
            create_annotated_video(
                cfg.output_frames_dir / vid,
                cfg.output_annotated_frames_dir,
                out_path,
                fps=int(cfg.frame_sampling_fps) or 10,
            )
            if verbose:
                print(f"  Video: {out_path}")

    def _generate_plots_and_report(self, all_rows, result, cfg, verbose):
        from .analysis import compute_statistics, generate_all_plots
        from .report import generate_experimental_report

        stats = compute_statistics(all_rows)
        result.stats = stats

        if cfg.save_plots:
            generate_all_plots(all_rows, cfg.output_plots_dir, cfg.dataset_name)
            if verbose:
                print("  Generated all plots")

        report_path = cfg.output_reports_dir / 'experimental_report.md'
        generate_experimental_report(
            stats, all_rows, cfg, report_path,
            video_summary_rows=result.video_summary_rows,
        )
        if verbose:
            print(f"  Report: {report_path}")

    # ------------------------------------------------------------------
    # Two-stage pipeline helpers
    # ------------------------------------------------------------------

    def _detect_and_pose(self, frame_images, cfg, verbose):
        """Two-stage: detect humans → filter → crop → pose → return detections.

        Returns a list of detection dicts compatible with the rest of
        the pipeline (same format as ``PoseEstimator.estimate_batch``).
        """
        all_detections = []

        for img_idx, img_path in enumerate(frame_images):
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Stage 1: Human detection
            raw_dets = self.detector.detect(
                image, conf_threshold=cfg.detection_confidence,
            )

            # Stage 1b: Filter invalid detections
            valid_dets, discarded = HumanDetector.filter_detections(
                raw_dets, image.shape, cfg,
            )

            if cfg.debug_detection:
                dbg_path = Path(str(img_path).replace('frames', 'debug_detections'))
                HumanDetector.debug_visualization(
                    image, valid_dets, discarded, [],
                    output_path=dbg_path,
                )

            # Stage 2: Crop-based pose estimation for each valid detection
            frame_detections = []
            for det in valid_dets:
                pose_result = self.crop_pose.estimate_on_crop(
                    image, det['bbox'], padding=cfg.bbox_padding,
                )

                if not pose_result['crop_valid']:
                    continue

                frame_detections.append({
                    'keypoints': pose_result['keypoints'],
                    'confidence': pose_result['confidence'],
                    'confidence_mean': pose_result['confidence_mean'],
                    'bbox': det['bbox'],
                    'detection_conf': det.get('confidence', 0.0),
                })

            all_detections.append({
                'image_path': str(img_path),
                'detections': frame_detections,
            })

            if verbose and (img_idx + 1) % max(1, len(frame_images) // 20) == 0:
                n_valid = len(valid_dets)
                n_disc = len(discarded)
                tag = f"  +{n_valid} valid" if not n_disc else f"  +{n_valid} valid -{n_disc} filtered"
                print(f"  [{img_idx + 1}/{len(frame_images)}]{tag}")

        return all_detections

    def run_comparison(self, dataset_name, verbose=True):
        """Run both one-stage and two-stage pipelines and compare results.

        Parameters
        ----------
        dataset_name : str
        verbose : bool

        Returns
        -------
        dict with ``one_stage_result``, ``two_stage_result``,
        ``comparison`` table.
        """
        print("=" * 65)
        print("  COMPARISON: One-Stage vs Two-Stage Pipeline")
        print("=" * 65)

        # Run one-stage
        cfg_one = ExperimentConfig(dataset_name=dataset_name)
        cfg_one.use_two_stage = False
        cfg_one.frame_sampling_fps = self.config.frame_sampling_fps
        cfg_one.max_frames_per_video = self.config.max_frames_per_video
        cfg_one.save_plots = False
        cfg_one.save_csv = False
        cfg_one.save_annotated_frames = False

        pipe_one = ErgoPipeline(config=cfg_one)
        t0 = time.time()
        res_one = pipe_one.run(dataset_name=dataset_name, verbose=False)
        t_one = time.time() - t0

        # Run two-stage
        cfg_two = ExperimentConfig(dataset_name=dataset_name)
        cfg_two.use_two_stage = True
        cfg_two.frame_sampling_fps = self.config.frame_sampling_fps
        cfg_two.max_frames_per_video = self.config.max_frames_per_video
        cfg_two.save_plots = False
        cfg_two.save_csv = False
        cfg_two.save_annotated_frames = False

        pipe_two = ErgoPipeline(config=cfg_two)
        t0 = time.time()
        res_two = pipe_two.run(dataset_name=dataset_name, verbose=False)
        t_two = time.time() - t0

        # Build comparison table
        rd1 = res_one.stats.get('risk_distribution', {})
        rd2 = res_two.stats.get('risk_distribution', {})

        total1 = res_one.total_postures
        total2 = res_two.total_postures

        comp = {
            'one_stage': {
                'total': total1,
                'LOW': rd1.get('LOW', {}).get('count', 0),
                'MEDIUM': rd1.get('MEDIUM', {}).get('count', 0),
                'HIGH': rd1.get('HIGH', {}).get('count', 0),
                'time_sec': round(t_one, 1),
                'model': 'yolov8n-pose (full frame)',
            },
            'two_stage': {
                'total': total2,
                'LOW': rd2.get('LOW', {}).get('count', 0),
                'MEDIUM': rd2.get('MEDIUM', {}).get('count', 0),
                'HIGH': rd2.get('HIGH', {}).get('count', 0),
                'time_sec': round(t_two, 1),
                'model': f'det={cfg_two.human_detector_model} pose={cfg_two.crop_pose_model}',
            },
        }

        if verbose:
            print()
            print(f"{'':>15s}  {'One-Stage (n-pose)':>20s}  {'Two-Stage':>20s}")
            print(f"{'─' * 57}")
            c1 = comp['one_stage']
            c2 = comp['two_stage']
            for level in ['LOW', 'MEDIUM', 'HIGH']:
                pct1 = (c1[level] / max(c1['total'], 1)) * 100
                pct2 = (c2[level] / max(c2['total'], 1)) * 100
                print(f"  {level:>10s}:  {c1[level]:>4d} ({pct1:5.1f}%)      "
                      f"{c2[level]:>4d} ({pct2:5.1f}%)")
            print()
            print(f"  Total postures:  {c1['total']:>4d}                  {c2['total']:>4d}")
            print(f"  Time:            {c1['time_sec']:>4.1f}s                {c2['time_sec']:>4.1f}s")
            print(f"  Model:           {c1['model']:35s}")
            print(f"                   {c2['model']:35s}")

        return comp


# ===================================================================
# Module-level helpers
# ===================================================================

def _val(ps, name):
    return ps.get(name, {}).get('value', '')


def _score(ps, name):
    return ps.get(name, {}).get('score', '')


def _label(ps, name):
    return ps.get(name, {}).get('label', '')


def _estimate_timestamp(frame_id, sampling_fps):
    digits = ''.join(c for c in frame_id if c.isdigit())
    try:
        idx = int(digits) if digits else 0
    except ValueError:
        idx = 0
    return idx / max(sampling_fps, 0.01)


def _mean_angle(rows, *cols):
    vals = []
    for r in rows:
        for col in cols:
            v = r.get(col)
            if v is not None and v != '' and v != 'None':
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
    return float(np.mean(vals)) if vals else 0.0


def _banner(dataset_name):
    print("=" * 70)
    print(f"  ErgoVision — Experimental Pipeline  |  Dataset: {dataset_name}")
    print("=" * 70)
