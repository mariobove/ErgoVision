"""
Markdown experimental report generation for ErgoVision.

Produces a self-contained ``experimental_report.md`` with dataset description,
setup, pipeline steps, quantitative results, failure analysis, discussion,
limitations, and reproducibility notes — ready for inclusion in a paper
appendix or supplementary material.
"""

import os
from pathlib import Path


def _link(path, label=None, report_dir=None):
    """Return a markdown image link if file exists, else a text reference.

    When *report_dir* is provided the link path is made relative to it,
    so the markdown renders correctly when the report is in a subdirectory.
    """
    p = Path(path)
    if p.exists():
        if report_dir is not None:
            try:
                rel = os.path.relpath(p, report_dir).replace('\\', '/')
            except ValueError:
                rel = p.as_posix()
        else:
            rel = p.as_posix()
        lbl = label or p.name
        return f"![{lbl}]({rel})"
    return f"*{label or p.name} (not generated)*"


def _fmt(val, decimals=1):
    """Format a number or return ``'N/A'``."""
    if val is None or val == '' or val == 'None':
        return 'N/A'
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def generate_experimental_report(stats, rows, config, output_path,
                                 video_summary_rows=None):
    """Write a complete experimental report markdown file.

    Parameters
    ----------
    stats : dict
        Output of ``analysis.compute_statistics()``.
    rows : list[dict]
        Frame-person results rows.
    config : ExperimentConfig
        The configuration used for this run.
    output_path : Path or str
        Full path for the ``.md`` file.
    video_summary_rows : list[dict] or None
        Per-video summary rows for the setup table.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = output_path.parent

    # Local wrapper so every _link call gets the report-relative path
    def _l(path, label=None):
        return _link(path, label=label, report_dir=report_dir)

    ds_name = config.dataset_name
    plots_dir = config.output_plots_dir
    examples_dir = config.output_examples_dir
    annotated_dir = config.output_annotated_frames_dir

    cfg = config

    # ---- helpers for inline stats ----
    rd = stats.get('risk_distribution', {})
    as_ = stats.get('angle_statistics', {})

    low = rd.get('LOW', {})
    med = rd.get('MEDIUM', {})
    high = rd.get('HIGH', {})

    total_postures = stats.get('total_postures', 0)
    valid_postures = stats.get('valid_postures', 0)
    discarded = stats.get('discarded_postures', 0)
    mean_kp_conf = stats.get('mean_keypoint_confidence', 0)
    ppf = stats.get('people_per_frame_summary', {})

    lines = []
    _w = lines.append

    _w("# Experimental Report")
    _w("")
    _w(f"**Pipeline:** ErgoVision v0.2.0  ")
    _w(f"**Date:** {_now()}  ")
    _w(f"**Dataset:** `{ds_name}`  ")
    _w("")

    # ---- 1. Dataset ----
    _w("## Dataset")
    _w("")
    unique_videos = set(
        r.get('video_id', '') for r in rows if r.get('video_id')
    )
    n_videos = len(unique_videos) - (1 if '' in unique_videos else 0)
    _w(f"The dataset `{ds_name}` contains **{n_videos} video(s)** "
       f"processed at **{cfg.frame_sampling_fps} FPS** "
       f"(max {cfg.max_frames_per_video} frames per video).  ")
    _w("")
    _w(f"- Frame sampling rate: {cfg.frame_sampling_fps} FPS  ")
    _w(f"- Max frames / video: {cfg.max_frames_per_video}  ")
    _w(f"- Detection confidence threshold: {cfg.detection_confidence}  ")
    _w(f"- Keypoint confidence threshold: {cfg.keypoint_confidence}  ")
    _w(f"- Total frames processed: {total_postures}  ")
    _w("")

    if video_summary_rows:
        _w("### Per-Video Processing Summary")
        _w("")
        _w("| Video ID | Frames Processed | Frames with People | Valid Postures | "
           "Low | Med | High | Mean Risk |")
        _w("|----------|-----------------|-------------------|---------------|-----|-----|------|----------|")
        for vs in video_summary_rows:
            _w(
                f"| {vs.get('video_id', '?')} "
                f"| {vs.get('processed_frames', '?')} "
                f"| {vs.get('frames_with_people', '?')} "
                f"| {vs.get('valid_postures', '?')} "
                f"| {vs.get('low_risk_count', 0)} "
                f"| {vs.get('medium_risk_count', 0)} "
                f"| {vs.get('high_risk_count', 0)} "
                f"| {_fmt(vs.get('mean_risk_score'))} |"
            )
        _w("")

    # ---- 2. Experimental Setup ----
    _w("## Experimental Setup")
    _w("")
    _w("### System Description")
    _w("")
    _w("The proposed method is a **vision-based RULA-inspired postural risk screening** ")
    _w("tool. It estimates the **observable postural component** of ergonomic risk ")
    _w("using a **Mamdani fuzzy inference system** applied to joint-angle features, ")
    _w("with continuous confidence propagation and temporal smoothing.  ")
    _w("")
    _w("**It does not implement full clinical RULA.** Force, load handling, muscle use, ")
    _w("repetition, and staticity are not automatically inferred and must be specified ")
    _w("manually for a complete ergonomic assessment.  ")
    _w("")
    _w("### Postural Risk Classification")
    _w("")
    _w("The **continuous severity score** (0–100) is computed via a Mamdani fuzzy ")
    _w("inference engine with trapezoidal membership functions calibrated on ")
    _w("RULA/REBA/OWAS cut-points from the ergonomics literature.  ")
    _w("Non-linear fuzzy rules capture biomechanical interactions between body regions, ")
    _w("replacing the linear weighted-sum approach.  ")
    _w("")
    _w("| Severity Range | Class | Interpretation |")
    _w("|--------------|-------|----------------|")
    _w("| 0–{cfg.severity_low_max} | Low Risk | Neutral postural alignment |")
    _w("| {cfg.severity_low_max}–{cfg.severity_medium_max} | Medium Risk | Moderate deviations — monitor |")
    _w("| >{cfg.severity_medium_max} | High Risk | Sustained severe postural load |")
    _w("")
    _w("Each pose receives a **continuous confidence score** (0–1) that encodes ")
    _w("keypoint coverage and detection quality.  Low-confidence estimates are still ")
    _w("reported with a caveat rather than discarded — unlike binary uncertain/gating ")
    _w("approaches.  ")
    _w("")
    _w("| Component | Detail |")
    _w("|-----------|--------|")
    _w("| Pose estimation | YOLOv8l-pose (cropped, pretrained, no fine-tuning) |")
    _w("| Detection | YOLOv8l (person class only, pretrained) |")
    _w("| Risk aggregation | Mamdani fuzzy inference (~25 rules) |")
    _w("| Membership functions | Trapezoidal, RULA-aligned thresholds |")
    _w("| Temporal smoothing | EMA (α={cfg.ema_alpha}) across frames |")
    _w("| Uncertainty handling | Continuous confidence (0–1) instead of binary discard |")
    _w("| Keypoints | 17 COCO keypoints |")
    _w("| Confidence threshold (detection) | {cfg.detection_confidence} |")
    _w("| Confidence threshold (keypoint) | {cfg.keypoint_confidence} |")
    _w("")

    _w("### Postural Features")
    _w("")
    _w("| Feature | Type | Description |")
    _w("|---------|------|------------|")
    _w("| Trunk angle | Primary | Deviation of torso from vertical (mid-shoulder to mid-hip) |")
    _w("| Neck angle | Primary | Deviation of head from vertical (nose to mid-shoulder) |")
    _w("| Upper arm angle (L/R) | Primary | Arm elevation from vertical (shoulder→elbow) |")
    _w("| Forearm angle (L/R) | Secondary | Elbow flexion, scored on deviation from 90° |")
    _w("| Knee angle (L/R) | Secondary | Knee flexion, scored on bend from straight |")
    _w("| Shoulder asymmetry | Secondary | Height difference between left and right shoulder |")
    _w("| Body inclination | Secondary | Lateral lean of torso (horizontal/height ratio) |")
    _w("")

    _w("### Fuzzy Rule Base (abbreviated)")
    _w("")
    _w("Risk is determined by ~25 fuzzy rules of the form:")
    _w("- IF trunk IS extreme → **Very High Risk** (single-region critical)  ")
    _w("- IF trunk IS high AND upper arm IS moderate → **Very High Risk** (interaction penalty)  ")
    _w("- IF trunk IS high → **High Risk**  ")
    _w("- IF trunk IS moderate AND neck IS moderate → **High Risk** (combined moderate)  ")
    _w("- IF trunk IS moderate → **Medium Risk**  ")
    _w("- IF knee IS extreme → **Medium Risk** (secondary escalation)  ")
    _w("- Default → **Low Risk**  ")
    _w("")

    # ---- 3. Pipeline ----
    _w("## Pipeline")
    _w("")
    _w("The ErgoVision pipeline consists of the following stages:")
    _w("")
    _w("1. **Frame extraction** — videos are sampled at "
       f"{cfg.frame_sampling_fps} FPS  ")
    _w("2. **People detection** — YOLOv8 detects persons in each frame  ")
    _w("3. **Pose estimation** — YOLOv8-pose predicts 17 keypoints per person  ")
    _w("4. **Angle computation** — geometric features derived from keypoints  ")
    _w("5. **Ergonomic risk evaluation** — rule-based RULA-inspired scoring  ")
    _w("6. **Output generation** — CSV, annotated frames/videos, plots, report  ")
    _w("")

    # ---- 4. Results ----
    _w("## Results")
    _w("")

    _w(f"- **Total postures evaluated:** {total_postures}  ")
    _w(f"- **Valid postures (all features available):** {valid_postures}  ")
    _w(f"- **Discarded postures (insufficient keypoints):** {discarded}  ")
    _w(f"- **Mean keypoint confidence:** {_fmt(mean_kp_conf)}  ")
    _w(f"- **People per frame:** mean {_fmt(ppf.get('mean'))}, "
       f"max {_fmt(ppf.get('max'))}  ")
    _w("")

    _w("### Risk Distribution")
    _w("")
    _w(f"| Level | Count | Percentage |")
    _w("|-------|-------|-----------|")
    _w(f"| **LOW** | {low.get('count', 0)} | {low.get('percentage', 0)}% |")
    _w(f"| **MEDIUM** | {med.get('count', 0)} | {med.get('percentage', 0)}% |")
    _w(f"| **HIGH** | {high.get('count', 0)} | {high.get('percentage', 0)}% |")
    _w("")

    # Overall risk distribution plot
    p_overall = plots_dir / 'risk_distribution_overall.png'
    _w(_l(p_overall, 'Overall Risk Distribution'))
    _w("")
    _w("")

    # Risk by video plot
    p_byvideo = plots_dir / 'risk_distribution_by_video.png'
    _w(_l(p_byvideo, 'Risk Distribution by Video'))
    _w("")
    _w("")

    # People per frame plot
    p_ppf = plots_dir / 'people_per_frame_distribution.png'
    _w(_l(p_ppf, 'People per Frame Distribution'))
    _w("")
    _w("")

    # Keypoint confidence plot
    p_kpconf = plots_dir / 'keypoint_confidence_distribution.png'
    _w(_l(p_kpconf, 'Keypoint Confidence Distribution'))
    _w("")
    _w("")

    # Angle statistics table
    _w("### Angle Statistics")
    _w("")
    _w("| Feature | Mean | Std | Min | P25 | P50 | P75 | Max |")
    _w("|---------|------|-----|-----|-----|-----|-----|-----|")
    for col in ['trunk_angle', 'neck_angle',
                'upper_arm_angle_left', 'upper_arm_angle_right',
                'forearm_angle_left', 'forearm_angle_right',
                'knee_angle_left', 'knee_angle_right',
                'shoulder_asymmetry', 'body_inclination']:
        s = as_.get(col, {})
        if s:
            _w(
                f"| {col} "
                f"| {_fmt(s.get('mean'))} "
                f"| {_fmt(s.get('std'))} "
                f"| {_fmt(s.get('min'))} "
                f"| {_fmt(s.get('p25'))} "
                f"| {_fmt(s.get('p50'))} "
                f"| {_fmt(s.get('p75'))} "
                f"| {_fmt(s.get('max'))} |"
            )
        else:
            _w(f"| {col} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    _w("")

    # Angle histogram plots
    _w("### Angle Histograms")
    _w("")
    for col in ['trunk_angle', 'neck_angle',
                'upper_arm_angle_left', 'upper_arm_angle_right',
                'forearm_angle_left', 'forearm_angle_right',
                'knee_angle_left', 'knee_angle_right',
                'shoulder_asymmetry', 'body_inclination']:
        p_hist = plots_dir / f'{col}_histogram.png'
        if p_hist.exists():
            _w(_l(p_hist, f'{col} Histogram'))
            _w("")
    _w("")

    # Visual examples
    _w("### Visual Examples by Risk Level")
    _w("")
    for level in ['LOW', 'MEDIUM', 'HIGH']:
        level_dir = examples_dir / level.lower()
        examples = sorted(level_dir.glob('*')) if level_dir.exists() else []
        if examples:
            _w(f"#### {level} Risk Examples")
            _w("")
            for ex in examples[:10]:
                _w(_l(ex, ex.name))
                _w("")
        else:
            _w(f"*No {level} risk examples saved.*")
            _w("")

    # ---- 5. Failure Cases ----
    _w("## Failure Cases")
    _w("")
    discarded_rows = [
        r for r in rows
        if r.get('discarded') in (True, 'True', 'true', '1')
    ]
    if discarded_rows:
        reasons = {}
        for r in discarded_rows:
            reason = r.get('discard_reason', 'unknown')
            reasons[reason] = reasons.get(reason, 0) + 1
        _w(f"**{len(discarded_rows)} posture(s) discarded.**  ")
        _w("")
        _w("| Discard Reason | Count |")
        _w("|---------------|-------|")
        for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
            _w(f"| {reason} | {cnt} |")
        _w("")
    else:
        _w("No postures were discarded.  ")
        _w("")

    fail_dir = examples_dir / 'failure'
    if fail_dir.exists():
        failures = sorted(fail_dir.glob('*'))
        if failures:
            _w("### Failure Case Visualisations")
            _w("")
            for f in failures[:10]:
                _w(_l(f, f.name))
                _w("")

    # ---- 6. Discussion ----
    _w("## Discussion")
    _w("")
    _w("### Interpretation of Results")
    _w("")
    pct_high = high.get('percentage', 0)
    pct_low = low.get('percentage', 0)
    pct_med = med.get('percentage', 0)
    _w(f"In this experiment, **{pct_low}%** of observations were classified "
       f"as Low Risk, **{pct_med}%** as Medium Risk, and **{pct_high}%** as "
       f"High Risk.  ")
    _w("")
    _w("These results represent the **observable postural component** of ")
    _w("ergonomic risk.  The continuous severity score (0-100) allows for ")
    _w("fine-grained tracking of postural load over time.  ")
    _w("")

    _w("### System Advantages")
    _w("")
    _w("- **Zero-shot generalisation:** the pipeline works on any dataset "
       "without dataset-specific fine-tuning.  ")
    _w("- **Reproducibility:** deterministic inference and rule-based "
       "scoring guarantee identical results across runs.  ")
    _w("- **Explainability:** each risk score is traced back to specific "
       "postural features with biomechanical severity values.  ")
    _w("- **Temporal robustness:** EMA smoothing prevents single-frame "
       "oscillations from changing the risk class.  ")
    _w("- **Uncertainty awareness:** low-quality poses are flagged rather "
       "than force-assigned a risk class.  ")
    _w("- **Resource efficiency:** runs on CPU for small batches; GPU "
       "optional for larger experiments.  ")
    _w("")

    _w("### Applicability to Industrial Environments")
    _w("")
    _w("The system is designed for Human-Centric Industry 5.0 settings, "
       "where non-intrusive vision-based monitoring can screen for risky "
       "postures without wearable sensors. The modular architecture allows "
       "it to be integrated into existing camera infrastructure.  ")
    _w("")
    _w("The framework should be interpreted as a **postural risk screening ")
    _w("tool** rather than a clinical-grade RULA replacement.  For a complete ")
    _w("ergonomic assessment, manual contextual factors (force, repetition, ")
    _w("duration) should be incorporated by an ergonomics expert.  ")
    _w("")

    # ---- 7. Limitations ----
    _w("## Limitations")
    _w("")
    _w("This framework should be interpreted as an **ergonomic risk screening tool** "
       "rather than a clinical-grade RULA replacement.  The following limitations apply:")
    _w("")
    _w("### 1. No Full RULA Compliance")
    _w("")
    _w("The proposed method does **not** implement full clinical RULA.  It provides ")
    _w("a vision-based RULA-inspired postural risk screening using joint-angle severity, ")
    _w("feature weighting, and temporal smoothing.  Specifically, it does **not** estimate:  ")
    _w("")
    _w("- **Force / load handling** — not inferable from vision alone  ")
    _w("- **Muscle use / static posture load** — requires electromyography or manual input  ")
    _w("- **Repetition / recovery** — requires temporal activity analysis  ")
    _w("- **Wrist posture** (flexion / deviation / rotation) — not reliably estimable from 2D  ")
    _w("- **Action Levels** — require contextual factors beyond postural analysis  ")
    _w("")
    _w("### 2. 2D Approximation")
    _w("Keypoints are estimated in 2D image coordinates.  Joint angles are computed ")
    _w("in the image plane and do not reflect true 3D anatomical angles.  Out-of-plane ")
    _w("rotations (e.g., trunk torsion) are not captured.  ")
    _w("")
    _w("### 3. No Ergonomic Expert Validation")
    _w("The risk classifications have not been validated against ground-truth ergonomic ")
    _w("assessments by certified professionals.  The system provides an **approximate** ")
    _w("postural risk estimate.  ")
    _w("")
    _w("### 4. Occlusion and Viewpoint Sensitivity")
    _w("Partially visible persons, self-occlusions, and extreme camera angles lead to ")
    _w("missing keypoints and reduced pose confidence.  The continuous confidence score ")
    _w("flags these cases without discarding the risk estimate.  ")
    _w("")
    _w("### 5. Single-View Limitation")
    _w("A single camera view may not capture all relevant postural information, ")
    _w("especially for asymmetric or complex industrial tasks.  Multi-view setups ")
    _w("would improve anatomical coverage.  ")
    _w("")

    # ---- 8. Reproducibility ----
    _w("## Reproducibility")
    _w("")
    _w("To reproduce this experiment:")
    _w("")
    _w("1. Place the dataset folder under ``data/<dataset_name>/``  ")
    _w("2. Run the pipeline:  ")
    _w("")
    _w("```python")
    _w("from ergovision.pipeline import ErgoPipeline")
    _w('pipeline = ErgoPipeline()')
    _w('pipeline.run(dataset_name="<dataset_name>")')
    _w("```")
    _w("")
    _w("### Output Structure")
    _w("")
    _w("```")
    _w(f"outputs/{ds_name}/")
    _w("    frames/             # extracted video frames")
    _w("    annotated_frames/   # frames with skeleton + risk overlay")
    _w("    annotated_videos/   # video with risk overlay")
    _w("    csv/                # frame_person_results.csv + video_summary.csv")
    _w("    plots/              # matplotlib figures")
    _w("    examples/           # risk-level example frames")
    _w("    reports/            # this report")
    _w("```")
    _w("")

    _w("### Key Parameters")
    _w("")
    _w("| Parameter | Value |")
    _w("|-----------|-------|")
    _w(f"| Frame sampling FPS | {cfg.frame_sampling_fps} |")
    _w(f"| Max frames / video | {cfg.max_frames_per_video} |")
    _w(f"| Detection confidence | {cfg.detection_confidence} |")
    _w(f"| Keypoint confidence | {cfg.keypoint_confidence} |")
    _w(f"| Pose model | {cfg.__class__.__name__} |")
    _w("")

    _w("---")
    _w(f"*Report generated by ErgoVision v0.2.0 on {_now()}*")
    _w("")

    output_path.write_text('\n'.join(lines), encoding='utf-8')


def _now():
    """Return current local datetime as a short string."""
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d %H:%M')
