#!/usr/bin/env python3
"""
generate_paper_figures.py — Paper-ready experimental figures and tables for ErgoVision.

Reads the CSV outputs from an ErgoVision pipeline run and generates:
  - Summary tables (CSV + Markdown)
  - Risk distribution charts
  - Per-video and per-workstation risk breakdowns
  - Angle distribution plots
  - Temporal risk evolution timelines
  - Pose quality metrics
  - Qualitative example indices
  - Failure case indices
  - Manual validation template
  - Paper-ready experimental summary report

Usage
-----
  python scripts/generate_paper_figures.py --dataset <dataset_name>
  python scripts/generate_paper_figures.py --input <frame_csv> --output <out_dir>

Dependencies
------------
  pandas, matplotlib (no seaborn, no heavy dependencies)
"""

import argparse
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Try pandas; give a clear error if missing ──────────────────────────
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required.  Install with:  pip install pandas")
    sys.exit(1)


# ======================================================================
# Helpers
# ======================================================================

def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _col(df, *names):
    """Return the first column from *names that exists in *df, or None."""
    for n in names:
        if n in df.columns:
            return n
    return None


def _safe_int(v):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _to_score(v):
    """Convert risk_score or action_level to 1/2/3."""
    s = _safe_int(v)
    return s if s in (1, 2, 3) else None


def _level_label(s):
    """Map 1/2/3 to short label."""
    return {1: 'LOW', 2: 'MEDIUM', 3: 'HIGH'}.get(s, 'UNKNOWN')


_RISK_COLORS = {'LOW': '#2ecc71', 'MEDIUM': '#f1c40f', 'HIGH': '#e74c3c',
                'UNKNOWN': '#95a5a6'}
_RISK_ORDER = ['LOW', 'MEDIUM', 'HIGH', 'UNKNOWN']


# ======================================================================
# 1. Dataset / Experiment Summary
# ======================================================================

def generate_dataset_summary(df, vdf, out_dir):
    """Write paper_table_dataset_summary.csv and .md."""
    total = len(df)
    risk_col = _col(df, 'risk_level', 'risk_class', 'final_risk_class')
    score_col = _col(df, 'risk_score', 'action_level')

    low = med = high = 0
    if score_col is not None:
        scores = df[score_col].apply(_to_score)
        low = (scores == 1).sum()
        med = (scores == 2).sum()
        high = (scores == 3).sum()
    elif risk_col is not None:
        rl = df[risk_col].astype(str).str.upper()
        for lv in ['LOW', 'MEDIUM', 'HIGH']:
            cnt = rl.str.contains(lv, na=False).sum()
            if lv == 'LOW':
                low = cnt
            elif lv == 'MEDIUM':
                med = cnt
            else:
                high = cnt

    n_videos = 0
    if vdf is not None and not vdf.empty:
        n_videos = len(vdf)
    elif _col(df, 'video_id'):
        n_videos = df['video_id'].nunique()

    kp_col = _col(df, 'mean_keypoint_confidence', 'keypoint_confidence')
    mean_kp = df[kp_col].mean() if kp_col and kp_col in df else None

    discarded = 0
    disc_col = _col(df, 'discarded')
    if disc_col:
        discarded = df[disc_col].astype(str).str.lower().isin(['true', '1', 'yes']).sum()

    fps_col = _col(df, 'fps_effective')
    fps = None
    if fps_col is None and vdf is not None and 'fps_effective' in vdf.columns:
        fps = vdf['fps_effective'].iloc[0]

    total_det = sum(_safe_int(v) or 0 for v in df.get('detection_confidence', [])
                    if _safe_float(v) is not None and _safe_float(v) > 0) if 'detection_confidence' in df.columns else ''

    rows = [
        ('Videos processed', n_videos),
        ('Frames processed', total),
        ('Valid postures', total - discarded),
        ('Discarded postures', discarded),
        ('LOW count', low),
        ('MEDIUM count', med),
        ('HIGH count', high),
        ('LOW %', round(low / max(total, 1) * 100, 1)),
        ('MEDIUM %', round(med / max(total, 1) * 100, 1)),
        ('HIGH %', round(high / max(total, 1) * 100, 1)),
        ('Mean keypoint confidence', round(mean_kp, 4) if mean_kp is not None else 'N/A'),
    ]
    if fps is not None:
        rows.append(('Effective FPS', round(fps, 2)))

    out_csv = Path(_ensure_dir(out_dir)) / 'paper_table_dataset_summary.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        f.write('Metric,Value\n')
        for k, v in rows:
            f.write(f'{k},{v}\n')

    out_md = out_csv.with_suffix('.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('## Dataset / Experiment Summary\n\n')
        f.write('| Metric | Value |\n|---|---|\n')
        for k, v in rows:
            f.write(f'| {k} | {v} |\n')

    return out_csv, out_md


# ======================================================================
# 2. Risk Distribution
# ======================================================================

def _get_risk_counts(df, score_col, risk_col):
    """Return Counter of LOW/MEDIUM/HIGH counts."""
    counts = Counter()
    if score_col is not None:
        for s in df[score_col].apply(_to_score):
            counts[_level_label(s)] += 1
    elif risk_col is not None:
        rl = df[risk_col].astype(str).str.upper()
        for lv in ['LOW', 'MEDIUM', 'HIGH']:
            counts[lv] = rl.str.contains(lv, na=False).sum()
    return counts


def plot_risk_distribution(df, out_dir, score_col, risk_col):
    """fig_risk_distribution.png"""
    counts = _get_risk_counts(df, score_col, risk_col)
    levels = [l for l in _RISK_ORDER if counts.get(l, 0) > 0 or l != 'UNKNOWN']
    values = [counts.get(l, 0) for l in levels]
    colors = [_RISK_COLORS.get(l, '#95a5a6') for l in levels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(levels, values, color=colors, edgecolor='gray', width=0.6)
    ax.set_ylabel('Count')
    ax.set_title('Postural Risk Distribution')
    for bar, v in zip(bars, values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                    str(v), ha='center', va='bottom', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = Path(out_dir) / 'fig_risk_distribution.png'
    plt.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close()
    return out


# ======================================================================
# 3. Risk Distribution per Video
# ======================================================================

def plot_risk_by_video(df, out_dir, score_col, risk_col):
    """fig_risk_distribution_by_video.png"""
    vid_col = _col(df, 'video_id')
    if vid_col is None:
        return None

    video_risk = {}
    for vid, grp in df.groupby(vid_col):
        counts = _get_risk_counts(grp, score_col, risk_col)
        video_risk[vid] = counts

    videos = sorted(video_risk.keys())
    if not videos:
        return None

    low = [video_risk[v].get('LOW', 0) for v in videos]
    med = [video_risk[v].get('MEDIUM', 0) for v in videos]
    high = [video_risk[v].get('HIGH', 0) for v in videos]

    fig, ax = plt.subplots(figsize=(max(8, len(videos) * 0.6), 5))
    x = np.arange(len(videos))
    width = 0.6
    ax.bar(x, low, width, label='LOW', color=_RISK_COLORS['LOW'], edgecolor='gray')
    ax.bar(x, med, width, bottom=low, label='MEDIUM', color=_RISK_COLORS['MEDIUM'], edgecolor='gray')
    ax.bar(x, high, width, bottom=[l + m for l, m in zip(low, med)],
           label='HIGH', color=_RISK_COLORS['HIGH'], edgecolor='gray')

    ax.set_ylabel('Count')
    ax.set_title('Risk Distribution by Video')
    ax.set_xticks(x)
    ax.set_xticklabels([str(v)[:25] for v in videos], rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = Path(out_dir) / 'fig_risk_distribution_by_video.png'
    plt.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close()
    return out


# ======================================================================
# 4. Risk Distribution per Workstation
# ======================================================================

def _extract_workstation(video_id):
    """Extract workstation code (WS10, WS20, WS30) from video_id."""
    m = re.search(r'(WS\d+)', str(video_id), re.IGNORECASE)
    return m.group(1).upper() if m else None


def plot_risk_by_workstation(df, out_dir, score_col, risk_col):
    """fig_risk_distribution_by_workstation.png"""
    vid_col = _col(df, 'video_id')
    if vid_col is None:
        return None

    df = df.copy()
    df['_ws'] = df[vid_col].apply(_extract_workstation)
    ws_df = df[df['_ws'].notna()]
    if ws_df.empty:
        return None

    ws_risk = {}
    for ws, grp in ws_df.groupby('_ws'):
        counts = _get_risk_counts(grp, score_col, risk_col)
        ws_risk[ws] = counts

    workstations = sorted(ws_risk.keys())
    low = [ws_risk[w].get('LOW', 0) for w in workstations]
    med = [ws_risk[w].get('MEDIUM', 0) for w in workstations]
    high = [ws_risk[w].get('HIGH', 0) for w in workstations]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(workstations))
    width = 0.5
    ax.bar(x, low, width, label='LOW', color=_RISK_COLORS['LOW'], edgecolor='gray')
    ax.bar(x, med, width, bottom=low, label='MEDIUM', color=_RISK_COLORS['MEDIUM'], edgecolor='gray')
    ax.bar(x, high, width, bottom=[l + m for l, m in zip(low, med)],
           label='HIGH', color=_RISK_COLORS['HIGH'], edgecolor='gray')

    ax.set_ylabel('Count')
    ax.set_xlabel('Workstation')
    ax.set_title('Risk Distribution by Workstation')
    ax.set_xticks(x)
    ax.set_xticklabels(workstations)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = Path(out_dir) / 'fig_risk_distribution_by_workstation.png'
    plt.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close()
    return out


# ======================================================================
# 5. Angle Distributions (violin / boxplot)
# ======================================================================

def plot_angle_distributions(df, out_dir):
    """fig_angle_distributions.png"""
    angle_cols = [c for c in ['trunk_angle', 'neck_angle',
                               'upper_arm_angle_left', 'upper_arm_angle_right',
                               'forearm_angle_left', 'forearm_angle_right',
                               'knee_angle_left', 'knee_angle_right',
                               'body_inclination']
                  if c in df.columns and df[c].notna().sum() > 10]

    if not angle_cols:
        return None

    data = [df[c].dropna().astype(float).clip(0, 180).values for c in angle_cols]
    labels = [c.replace('_', ' ').title() for c in angle_cols]

    fig, ax = plt.subplots(figsize=(max(10, len(angle_cols) * 1.2), 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=4))
    for patch, col in zip(bp['boxes'], ['#3498db'] * len(angle_cols)):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Joint Angle Distributions')
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = Path(out_dir) / 'fig_angle_distributions.png'
    plt.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close()
    return out


# ======================================================================
# 6. Temporal Risk Evolution
# ======================================================================

def plot_timelines(df, out_dir, score_col, risk_col):
    """Per-video timelines + representative aggregate."""
    vid_col = _col(df, 'video_id')
    time_col = _col(df, 'timestamp_sec', 'frame_id')
    if vid_col is None or time_col is None:
        return None, None

    timelines_dir = _ensure_dir(os.path.join(out_dir, 'timelines'))
    generated = []
    max_transitions = -1
    rep_video = None
    rep_scores = None
    rep_times = None

    for vid, grp in df.groupby(vid_col):
        grp = grp.sort_values(time_col)
        times = grp[time_col].values
        scores = grp[score_col].apply(_to_score).values if score_col else None
        if scores is None and risk_col:
            scores = np.array([_to_score(s) if s in (1, 2, 3) else None
                               for s in grp[risk_col].apply(_safe_int)])

        if scores is None or len(scores) < 3:
            continue

        fig, ax = plt.subplots(figsize=(10, 3))
        colors = [_RISK_COLORS[_level_label(s)] for s in scores]
        ax.scatter(range(len(scores)), scores, c=colors, s=8, alpha=0.7)
        ax.plot(range(len(scores)), scores, color='gray', linewidth=0.5, alpha=0.5)
        ax.set_ylabel('Risk Score')
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['LOW', 'MEDIUM', 'HIGH'])
        ax.set_xlabel('Frame')
        ax.set_title(f'Temporal Risk Evolution — {str(vid)[:50]}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        safe_vid = re.sub(r'[^a-zA-Z0-9_-]', '_', str(vid))[:60]
        out = Path(timelines_dir) / f'fig_timeline_{safe_vid}.png'
        plt.savefig(str(out), dpi=300, bbox_inches='tight')
        plt.close()
        generated.append(out)

        # Count transitions for representative selection
        n_trans = sum(1 for i in range(1, len(scores))
                      if scores[i] is not None and scores[i - 1] is not None
                      and scores[i] != scores[i - 1])
        if n_trans > max_transitions:
            max_transitions = n_trans
            rep_video = vid
            rep_scores = scores
            rep_times = times

    # Representative
    rep_out = None
    if rep_scores is not None and len(rep_scores) >= 3:
        fig, ax = plt.subplots(figsize=(10, 3))
        colors = [_RISK_COLORS[_level_label(s)] for s in rep_scores]
        ax.scatter(range(len(rep_scores)), rep_scores, c=colors, s=10, alpha=0.7)
        ax.plot(range(len(rep_scores)), rep_scores, color='gray', linewidth=0.5, alpha=0.5)
        ax.set_ylabel('Risk Score')
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['LOW', 'MEDIUM', 'HIGH'])
        ax.set_xlabel('Frame')
        ax.set_title(f'Temporal Risk Evolution (representative: {str(rep_video)[:50]})')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        rep_out = Path(out_dir) / 'fig_temporal_risk_evolution_representative.png'
        plt.savefig(str(rep_out), dpi=300, bbox_inches='tight')
        plt.close()

    return generated, rep_out


# ======================================================================
# 7. Pose Quality Metrics
# ======================================================================

def plot_pose_quality(df, out_dir):
    """fig_pose_quality_confidence.png, fig_valid_keypoints_distribution.png, fig_pose_acceptance_summary.png"""
    generated = []

    # Confidence distribution
    conf_col = _col(df, 'mean_keypoint_confidence', 'pose_confidence', 'confidence')
    if conf_col and conf_col in df.columns:
        vals = df[conf_col].dropna().astype(float)
        if len(vals) > 10:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(vals, bins=40, color='#9b59b6', edgecolor='white', alpha=0.85)
            ax.axvline(vals.mean(), color='red', linestyle='--', linewidth=1.5,
                       label=f'Mean: {vals.mean():.2f}')
            ax.set_xlabel('Mean Keypoint Confidence')
            ax.set_ylabel('Frequency')
            ax.set_title('Keypoint Confidence Distribution')
            ax.legend(fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            out = Path(out_dir) / 'fig_pose_quality_confidence.png'
            plt.savefig(str(out), dpi=300, bbox_inches='tight')
            plt.close()
            generated.append(out)

    # Valid keypoints distribution
    kp_col = _col(df, 'valid_keypoints_count', 'valid_keypoints')
    if kp_col and kp_col in df.columns:
        vals = df[kp_col].dropna().astype(int)
        if len(vals) > 5:
            counts = Counter(vals)
            x = sorted(counts.keys())
            y = [counts[k] for k in x]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x, y, color='#3498db', edgecolor='gray', width=0.6)
            ax.set_xlabel('Valid Keypoints')
            ax.set_ylabel('Number of Frames')
            ax.set_title('Valid Keypoints Distribution')
            ax.set_xticks(x)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            out = Path(out_dir) / 'fig_valid_keypoints_distribution.png'
            plt.savefig(str(out), dpi=300, bbox_inches='tight')
            plt.close()
            generated.append(out)

    # Acceptance summary
    disc_col = _col(df, 'discarded')
    if disc_col:
        accepted = (~df[disc_col].astype(str).str.lower().isin(['true', '1', 'yes'])).sum()
        rejected = df[disc_col].astype(str).str.lower().isin(['true', '1', 'yes']).sum()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(['Accepted', 'Discarded'], [accepted, rejected],
               color=['#2ecc71', '#e74c3c'], edgecolor='gray', width=0.5)
        ax.set_ylabel('Count')
        ax.set_title('Pose Acceptance Summary')
        for i, v in enumerate([accepted, rejected]):
            ax.text(i, v + max(accepted, rejected) * 0.01, str(v),
                    ha='center', va='bottom', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        out = Path(out_dir) / 'fig_pose_acceptance_summary.png'
        plt.savefig(str(out), dpi=300, bbox_inches='tight')
        plt.close()
        generated.append(out)

    return generated


# ======================================================================
# 8. Qualitative Examples Index
# ======================================================================

def generate_examples_index(df, out_dir, score_col, risk_col, n_per=3):
    """paper_examples_index.csv and .md"""
    vid_col = _col(df, 'video_id')
    time_col = _col(df, 'timestamp_sec', 'frame_id')
    frame_col = _col(df, 'frame_id')
    annot_col = _col(df, 'annotated_frame_path', 'annotated_image', '_annotated_frame_path')
    reason_col = _col(df, 'action_level_reason', 'explanation', 'high_reason', 'medium_reason')

    rows = []
    for level in ['LOW', 'MEDIUM', 'HIGH']:
        subset = df[df['risk_level'].astype(str).str.contains(level, na=False)]
        if subset.empty:
            continue
        for _, r in subset.head(n_per).iterrows():
            row = {
                'video_id': r.get(vid_col, ''),
                'frame_id': r.get(frame_col, ''),
                'timestamp_sec': r.get(time_col, ''),
                'risk_level': level,
                'risk_score': r.get(score_col, ''),
            }
            if reason_col:
                row['reason'] = str(r.get(reason_col, ''))[:120]
            row['trunk_angle'] = r.get('trunk_angle', '')
            row['neck_angle'] = r.get('neck_angle', '')
            ua = []
            for c in ['upper_arm_angle_left', 'upper_arm_angle_right']:
                v = _safe_float(r.get(c))
                if v is not None:
                    ua.append(v)
            row['upper_arm_max'] = max(ua) if ua else ''
            if annot_col and annot_col in df.columns:
                row['image_path'] = r.get(annot_col, '')
            rows.append(row)

    if not rows:
        return None, None

    out_csv = Path(_ensure_dir(out_dir)) / 'paper_examples_index.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8')

    out_md = out_csv.with_suffix('.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('## Qualitative Examples Index\n\n')
        f.write('| Video ID | Frame | Risk Level | Score | Trunk | Neck | Upper Arm Max | Reason |\n')
        f.write('|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            f.write(f"| {r.get('video_id','')} | {r.get('frame_id','')} "
                    f"| {r.get('risk_level','')} | {r.get('risk_score','')} "
                    f"| {r.get('trunk_angle','')} | {r.get('neck_angle','')} "
                    f"| {r.get('upper_arm_max','')} "
                    f"| {str(r.get('reason',''))[:60]} |\n")

    return out_csv, out_md


# ======================================================================
# 9. Failure Cases Index
# ======================================================================

def generate_failure_index(df, out_dir):
    """paper_failure_cases_index.csv and .md"""
    # Collect rows that are discarded, uncertain, low confidence, or mapping-inconsistent
    disc_col = _col(df, 'discarded')
    uncert_col = _col(df, 'uncertain', 'uncertainty')
    conf_col = _col(df, 'pose_confidence', 'confidence', 'mean_keypoint_confidence')
    reason_col = _col(df, 'discard_reason', 'uncertainty_reason', 'action_level_reason')
    mapping_col = _col(df, 'mapping_consistent')
    vid_col = _col(df, 'video_id')
    frame_col = _col(df, 'frame_id')
    annot_col = _col(df, 'annotated_frame_path', '_annotated_frame_path')

    mask = pd.Series([False] * len(df))
    reasons = {}

    if disc_col:
        d = df[disc_col].astype(str).str.lower().isin(['true', '1', 'yes'])
        mask |= d
        for idx in df[d].index:
            r = str(df.loc[idx, reason_col])[:100] if reason_col and reason_col in df.columns else 'discarded'
            reasons[idx] = r

    if uncert_col:
        u = df[uncert_col].astype(str).str.lower().isin(['true', '1', 'yes'])
        mask |= u
        for idx in df[u].index:
            r = str(df.loc[idx, reason_col])[:100] if reason_col and reason_col in df.columns else 'uncertain'
            reasons[idx] = r

    if conf_col and conf_col in df.columns:
        low_conf = df[conf_col].astype(float) < 0.3
        mask |= low_conf
        for idx in df[low_conf].index:
            if idx not in reasons:
                reasons[idx] = f'low_confidence: {df.loc[idx, conf_col]:.2f}'

    if mapping_col and mapping_col in df.columns:
        inc = df[mapping_col].astype(str).str.lower() == 'false'
        mask |= inc
        for idx in df[inc].index:
            reasons[idx] = 'mapping_inconsistent'

    failures = df[mask].copy()
    if failures.empty:
        return None, None

    rows = []
    for idx, r in failures.iterrows():
        rows.append({
            'video_id': r.get(vid_col, ''),
            'frame_id': r.get(frame_col, ''),
            'reason': reasons.get(idx, 'unknown'),
            'confidence': r.get(conf_col, '') if conf_col else '',
            'image_path': r.get(annot_col, '') if annot_col else '',
        })

    out_csv = Path(_ensure_dir(out_dir)) / 'paper_failure_cases_index.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8')

    out_md = out_csv.with_suffix('.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('## Failure Cases Index\n\n')
        f.write('| Video ID | Frame | Reason | Confidence |\n|---|---|---|---|\n')
        for r in rows:
            f.write(f"| {r['video_id']} | {r['frame_id']} | {r['reason']} | {r['confidence']} |\n")

    return out_csv, out_md


# ======================================================================
# 10. Manual Validation Template
# ======================================================================

def generate_validation_template(df, out_dir, score_col, risk_col, total_samples=100):
    """manual_validation_template.csv with balanced sampling."""
    vid_col = _col(df, 'video_id')
    frame_col = _col(df, 'frame_id')
    time_col = _col(df, 'timestamp_sec', 'frame_id')
    annot_col = _col(df, 'annotated_frame_path', '_annotated_frame_path')

    # Separate risk levels
    low_df = df[df[risk_col].astype(str).str.contains('LOW', na=False)] if risk_col else pd.DataFrame()
    med_df = df[df[risk_col].astype(str).str.contains('MEDIUM', na=False)] if risk_col else pd.DataFrame()
    high_df = df[df[risk_col].astype(str).str.contains('HIGH', na=False)] if risk_col else pd.DataFrame()

    # Fall back to score column
    if low_df.empty and score_col:
        scores = df[score_col].apply(_to_score)
        low_df = df[scores == 1]
        med_df = df[scores == 2]
        high_df = df[scores == 3]

    n_high = min(20, len(high_df), total_samples // 5)
    n_low = min(40, len(low_df), (total_samples - n_high) // 2)
    n_med = min(40, len(med_df), total_samples - n_high - n_low)

    # Adjust if insufficient samples
    if len(low_df) < n_low:
        n_med = min(n_med + (n_low - len(low_df)), len(med_df))
        n_low = len(low_df)
    if len(med_df) < n_med:
        n_high = min(n_high + (n_med - len(med_df)), len(high_df))
        n_med = len(med_df)
    n_total = n_low + n_med + n_high

    samples = []
    for i, (sub, level) in enumerate([(low_df, 'LOW'), (med_df, 'MEDIUM'), (high_df, 'HIGH')]):
        n = [n_low, n_med, n_high][i]
        if n == 0 or sub.empty:
            continue
        chosen = sub.sample(n=min(n, len(sub)), random_state=42)
        for _, r in chosen.iterrows():
            samples.append({
                'sample_id': len(samples) + 1,
                'video_id': r.get(vid_col, ''),
                'frame_id': r.get(frame_col, ''),
                'timestamp_sec': r.get(time_col, ''),
                'predicted_risk_level': level,
                'predicted_risk_score': _to_score(r.get(score_col)) if score_col else '',
                'annotated_image_path': r.get(annot_col, '') if annot_col else '',
                'human_label': '',
                'notes': '',
            })

    out_csv = Path(_ensure_dir(out_dir)) / 'manual_validation_template.csv'
    pd.DataFrame(samples).to_csv(out_csv, index=False, encoding='utf-8')
    return out_csv


# ======================================================================
# 11. Paper-ready Summary Report
# ======================================================================

def generate_paper_report(summary_csv, figures, tables, warnings_list, out_dir):
    """experimental_results_summary.md"""
    # Read summary
    summary = {}
    if summary_csv and os.path.exists(summary_csv):
        with open(summary_csv, encoding='utf-8') as f:
            for line in f:
                if ',' in line:
                    k, _, v = line.strip().partition(',')
                    summary[k.strip()] = v.strip()

    out = Path(_ensure_dir(out_dir)) / 'experimental_results_summary.md'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('# Experimental Results Summary\n\n')
        f.write(f'*Generated by ErgoVision paper figures script*\n\n')

        f.write('## Dataset Summary\n\n')
        f.write('| Metric | Value |\n|---|---|\n')
        for k, v in summary.items():
            f.write(f'| {k} | {v} |\n')

        f.write('\n## Risk Distribution\n\n')
        for k in ['LOW count', 'MEDIUM count', 'HIGH count',
                  'LOW %', 'MEDIUM %', 'HIGH %']:
            if k in summary:
                f.write(f'- **{k}**: {summary[k]}\n')

        f.write('\n## Observations\n\n')
        high_pct = _safe_float(summary.get('HIGH %', '0'))
        if high_pct is not None and high_pct > 50:
            f.write(f'- The majority of postures ({high_pct:.0f}%) were classified as HIGH, '
                    f'indicating sustained severe postural load across the dataset.\n')
        elif high_pct is not None and high_pct > 20:
            f.write(f'- A significant proportion of postures ({high_pct:.0f}%) were classified as HIGH, '
                    f'warranting ergonomic investigation.\n')
        else:
            f.write(f'- LOW and MEDIUM risk postures dominate the dataset.\n')

        low_pct = _safe_float(summary.get('LOW %', '0'))
        if low_pct is not None and low_pct > 70:
            f.write('- Most postures are LOW risk, indicating predominantly neutral postural alignment.\n')

        if 'Mean keypoint confidence' in summary:
            f.write(f'- Mean keypoint confidence: {summary["Mean keypoint confidence"]}.\n')

        if summary.get('Discarded postures', '0') != '0':
            f.write(f'- {summary["Discarded postures"]} postures were discarded '
                    f'(insufficient keypoints for reliable scoring).\n')

        f.write('\n## Generated Figures\n\n')
        for fpath in sorted(figures):
            f.write(f'- `{os.path.relpath(fpath, out_dir)}`\n')

        f.write('\n## Generated Tables\n\n')
        for tpath in sorted(tables):
            f.write(f'- `{os.path.relpath(tpath, out_dir)}`\n')

        f.write('\n## Limitations\n\n')
        f.write('- These results represent the **observable postural component** only.\n')
        f.write('- Force, load, repetition, and muscle use are NOT automatically inferred.\n')
        f.write('- The system is RULA-inspired, not full clinical RULA.\n')
        f.write('- 2D joint angles are approximations of true 3D anatomical angles.\n')

        if warnings_list:
            f.write('\n## Warnings\n\n')
            for w in warnings_list:
                f.write(f'- {w}\n')

    return out


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate paper-ready figures and tables from ErgoVision CSV outputs.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (looks for outputs/<dataset>/csv/)')
    group.add_argument('--input', type=str, default=None,
                       help='Path to frame_person_results.csv')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: outputs/<dataset>/paper_figures)')
    parser.add_argument('--video-summary', type=str, default=None,
                       help='Path to video_summary.csv (optional)')
    args = parser.parse_args()

    # Resolve input / output paths
    if args.dataset:
        base = Path('outputs') / args.dataset
        frame_csv = base / 'csv' / 'frame_person_results.csv'
        video_summary_csv = args.video_summary or (base / 'csv' / 'video_summary.csv')
        out_dir = args.output or (base / 'paper_figures')
    elif args.input:
        frame_csv = Path(args.input)
        video_summary_csv = args.video_summary or frame_csv.parent / 'video_summary.csv'
        out_dir = Path(args.output or (frame_csv.parent.parent / 'paper_figures'))
    else:
        # Try default
        for ds in ['mp4', 'default']:
            p = Path('outputs') / ds / 'csv' / 'frame_person_results.csv'
            if p.exists():
                frame_csv = p
                video_summary_csv = p.parent / 'video_summary.csv'
                out_dir = p.parent.parent / 'paper_figures'
                break
        else:
            parser.print_help()
            print('\nERROR: no dataset specified and no default CSV found.')
            print('Provide --dataset or --input.')
            sys.exit(1)

    if not frame_csv.exists():
        print(f'ERROR: frame CSV not found: {frame_csv}')
        sys.exit(1)

    _ensure_dir(out_dir)
    warnings_list = []
    figures = []
    tables = []

    print(f'Reading: {frame_csv}')
    df = pd.read_csv(frame_csv, encoding='utf-8', low_memory=False)
    print(f'  Rows: {len(df)}')
    print(f'  Columns: {list(df.columns)}')

    # Detect score and risk columns
    score_col = _col(df, 'risk_score', 'action_level', 'approximate_action_level', 'final_score')
    risk_col = _col(df, 'risk_level', 'final_risk_class', 'risk_class')
    if score_col:
        print(f'  Using score column: {score_col}')
    if risk_col:
        print(f'  Using risk column: {risk_col}')

    # Load video summary
    vdf = None
    if video_summary_csv and os.path.exists(video_summary_csv):
        vdf = pd.read_csv(video_summary_csv, encoding='utf-8', low_memory=False)
        print(f'  Video summary: {len(vdf)} videos')
    else:
        warnings_list.append(f'Video summary not found: {video_summary_csv}')

    # 1. Dataset summary
    try:
        s_csv, s_md = generate_dataset_summary(df, vdf, out_dir)
        tables.extend([s_csv, s_md])
        print(f'  [1/11] Dataset summary → {s_csv.name}')
    except Exception as e:
        warnings_list.append(f'Dataset summary failed: {e}')

    # 2. Risk distribution
    try:
        f = plot_risk_distribution(df, out_dir, score_col, risk_col)
        if f:
            figures.append(f)
            print(f'  [2/11] Risk distribution → {f.name}')
    except Exception as e:
        warnings_list.append(f'Risk distribution failed: {e}')

    # 3. Risk by video
    try:
        f = plot_risk_by_video(df, out_dir, score_col, risk_col)
        if f:
            figures.append(f)
            print(f'  [3/11] Risk by video → {f.name}')
    except Exception as e:
        warnings_list.append(f'Risk by video failed: {e}')

    # 4. Risk by workstation
    try:
        f = plot_risk_by_workstation(df, out_dir, score_col, risk_col)
        if f:
            figures.append(f)
            print(f'  [4/11] Risk by workstation → {f.name}')
    except Exception as e:
        warnings_list.append(f'Risk by workstation failed: {e}')

    # 5. Angle distributions
    try:
        f = plot_angle_distributions(df, out_dir)
        if f:
            figures.append(f)
            print(f'  [5/11] Angle distributions → {f.name}')
    except Exception as e:
        warnings_list.append(f'Angle distributions failed: {e}')

    # 6. Temporal timelines
    try:
        timelines, rep = plot_timelines(df, out_dir, score_col, risk_col)
        if timelines:
            figures.extend(timelines)
            print(f'  [6/11] {len(timelines)} timeline(s)')
        if rep:
            figures.append(rep)
            print(f'         Representative → {rep.name}')
    except Exception as e:
        warnings_list.append(f'Temporal timelines failed: {e}')

    # 7. Pose quality
    try:
        pq = plot_pose_quality(df, out_dir)
        if pq:
            figures.extend(pq)
            print(f'  [7/11] {len(pq)} pose quality figure(s)')
    except Exception as e:
        warnings_list.append(f'Pose quality failed: {e}')

    # 8. Examples index
    try:
        ex_csv, ex_md = generate_examples_index(df, out_dir, score_col, risk_col)
        if ex_csv:
            tables.extend([ex_csv, ex_md])
            print(f'  [8/11] Examples index → {ex_csv.name}')
    except Exception as e:
        warnings_list.append(f'Examples index failed: {e}')

    # 9. Failure cases index
    try:
        fc_csv, fc_md = generate_failure_index(df, out_dir)
        if fc_csv:
            tables.extend([fc_csv, fc_md])
            print(f'  [9/11] Failure cases index → {fc_csv.name}')
    except Exception as e:
        warnings_list.append(f'Failure index failed: {e}')

    # 10. Validation template
    try:
        vt = generate_validation_template(df, out_dir, score_col, risk_col)
        if vt:
            tables.append(vt)
            print(f' [10/11] Validation template → {vt.name}')
    except Exception as e:
        warnings_list.append(f'Validation template failed: {e}')

    # 11. Paper report
    try:
        s_csv_path = s_csv if 's_csv' in dir() else None
        report = generate_paper_report(s_csv_path, figures, tables, warnings_list, out_dir)
        tables.append(report)
        print(f' [11/11] Paper report → {report.name}')
    except Exception as e:
        warnings_list.append(f'Paper report failed: {e}')

    # Summary
    print(f'\n{"=" * 50}')
    print(f'  Paper generation complete.')
    print(f'  Output: {out_dir.resolve()}')
    print(f'  Figures:   {len(figures)}')
    print(f'  Tables:    {len(tables)}')
    if warnings_list:
        print(f'  Warnings:  {len(warnings_list)}')
        for w in warnings_list:
            print(f'    - {w}')
    print(f'{"=" * 50}')


if __name__ == '__main__':
    main()
