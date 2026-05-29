"""
Statistical analysis and visualisation for ErgoVision experiments.

All plots use **matplotlib only** (no seaborn) so no extra dependencies.
"""

import numpy as np
from pathlib import Path
from collections import Counter, OrderedDict

# matplotlib imported lazily in each plotting function so the module can
# be imported in headless environments that do not have a display backend.


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _safe_stat(values):
    """Return common descriptive statistics for a 1-D array-like."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {}
    return OrderedDict([
        ('mean', float(np.mean(arr))),
        ('std', float(np.std(arr))),
        ('min', float(np.min(arr))),
        ('p25', float(np.percentile(arr, 25))),
        ('p50', float(np.percentile(arr, 50))),
        ('p75', float(np.percentile(arr, 75))),
        ('max', float(np.max(arr))),
    ])


def compute_statistics(rows):
    """Compute aggregate experimental statistics from frame-person results.

    Parameters
    ----------
    rows : list[dict]
        Each dict is one row of *frame_person_results.csv*.

    Returns
    -------
    dict with keys: ``angle_statistics``, ``risk_distribution``,
    ``total_postures``, ``valid_postures``, ``discarded_postures``,
    ``people_per_frame``, ``keypoint_confidence``.
    """
    # Convert to numpy arrays for numeric columns
    angle_cols = [
        'trunk_angle', 'neck_angle',
        'upper_arm_angle_left', 'upper_arm_angle_right',
        'forearm_angle_left', 'forearm_angle_right',
        'knee_angle_left', 'knee_angle_right',
        'shoulder_asymmetry', 'body_inclination',
    ]
    angle_stats = {}
    for col in angle_cols:
        vals = []
        for r in rows:
            v = r.get(col)
            if v is not None and v != '' and v != 'None':
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        if vals:
            angle_stats[col] = _safe_stat(vals)

    # Risk distribution (from numeric risk_score — robust to label changes)
    risk_scores = []
    for r in rows:
        s = r.get('risk_score')
        if s is not None and str(s).strip() and str(s) != 'None':
            try:
                risk_scores.append(int(s))
            except (ValueError, TypeError):
                pass
    risk_counts = Counter(risk_scores)
    total = len(rows)
    risk_dist = {}
    for score, label in [(1, 'LOW'), (2, 'MEDIUM'), (3, 'HIGH')]:
        cnt = risk_counts.get(score, 0)
        risk_dist[label] = {
            'count': cnt,
            'percentage': round((cnt / total * 100) if total > 0 else 0, 2),
        }

    # Keypoint confidence
    kp_confs = []
    for r in rows:
        v = r.get('mean_keypoint_confidence')
        if v is not None and v != '' and v != 'None':
            try:
                kp_confs.append(float(v))
            except (ValueError, TypeError):
                pass

    # Pose confidence (fuzzy system)
    pose_confs = []
    for r in rows:
        v = r.get('pose_confidence')
        if v is not None and v != '' and v != 'None':
            try:
                pose_confs.append(float(v))
            except (ValueError, TypeError):
                pass

    # People per frame (unique per video+frame combo)
    people_per_frame = {}
    for r in rows:
        key = f"{r.get('video_id', '')}_{r.get('frame_id', '')}"
        people_per_frame[key] = people_per_frame.get(key, 0) + 1

    stats = {
        'angle_statistics': angle_stats,
        'risk_distribution': risk_dist,
        'total_postures': total,
        'valid_postures': sum(
            1 for r in rows
            if r.get('discarded') in (False, 'False', 'false', '')
        ),
        'discarded_postures': sum(
            1 for r in rows
            if r.get('discarded') in (True, 'True', 'true')
        ),
        'mean_keypoint_confidence': (
            float(np.mean(kp_confs)) if kp_confs else 0.0
        ),
        'mean_pose_confidence': (
            float(np.mean(pose_confs)) if pose_confs else 0.0
        ),
        'people_per_frame_summary': _safe_stat(list(people_per_frame.values())),
    }

    return stats


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _import_plt():
    """Lazy-import matplotlib pyplot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_risk_distribution(rows, output_path,
                           title='Risk Level Distribution'):
    """Bar chart of LOW / MEDIUM / HIGH counts."""
    plt = _import_plt()

    # Use numeric risk_score for robustness against label changes
    scores = []
    for r in rows:
        s = r.get('risk_score')
        if s is not None and str(s).strip() and str(s) != 'None':
            try:
                scores.append(int(s))
            except (ValueError, TypeError):
                pass
    sc = Counter(scores)
    levels = ['LOW', 'MEDIUM', 'HIGH']
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    values = [sc.get(1, 0), sc.get(2, 0), sc.get(3, 0)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(levels, values, color=colors, edgecolor='gray', width=0.6)
    ax.set_ylabel('Count')
    ax.set_title(title)
    max_val = max(values) if values else 1
    for bar, v in zip(bars, values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.01,
                    str(v), ha='center', va='bottom', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def plot_risk_by_video(rows, output_path):
    """Stacked bar chart of risk distribution per video."""
    plt = _import_plt()

    video_risk = {}
    for r in rows:
        vid = r.get('video_id', 'unknown')
        raw_level = r.get('risk_level', 'UNKNOWN')
        # Normalise level labels for backward compat
        if raw_level.startswith('AL1'):
            level = 'LOW'
        elif raw_level.startswith('AL2'):
            level = 'MEDIUM'
        elif raw_level.startswith('AL3'):
            level = 'HIGH'
        else:
            level = raw_level
        if vid not in video_risk:
            video_risk[vid] = Counter()
        video_risk[vid][level] += 1

    # Sort videos by name
    videos = sorted(video_risk.keys())
    low = [video_risk[v].get('LOW', 0) for v in videos]
    med = [video_risk[v].get('MEDIUM', 0) for v in videos]
    high = [video_risk[v].get('HIGH', 0) for v in videos]

    fig, ax = plt.subplots(figsize=(max(10, len(videos) * 0.8), 5))
    x = np.arange(len(videos))
    width = 0.6
    ax.bar(x, low, width, label='LOW', color='#2ecc71', edgecolor='gray')
    ax.bar(x, med, width, bottom=low, label='MEDIUM', color='#f1c40f', edgecolor='gray')
    ax.bar(x, high, width, bottom=[l + m for l, m in zip(low, med)],
           label='HIGH', color='#e74c3c', edgecolor='gray')

    ax.set_ylabel('Count')
    ax.set_title('Risk Distribution by Video')
    ax.set_xticks(x)
    ax.set_xticklabels([v[:20] for v in videos], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def plot_histogram(values, output_path, xlabel='Value', title='Distribution',
                   bins=50, color='steelblue'):
    """Single histogram with mean line."""
    plt = _import_plt()

    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(arr, bins=bins, color=color, edgecolor='white', alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.axvline(arr.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f"Mean: {arr.mean():.1f}")
    ax.axvline(arr.mean() - arr.std(), color='orange', linestyle=':',
               linewidth=1.2, label=f"±1 SD")
    ax.axvline(arr.mean() + arr.std(), color='orange', linestyle=':',
               linewidth=1.2)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def plot_people_per_frame(rows, output_path):
    """Bar chart of number of people detected per frame."""
    plt = _import_plt()

    people_count = Counter()
    for r in rows:
        fid = r.get('frame_id', '')
        people_count[fid] += 1

    counts = list(people_count.values())
    if not counts:
        return

    dist = Counter(counts)
    x = sorted(dist.keys())
    y = [dist[k] for k in x]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, y, color='#3498db', edgecolor='gray', width=0.6)
    ax.set_xlabel('People per Frame')
    ax.set_ylabel('Number of Frames')
    ax.set_title('People per Frame Distribution')
    ax.set_xticks(x)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + max(y) * 0.01, str(yi), ha='center', va='bottom',
                fontsize=9)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def plot_keypoint_confidence(rows, output_path):
    """Histogram of mean keypoint confidence across detections."""
    confs = []
    for r in rows:
        v = r.get('mean_keypoint_confidence')
        if v is not None and v != '' and v != 'None':
            try:
                confs.append(float(v))
            except (ValueError, TypeError):
                pass
    if confs:
        plot_histogram(confs, output_path,
                       xlabel='Mean Keypoint Confidence',
                       title='Keypoint Confidence Distribution',
                       bins=30, color='#9b59b6')


def plot_top_risk_frames(rows, output_path, top_n=20):
    """Bar chart of top highest-risk frames."""
    plt = _import_plt()

    scored = []
    for r in rows:
        score = r.get('risk_score')
        fid = r.get('frame_id', '')
        vid = r.get('video_id', '')
        if score is not None and score != '' and score != 'None':
            try:
                scored.append((int(score), fid, vid))
            except (ValueError, TypeError):
                pass

    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:top_n]

    if not top:
        return

    labels = [f"{v}:{f}" for _, f, v in top]
    scores = [s for s, _, _ in top]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#e74c3c' if s >= 3 else '#f1c40f' if s == 2 else '#2ecc71'
              for s in scores]
    ax.barh(range(len(labels)), scores, color=colors, edgecolor='gray')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Risk Score')
    ax.set_title(f'Top {top_n} Highest-Risk Frames')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_angle_histograms(rows, output_dir):
    """Generate histograms for all angle features into *output_dir*."""
    angle_cols = [
        ('trunk_angle', '#2c3e50'),
        ('neck_angle', '#8e44ad'),
        ('upper_arm_angle_left', '#2980b9'),
        ('upper_arm_angle_right', '#2980b9'),
        ('forearm_angle_left', '#27ae60'),
        ('forearm_angle_right', '#27ae60'),
        ('knee_angle_left', '#d35400'),
        ('knee_angle_right', '#d35400'),
        ('shoulder_asymmetry', '#c0392b'),
        ('body_inclination', '#7f8c8d'),
    ]

    output_dir = Path(output_dir)
    for col, color in angle_cols:
        vals = []
        for r in rows:
            v = r.get(col)
            if v is not None and v != '' and v != 'None':
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        if vals:
            plot_path = output_dir / f'{col}_histogram.png'
            plot_histogram(vals, plot_path,
                           xlabel=f'{col} (degrees)',
                           title=f'{col} Distribution',
                           bins=50, color=color)


def generate_all_plots(rows, output_dir, dataset_name='dataset'):
    """Convenience: run every plotting function for a set of results.

    Parameters
    ----------
    rows : list[dict]
        Frame-person results rows.
    output_dir : Path or str
        Directory where plots should be saved.
    dataset_name : str
        Used in titles.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_risk_distribution(rows, output_dir / 'risk_distribution_overall.png',
                           title=f'Risk Level Distribution — {dataset_name}')

    plot_risk_by_video(rows, output_dir / 'risk_distribution_by_video.png')

    plot_people_per_frame(rows, output_dir / 'people_per_frame_distribution.png')

    plot_keypoint_confidence(rows, output_dir / 'keypoint_confidence_distribution.png')

    plot_all_angle_histograms(rows, output_dir)

    plot_top_risk_frames(rows, output_dir / 'top_risk_frames.png')
