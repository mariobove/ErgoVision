"""
Report generation for ErgoVision experiments.

Generates runtime reports, robustness CSV tables, risk-distribution tables,
feature-availability tables, and the paper-ready experimental summary text.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class ExperimentalReports:
    """Collection of static methods that write report files to disk."""

    REPORT_HEADER = (
        "===============================================================================\n"
        "ErgoVision — RULA-inspired lightweight ergonomic risk estimation\n"
        "===============================================================================\n"
    )

    # ------------------------------------------------------------------
    # Runtime report
    # ------------------------------------------------------------------

    @staticmethod
    def write_runtime_report(metrics, output_path):
        """
        Write a human-readable runtime and detection report.

        Parameters
        ----------
        metrics : dict
            Output of ``RobustnessMetrics.summarize()``.
        output_path : str or Path
        """
        lines = [
            ExperimentalReports.REPORT_HEADER,
            "Experiment 2: Assembly101 Robustness — Runtime Report",
            "",
            f"Total videos processed          : {metrics.get('total_videos', 'N/A')}",
            f"Total frames extracted          : {metrics.get('total_frames_extracted', 'N/A')}",
            f"Total frames processed          : {metrics['total_frames']}",
            f"Frames with person detected     : {metrics['frames_with_detection']}",
            f"Frames with no person detected  : {metrics['frames_no_person']}",
            "",
            "--- Detection rates ---",
            f"Successful pose detection rate  : {metrics['pose_detection_rate']:.2%}",
            f"No-person detection rate        : {metrics['no_person_detection_rate']:.2%}",
            f"Missing keypoint rate (overall) : {metrics['missing_keypoint_rate']:.2%}",
            "",
            "--- Inference speed ---",
            f"Average inference time per frame: {metrics['avg_inference_time_seconds']:.4f} s",
            f"FPS (frames per second)         : {metrics['fps']:.2f}",
            "",
            "--- Feature availability ---",
        ]

        for feat, counts in metrics.get('feature_availability', {}).items():
            lines.append(
                f"  {feat:25s}: "
                f"available {counts['available']:5d} / "
                f"unavailable {counts['unavailable']:5d}  "
                f"(rate: {counts['availability_rate']:.2%})"
            )

        lines += [
            "",
            "--- Risk distribution ---",
        ]
        for cls, info in metrics.get('risk_distribution', {}).items():
            lines.append(
                f"  {cls:15s}: {info['count']:4d} ({info['percentage']:.1f}%)"
            )

        lines += [
            "",
            f"Total failure cases recorded    : {metrics['total_failure_cases']}",
            "",
            "Experimental note:",
            "  This is a RULA-inspired lightweight ergonomic risk estimation pipeline.",
            "  It does NOT claim full clinical RULA compliance or medical certification.",
            "  Results are for research and feasibility evaluation only.",
            "",
            f"  Generated: 2026-05-28",
            "===============================================================================",
        ]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('\n'.join(lines), encoding='utf-8')

    # ------------------------------------------------------------------
    # Robustness CSV (per-frame or per-video breakdown)
    # ------------------------------------------------------------------

    @staticmethod
    def write_robustness_csv(video_summaries, output_path):
        """
        Write a CSV with per-video robustness metrics.

        Parameters
        ----------
        video_summaries : list of dict
            Each dict has keys: video_name, total_frames, frames_with_detection,
            frames_no_person, avg_inference_time, risk_low, risk_medium, risk_high.
        output_path : str or Path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            'video_name',
            'frames_processed',
            'frames_with_detection',
            'frames_no_person',
            'pose_detection_rate',
            'avg_inference_time_s',
            'fps',
            'risk_low',
            'risk_medium',
            'risk_high',
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for vs in video_summaries:
                total = max(vs.get('frames_processed', 1), 1)
                writer.writerow({
                    'video_name': vs.get('video_name', ''),
                    'frames_processed': vs.get('frames_processed', 0),
                    'frames_with_detection': vs.get('frames_with_detection', 0),
                    'frames_no_person': vs.get('frames_no_person', 0),
                    'pose_detection_rate': round(
                        vs.get('frames_with_detection', 0) / total, 4
                    ),
                    'avg_inference_time_s': round(
                        vs.get('avg_inference_time', 0.0), 4
                    ),
                    'fps': round(
                        1.0 / max(vs.get('avg_inference_time', 0.001), 0.001), 2
                    ),
                    'risk_low': vs.get('risk_low', 0),
                    'risk_medium': vs.get('risk_medium', 0),
                    'risk_high': vs.get('risk_high', 0),
                })

    # ------------------------------------------------------------------
    # Risk-distribution CSV
    # ------------------------------------------------------------------

    @staticmethod
    def write_risk_distribution_csv(metrics, output_path):
        """Write a simple two-column CSV: risk_class, count."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['risk_class', 'count'])
            for cls in ['Low Risk', 'Medium Risk', 'High Risk']:
                info = metrics['risk_distribution'].get(cls, {})
                writer.writerow([cls, info.get('count', 0)])

    # ------------------------------------------------------------------
    # Feature-availability CSV
    # ------------------------------------------------------------------

    @staticmethod
    def write_feature_availability_csv(metrics, output_path):
        """Write feature availability as CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['feature', 'available', 'unavailable', 'availability_rate'])
            for feat, counts in metrics['feature_availability'].items():
                writer.writerow([
                    feat,
                    counts['available'],
                    counts['unavailable'],
                    f"{counts['availability_rate']:.2%}",
                ])

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    @staticmethod
    def plot_risk_distribution(metrics, output_path):
        """Bar chart of Low / Medium / High risk counts."""
        classes = ['Low Risk', 'Medium Risk', 'High Risk']
        counts = [
            metrics['risk_distribution'].get(c, {}).get('count', 0)
            for c in classes
        ]
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(classes, counts, color=colors, edgecolor='gray')
        ax.set_ylabel('Number of frames / detections')
        ax.set_title('Ergonomic Risk Distribution — Assembly101')
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(cnt), ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_feature_availability(metrics, output_path):
        """Horizontal bar chart of feature availability rates."""
        features = list(metrics['feature_availability'].keys())
        rates = [
            metrics['feature_availability'][f]['availability_rate']
            for f in features
        ]
        labels = [f.replace('_', ' ').title() for f in features]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(labels, rates, color='#3498db', edgecolor='gray')
        ax.set_xlabel('Availability rate')
        ax.set_title('Per-Feature Keypoint Availability — Assembly101')
        ax.set_xlim(0, 1.05)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{rate:.0%}', va='center', fontsize=9)
        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Paper-ready experimental summary
    # ------------------------------------------------------------------

    @staticmethod
    def write_paper_summary(metrics, output_path, extra=None):
        """
        Generate the paper-ready experimental summary text.

        ``extra`` is an optional dict that can provide values not present in
        the metrics dict (e.g. number of videos, total frames extracted).

        The wording is designed for inclusion in a ``Results`` section and
        follows the required disclaimer.
        """
        extra = extra or {}

        summary = (
            "===============================================================================\n"
            "ErgoVision — Paper-Ready Experimental Summary\n"
            "===============================================================================\n"
            "\n"
            "The experimental phase does not aim to train or benchmark a new pose "
            "estimation model. Instead, it evaluates the feasibility, interpretability "
            "and computational viability of a RULA-inspired lightweight ergonomic risk "
            "estimation pipeline based on pretrained human pose estimation.\n"
            "\n"
            "--- Dataset ---\n"
            f"  Videos processed            : {extra.get('total_videos', 'N/A')}\n"
            f"  Frames extracted            : {extra.get('total_frames_extracted', 'N/A')}\n"
            f"  Frames analysed             : {metrics['total_frames']}\n"
            "\n"
            "--- Pose detection ---\n"
            f"  Successful detection rate   : {metrics['pose_detection_rate']:.2%}\n"
            f"  No-person frames            : {metrics['frames_no_person']} "
            f"({metrics['no_person_detection_rate']:.2%})\n"
            f"  Missing keypoint rate       : {metrics['missing_keypoint_rate']:.2%}\n"
            "\n"
            "--- Feature availability ---\n"
        )

        for feat, counts in metrics.get('feature_availability', {}).items():
            summary += (
                f"  {feat:25s}: "
                f"{counts['availability_rate']:.1%} availability\n"
            )

        summary += "\n--- Risk distribution ---\n"
        for cls in ['Low Risk', 'Medium Risk', 'High Risk']:
            info = metrics['risk_distribution'].get(cls, {})
            summary += (
                f"  {cls:15s}: {info.get('count', 0)} frames "
                f"({info.get('percentage', 0):.1f}%)\n"
            )

        summary += (
            "\n"
            "--- Runtime ---\n"
            f"  Average inference time      : {metrics['avg_inference_time_seconds']:.3f} s\n"
            f"  Throughput                  : {metrics['fps']:.1f} FPS\n"
            "\n"
            "--- Failure analysis ---\n"
            f"  Total failure cases         : {metrics['total_failure_cases']}\n"
            "\n"
            "--- Limitations ---\n"
            "  This pipeline implements a RULA-inspired lightweight ergonomic risk\n"
            "  estimation. It does NOT claim full clinical RULA compliance, medical\n"
            "  certification, or occupational health approval. The scoring is based on\n"
            "  a simplified rule-based approximation of RULA principles using only\n"
            "  five postural features. Results should be interpreted as research-grade\n"
            "  feasibility indicators, not as clinical or certified ergonomic assessments.\n"
            "  No ground-truth ergonomic labels were available; therefore accuracy,\n"
            "  precision, recall, and F1 metrics are not reported.\n"
            "\n"
            "--- Citation ---\n"
            "  Assembly101 dataset:\n"
            "    Sener, F., Chatterjee, D., Shelepov, E., He, K., Singhania, D.,\n"
            "    Wang, R., Yao, A., and Gall, J. (2022). Assembly101: A Large-Scale\n"
            "    Multi-View Video Dataset for Understanding Procedural Activities.\n"
            "    In Proceedings of the IEEE/CVF Conference on Computer Vision and\n"
            "    Pattern Recognition (CVPR).\n"
            "\n"
            "  Generated on 2026-05-28 by ErgoVision v0.1.0\n"
            "===============================================================================\n"
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary, encoding='utf-8')
