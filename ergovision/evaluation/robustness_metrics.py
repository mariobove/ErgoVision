"""
Robustness metrics for Assembly101 experiments.

Computes detection rates, feature availability, inference speed, and risk
distribution from per-frame inference results.
"""

import numpy as np


class RobustnessMetrics:
    """
    Aggregates per-frame inference results into summary robustness metrics.

    Usage
    -----
    >>> metrics = RobustnessMetrics()
    >>> for frame_result in inference_results:
    ...     metrics.update(frame_result)
    >>> report = metrics.summarize()
    """

    def __init__(self):
        # Counters
        self.total_frames = 0
        self.frames_with_detection = 0
        self.frames_no_person = 0
        self.inference_times = []       # seconds per frame

        # Feature tracking: for each of the 5 features, how many frames had
        # it available vs unavailable
        self.feature_availability = {
            'torso_angle':         {'available': 0, 'unavailable': 0},
            'neck_angle':          {'available': 0, 'unavailable': 0},
            'knee_angle':          {'available': 0, 'unavailable': 0},
            'shoulder_asymmetry':  {'available': 0, 'unavailable': 0},
            'body_inclination':    {'available': 0, 'unavailable': 0},
        }

        # Risk distribution (final risk class per detected person)
        self.risk_counts = {
            'Low Risk': 0,
            'Medium Risk': 0,
            'High Risk': 0,
        }

        # Failure case tracking
        self.failure_cases = []       # (frame_id, reason, detail)

    def update(self, frame_result):
        """
        Incorporate a single frame's inference result.

        Parameters
        ----------
        frame_result : dict
            Must contain keys:
              - frame_id : str
              - has_person : bool
              - inference_time : float (seconds)
              - detections : list of detection dicts (each with a ``score``
                result containing ``partial_scores``, ``unavailable_features``,
                ``final_risk_class``)
        """
        self.total_frames += 1

        if frame_result.get('inference_time') is not None:
            self.inference_times.append(frame_result['inference_time'])

        if not frame_result.get('has_person', False):
            self.frames_no_person += 1
            self.failure_cases.append((
                frame_result.get('frame_id', 'unknown'),
                'no_person_detected',
                'No person detected in frame',
            ))
            return

        self.frames_with_detection += 1

        for person in frame_result.get('detections', []):
            # Risk distribution
            rc = person.get('final_risk_class', 'Low Risk')
            if rc in self.risk_counts:
                self.risk_counts[rc] += 1

            # Feature availability
            for feat_name in self.feature_availability:
                if feat_name in person.get('unavailable_features', []):
                    self.feature_availability[feat_name]['unavailable'] += 1
                else:
                    self.feature_availability[feat_name]['available'] += 1

            # Track failure cases for missing keypoints
            unavailable = person.get('unavailable_features', [])
            if len(unavailable) >= 3:
                self.failure_cases.append((
                    frame_result.get('frame_id', 'unknown'),
                    'missing_keypoints',
                    f'{len(unavailable)} features unavailable: {", ".join(unavailable)}',
                ))

    def summarize(self):
        """
        Compute all aggregate robustness metrics.

        Returns
        -------
        dict
        """
        total = max(self.total_frames, 1)

        pose_detection_rate = (
            self.frames_with_detection / total
        )
        no_person_rate = (
            self.frames_no_person / total
        )

        # Feature availability rates (across all detected persons)
        feature_rates = {}
        for feat, counts in self.feature_availability.items():
            feat_total = counts['available'] + counts['unavailable']
            rate = (
                counts['available'] / max(feat_total, 1)
            )
            feature_rates[feat] = {
                'available': counts['available'],
                'unavailable': counts['unavailable'],
                'availability_rate': round(rate, 4),
            }

        # Overall missing-keypoint rate = ratio of unavailable feature
        # instances across all 5 features
        total_feature_checks = sum(
            c['available'] + c['unavailable']
            for c in self.feature_availability.values()
        )
        total_unavailable = sum(
            c['unavailable']
            for c in self.feature_availability.values()
        )
        missing_keypoint_rate = (
            total_unavailable / max(total_feature_checks, 1)
        )

        # Inference speed
        inference_times = self.inference_times or [0.0]
        avg_inference_time = float(np.mean(inference_times))
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0

        # Risk distribution
        total_people = sum(self.risk_counts.values()) or 1
        risk_distribution = {
            cls: {
                'count': cnt,
                'percentage': round(cnt / total_people * 100, 2),
            }
            for cls, cnt in self.risk_counts.items()
        }

        return {
            'total_frames': self.total_frames,
            'frames_with_detection': self.frames_with_detection,
            'frames_no_person': self.frames_no_person,
            'pose_detection_rate': round(pose_detection_rate, 4),
            'no_person_detection_rate': round(no_person_rate, 4),
            'missing_keypoint_rate': round(missing_keypoint_rate, 4),
            'feature_availability': feature_rates,
            'risk_distribution': risk_distribution,
            'total_people_detected': total_people,
            'avg_inference_time_seconds': round(avg_inference_time, 4),
            'fps': round(fps, 2),
            'total_failure_cases': len(self.failure_cases),
            'failure_cases': self.failure_cases[:100],  # cap at 100 entries
        }
