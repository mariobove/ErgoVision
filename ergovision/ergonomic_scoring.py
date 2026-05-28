"""
RULA-inspired lightweight ergonomic risk estimation.

This module implements a transparent, rule-based risk estimator inspired by
the Rapid Upper Limb Assessment (RULA) method.  It is NOT a full clinical RULA
implementation and does NOT claim clinical validation.

Method
------
For each detected person the pipeline:
  1. Extracts up to five postural features from COCO keypoints.
  2. Assigns each feature a partial score (1 = low risk, 2 = medium risk,
     3 = high risk) based on fixed angle / asymmetry thresholds.
  3. Computes the final score as the **maximum** of all available partial
     scores.  Using the maximum ensures that a single critical posture is
     not hidden by averaging.
  4. Maps the final score to a risk class:
       - 1 → Low Risk
       - 2 → Medium Risk
       - 3 → High Risk
  5. Returns an explanation that names the deciding feature(s) and the
     threshold that was exceeded.

Features that cannot be computed (missing keypoints, low confidence) are
excluded from the final score and listed in ``unavailable_features``.
"""

import numpy as np
from .config import (
    TORSO_THRESHOLDS,
    NECK_THRESHOLDS,
    KNEE_THRESHOLDS,
    SHOULDER_ASYMMETRY_THRESHOLDS,
    INCLINATION_THRESHOLDS,
    RISK_CLASSES,
    KEYPOINT_INDICES as KPT,
)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def calculate_angle(p1, p2, p3):
    """
    Interior angle (degrees) at p2 formed by vectors p2→p1 and p2→p3.

    Used for knee angle:  hip → knee → ankle returns the leg bend angle.
    Returns 0.0 when points are degenerate.
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    cos_a = np.dot(v1, v2) / (n1 * n2)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def angle_from_vertical(p_upper, p_lower):
    """
    Deviation (degrees) of the vector *p_lower → p_upper* from straight-up
    vertical in image coordinates.

    0°  = perfectly upright.
    90° = horizontal (e.g. person leaning 90° forward).
    """
    dx = p_upper[0] - p_lower[0]
    dy = p_upper[1] - p_lower[1]   # negative = upward in image space

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0

    vec = np.array([dx, dy])
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    vertical = np.array([0, -1])    # pointing up in image coords

    cos_a = np.dot(vec, vertical)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# ---------------------------------------------------------------------------
# Threshold → score  (1, 2, 3)
# ---------------------------------------------------------------------------

def _value_to_score(value, thresholds):
    """
    Map a continuous feature value to a discrete risk score.

    Parameters
    ----------
    value : float
        The computed feature value (angle in degrees or asymmetry %).
    thresholds : dict
        With keys ``'low'``, ``'medium'``, ``'high'``, each a ``(lo, hi)``
        tuple defining the range for score 1, 2, 3 respectively.

    Returns
    -------
    int
        1 (Low Risk), 2 (Medium Risk), or 3 (High Risk).
    """
    for score, level in enumerate(['low', 'medium', 'high'], start=1):
        lo, hi = thresholds[level]
        if lo <= value < hi:
            return score
    return 3  # default: High Risk


# ---------------------------------------------------------------------------
# ErgonomicScorer
# ---------------------------------------------------------------------------

class ErgonomicScorer:
    """
    RULA-inspired lightweight ergonomic risk estimator.

    Computes partial risk scores for torso angle, neck angle, knee angle,
    shoulder asymmetry, and body inclination, then produces a final score
    equal to the **maximum** of all available partial scores.

    Usage
    -----
    >>> scorer = ErgonomicScorer()
    >>> result = scorer.score(keypoints)   # keypoints: (17, 2) or (17, 3)
    """

    def __init__(self):
        # Ordered list of feature keys — controls iteration order in score()
        self.feature_names = [
            'torso_angle',
            'neck_angle',
            'knee_angle',
            'shoulder_asymmetry',
            'body_inclination',
        ]

    # ------------------------------------------------------------------
    # Keypoint accessor
    # ------------------------------------------------------------------

    @staticmethod
    def _get_kp(keypoints, idx, min_confidence=0.3):
        """
        Extract a single keypoint coordinate.

        Returns (x, y) as a 1-D array, or ``None`` if the keypoint is at
        the origin (0, 0) or has confidence below *min_confidence*.
        """
        x = float(keypoints[idx, 0])
        y = float(keypoints[idx, 1])
        if x == 0 and y == 0:
            return None
        if keypoints.shape[1] >= 3 and float(keypoints[idx, 2]) < min_confidence:
            return None
        return np.array([x, y])

    # ------------------------------------------------------------------
    # Per-feature computation
    # ------------------------------------------------------------------

    def _compute_torso(self, ls, rs, lh, rh):
        """
        Compute torso angle: deviation of the spine from vertical.

        Uses the midpoint of the shoulders and the midpoint of the hips
        to approximate the spine line.

        Torso angle thresholds:
            < 20°  → score 1 (upright)
            20–60° → score 2 (moderate lean)
            > 60°  → score 3 (severe lean)
        """
        if ls is None or rs is None or lh is None or rh is None:
            return None, None, 'torso_angle'

        mid_shoulder = (ls + rs) / 2.0
        mid_hip = (lh + rh) / 2.0
        value = angle_from_vertical(mid_shoulder, mid_hip)
        score = _value_to_score(value, TORSO_THRESHOLDS)

        explanation = (
            f"Torso angle is {value:.1f}° "
            f"({RISK_CLASSES[score - 1]}). "
            f"Thresholds: < 20° = Low, 20–60° = Medium, > 60° = High."
        )
        return value, score, explanation

    def _compute_neck(self, nose, ls, rs):
        """
        Compute neck angle: deviation of the head from vertical.

        Uses the vector from mid-shoulder to nose.

        Neck angle thresholds:
            < 10°  → score 1 (neutral)
            10–30° → score 2 (moderate flexion/extension)
            > 30°  → score 3 (severe flexion/extension)
        """
        if nose is None or ls is None or rs is None:
            return None, None, 'neck_angle'

        mid_shoulder = (ls + rs) / 2.0
        value = angle_from_vertical(nose, mid_shoulder)
        score = _value_to_score(value, NECK_THRESHOLDS)

        explanation = (
            f"Neck angle is {value:.1f}° "
            f"({RISK_CLASSES[score - 1]}). "
            f"Thresholds: < 10° = Low, 10–30° = Medium, > 30° = High."
        )
        return value, score, explanation

    def _compute_knee(self, lh, lk, la, rh, rk, ra):
        """
        Compute knee angle from left and right legs (averaged).

        Knee angle thresholds (applied to the *bend* = 180 - angle):
            normal / extended          → score 1
            moderate flexion           → score 2
            strong flexion / unstable  → score 3

        A straight leg (~180°) gives a small bend (~0°) → Low Risk.
        A deeply bent leg (~90°) gives a large bend (~90°) → High Risk.
        """
        angles = []
        for hip, knee, ankle in [(lh, lk, la), (rh, rk, ra)]:
            if hip is not None and knee is not None and ankle is not None:
                angles.append(calculate_angle(hip, knee, ankle))

        if not angles:
            return None, None, 'knee_angle'

        avg_angle = float(np.mean(angles))
        bend = 180 - avg_angle                     # 0 = straight, 90 = bent
        score = _value_to_score(bend, KNEE_THRESHOLDS)

        explanation = (
            f"Average knee angle is {avg_angle:.1f}° "
            f"(bend = {bend:.1f}°, {RISK_CLASSES[score - 1]}). "
            f"Extended/normal posture = Low, moderate flexion = Medium, "
            f"strong flexion = High."
        )
        return avg_angle, score, explanation

    def _compute_shoulder_asymmetry(self, ls, rs, lh, rh):
        """
        Compute shoulder asymmetry: vertical height difference between
        left and right shoulders, normalised by torso length (%).

        Asymmetry thresholds:
            < 10%  → score 1 (low asymmetry)
            10–30% → score 2 (moderate asymmetry)
            > 30%  → score 3 (high asymmetry)
        """
        if ls is None or rs is None:
            return None, None, 'shoulder_asymmetry'

        y_diff = abs(float(ls[1] - rs[1]))

        # Normalise by torso length for scale-invariance
        if lh is not None and rh is not None:
            mid_shoulder = (ls + rs) / 2.0
            mid_hip = (lh + rh) / 2.0
            torso_len = float(np.linalg.norm(mid_shoulder - mid_hip))
            pct = (y_diff / torso_len * 100) if torso_len > 0 else 0.0
        else:
            pct = float(y_diff)

        score = _value_to_score(pct, SHOULDER_ASYMMETRY_THRESHOLDS)

        explanation = (
            f"Shoulder asymmetry is {pct:.1f}% "
            f"({RISK_CLASSES[score - 1]}). "
            f"Thresholds: < 10% = Low, 10–30% = Medium, > 30% = High."
        )
        return pct, score, explanation

    def _compute_inclination(self, ls, rs, lh, rh):
        """
        Compute body lateral inclination: horizontal displacement of the
        shoulder midpoint relative to the hip midpoint, as a percentage
        of the vertical torso span.

        Inclination thresholds:
            < 10%  → score 1 (upright)
            10–30% → score 2 (moderate lean)
            > 30%  → score 3 (severe lean)
        """
        if ls is None or rs is None or lh is None or rh is None:
            return None, None, 'body_inclination'

        mid_shoulder = (ls + rs) / 2.0
        mid_hip = (lh + rh) / 2.0
        vert_span = abs(mid_shoulder[1] - mid_hip[1])

        if vert_span > 0:
            disp = abs(mid_shoulder[0] - mid_hip[0])
            pct = (disp / vert_span) * 100
        else:
            pct = 0.0

        score = _value_to_score(pct, INCLINATION_THRESHOLDS)

        explanation = (
            f"Body lateral inclination is {pct:.1f}% "
            f"({RISK_CLASSES[score - 1]}). "
            f"Thresholds: < 10% = Low, 10–30% = Medium, > 30% = High."
        )
        return pct, score, explanation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, keypoints):
        """
        Compute the RULA-inspired lightweight ergonomic risk for one person.

        Parameters
        ----------
        keypoints : np.ndarray, shape (17, 2) or (17, 3)
            COCO-format pose keypoints from YOLOv8-pose.  If the array has
            3 columns the third column is treated as confidence.

        Returns
        -------
        dict with keys:
            final_risk_class      : str — ``'Low Risk'`` | ``'Medium Risk'`` |
                                    ``'High Risk'``
            final_score           : int — 1, 2, or 3  (the maximum of all
                                    available partial scores)
            partial_scores        : dict — feature name → dict with keys
                                    ``value``, ``score``, ``label``,
                                    ``explanation``
            explanation           : str — human-readable description of how
                                    the final score was determined, naming the
                                    deciding feature(s) and thresholds
            unavailable_features  : list of str — feature names that could not
                                    be computed due to missing / low-confidence
                                    keypoints
        """
        # Extract relevant keypoints
        ls = self._get_kp(keypoints, KPT['left_shoulder'])
        rs = self._get_kp(keypoints, KPT['right_shoulder'])
        lh = self._get_kp(keypoints, KPT['left_hip'])
        rh = self._get_kp(keypoints, KPT['right_hip'])
        nose = self._get_kp(keypoints, KPT['nose'])
        lk = self._get_kp(keypoints, KPT['left_knee'])
        rk = self._get_kp(keypoints, KPT['right_knee'])
        la = self._get_kp(keypoints, KPT['left_ankle'])
        ra = self._get_kp(keypoints, KPT['right_ankle'])

        # ---- Compute each feature ----
        feat_results = {}
        unavailable_features = []

        def _compute(key, result):
            value, score, explanation = result
            if value is None:
                unavailable_features.append(key)
                return
            feat_results[key] = {
                'value': round(float(value), 2),
                'score': int(score),
                'label': RISK_CLASSES[int(score) - 1],
                'explanation': str(explanation),
            }

        _compute('torso_angle',          self._compute_torso(ls, rs, lh, rh))
        _compute('neck_angle',           self._compute_neck(nose, ls, rs))
        _compute('knee_angle',           self._compute_knee(lh, lk, la, rh, rk, ra))
        _compute('shoulder_asymmetry',   self._compute_shoulder_asymmetry(ls, rs, lh, rh))
        _compute('body_inclination',     self._compute_inclination(ls, rs, lh, rh))

        # ---- Aggregate: use the MAXIMUM score across available features ----
        available_scores = [
            f['score'] for f in feat_results.values()
        ]

        if available_scores:
            final_score = max(available_scores)
        else:
            # No features could be computed at all — fall back to Low Risk
            # (with an explanation that no postural data was available)
            final_score = 1

        final_risk_class = RISK_CLASSES[final_score - 1]

        # ---- Build explanation ----
        # Find which feature(s) determined the final score
        deciding_features = [
            name for name, fd in feat_results.items()
            if fd['score'] == final_score
        ]

        if not feat_results:
            explanation = (
                f"{final_risk_class} — none of the five postural features "
                f"could be computed (all keypoints missing or low confidence)."
            )
        elif final_score == 1:
            explanation = (
                f"{final_risk_class} — all available postural features are "
                f"within low-risk thresholds."
            )
        elif len(deciding_features) == 1:
            feat_name = deciding_features[0]
            fd = feat_results[feat_name]
            explanation = (
                f"{final_risk_class} because "
                f"{feat_name.replace('_', ' ')} is {fd['value']}"
                f"{'°' if 'angle' in feat_name else '%'}, "
                f"reaching score {final_score} "
                f"({fd['label']}). "
            )
        else:
            # Multiple features tied at the max score
            parts = []
            for name in deciding_features:
                fd = feat_results[name]
                parts.append(
                    f"{name.replace('_', ' ')} = {fd['value']}"
                    f"{'°' if 'angle' in name else '%'} "
                    f"(score {fd['score']}, {fd['label']})"
                )
            explanation = (
                f"{final_risk_class} — multiple features reached the "
                f"maximum score: {'; '.join(parts)}. "
            )

        # Append warning about unavailable features
        if unavailable_features:
            explanation += (
                f"Note: the following features could not be computed "
                f"(missing keypoints): {', '.join(unavailable_features)}."
            )

        return {
            'final_risk_class': final_risk_class,
            'final_score': final_score,
            'partial_scores': feat_results,
            'explanation': explanation,
            'unavailable_features': unavailable_features,
        }
