"""
Lightweight RULA-inspired ergonomic risk scoring.

This is *not* a full clinical RULA implementation.  It applies interpretable,
rule-based thresholds to pose keypoints and produces a three-level risk
classification along with per-feature explanations.
"""

import numpy as np
from .config import (
    TORSO_THRESHOLDS,
    NECK_THRESHOLDS,
    KNEE_THRESHOLDS,
    SHOULDER_ASYMMETRY_THRESHOLDS,
    INCLINATION_THRESHOLDS,
    FEATURE_WEIGHTS,
    KEYPOINT_INDICES as KPT,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def calculate_angle(p1, p2, p3):
    """
    Angle (degrees) at p2 formed by vectors p2→p1 and p2→p3.
    Returns 0 for degenerate (collinear / zero-length) inputs.
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
    Angle (degrees) of the vector *p_lower → p_upper* relative to upward
    vertical.  0° means the vector points straight up in image coordinates
    (i.e. the person is upright).
    """
    dx = p_upper[0] - p_lower[0]
    dy = p_upper[1] - p_lower[1]          # negative = upward in image space

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0

    vec = np.array([dx, dy])
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    vertical = np.array([0, -1])           # pointing up in image coords

    cos_a = np.dot(vec, vertical)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _threshold_to_score(value, thresholds):
    """Map a continuous value to 0 (low), 1 (medium), or 2 (high)."""
    for level, (lo, hi) in enumerate([
            thresholds['low'],
            thresholds['medium'],
            thresholds['high'],
    ]):
        if lo <= value < hi:
            return level
    return 2


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class ErgonomicScorer:
    """
    Rule-based ergonomic risk scorer inspired by RULA.

    Computes five interpretable postural features and combines them into a
    single risk class with a human-readable explanation.
    """

    def __init__(self):
        self.feature_names = [
            'torso_angle',
            'neck_angle',
            'knee_angle',
            'shoulder_asymmetry',
            'body_inclination',
        ]

    # -- keypoint accessor ------------------------------------------------

    @staticmethod
    def _get_kp(keypoints, idx, min_confidence=0.3):
        """Return (x, y) for keypoint *idx* or None if missing / low-conf."""
        # Handle both (17,2) and (17,3) arrays
        x, y = float(keypoints[idx, 0]), float(keypoints[idx, 1])
        if x == 0 and y == 0:
            return None
        if keypoints.shape[1] >= 3 and keypoints[idx, 2] < min_confidence:
            return None
        return np.array([x, y])

    # -- per-feature computation ------------------------------------------

    def compute_features(self, keypoints):
        """
        Compute all five ergonomic features.

        Returns
        -------
        dict of name -> (value, risk_level, explanation_string)
        """
        features = {}

        ls = self._get_kp(keypoints, KPT['left_shoulder'])
        rs = self._get_kp(keypoints, KPT['right_shoulder'])
        lh = self._get_kp(keypoints, KPT['left_hip'])
        rh = self._get_kp(keypoints, KPT['right_hip'])
        nose = self._get_kp(keypoints, KPT['nose'])
        lk = self._get_kp(keypoints, KPT['left_knee'])
        rk = self._get_kp(keypoints, KPT['right_knee'])
        la = self._get_kp(keypoints, KPT['left_ankle'])
        ra = self._get_kp(keypoints, KPT['right_ankle'])

        # --- Torso angle -------------------------------------------------
        if all(x is not None for x in (ls, rs, lh, rh)):
            mid_shoulder = (ls + rs) / 2.0
            mid_hip = (lh + rh) / 2.0
            val = angle_from_vertical(mid_shoulder, mid_hip)
            score = _threshold_to_score(val, TORSO_THRESHOLDS)
            features['torso_angle'] = (
                val, score,
                f"Torso deviates {val:.1f}° from vertical"
            )
        else:
            features['torso_angle'] = (0.0, 0, "Torso angle: not available (missing shoulder/hip keypoints)")

        # --- Neck angle --------------------------------------------------
        if nose is not None and ls is not None and rs is not None:
            mid_shoulder = (ls + rs) / 2.0
            val = angle_from_vertical(nose, mid_shoulder)
            score = _threshold_to_score(val, NECK_THRESHOLDS)
            features['neck_angle'] = (
                val, score,
                f"Neck deviates {val:.1f}° from vertical"
            )
        else:
            features['neck_angle'] = (0.0, 0, "Neck angle: not available (missing nose/shoulder keypoints)")

        # --- Knee angle --------------------------------------------------
        knee_vals = []
        for hip, knee, ankle in [(lh, lk, la), (rh, rk, ra)]:
            if all(x is not None for x in (hip, knee, ankle)):
                knee_vals.append(calculate_angle(hip, knee, ankle))

        if knee_vals:
            avg = float(np.mean(knee_vals))
            # Treat *bent* knees as risky → score on (180 - angle)
            score = _threshold_to_score(180 - avg, KNEE_THRESHOLDS)
            features['knee_angle'] = (
                avg, score,
                f"Average knee angle: {avg:.1f}°"
            )
        else:
            features['knee_angle'] = (180.0, 0, "Knee angle: not available (missing leg keypoints)")

        # --- Shoulder asymmetry ------------------------------------------
        if ls is not None and rs is not None:
            y_diff = abs(ls[1] - rs[1])
            # Normalise by torso length for scale invariance
            if lh is not None and rh is not None:
                mid_hip = (lh + rh) / 2.0
                mid_shoulder = (ls + rs) / 2.0
                torso_len = float(np.linalg.norm(mid_shoulder - mid_hip))
                pct = (y_diff / torso_len * 100) if torso_len > 0 else 0.0
            else:
                pct = float(y_diff)

            score = _threshold_to_score(pct, SHOULDER_ASYMMETRY_THRESHOLDS)
            features['shoulder_asymmetry'] = (
                pct, score,
                f"Shoulder height difference: {pct:.1f}% of torso length"
            )
        else:
            features['shoulder_asymmetry'] = (0.0, 0, "Shoulder asymmetry: not available (missing shoulder keypoints)")

        # --- Body inclination (lateral lean) -----------------------------
        if all(x is not None for x in (ls, rs, lh, rh)):
            mid_shoulder = (ls + rs) / 2.0
            mid_hip = (lh + rh) / 2.0
            vert_span = abs(mid_shoulder[1] - mid_hip[1])
            if vert_span > 0:
                # Horizontal displacement of midpoints / vertical span
                disp = abs(mid_shoulder[0] - mid_hip[0])
                pct = (disp / vert_span) * 100
            else:
                pct = 0.0

            score = _threshold_to_score(pct, INCLINATION_THRESHOLDS)
            features['body_inclination'] = (
                pct, score,
                f"Body lateral inclination: {pct:.1f}%"
            )
        else:
            features['body_inclination'] = (0.0, 0, "Body inclination: not available (missing shoulder/hip keypoints)")

        return features

    # -- aggregate scoring ------------------------------------------------

    def score(self, keypoints):
        """
        Compute the overall ergonomic risk.

        Parameters
        ----------
        keypoints : np.ndarray, shape (17, 2) or (17, 3)
            Pose keypoints from YOLOv8-pose.  If shape is (17, 3) the third
            column is treated as confidence.

        Returns
        -------
        dict with keys:
          - risk_class   : "Low Risk" | "Medium Risk" | "High Risk"
          - risk_score   : float in [0, 1]
          - features     : dict of name -> {value, score, explanation}
          - explanations : list of per-feature explanation strings
        """
        features = self.compute_features(keypoints)

        weighted_sum = 0.0
        total_weight = 0.0
        explanations = []
        feature_details = {}

        for fname in self.feature_names:
            value, score, explanation = features[fname]
            w = FEATURE_WEIGHTS.get(fname, 0.0)
            weighted_sum += w * score
            total_weight += w
            feature_details[fname] = {
                'value': round(float(value), 2),
                'score': int(score),
                'explanation': explanation,
            }
            explanations.append(explanation)

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        normalised = final_score / 2.0          # map 0-2 → 0-1

        if normalised < 1 / 3:
            risk_class = 'Low Risk'
        elif normalised < 2 / 3:
            risk_class = 'Medium Risk'
        else:
            risk_class = 'High Risk'

        return {
            'risk_class': risk_class,
            'risk_score': round(normalised, 4),
            'features': feature_details,
            'explanations': explanations,
        }
