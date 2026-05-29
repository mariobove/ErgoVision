"""
Fuzzy RULA-inspired postural risk screening (SOTA-aligned).

Replaces the previous weighted-linear-sum approach with a **Mamdani fuzzy
inference system** that captures non-linear interactions between body regions
— the dominant paradigm in state-of-the-art vision-based ergonomic assessment
(Li et al. 2024, *Automation in Construction*; Menanno et al. 2024,
*Applied Sciences*).

Key improvements over the linear weighted-sum approach:

1. **Fuzzy inference** — rules capture biomechanical interactions (e.g.
   trunk + upper arm together produce more risk than the sum of their parts).

2. **Trapezoidal membership functions** — smooth transitions instead of
   hard thresholds; calibrated on RULA / REBA / OWAS cut-points.

3. **Continuous confidence** — pose confidence is propagated alongside
   the risk score instead of a binary UNCERTAIN gate.  Every frame receives
   a risk estimate; the confidence tells you how much to trust it.

4. **Separated uncertainty** — the risk score is never masked; low-confidence
   poses are still scored and reported with a caveat.

The system **does not** implement full clinical RULA: force, repetition,
static posture, and wrist rotation are not automatically inferred.

References
----------
- Li, Z. et al. (2024). Data-driven ergonomic assessment of construction
  workers. *Automation in Construction*, 165, 105561.
- Menanno, M. et al. (2024). An Ergonomic Risk Assessment System Based on
  3D Human Pose Estimation and Collaborative Robot. *Applied Sciences*,
  14(11), 4823.
"""

import numpy as np
from .config import (
    FUZZY_MEMBERSHIPS,
    RISK_MEMBERSHIPS,
    FUZZY_RULES,
    TORSO_THRESHOLDS,
    NECK_THRESHOLDS,
    KNEE_THRESHOLDS,
    SHOULDER_ASYMMETRY_THRESHOLDS,
    INCLINATION_THRESHOLDS,
    UPPER_ARM_THRESHOLDS,
    FOREARM_THRESHOLDS,
    RISK_CLASSES,
    RISK_LEVEL_SHORT,
    KEYPOINT_INDICES as KPT,
)

# ===================================================================
# Fuzzy Membership Function
# ===================================================================

class TrapezoidalMF:
    """Trapezoidal membership function with parameters (a, b, c, d).

    ::

       mu
      1 ──────────
       |          |
       |          |
     0 ──┐      ┌──
          a  b  c  d
    """

    __slots__ = ('a', 'b', 'c', 'd')

    def __init__(self, a, b, c, d):
        self.a, self.b, self.c, self.d = a, b, c, d

    def __call__(self, x):
        if x is None or np.isnan(x):
            return 0.0
        if x < self.a or x > self.d:
            return 0.0
        # Rising edge: a -> b  (if a == b and x == a, skip to plateau)
        if self.a <= x < self.b:
            return (x - self.a) / (self.b - self.a)
        # Plateau
        if self.b <= x <= self.c:
            return 1.0
        # Falling edge: c -> d
        if self.c < x < self.d:
            if np.isinf(self.d) or np.isinf(self.c):
                return 1.0  # plateau continues for unbounded tails
            return (self.d - x) / (self.d - self.c)
        return 0.0


# ===================================================================
# Mamdani Fuzzy Inference System
# ===================================================================

class MamdaniFIS:
    """Mamdani fuzzy inference engine.

    Parameters
    ----------
    input_mfs : dict
        ``{feature_name: {linguistic_var: (a,b,c,d)}}``
    rules : list of (list of (feature, lingvar), consequent)
        Antecedents are AND-connected.
    output_mfs : dict
        ``{linguistic_var: (a,b,c,d)}`` over the risk domain [0, 100].
    """

    def __init__(self, input_mfs, rules, output_mfs):
        # Convert raw tuples to TrapezoidalMF
        self._in_mf = {}
        for feat, lingvars in input_mfs.items():
            self._in_mf[feat] = {
                label: TrapezoidalMF(*params)
                for label, params in lingvars.items()
            }
        self._rules = rules
        self._out_mf = {
            label: TrapezoidalMF(*params)
            for label, params in output_mfs.items()
        }
        # Precompute centroid of each output MF for weighted-average defuzz
        self._out_centroid = {
            label: self._trapz_centroid(mf)
            for label, mf in self._out_mf.items()
        }

    # -- fuzzification ---------------------------------------------------

    def fuzzify(self, inputs):
        """Return ``{feature: {lingvar: mu}}``."""
        result = {}
        for feat, val in inputs.items():
            if val is None:
                continue
            mfs = self._in_mf.get(feat)
            if mfs is None:
                continue
            result[feat] = {lbl: mf(val) for lbl, mf in mfs.items()}
        return result

    # -- inference -------------------------------------------------------

    def infer(self, fuzzified):
        """Return ``{consequent: firing_strength}``."""
        outputs = {}
        for ante_list, consequent in self._rules:
            if not ante_list:
                firing = 1.0  # default rule always fires
            else:
                firing = 1.0
                for feat, label in ante_list:
                    mu = fuzzified.get(feat, {}).get(label, 0.0)
                    firing = min(firing, mu)
                    if firing < 1e-12:
                        break  # short-circuit
            if firing > 1e-12:
                outputs[consequent] = max(outputs.get(consequent, 0.0), firing)
        return outputs

    # -- defuzzification -------------------------------------------------

    def defuzzify(self, rule_outputs):
        """Weighted average of output MF centroids (simplified centroid).

        Returns a continuous risk score in [0, 100].
        Returns 0 when no rules fire (no postural deviation detected).
        """
        if not rule_outputs:
            return 0.0  # no rules fired → minimal risk
        num = 0.0
        den = 0.0
        for label, firing in rule_outputs.items():
            c = self._out_centroid.get(label)
            if c is not None:
                num += firing * c
                den += firing
        if den > 1e-12:
            return round(num / den, 1)
        return 0.0

    # -- convenience -----------------------------------------------------

    def evaluate(self, inputs):
        """One-shot fuzzify → infer → defuzzify.

        Parameters
        ----------
        inputs : dict
            ``{feature_name: float_value}``

        Returns
        -------
        risk_score : float in [0, 100]
        rule_outputs : dict — firing strengths for interpretability
        """
        fuzzified = self.fuzzify(inputs)
        rule_outputs = self.infer(fuzzified)
        return self.defuzzify(rule_outputs), rule_outputs

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _trapz_centroid(mf):
        """Centroid (centre of mass) of a trapezoid."""
        a, b, c, d = mf.a, mf.b, mf.c, mf.d
        if b <= a and c >= d:          # rectangle
            return (a + d) / 2.0
        if b <= a and c == d and d <= a:
            return a
        # Full formula: centroid_y * area = ∫ x·mu(x) dx
        # For trapezoid a-b-c-d at height 1:
        # area = (b-a)/2 + (c-b) + (d-c)/2 = (c + d - a - b) / 2
        # first moment M = ∫ x·mu(x) dx  (skipping the algebra here)
        # Common simplification: use (b + c) / 2 when b > a and d > c
        if (b - a) < 1e-10 and (d - c) < 1e-10:
            return (b + c) / 2.0
        # Full centroid formula for general trapezoid
        # M = (c^2 - b^2 + c*d - a*b) / 6 + (b^2 - a^2) / 2 + (d^2 - c^2) / ... no
        # Let's use a robust numeric approach:
        xs = np.linspace(a, d, 201)
        mus = np.array([mf(x) for x in xs])
        if mus.sum() < 1e-12:
            return (a + d) / 2.0
        return float(np.dot(xs, mus) / mus.sum())


# Singleton FIS instance (lazy-built on first use)
_FIS = None

def _get_fis():
    global _FIS
    if _FIS is None:
        _FIS = MamdaniFIS(FUZZY_MEMBERSHIPS, FUZZY_RULES, RISK_MEMBERSHIPS)
    return _FIS


# ===================================================================
# Geometry helpers (unchanged from original)
# ===================================================================

def calculate_angle(p1, p2, p3):
    """Angle (degrees) at p2 formed by vectors p2->p1 and p2->p3."""
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
    """Angle (degrees) of *p_lower -> p_upper* relative to upward vertical."""
    dx = p_upper[0] - p_lower[0]
    dy = p_upper[1] - p_lower[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    vec = np.array([dx, dy])
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    vertical = np.array([0, -1])
    cos_a = np.dot(vec, vertical)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# ===================================================================
# Feature severity (continuous, from fuzzy memberships)
# ===================================================================

def _feature_severity_fuzzy(value, membership_dict):
    """Continuous severity in [0, 100] = max non-neutral membership × 100.

    This replaces the old piecewise-linear ``_continuous_severity``.
    """
    if value is None or np.isnan(value):
        return 0.0
    max_mu = 0.0
    for label, params in membership_dict.items():
        if label == 'neutral':
            continue
        mf = TrapezoidalMF(*params)
        max_mu = max(max_mu, mf(value))
    return round(min(max_mu * 100.0, 100.0), 1)


def _legacy_discrete(value, thresholds):
    """Discrete score 1-3 using legacy thresholds (backward compat)."""
    if value is None or np.isnan(value):
        return 1
    lo_lo, lo_hi = thresholds['low']
    med_lo, med_hi = thresholds['medium']
    hi_lo, hi_hi = thresholds['high']
    if lo_lo <= value < lo_hi:
        return 1
    elif med_lo <= value < med_hi:
        return 2
    elif hi_lo <= value < hi_hi:
        return 3
    else:
        return 3


# ===================================================================
# Pose confidence (continuous, replaces binary UNCERTAIN)
# ===================================================================

def compute_pose_confidence(keypoints, min_confidence=0.3):
    """Continuous confidence score in [0, 1] for a pose estimate.

    Combines two factors:
    - **Coverage**: fraction of the 17 keypoints that are valid
      (sigmoid-mapped so 8/17 ≈ 0.5, 13/17 ≈ 0.92).
    - **Quality**: mean detection confidence of valid keypoints
      (capped at 0.5 which is treated as "full" for YOLO-pose).

    Returns
    -------
    confidence : float in [0, 1]
    n_valid : int
    mean_conf : float
    """
    valid_confs = []
    n_valid = 0
    for i in range(min(17, keypoints.shape[0])):
        x, y = float(keypoints[i, 0]), float(keypoints[i, 1])
        if x == 0 and y == 0:
            continue
        if keypoints.shape[1] >= 3:
            c = float(keypoints[i, 2])
            if c < min_confidence:
                continue
            valid_confs.append(c)
        n_valid += 1

    mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0

    # Coverage: sigmoid centred at 8 valid keypoints
    #   sig(0) ≈ 0.02, sig(8) ≈ 0.50, sig(13) ≈ 0.92, sig(17) ≈ 0.999
    coverage = 1.0 / (1.0 + np.exp(-(n_valid - 8) * 0.6))

    # Quality: mean confidence normalised so that 0.50 → 1.0
    quality = min(mean_conf / 0.50, 1.0) if mean_conf > 0 else 0.0

    # Geometric mean: a low score on either factor pulls confidence down
    confidence = float(np.sqrt(coverage * quality))
    return min(max(confidence, 0.0), 1.0), n_valid, mean_conf


# ===================================================================
# Ergonomic Scorer (fuzzy)
# ===================================================================

PRIMARY_FEATURES = {
    'trunk_angle',
    'neck_angle',
    'upper_arm_angle_left',
    'upper_arm_angle_right',
}

SECONDARY_FEATURES = {
    'forearm_angle_left',
    'forearm_angle_right',
    'knee_angle_left',
    'knee_angle_right',
    'shoulder_asymmetry',
    'body_inclination',
}

FEATURE_WEIGHTS = {
    'trunk_angle': 0.30,
    'neck_angle': 0.20,
    'upper_arm_angle_left': 0.10,
    'upper_arm_angle_right': 0.10,
    'forearm_angle_left': 0.05,
    'forearm_angle_right': 0.05,
    'knee_angle_left': 0.05,
    'knee_angle_right': 0.05,
    'shoulder_asymmetry': 0.05,
    'body_inclination': 0.05,
}


class ErgonomicScorer:
    """Fuzzy rule-based RULA-inspired postural risk screening.

    Uses a **Mamdani fuzzy inference system** to aggregate joint-angle
    features into a continuous risk score, replacing the linear weighted-sum
    with non-linear rule-based reasoning.

    Every pose receives a risk estimate.  The ``confidence`` field (0-1)
    tells you how reliable that estimate is — *not* whether the estimate
    exists.
    """

    PRIMARY = PRIMARY_FEATURES
    SECONDARY = SECONDARY_FEATURES
    WEIGHTS = FEATURE_WEIGHTS

    # -- keypoint access -------------------------------------------------

    @staticmethod
    def _get_kp(keypoints, idx, min_confidence=0.3):
        """Return (x, y) for keypoint *idx* or None."""
        x, y = float(keypoints[idx, 0]), float(keypoints[idx, 1])
        if x == 0 and y == 0:
            return None
        if keypoints.shape[1] >= 3 and keypoints[idx, 2] < min_confidence:
            return None
        return np.array([x, y])

    # -- per-feature computation -----------------------------------------

    def _compute_torso(self, keypoints):
        ls = self._get_kp(keypoints, KPT['left_shoulder'])
        rs = self._get_kp(keypoints, KPT['right_shoulder'])
        lh = self._get_kp(keypoints, KPT['left_hip'])
        rh = self._get_kp(keypoints, KPT['right_hip'])
        if all(x is not None for x in (ls, rs, lh, rh)):
            mid_shoulder = (ls + rs) / 2.0
            mid_hip = (lh + rh) / 2.0
            val = angle_from_vertical(mid_shoulder, mid_hip)
            sev = _feature_severity_fuzzy(val, FUZZY_MEMBERSHIPS['trunk_angle'])
            disc = _legacy_discrete(val, TORSO_THRESHOLDS)
            return {
                'value': round(val, 1), 'score': disc, 'severity': sev,
                'label': RISK_CLASSES[min(disc, 3) - 1],
                'explanation': f'Torso {val:.0f}° from vertical [severity {sev:.0f}/100]',
            }
        return None

    def _compute_neck(self, keypoints):
        nose = self._get_kp(keypoints, KPT['nose'])
        ls = self._get_kp(keypoints, KPT['left_shoulder'])
        rs = self._get_kp(keypoints, KPT['right_shoulder'])
        if all(x is not None for x in (nose, ls, rs)):
            mid_shoulder = (ls + rs) / 2.0
            val = angle_from_vertical(nose, mid_shoulder)
            sev = _feature_severity_fuzzy(val, FUZZY_MEMBERSHIPS['neck_angle'])
            disc = _legacy_discrete(val, NECK_THRESHOLDS)
            return {
                'value': round(val, 1), 'score': disc, 'severity': sev,
                'label': RISK_CLASSES[min(disc, 3) - 1],
                'explanation': f'Neck {val:.0f}° from vertical [severity {sev:.0f}/100]',
            }
        return None

    def _compute_knee_side(self, keypoints, hip_idx, knee_idx, ankle_idx, side):
        hip = self._get_kp(keypoints, hip_idx)
        knee = self._get_kp(keypoints, knee_idx)
        ankle = self._get_kp(keypoints, ankle_idx)
        if all(x is not None for x in (hip, knee, ankle)):
            val = calculate_angle(hip, knee, ankle)
            bend = 180 - val  # deviation from straight
            sev = _feature_severity_fuzzy(bend, FUZZY_MEMBERSHIPS['knee_bend'])
            disc = _legacy_discrete(bend, KNEE_THRESHOLDS)
            return {
                'value': round(val, 1), 'score': disc, 'severity': sev,
                'label': RISK_CLASSES[min(disc, 3) - 1],
                'explanation': f'Knee ({side}) bend {bend:.0f}° [severity {sev:.0f}/100]',
            }
        return None

    def _compute_upper_arm(self, keypoints, shoulder_idx, elbow_idx, side):
        shoulder = self._get_kp(keypoints, shoulder_idx)
        elbow = self._get_kp(keypoints, elbow_idx)
        if shoulder is not None and elbow is not None:
            val = angle_from_vertical(shoulder, elbow)
            sev = _feature_severity_fuzzy(val, FUZZY_MEMBERSHIPS['upper_arm_angle'])
            disc = _legacy_discrete(val, UPPER_ARM_THRESHOLDS)
            return {
                'value': round(val, 1), 'score': disc, 'severity': sev,
                'label': RISK_CLASSES[min(disc, 3) - 1],
                'explanation': f'Upper arm ({side}) {val:.0f}° from vertical [severity {sev:.0f}/100]',
            }
        return None

    def _compute_forearm(self, keypoints, shoulder_idx, elbow_idx, wrist_idx, side):
        shoulder = self._get_kp(keypoints, shoulder_idx)
        elbow = self._get_kp(keypoints, elbow_idx)
        wrist = self._get_kp(keypoints, wrist_idx)
        if all(x is not None for x in (shoulder, elbow, wrist)):
            angle = calculate_angle(shoulder, elbow, wrist)
            deviation = abs(angle - 90)
            sev = _feature_severity_fuzzy(deviation, FUZZY_MEMBERSHIPS['forearm_deviation'])
            disc = _legacy_discrete(deviation, FOREARM_THRESHOLDS)
            return {
                'value': round(angle, 1), 'score': disc, 'severity': sev,
                'label': RISK_CLASSES[min(disc, 3) - 1],
                'explanation': f'Forearm ({side}) {angle:.0f}° [severity {sev:.0f}/100]',
            }
        return None

    def _compute_shoulder_asymmetry(self, keypoints):
        ls = self._get_kp(keypoints, KPT['left_shoulder'])
        rs = self._get_kp(keypoints, KPT['right_shoulder'])
        lh = self._get_kp(keypoints, KPT['left_hip'])
        rh = self._get_kp(keypoints, KPT['right_hip'])
        if ls is not None and rs is not None:
            y_diff = abs(ls[1] - rs[1])
            if lh is not None and rh is not None:
                mid_hip = (lh + rh) / 2.0
                mid_shoulder = (ls + rs) / 2.0
                torso_len = float(np.linalg.norm(mid_shoulder - mid_hip))
                pct = (y_diff / torso_len * 100) if torso_len > 0 else 0.0
            else:
                pct = float(y_diff)
            sev = _feature_severity_fuzzy(pct, FUZZY_MEMBERSHIPS['shoulder_asymmetry'])
            disc = _legacy_discrete(pct, SHOULDER_ASYMMETRY_THRESHOLDS)
            return {
                'value': round(pct, 1), 'score': disc, 'severity': sev,
                'label': RISK_CLASSES[min(disc, 3) - 1],
                'explanation': f'Shoulder asymmetry {pct:.0f}% [severity {sev:.0f}/100]',
            }
        return None

    def _compute_inclination(self, keypoints):
        ls = self._get_kp(keypoints, KPT['left_shoulder'])
        rs = self._get_kp(keypoints, KPT['right_shoulder'])
        lh = self._get_kp(keypoints, KPT['left_hip'])
        rh = self._get_kp(keypoints, KPT['right_hip'])
        if all(x is not None for x in (ls, rs, lh, rh)):
            mid_shoulder = (ls + rs) / 2.0
            mid_hip = (lh + rh) / 2.0
            vert_span = abs(mid_shoulder[1] - mid_hip[1])
            if vert_span > 0:
                disp = abs(mid_shoulder[0] - mid_hip[0])
                pct = (disp / vert_span) * 100
            else:
                pct = 0.0
            sev = _feature_severity_fuzzy(pct, FUZZY_MEMBERSHIPS['body_inclination'])
            disc = _legacy_discrete(pct, INCLINATION_THRESHOLDS)
            return {
                'value': round(pct, 1), 'score': disc, 'severity': sev,
                'label': RISK_CLASSES[min(disc, 3) - 1],
                'explanation': f'Body lateral inclination {pct:.0f}% [severity {sev:.0f}/100]',
            }
        return None

    # -- public API ------------------------------------------------------

    def score(self, keypoints, manual_context=None):
        """Compute fuzzy postural risk for a single frame.

        Parameters
        ----------
        keypoints : np.ndarray, shape (17, 2) or (17, 3)
        manual_context : dict or None

        Returns
        -------
        dict with keys:
          - final_risk_class      : 'Low Risk' | 'Medium Risk' | 'High Risk'
          - final_score           : 1 | 2 | 3
          - continuous_severity   : float (0-100, fuzzy inference)
          - confidence            : float (0-1)
          - feature_severities    : dict feature -> severity (0-100)
          - partial_scores        : dict feature -> {value, score, severity, ...}
          - explanation           : str
          - risk_drivers          : dict {primary: [...], secondary: [...]}
          - rule_firings          : dict {consequent: firing_strength}
          - unavailable_features  : list[str]
        """
        partial_scores = {}
        unavailable = []

        # -- Confidence (continuous, never blocks scoring) --------------
        confidence, n_valid_kp, mean_kp_conf = compute_pose_confidence(keypoints)
        is_low_confidence = confidence < 0.25

        # -- Compute all feature angles ---------------------------------
        features = [
            ('trunk_angle', self._compute_torso),
            ('neck_angle', self._compute_neck),
            ('upper_arm_angle_left',
             lambda k: self._compute_upper_arm(
                 k, KPT['left_shoulder'], KPT['left_elbow'], 'left')),
            ('upper_arm_angle_right',
             lambda k: self._compute_upper_arm(
                 k, KPT['right_shoulder'], KPT['right_elbow'], 'right')),
            ('forearm_angle_left',
             lambda k: self._compute_forearm(
                 k, KPT['left_shoulder'], KPT['left_elbow'],
                 KPT['left_wrist'], 'left')),
            ('forearm_angle_right',
             lambda k: self._compute_forearm(
                 k, KPT['right_shoulder'], KPT['right_elbow'],
                 KPT['right_wrist'], 'right')),
            ('knee_angle_left',
             lambda k: self._compute_knee_side(
                 k, KPT['left_hip'], KPT['left_knee'],
                 KPT['left_ankle'], 'left')),
            ('knee_angle_right',
             lambda k: self._compute_knee_side(
                 k, KPT['right_hip'], KPT['right_knee'],
                 KPT['right_ankle'], 'right')),
            ('shoulder_asymmetry', self._compute_shoulder_asymmetry),
            ('body_inclination', self._compute_inclination),
        ]

        for name, method in features:
            try:
                result = method(keypoints)
            except Exception:
                result = None

            if result is not None:
                partial_scores[name] = result
            else:
                partial_scores[name] = {
                    'value': None, 'score': None, 'severity': 0.0,
                    'label': None,
                    'explanation': f'{name}: unavailable.',
                }
                unavailable.append(name)

        feature_severities = {
            n: ps.get('severity', 0.0) or 0.0
            for n, ps in partial_scores.items()
        }

        # Preprocess paired features: worst side per pair
        def _val_angle(*names):
            """Raw angle value (for trunk, neck, upper arm)."""
            vals = [partial_scores[n]['value'] for n in names
                    if partial_scores.get(n, {}).get('value') is not None]
            return max(vals) if vals else None

        def _forearm_deviation(name):
            """Forearm: elbow interior angle → deviation from 90°."""
            ps = partial_scores.get(name, {})
            v = ps.get('value')
            if v is not None:
                return abs(v - 90)
            return None

        def _knee_bend(name):
            """Knee: interior angle → bend from straight."""
            ps = partial_scores.get(name, {})
            v = ps.get('value')
            if v is not None:
                return 180.0 - v
            return None

        fuzzy_inputs = {}
        fuzzy_inputs['trunk_angle'] = _val_angle('trunk_angle')
        fuzzy_inputs['neck_angle'] = _val_angle('neck_angle')
        fuzzy_inputs['upper_arm_angle'] = _val_angle(
            'upper_arm_angle_left', 'upper_arm_angle_right')
        fuzzy_inputs['forearm_deviation'] = max(
            filter(None, [_forearm_deviation('forearm_angle_left'),
                          _forearm_deviation('forearm_angle_right')]), default=None)
        fuzzy_inputs['knee_bend'] = max(
            filter(None, [_knee_bend('knee_angle_left'),
                          _knee_bend('knee_angle_right')]), default=None)

        # Need at least trunk or neck to evaluate
        if fuzzy_inputs.get('trunk_angle') is None and fuzzy_inputs.get('neck_angle') is None:
            # Fallback: use weighted sum (old method) when even primary
            # features are missing — shouldn't happen with valid detections
            total_w = sum(FEATURE_WEIGHTS.get(n, 0) for n in partial_scores)
            continuous_severity = sum(
                FEATURE_WEIGHTS.get(n, 0) * (ps.get('severity', 0.0) or 0.0)
                for n, ps in partial_scores.items()
            ) / max(total_w, 0.01)
            rule_outputs = {}
        else:
            continuous_severity, rule_outputs = _get_fis().evaluate(fuzzy_inputs)

        # -- Discrete classification from continuous severity ------------
        if continuous_severity >= 65:
            final_score = 3
            final_risk_class = RISK_CLASSES[2]
        elif continuous_severity >= 35:
            final_score = 2
            final_risk_class = RISK_CLASSES[1]
        else:
            final_score = 1
            final_risk_class = RISK_CLASSES[0]

        # -- Override to UNCERTAIN only when truly degenerate ------------
        uncertain = False
        uncertainty_reason = ''
        if n_valid_kp < 5:
            uncertain = True
            uncertainty_reason = (
                f'Very few reliable keypoints ({n_valid_kp}/17) — '
                f'estimate may be unreliable.'
            )
            # Still keep the score, just flag it

        # -- Risk drivers ------------------------------------------------
        primary_drivers = []
        secondary_drivers = []
        for name, ps in partial_scores.items():
            sev = ps.get('severity', 0.0) or 0.0
            if sev > 55:
                if name in PRIMARY_FEATURES:
                    primary_drivers.append(name)
                else:
                    secondary_drivers.append(name)

        # -- Explanation -------------------------------------------------
        parts = []
        if uncertain:
            risk_tag = f'ESTIMATE ({final_risk_class})' if confidence < 0.25 else final_risk_class
            parts.append(
                f'Low-confidence estimate: {uncertainty_reason} '
                f'Best estimate: {final_risk_class} '
                f'(severity {continuous_severity:.0f}/100, '
                f'confidence {confidence:.2f}).'
            )
        elif final_score == 3:
            parts.append(f'HIGH postural risk (severity {continuous_severity:.0f}/100) — sustained severe postural load.')
            if primary_drivers:
                parts.append(f'Primary drivers: {", ".join(primary_drivers)}.')
            if secondary_drivers:
                parts.append(f'Secondary indicators: {", ".join(secondary_drivers)}.')
        elif final_score == 2:
            parts.append(f'MEDIUM postural risk (severity {continuous_severity:.0f}/100) — moderate postural deviations.')
            if primary_drivers:
                parts.append(f'Primary: {", ".join(primary_drivers)}.')
        else:
            parts.append(f'LOW postural risk (severity {continuous_severity:.0f}/100) — neutral postural alignment.')

        if confidence < 0.5 and not uncertain:
            parts.append(f'[Reduced confidence: {confidence:.2f}]')

        if manual_context:
            active = [k for k, v in manual_context.items() if v is not None]
            if active:
                parts.append(
                    f'Manual context: {", ".join(active)}. '
                    f'These factors are NOT automatically inferred.'
                )

        explanation = ' '.join(parts)

        return {
            'final_risk_class': final_risk_class,
            'final_score': final_score,
            'continuous_severity': round(continuous_severity, 1),
            'confidence': round(confidence, 3),
            'feature_severities': feature_severities,
            'partial_scores': partial_scores,
            'explanation': explanation,
            'risk_drivers': {
                'primary': primary_drivers,
                'secondary': secondary_drivers,
            },
            'rule_firings': {k: round(v, 3) for k, v in rule_outputs.items()},
            'uncertain': uncertain,
            'uncertainty_reason': uncertainty_reason,
            'unavailable_features': unavailable,
        }


# ===================================================================
# Temporal smoothing (EMA)
# ===================================================================

class TemporalSmoothedScorer:
    """Wraps :class:`ErgonomicScorer` with EMA temporal smoothing.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor (0-1).  Lower = smoother (default 0.35).
    severity_low_max : float
        Threshold below which smoothed score → LOW (default 35).
    severity_medium_max : float
        Threshold below which smoothed score → MEDIUM (default 65).
    """

    def __init__(self, alpha=0.35, severity_low_max=35, severity_medium_max=65):
        self.scorer = ErgonomicScorer()
        self.alpha = alpha
        self.severity_low_max = severity_low_max
        self.severity_medium_max = severity_medium_max
        self._state = {}

    def score(self, keypoints, tracking_key=None, manual_context=None):
        """Compute temporally smoothed postural risk.

        Returns same dict as :meth:`ErgonomicScorer.score` plus:
          - raw_severity        : float (before smoothing)
          - smoothed_severity   : float (after EMA)
          - persistence_frames  : int
        """
        raw = self.scorer.score(keypoints, manual_context=manual_context)
        raw_severity = raw['continuous_severity']

        if tracking_key is None:
            raw['smoothed_severity'] = raw_severity
            raw['persistence_frames'] = 1
            return raw

        prev_ema, prev_class, persistence = self._state.get(
            tracking_key, (raw_severity, raw['final_score'], 0)
        )

        smoothed = self.alpha * raw_severity + (1 - self.alpha) * prev_ema
        persistence = persistence + 1
        self._state[tracking_key] = (smoothed, raw['final_score'], persistence)

        # Re-classify from smoothed severity
        if raw['uncertain'] and raw['final_score'] == 0:
            final_score = 0
            final_class = 'Uncertain'
        elif smoothed > self.severity_medium_max:
            final_score = 3
            final_class = RISK_CLASSES[2]
        elif smoothed > self.severity_low_max:
            final_score = 2
            final_class = RISK_CLASSES[1]
        else:
            final_score = 1
            final_class = RISK_CLASSES[0]

        raw['final_score'] = final_score
        raw['final_risk_class'] = final_class
        raw['raw_severity'] = round(raw_severity, 1)
        raw['smoothed_severity'] = round(smoothed, 1)
        raw['persistence_frames'] = persistence

        raw['explanation'] += (
            f' [temporal: raw {raw_severity:.0f} → '
            f'smoothed {smoothed:.0f}, '
            f'persistence {persistence} frame(s)]'
        )
        return raw

    def reset(self, tracking_key=None):
        if tracking_key is None:
            self._state.clear()
        else:
            self._state.pop(tracking_key, None)


def temporal_smooth_severity(results_rows, alpha=0.35,
                              severity_low_max=35, severity_medium_max=65):
    """Post-process per-person rows with EMA smoothing (post-hoc)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results_rows:
        groups[r.get('video_id', 'default')].append(r)

    smoothed_rows = []
    for vid, rows in groups.items():
        rows.sort(key=lambda x: x.get('frame_id', ''))
        ema = None
        for r in rows:
            sev = r.get('postural_severity', 50.0)
            if sev is None or (isinstance(sev, str) and not sev.strip()):
                sev = 50.0
            sev = float(sev)
            ema = sev if ema is None else alpha * sev + (1 - alpha) * ema

            r['smoothed_severity'] = round(ema, 1)
            r['raw_severity'] = round(sev, 1)

            risk_level = r.get('risk_level', '')
            if risk_level == 'UNCERTAIN':
                pass
            elif ema > severity_medium_max:
                r['risk_score'] = 3
                r['risk_level'] = RISK_LEVEL_SHORT.get(3, 'HIGH')
            elif ema > severity_low_max:
                r['risk_score'] = 2
                r['risk_level'] = RISK_LEVEL_SHORT.get(2, 'MEDIUM')
            else:
                r['risk_score'] = 1
                r['risk_level'] = RISK_LEVEL_SHORT.get(1, 'LOW')

            smoothed_rows.append(r)

    return smoothed_rows
