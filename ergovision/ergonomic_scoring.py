"""
RULA-inspired postural risk screening via segment scores and Action Levels.

Estimates the **observable postural component** of ergonomic risk from 2D
keypoints using discrete segment-level scores (1-4 per body region) and
explicit aggregation rules to produce approximate RULA Action Levels.

The system is **not** full clinical RULA: force, load, repetition, muscle
use, wrist twist, and true 3D rotation are not inferred automatically.

References
----------
- McAtamney & Corlett (1993). RULA. *Applied Ergonomics*, 24(2), 91--99.
- Li, Martin & Xu (2020). Vision-based real-time RULA. *Applied Ergonomics*, 87, 103138.
- Massiris Fernández et al. (2020). CV-based RULA. *Computers & Industrial Engineering*, 149, 106816.
- Deshpande et al. (2025). ML for HPE-based ERA (review). *Discover AI*, 5, 287.
"""

import numpy as np
from .config import (
    RISK_CLASSES,
    RISK_LEVEL_SHORT,
    KEYPOINT_INDICES as KPT,
)

# ===================================================================
# Vector geometry helpers
# ===================================================================

def angle_between(v1, v2):
    """Angle (degrees) between two vectors in Euclidean space."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def calculate_angle(p1, p2, p3):
    """Angle (degrees) at p2 from vectors p2->p1 and p2->p3."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    return angle_between(v1, v2)


# ===================================================================
# Pose confidence
# ===================================================================

def compute_pose_confidence(keypoints, min_confidence=0.3):
    """Continuous confidence score in [0, 1]."""
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
    coverage = 1.0 / (1.0 + np.exp(-(n_valid - 8) * 0.6))
    quality = min(mean_conf / 0.50, 1.0) if mean_conf > 0 else 0.0
    confidence = float(np.sqrt(coverage * quality))
    return min(max(confidence, 0.0), 1.0), n_valid, mean_conf


# ===================================================================
# Mapping: colour per risk level (central mapping — single source of truth)
# ===================================================================

RISK_TO_OVERLAY_COLOUR = {
    1: (0, 255, 0),     # GREEN  for AL1 / LOW
    2: (0, 255, 255),   # YELLOW for AL2 / MEDIUM
    3: (0, 0, 255),     # RED    for AL3+ / HIGH
}

RISK_TO_AL_LABEL = {
    1: 'AL1 (Low Risk)',
    2: 'AL2 (Medium Risk)',
    3: 'AL3+ (High Risk)',
}

RISK_TO_LEVEL_SHORT = {
    1: 'AL1 (LOW)',
    2: 'AL2 (MEDIUM)',
    3: 'AL3+ (HIGH)',
}


# ===================================================================
# Segment score helpers
# ===================================================================

def _trunk_score(angle):
    if angle is None or angle < 20:
        return 1
    if angle < 60:
        return 2
    return 3

def _neck_score(angle):
    if angle is None or angle < 20:
        return 1
    if angle < 45:
        return 2
    return 3

def _upper_arm_score(angle):
    if angle is None or angle < 20:
        return 1
    if angle < 45:
        return 2
    if angle < 90:
        return 3
    return 4

def _forearm_score(angle):
    """RULA-style: 1 if 60-100°, else 2."""
    if angle is None:
        return 1
    return 1 if 60 <= angle <= 100 else 2

def _leg_score(knee_bend_max):
    if knee_bend_max is None or knee_bend_max < 30:
        return 1
    if knee_bend_max < 60:
        return 2
    return 3

def _continuous_severity(angle, low, high, cap=100.0):
    """Diagnostic continuous severity 0-100 (not used for final decision)."""
    if angle is None or angle <= low:
        return 0.0
    if angle >= cap:
        return 100.0
    t = (angle - low) / max(cap - low, 1.0)
    return round(min(100.0, 100.0 * (t ** 0.7)), 1)


# ===================================================================
# Ergonomic Scorer — RULA-inspired Action Level
# ===================================================================

PRIMARY_FEATURES = {
    'trunk_angle', 'neck_angle',
    'upper_arm_angle_left', 'upper_arm_angle_right',
}

SECONDARY_FEATURES = {
    'forearm_angle_left', 'forearm_angle_right',
    'knee_angle_left', 'knee_angle_right',
    'shoulder_asymmetry', 'body_inclination',
}


class ErgonomicScorer:
    """RULA-inspired postural risk scorer producing approximate Action Levels.

    **Final decision (in order):**
    1. Neutral posture gate → AL1
    2. HIGH rules:
       - trunk_angle >= 60°
       - trunk_angle >= 45° AND upper_arm_max >= 60°
       - upper_arm_max >= 90° AND trunk_angle >= 30°
       - two or more severe primary deviations
       (neck alone capped to MEDIUM)
    3. MEDIUM rules:
       - trunk_angle >= 30°
       - upper_arm_max >= 45°
       - neck_angle >= 45° (if trunk/arms not severe)
       - two or more moderate primary deviations
    4. LOW (fallback)

    Output includes per-segment scores (1-3/4) and the continuous severity
    score as diagnostic-only columns.
    """

    PRIMARY = PRIMARY_FEATURES
    SECONDARY = SECONDARY_FEATURES

    @staticmethod
    def _get_kp(keypoints, idx, min_confidence=0.3):
        x, y = float(keypoints[idx, 0]), float(keypoints[idx, 1])
        if x == 0 and y == 0:
            return None
        if keypoints.shape[1] >= 3 and keypoints[idx, 2] < min_confidence:
            return None
        return np.array([x, y])

    @staticmethod
    def _midpoint(p1, p2):
        if p1 is not None and p2 is not None:
            return (p1 + p2) / 2.0
        return p1 if p1 is not None else p2

    # -- per-feature computation -----------------------------------------

    def _compute_torso(self, keypoints):
        ls = self._get_kp(keypoints, KPT['left_shoulder'])
        rs = self._get_kp(keypoints, KPT['right_shoulder'])
        lh = self._get_kp(keypoints, KPT['left_hip'])
        rh = self._get_kp(keypoints, KPT['right_hip'])
        if all(x is not None for x in (ls, rs, lh, rh)):
            ms = (ls + rs) / 2.0
            mh = (lh + rh) / 2.0
            trunk_vec = ms - mh  # upward pointing
            vertical = np.array([0, -1])
            val = angle_between(trunk_vec, vertical)
            score = _trunk_score(val)
            sev = _continuous_severity(val, 20, 80)
            return {
                'value': round(val, 1), 'score': score, 'severity': sev,
                'label': RISK_CLASSES[min(score, 3) - 1],
                'explanation': f'Torso {val:.0f}° [segment score {score}]',
            }
        return None

    def _compute_neck(self, keypoints):
        """Neck angle relative to trunk with neutral_offset = 15°."""
        nose = self._get_kp(keypoints, KPT['nose'])
        ls = self._get_kp(keypoints, KPT['left_shoulder'])
        rs = self._get_kp(keypoints, KPT['right_shoulder'])
        if all(x is not None for x in (nose, ls, rs)):
            ms = (ls + rs) / 2.0
            trunk_vec = ms - self._trunk_mid_hip  # upward
            neck_vec = nose - ms
            raw = angle_between(neck_vec, trunk_vec)
            # Apply neutral offset (physiological lordosis ~15°)
            val = max(0.0, raw - 15.0)
            score = _neck_score(val)
            sev = _continuous_severity(val, 15, 60)
            return {
                'value': round(val, 1), 'score': score, 'severity': sev,
                'label': RISK_CLASSES[min(score, 3) - 1],
                'explanation': f'Neck {val:.0f}° (raw {raw:.0f}°) [segment score {score}]',
                'neck_trunk_relative': round(val, 1),
                'neck_raw': round(raw, 1),
            }
        return None

    def _compute_upper_arm(self, keypoints, shoulder_idx, elbow_idx, side):
        """Upper arm elevation relative to trunk_down vector."""
        shoulder = self._get_kp(keypoints, shoulder_idx)
        elbow = self._get_kp(keypoints, elbow_idx)
        if shoulder is not None and elbow is not None:
            trunk_down = self._trunk_mid_hip - self._trunk_mid_shoulder
            ua_vec = elbow - shoulder
            val = angle_between(ua_vec, trunk_down)
            score = _upper_arm_score(val)
            sev = _continuous_severity(val, 20, 90)
            return {
                'value': round(val, 1), 'score': score, 'severity': sev,
                'label': RISK_CLASSES[min(score, 3) - 1],
                'explanation': f'Upper arm ({side}) {val:.0f}° [segment score {score}]',
            }
        return None

    def _compute_forearm(self, keypoints, shoulder_idx, elbow_idx, wrist_idx, side):
        """Forearm: elbow flexion angle (interior angle at elbow)."""
        shoulder = self._get_kp(keypoints, shoulder_idx)
        elbow = self._get_kp(keypoints, elbow_idx)
        wrist = self._get_kp(keypoints, wrist_idx)
        if all(x is not None for x in (shoulder, elbow, wrist)):
            val = calculate_angle(shoulder, elbow, wrist)
            score = _forearm_score(val)
            return {
                'value': round(val, 1), 'score': score,
                'severity': 0.0,  # diagnostic only
                'label': 'Forearm',
                'explanation': f'Forearm ({side}) {val:.0f}° [segment score {score}]',
            }
        return None

    def _compute_knee_side(self, keypoints, hip_idx, knee_idx, ankle_idx, side):
        hip = self._get_kp(keypoints, hip_idx)
        knee = self._get_kp(keypoints, knee_idx)
        ankle = self._get_kp(keypoints, ankle_idx)
        if all(x is not None for x in (hip, knee, ankle)):
            val = calculate_angle(hip, knee, ankle)
            bend = 180.0 - val
            return {
                'value': round(val, 1), 'bend': round(bend, 1),
                'score': _leg_score(bend), 'severity': 0.0,
                'label': 'Knee',
                'explanation': f'Knee ({side}) bend {bend:.0f}° [segment score {_leg_score(bend)}]',
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
                mh = (lh + rh) / 2.0
                ms = (ls + rs) / 2.0
                tl = float(np.linalg.norm(ms - mh))
                pct = (y_diff / tl * 100) if tl > 0 else 0.0
            else:
                pct = float(y_diff)
            return {'value': round(pct, 1), 'score': 1, 'severity': 0.0,
                    'label': 'Asymmetry',
                    'explanation': f'Shoulder asymmetry {pct:.0f}%'}
        return None

    def _compute_inclination(self, keypoints):
        ls = self._get_kp(keypoints, KPT['left_shoulder'])
        rs = self._get_kp(keypoints, KPT['right_shoulder'])
        lh = self._get_kp(keypoints, KPT['left_hip'])
        rh = self._get_kp(keypoints, KPT['right_hip'])
        if all(x is not None for x in (ls, rs, lh, rh)):
            ms = (ls + rs) / 2.0
            mh = (lh + rh) / 2.0
            vs = abs(ms[1] - mh[1])
            pct = (abs(ms[0] - mh[0]) / vs * 100) if vs > 0 else 0.0
            return {'value': round(pct, 1), 'score': 1, 'severity': 0.0,
                    'label': 'Inclination',
                    'explanation': f'Body lateral inclination {pct:.0f}%'}
        return None

    # -- public API ------------------------------------------------------

    def score(self, keypoints, manual_context=None):
        """Compute approximate RULA Action Level for a single frame.

        Returns
        -------
        dict with fields:
          final_risk_class, final_score, action_level, overlay_color
          trunk/neck/upper_arm/forearm angle and score
          approximate_action_level, action_level_reason
          neutral_gate_applied, neck_capped
          primary_risk_drivers, secondary_risk_drivers
          continuous_severity (diagnostic only)
          mapping_consistent: bool
        """
        partial_scores = {}
        unavailable = []

        confidence, n_valid_kp, mean_kp_conf = compute_pose_confidence(keypoints)

        # Cache trunk midpoints for neck and arm computation
        ls = self._get_kp(keypoints, KPT['left_shoulder'])
        rs = self._get_kp(keypoints, KPT['right_shoulder'])
        lh = self._get_kp(keypoints, KPT['left_hip'])
        rh = self._get_kp(keypoints, KPT['right_hip'])
        self._trunk_mid_shoulder = self._midpoint(ls, rs)
        self._trunk_mid_hip = self._midpoint(lh, rh)

        # -- Compute all feature angles ---------------------------------
        features = [
            ('trunk_angle', self._compute_torso),
            ('neck_angle', self._compute_neck),
            ('upper_arm_angle_left',
             lambda k: self._compute_upper_arm(k, KPT['left_shoulder'], KPT['left_elbow'], 'left')),
            ('upper_arm_angle_right',
             lambda k: self._compute_upper_arm(k, KPT['right_shoulder'], KPT['right_elbow'], 'right')),
            ('forearm_angle_left',
             lambda k: self._compute_forearm(k, KPT['left_shoulder'], KPT['left_elbow'], KPT['left_wrist'], 'left')),
            ('forearm_angle_right',
             lambda k: self._compute_forearm(k, KPT['right_shoulder'], KPT['right_elbow'], KPT['right_wrist'], 'right')),
            ('knee_angle_left',
             lambda k: self._compute_knee_side(k, KPT['left_hip'], KPT['left_knee'], KPT['left_ankle'], 'left')),
            ('knee_angle_right',
             lambda k: self._compute_knee_side(k, KPT['right_hip'], KPT['right_knee'], KPT['right_ankle'], 'right')),
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
                    'label': None, 'explanation': f'{name}: unavailable.',
                }
                unavailable.append(name)

        # -- Extract angle values for decision making --------------------
        def _val(name):
            ps = partial_scores.get(name, {})
            return ps.get('value') if ps else None

        trunk_val = _val('trunk_angle')
        neck_val = _val('neck_angle')
        ua_left = _val('upper_arm_angle_left')
        ua_right = _val('upper_arm_angle_right')
        incl_val = _val('body_inclination')

        ua_vals = [v for v in [ua_left, ua_right] if v is not None]
        ua_max = max(ua_vals) if ua_vals else None

        # -- Segment scores ---------------------------------------------
        trunk_seg = _trunk_score(trunk_val)
        neck_seg = _neck_score(neck_val)
        ua_left_seg = _upper_arm_score(ua_left)
        ua_right_seg = _upper_arm_score(ua_right)
        ua_seg = max(ua_left_seg, ua_right_seg)

        fa_left_seg = _forearm_score(_val('forearm_angle_left'))
        fa_right_seg = _forearm_score(_val('forearm_angle_right'))
        fa_seg = max(fa_left_seg, fa_right_seg)

        # Knee bend from partial scores
        def _knee_bend(side):
            ps = partial_scores.get(f'knee_angle_{side}', {})
            return ps.get('bend') if ps else None

        kb_vals = [v for v in [_knee_bend('left'), _knee_bend('right')] if v is not None]
        kb_max = max(kb_vals) if kb_vals else None
        leg_seg = _leg_score(kb_max)

        # -- Severity (diagnostic only, not used for final decision) -----
        diag_severities = {}
        diag_severities['trunk'] = _continuous_severity(trunk_val, 20, 80)
        diag_severities['neck'] = _continuous_severity(neck_val, 15, 60)
        diag_severities['upper_arm'] = _continuous_severity(ua_max, 20, 90)
        diag_continuous = float(np.mean(list(diag_severities.values()))) if diag_severities else 0.0

        # ---- 1. NEUTRAL POSTURE GATE ---------------------------------
        neutral_gate_applied = False
        neutral_gate = (
            trunk_val is not None and trunk_val < 20
            and ua_max is not None and ua_max < 45
            and neck_val is not None and neck_val < 30
            and incl_val is not None and incl_val < 20
        )
        if neutral_gate and not (manual_context and any(v is not None for v in manual_context.values())):
            neutral_gate_applied = True
            action_level = 1
            final_score = 1
            al_reason = 'neutral_gate'
            return self._build_result(
                action_level, final_score, al_reason, diag_continuous,
                trunk_seg, neck_seg, ua_seg, fa_seg, leg_seg,
                trunk_val, neck_val, ua_max, ua_left, ua_right,
                partial_scores, diag_severities,
                neutral_gate_applied=True, neck_capped=False,
                primary_drivers=[], secondary_drivers=[], n_valid_kp=n_valid_kp,
                confidence=confidence, manual_context=manual_context,
            )

        # ---- 2. HIGH CHECK (explicit rules, raw angles) ---------------
        is_high = False
        is_medium = False
        al_reason = ''
        neck_capped = False
        primary_drivers = []
        secondary_drivers = []

        # Count severe (score >= 3) primary features
        severe_primaries = 0
        if trunk_seg >= 3:
            severe_primaries += 1
            primary_drivers.append('trunk')
        if neck_seg >= 3:
            severe_primaries += 1
        if ua_seg >= 3:
            severe_primaries += 1
            primary_drivers.append('upper_arm')

        if trunk_val is not None and trunk_val >= 60:
            is_high = True
            al_reason = 'trunk_angle >= 60deg'
        elif trunk_val is not None and trunk_val >= 45 and ua_max is not None and ua_max >= 60:
            is_high = True
            al_reason = 'trunk >= 45deg AND upper_arm >= 60deg'
        elif ua_max is not None and ua_max >= 90 and trunk_val is not None and trunk_val >= 30:
            is_high = True
            al_reason = 'upper_arm >= 90deg AND trunk >= 30deg'
        elif severe_primaries >= 2:
            is_high = True
            al_reason = f'{severe_primaries} severe primary deviations'

        # Neck-only HIGH cap
        if is_high and neck_seg >= 3 and trunk_seg < 3 and ua_seg < 3:
            is_high = False
            is_medium = True
            neck_capped = True
            al_reason = 'neck_high_capped: neck alone cannot produce HIGH'

        # ---- 3. MEDIUM CHECK ------------------------------------------
        if not is_high and not is_medium:
            # Count moderate (score >= 2) primary features
            moderate_primaries = 0
            if trunk_seg >= 2: moderate_primaries += 1
            if neck_seg >= 2: moderate_primaries += 1
            if ua_seg >= 2: moderate_primaries += 1

            trunk_mod = trunk_val is not None and trunk_val >= 30
            arm_mod = ua_max is not None and ua_max >= 45
            neck_mod = neck_val is not None and neck_val >= 45

            if trunk_mod:
                is_medium = True
                al_reason = f'trunk_angle {trunk_val:.0f}deg >= 30'
            elif arm_mod:
                is_medium = True
                al_reason = f'upper_arm {ua_max:.0f}deg >= 45'
            elif neck_mod:
                is_medium = True
                neck_capped = True
                al_reason = f'neck_angle {neck_val:.0f}deg >= 45 (capped)'
            elif moderate_primaries >= 2:
                is_medium = True
                al_reason = f'{moderate_primaries} moderate primary deviations'

        # ---- 4. FINAL ACTION LEVEL ------------------------------------
        if is_high:
            action_level = 3
            final_score = 3
        elif is_medium:
            action_level = 2
            final_score = 2
        else:
            action_level = 1
            final_score = 1
            al_reason = 'no significant postural deviation'

        # Primary risk drivers (for debug)
        if trunk_seg >= 3 and 'trunk' not in primary_drivers:
            primary_drivers.append('trunk')
        if ua_seg >= 3 and 'upper_arm' not in primary_drivers:
            primary_drivers.append('upper_arm')
        if neck_seg >= 3 and not neck_capped:
            primary_drivers.append('neck')

        return self._build_result(
            action_level, final_score, al_reason, diag_continuous,
            trunk_seg, neck_seg, ua_seg, fa_seg, leg_seg,
            trunk_val, neck_val, ua_max, ua_left, ua_right,
            partial_scores, diag_severities,
            neutral_gate_applied=False, neck_capped=neck_capped,
            primary_drivers=primary_drivers,
            secondary_drivers=[], n_valid_kp=n_valid_kp,
            confidence=confidence, manual_context=manual_context,
        )

    # -- result builder -------------------------------------------------

    @staticmethod
    def _build_result(action_level, final_score, al_reason, diag_continuous,
                      trunk_seg, neck_seg, ua_seg, fa_seg, leg_seg,
                      trunk_val, neck_val, ua_max, ua_left, ua_right,
                      partial_scores, diag_severities,
                      neutral_gate_applied, neck_capped,
                      primary_drivers, secondary_drivers,
                      n_valid_kp, confidence, manual_context):
        """Build the output dict with mapping consistency check."""

        uncertain = n_valid_kp < 5
        risk_class = RISK_TO_AL_LABEL[action_level]
        risk_level_short = RISK_TO_LEVEL_SHORT[action_level]
        overlay_color = RISK_TO_OVERLAY_COLOUR[action_level]

        # Mapping consistency: final_score must match action_level
        mapping_consistent = (final_score == action_level)

        # Explanation
        parts = []
        if uncertain:
            parts.append(f'Low-confidence estimate: {risk_class} (confidence {confidence:.2f}).')
        elif neutral_gate_applied:
            parts.append(f'{risk_class} — neutral posture gate.')
        elif action_level == 3:
            parts.append(f'{risk_class} — {al_reason}.')
            if primary_drivers:
                parts.append(f'Primary: {", ".join(primary_drivers)}.')
            if neck_capped:
                parts.append('[neck_capped]')
        elif action_level == 2:
            parts.append(f'{risk_class} — {al_reason}.')
            if primary_drivers:
                parts.append(f'Primary: {", ".join(primary_drivers)}.')
            if neck_capped:
                parts.append('[neck_capped]')
        else:
            parts.append(f'{risk_class} — {al_reason}.')

        if confidence < 0.5 and not uncertain:
            parts.append(f'[Reduced confidence: {confidence:.2f}]')
        explanation = ' '.join(parts)

        return {
            # Core fields
            'final_risk_class': risk_class,
            'final_score': final_score,
            'risk_level': risk_level_short,
            'action_level': action_level,
            'overlay_color': overlay_color,
            'continuous_severity': round(diag_continuous, 1),
            'confidence': round(confidence, 3),
            'explanation': explanation,
            'partial_scores': partial_scores,
            'unavailable_features': [
                n for n, ps in partial_scores.items()
                if ps.get('value') is None
            ],

            # Segment scores
            'trunk_score': trunk_seg,
            'neck_score': neck_seg,
            'upper_arm_score': ua_seg,
            'forearm_score': fa_seg,
            'leg_score': leg_seg,

            # Key raw angles (for CSV)
            'trunk_angle': trunk_val,
            'neck_angle': neck_val,
            'upper_arm_angle_left': ua_left,
            'upper_arm_angle_right': ua_right,

            # Debug
            'approximate_action_level': action_level,
            'action_level_reason': al_reason,
            'neutral_gate_applied': neutral_gate_applied,
            'neck_capped': neck_capped,
            'primary_risk_drivers': primary_drivers,
            'secondary_risk_drivers': secondary_drivers,
            'mapping_consistent': mapping_consistent,
            'diagnostic_severities': diag_severities,

            # Legacy interface
            'uncertain': uncertain,
            'uncertainty_reason': (
                f'Very few reliable keypoints ({n_valid_kp}/17).' if uncertain else ''
            ),
        }


# ===================================================================
# Temporal smoothing (asymmetric EMA — works on continuous_severity)
# ===================================================================

class TemporalSmoothedScorer:
    """Wraps :class:`ErgonomicScorer` with asymmetric EMA."""

    def __init__(self, alpha=0.25, decay_alpha=0.50,
                 severity_low_max=35, severity_medium_max=65):
        self.scorer = ErgonomicScorer()
        self.alpha = alpha
        self.decay_alpha = decay_alpha
        self.severity_low_max = severity_low_max
        self.severity_medium_max = severity_medium_max
        self._state = {}

    def score(self, keypoints, tracking_key=None, manual_context=None):
        raw = self.scorer.score(keypoints, manual_context=manual_context)
        raw_sev = raw['continuous_severity']

        if tracking_key is None:
            raw['smoothed_severity'] = raw_sev
            raw['persistence_frames'] = 1
            return raw

        prev_ema, prev_class, persistence = self._state.get(
            tracking_key, (raw_sev, raw['final_score'], 0)
        )

        alpha_eff = self.decay_alpha if raw_sev < prev_ema else self.alpha
        smoothed = alpha_eff * raw_sev + (1 - alpha_eff) * prev_ema
        persistence += 1
        self._state[tracking_key] = (smoothed, raw['final_score'], persistence)

        raw_risk_level = raw['final_risk_class']
        if raw['uncertain'] and raw['final_score'] == 0:
            final_score = 0
            final_class = 'Uncertain'
        elif smoothed > self.severity_medium_max:
            final_score, final_class = 3, RISK_TO_AL_LABEL[3]
        elif smoothed > self.severity_low_max:
            final_score, final_class = 2, RISK_TO_AL_LABEL[2]
        else:
            final_score, final_class = 1, RISK_TO_AL_LABEL[1]

        raw['final_score'] = final_score
        raw['final_risk_class'] = final_class
        raw['risk_level'] = RISK_TO_LEVEL_SHORT[final_score]
        raw['action_level'] = final_score
        raw['raw_severity'] = round(raw_sev, 1)
        raw['smoothed_severity'] = round(smoothed, 1)
        raw['persistence_frames'] = persistence
        raw['ema_alpha_used'] = round(alpha_eff, 2)

        raw['explanation'] += (
            f' [temporal: raw {raw_sev:.0f} → '
            f'smoothed {smoothed:.0f} (α={alpha_eff:.2f}), '
            f'persistence {persistence} frame(s)]'
        )
        return raw

    def reset(self, tracking_key=None):
        if tracking_key is None:
            self._state.clear()
        else:
            self._state.pop(tracking_key, None)


def temporal_smooth_severity(results_rows, alpha=0.25, decay_alpha=0.50,
                              severity_low_max=35, severity_medium_max=65):
    """Post-process rows with asymmetric EMA (post-hoc)."""
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
            if ema is None:
                ema = sev
            else:
                alpha_eff = decay_alpha if sev < ema else alpha
                ema = alpha_eff * sev + (1 - alpha_eff) * ema

            r['smoothed_severity'] = round(ema, 1)
            r['raw_severity'] = round(sev, 1)

            # Re-classify from EMA severity (only affects risk_score/risk_level)
            ema_score = 3 if ema > severity_medium_max else (2 if ema > severity_low_max else 1)
            r['risk_score'] = ema_score
            r['risk_level'] = RISK_TO_LEVEL_SHORT[ema_score]
            r['action_level'] = ema_score

            smoothed_rows.append(r)

    return smoothed_rows
