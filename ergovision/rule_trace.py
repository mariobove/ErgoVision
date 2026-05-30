"""
Rule trace generator for ErgoVision interpretability.

Produces a structured JSON trace of every rule evaluated during scoring,
including operand values, thresholds, and firing status, so each risk
decision can be fully audited frame-by-frame.

Saved during pipeline execution as ``rule_trace.json`` in the output dir.
"""

from pathlib import Path
from typing import Any, Optional


# ===================================================================
# Per-frame rule trace builder
# ===================================================================

def _val(partial_scores: dict, name: str) -> Optional[float]:
    """Extract a float angle value from partial_scores."""
    ps = partial_scores.get(name, {})
    v = ps.get('value') if ps else None
    return float(v) if v is not None else None


def _op(variable: str, value: Optional[float], threshold: float,
        comparison: str) -> dict:
    """Build an operand dict with pass/fail judgement."""
    if value is None:
        return {
            "variable": variable,
            "value": None,
            "threshold": threshold,
            "comparison": comparison,
            "passed": False,
            "available": False,
        }
    if comparison == '<':
        passed = value < threshold
    elif comparison == '<=':
        passed = value <= threshold
    elif comparison == '>=':
        passed = value >= threshold
    elif comparison == '>':
        passed = value > threshold
    elif comparison == '==':
        passed = value == threshold
    else:
        passed = False
    return {
        "variable": variable,
        "value": round(value, 1),
        "threshold": threshold,
        "comparison": comparison,
        "passed": passed,
        "available": True,
    }


def _single_op_rule(rule_id: str, name: str, description: str,
                    condition: str, consequence_label: str, category: str,
                    variable: str, value: Optional[float],
                    threshold: float, comparison: str) -> dict:
    """Build a trace entry for a single-operand rule."""
    operand = _op(variable, value, threshold, comparison)
    triggered = operand['passed']
    return {
        "id": rule_id,
        "name": name,
        "description": description,
        "condition": condition,
        "category": category,
        "operands": [operand],
        "passed": triggered,
        "triggered": triggered,
        "consequence": f"→ {consequence_label}" if triggered else None,
    }


def build_person_rule_trace(score_result: dict,
                            partial_scores: dict,
                            frame_id: str = '',
                            person_id: int = 0) -> dict:
    """Build a structured rule trace for one person detection.

    Parameters
    ----------
    score_result : dict
        Output from ``ErgonomicScorer.score()``.
    partial_scores : dict
        Per-segment angle and score data (same as
        ``score_result['partial_scores']``).
    frame_id : str
        Frame identifier (for context).
    person_id : int
        Person index within the frame.

    Returns
    -------
    dict — full rule trace.
    """
    # --- Extract feature values ---
    trunk_val = _val(partial_scores, 'trunk_angle')
    neck_val = _val(partial_scores, 'neck_angle')
    ua_left = _val(partial_scores, 'upper_arm_angle_left')
    ua_right = _val(partial_scores, 'upper_arm_angle_right')
    incl_val = _val(partial_scores, 'body_inclination')

    ua_vals = [v for v in [ua_left, ua_right] if v is not None]
    ua_max = max(ua_vals) if ua_vals else None

    # --- Segment scores (from score_result) ---
    trunk_seg = score_result.get('trunk_score', 1)
    neck_seg = score_result.get('neck_score', 1)
    ua_seg = score_result.get('upper_arm_score', 1)

    rules = []

    # ---- 1. Neutral Posture Gate ----
    operands = [
        _op('trunk_val < 20', trunk_val, 20, '<'),
        _op('ua_max < 45', ua_max, 45, '<'),
        _op('neck_val < 30', neck_val, 30, '<'),
        _op('incl_val < 20', incl_val, 20, '<'),
    ]
    neutral_gate_passed = all(
        o['passed'] for o in operands if o['available']
    ) and all(o['available'] for o in operands)
    rules.append({
        "id": "neutral_gate",
        "name": "Neutral Posture Gate",
        "description": "If all body segments are within neutral range, classify as LOW.",
        "condition": "trunk < 20° AND upper_arm < 45° AND neck < 30° AND inclination < 20°",
        "category": "gate",
        "operands": operands,
        "passed": neutral_gate_passed,
        "triggered": neutral_gate_passed,
        "consequence": "→ AL1 (LOW) — neutral posture" if neutral_gate_passed else None,
    })

    # ---- 2-5. HIGH rules ----
    rules.append(_single_op_rule(
        "high_trunk_60", "HIGH: trunk >= 60°",
        "Severe trunk flexion.",
        "trunk_angle >= 60°", "HIGH", "high",
        "trunk_angle >= 60", trunk_val, 60, '>=',
    ))

    ops_h45a60 = [
        _op('trunk_angle >= 45', trunk_val, 45, '>='),
        _op('upper_arm_max >= 60', ua_max, 60, '>='),
    ]
    h45a60 = all(o['passed'] for o in ops_h45a60)
    rules.append({
        "id": "high_trunk_45_arm_60",
        "name": "HIGH: trunk >= 45° AND upper_arm >= 60°",
        "description": "Moderate trunk flexion combined with arm elevation.",
        "condition": "trunk_angle >= 45° AND upper_arm_max >= 60°",
        "category": "high",
        "operands": ops_h45a60,
        "passed": h45a60,
        "triggered": h45a60,
        "consequence": "→ HIGH" if h45a60 else None,
    })

    ops_a90t30 = [
        _op('upper_arm_max >= 90', ua_max, 90, '>='),
        _op('trunk_angle >= 30', trunk_val, 30, '>='),
    ]
    a90t30 = all(o['passed'] for o in ops_a90t30)
    rules.append({
        "id": "high_arm_90_trunk_30",
        "name": "HIGH: upper_arm >= 90° AND trunk >= 30°",
        "description": "Severe arm elevation with moderate trunk flexion.",
        "condition": "upper_arm_max >= 90° AND trunk_angle >= 30°",
        "category": "high",
        "operands": ops_a90t30,
        "passed": a90t30,
        "triggered": a90t30,
        "consequence": "→ HIGH" if a90t30 else None,
    })

    severe_count = sum([1 for s in [trunk_seg, neck_seg, ua_seg] if s >= 3])
    ops_severe = [
        {"variable": "trunk_score", "value": trunk_seg, "threshold": 3,
         "comparison": ">=", "passed": trunk_seg >= 3, "available": True},
        {"variable": "neck_score", "value": neck_seg, "threshold": 3,
         "comparison": ">=", "passed": neck_seg >= 3, "available": True},
        {"variable": "upper_arm_score", "value": ua_seg, "threshold": 3,
         "comparison": ">=", "passed": ua_seg >= 3, "available": True},
        {"variable": "severe_count (≥2 triggers)", "value": severe_count,
         "threshold": 2, "comparison": ">=", "passed": severe_count >= 2,
         "available": True},
    ]
    high_severe = severe_count >= 2
    rules.append({
        "id": "high_severe_primaries",
        "name": "HIGH: ≥ 2 severe primary deviations",
        "description": "Two or more primary segments (trunk, neck, upper arm) have score ≥ 3.",
        "condition": "count(segment_score >= 3) >= 2",
        "category": "high",
        "operands": ops_severe,
        "passed": high_severe,
        "triggered": high_severe,
        "consequence": "→ HIGH" if high_severe else None,
    })

    # ---- 6. Neck HIGH cap ----
    neck_capped = score_result.get('neck_capped', False)
    ops_neck_cap = [
        _op('neck_score >= 3', neck_seg, 3, '>='),
        _op('trunk_score < 3', trunk_seg, 3, '<'),
        _op('upper_arm_score < 3', ua_seg, 3, '<'),
    ]
    rules.append({
        "id": "neck_high_cap",
        "name": "Neck HIGH Cap",
        "description": "Neck alone cannot produce HIGH. If neck is severe but trunk and arms are not, cap to MEDIUM.",
        "condition": "neck_score >= 3 AND trunk_score < 3 AND upper_arm_score < 3",
        "category": "cap",
        "operands": ops_neck_cap,
        "passed": neck_capped,
        "triggered": neck_capped,
        "consequence": "→ MEDIUM (neck capped from HIGH)" if neck_capped else None,
    })

    # ---- 7-10. MEDIUM rules ----
    rules.append(_single_op_rule(
        "medium_trunk_30", "MEDIUM: trunk >= 30°",
        "Moderate trunk flexion.",
        "trunk_angle >= 30°", "MEDIUM", "medium",
        "trunk_angle >= 30", trunk_val, 30, '>=',
    ))

    rules.append(_single_op_rule(
        "medium_arm_45", "MEDIUM: upper_arm >= 45°",
        "Moderate arm elevation.",
        "upper_arm_max >= 45°", "MEDIUM", "medium",
        "upper_arm_max >= 45", ua_max, 45, '>=',
    ))

    rules.append(_single_op_rule(
        "medium_neck_45", "MEDIUM: neck >= 45° (capped)",
        "Moderate neck flexion — automatically capped to MEDIUM.",
        "neck_angle >= 45°", "MEDIUM", "medium",
        "neck_angle >= 45", neck_val, 45, '>=',
    ))

    moderate_count = sum([1 for s in [trunk_seg, neck_seg, ua_seg] if s >= 2])
    ops_moderate = [
        {"variable": "trunk_score", "value": trunk_seg, "threshold": 2,
         "comparison": ">=", "passed": trunk_seg >= 2, "available": True},
        {"variable": "neck_score", "value": neck_seg, "threshold": 2,
         "comparison": ">=", "passed": neck_seg >= 2, "available": True},
        {"variable": "upper_arm_score", "value": ua_seg, "threshold": 2,
         "comparison": ">=", "passed": ua_seg >= 2, "available": True},
        {"variable": "moderate_count (≥2 triggers)", "value": moderate_count,
         "threshold": 2, "comparison": ">=", "passed": moderate_count >= 2,
         "available": True},
    ]
    med_moderate = moderate_count >= 2
    rules.append({
        "id": "medium_moderate_primaries",
        "name": "MEDIUM: ≥ 2 moderate primary deviations",
        "description": "Two or more primary segments have score ≥ 2.",
        "condition": "count(segment_score >= 2) >= 2",
        "category": "medium",
        "operands": ops_moderate,
        "passed": med_moderate,
        "triggered": med_moderate,
        "consequence": "→ MEDIUM" if med_moderate else None,
    })

    # ---- 11. LOW fallback ----
    high_triggered = any(
        r['triggered'] for r in rules
        if r['id'] in ('high_trunk_60', 'high_trunk_45_arm_60',
                       'high_arm_90_trunk_30', 'high_severe_primaries')
    )
    med_triggered = any(
        r['triggered'] for r in rules
        if r['id'] in ('medium_trunk_30', 'medium_arm_45',
                       'medium_neck_45', 'medium_moderate_primaries')
    )
    low_fallback = not neutral_gate_passed and not high_triggered and not med_triggered

    rules.append({
        "id": "low_fallback",
        "name": "LOW (fallback)",
        "description": "No significant postural deviation detected.",
        "condition": "no HIGH or MEDIUM rule triggered",
        "category": "low",
        "operands": [
            {"variable": "neutral_gate_passed", "value": neutral_gate_passed,
             "threshold": True, "comparison": "==",
             "passed": neutral_gate_passed, "available": True},
            {"variable": "any_HIGH_rule_triggered", "value": high_triggered,
             "threshold": False, "comparison": "==",
             "passed": not high_triggered, "available": True},
            {"variable": "any_MEDIUM_rule_triggered", "value": med_triggered,
             "threshold": False, "comparison": "==",
             "passed": not med_triggered, "available": True},
        ],
        "passed": low_fallback,
        "triggered": low_fallback,
        "consequence": "→ AL1 (LOW) — fallback" if low_fallback else None,
    })

    # ---- Build decision path ----
    action_level = score_result.get('action_level', 1)
    decision_path = []

    decision_path.append({
        "step": 1,
        "evaluation": "Neutral Posture Gate",
        "outcome": "PASSED → AL1 (LOW)" if neutral_gate_passed else "FAILED → continue",
    })

    if not neutral_gate_passed:
        high_fired = [r for r in rules if r['triggered'] and r['category'] == 'high']
        if high_fired:
            for r in high_fired:
                decision_path.append({
                    "step": len(decision_path) + 1,
                    "evaluation": f"HIGH rule: {r['name']}",
                    "outcome": "TRIGGERED",
                })
            if neck_capped:
                decision_path.append({
                    "step": len(decision_path) + 1,
                    "evaluation": "Neck HIGH Cap",
                    "outcome": "APPLIED → downgraded to MEDIUM",
                })
            else:
                decision_path.append({
                    "step": len(decision_path) + 1,
                    "evaluation": "Final action level",
                    "outcome": "→ HIGH (AL3)",
                })
        else:
            decision_path.append({
                "step": len(decision_path) + 1,
                "evaluation": "All HIGH rules",
                "outcome": "none triggered → continue",
            })

    if not neutral_gate_passed and not high_triggered:
        med_fired = [r for r in rules if r['triggered'] and r['category'] == 'medium']
        if med_fired:
            for r in med_fired:
                decision_path.append({
                    "step": len(decision_path) + 1,
                    "evaluation": f"MEDIUM rule: {r['name']}",
                    "outcome": "TRIGGERED",
                })
            decision_path.append({
                "step": len(decision_path) + 1,
                "evaluation": "Final action level",
                "outcome": "→ MEDIUM (AL2)",
            })
        elif not high_triggered:
            decision_path.append({
                "step": len(decision_path) + 1,
                "evaluation": "All MEDIUM rules",
                "outcome": "none triggered → continue",
            })
            decision_path.append({
                "step": len(decision_path) + 1,
                "evaluation": "LOW fallback",
                "outcome": "→ LOW (AL1)",
            })

    return {
        "frame_id": frame_id,
        "person_id": person_id,
        "action_level": action_level,
        "risk_class": score_result.get('final_risk_class', ''),
        "risk_score": score_result.get('final_score', 1),
        "risk_level_short": score_result.get('risk_level', ''),
        "action_level_reason": score_result.get('action_level_reason', ''),
        "neutral_gate_applied": score_result.get('neutral_gate_applied', False),
        "neck_capped": score_result.get('neck_capped', False),
        "explanation": score_result.get('explanation', ''),
        "angle_values": {
            "trunk_angle": trunk_val,
            "neck_angle": neck_val,
            "upper_arm_angle_left": _val(partial_scores, 'upper_arm_angle_left'),
            "upper_arm_angle_right": _val(partial_scores, 'upper_arm_angle_right'),
            "forearm_angle_left": _val(partial_scores, 'forearm_angle_left'),
            "forearm_angle_right": _val(partial_scores, 'forearm_angle_right'),
            "knee_bend_left": _get_bend(partial_scores, 'knee_angle_left'),
            "knee_bend_right": _get_bend(partial_scores, 'knee_angle_right'),
            "shoulder_asymmetry": _val(partial_scores, 'shoulder_asymmetry'),
            "body_inclination": _val(partial_scores, 'body_inclination'),
        },
        "segment_scores": {
            "trunk": trunk_seg,
            "neck": neck_seg,
            "upper_arm": ua_seg,
            "forearm": score_result.get('forearm_score', 1),
            "leg": score_result.get('leg_score', 1),
        },
        "rules": rules,
        "decision_path": decision_path,
    }


def _get_bend(partial_scores: dict, name: str) -> Optional[float]:
    """Extract knee bend angle (not the joint angle)."""
    ps = partial_scores.get(name, {})
    if ps and 'bend' in ps:
        v = ps['bend']
        return round(float(v), 1) if v is not None else None
    # fallback: compute from value (180 - joint_angle)
    v = ps.get('value') if ps else None
    return round(180.0 - float(v), 1) if v is not None else None


# ===================================================================
# Batch collection and file saving
# ===================================================================

def save_rule_trace(all_traces: list[dict], output_path: Path) -> None:
    """Save the full rule trace collection as pretty-printed JSON.

    Parameters
    ----------
    all_traces : list[dict]
        List of per-person rule traces (output of
        ``build_person_rule_trace``).
    output_path : Path
        Destination path (e.g. ``outputs/<dataset>/rule_trace.json``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Summary statistics
    action_levels = {}
    triggered_rules = {}
    for t in all_traces:
        al = t['action_level']
        action_levels[al] = action_levels.get(al, 0) + 1
        for r in t['rules']:
            if r['triggered']:
                rid = r['id']
                triggered_rules[rid] = triggered_rules.get(rid, 0) + 1

    document = {
        "metadata": {
            "total_persons_traced": len(all_traces),
            "action_level_distribution": {
                str(k): v for k, v in sorted(action_levels.items())
            },
            "rules_triggered_count": triggered_rules,
        },
        "rules_dictionary": [
            {
                "id": "neutral_gate",
                "name": "Neutral Posture Gate",
                "condition": "trunk < 20° AND upper_arm < 45° AND neck < 30° AND inclination < 20°",
                "consequence": "AL1 (LOW) immediately",
            },
            {
                "id": "high_trunk_60",
                "name": "HIGH: trunk >= 60°",
                "condition": "trunk_angle >= 60°",
                "consequence": "HIGH (AL3)",
            },
            {
                "id": "high_trunk_45_arm_60",
                "name": "HIGH: trunk >= 45° AND upper_arm >= 60°",
                "condition": "trunk_angle >= 45° AND upper_arm_max >= 60°",
                "consequence": "HIGH (AL3)",
            },
            {
                "id": "high_arm_90_trunk_30",
                "name": "HIGH: upper_arm >= 90° AND trunk >= 30°",
                "condition": "upper_arm_max >= 90° AND trunk_angle >= 30°",
                "consequence": "HIGH (AL3)",
            },
            {
                "id": "high_severe_primaries",
                "name": "HIGH: ≥ 2 severe primary deviations",
                "condition": "count(segment_score >= 3) >= 2",
                "consequence": "HIGH (AL3)",
            },
            {
                "id": "neck_high_cap",
                "name": "Neck HIGH Cap",
                "condition": "neck_score >= 3 AND trunk_score < 3 AND upper_arm_score < 3",
                "consequence": "Downgrade HIGH→MEDIUM",
            },
            {
                "id": "medium_trunk_30",
                "name": "MEDIUM: trunk >= 30°",
                "condition": "trunk_angle >= 30°",
                "consequence": "MEDIUM (AL2)",
            },
            {
                "id": "medium_arm_45",
                "name": "MEDIUM: upper_arm >= 45°",
                "condition": "upper_arm_max >= 45°",
                "consequence": "MEDIUM (AL2)",
            },
            {
                "id": "medium_neck_45",
                "name": "MEDIUM: neck >= 45° (capped)",
                "condition": "neck_angle >= 45°",
                "consequence": "MEDIUM (AL2)",
            },
            {
                "id": "medium_moderate_primaries",
                "name": "MEDIUM: ≥ 2 moderate primary deviations",
                "condition": "count(segment_score >= 2) >= 2",
                "consequence": "MEDIUM (AL2)",
            },
            {
                "id": "low_fallback",
                "name": "LOW (fallback)",
                "condition": "no HIGH or MEDIUM rule triggered",
                "consequence": "LOW (AL1)",
            },
        ],
        "frames": all_traces,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(document, f, indent=2, ensure_ascii=False)

    print(f"  Saved rule trace: {output_path} ({len(all_traces)} persons)")
