"""
decision_engine.py
------------------
Requirement 5: One Final Decision Per Arm
Combines all four signals into a single decision:
    EMERGENCY_STOP      – immediate danger, halt NOW
    SCHEDULE_MAINTENANCE – wear detected AND maintenance logs confirm issues
    MONITOR             – either ML or maintenance logs raise concern (but not both)
    SAFE                – all signals clear

Decision Priority (waterfall):
    1. Safety threshold crossed → EMERGENCY_STOP  (overrides everything)
    2. Spoof detected with stress context → EMERGENCY_STOP
    3. Wear detected AND maintenance concern → SCHEDULE_MAINTENANCE
    4. Wear detected OR maintenance concern → MONITOR
    5. All clear → SAFE
"""

DECISIONS = ["SAFE", "MONITOR", "SCHEDULE_MAINTENANCE", "EMERGENCY_STOP"]


def make_decision(
    safety_result:      dict,
    spoof_result:       dict,
    wear_result:        dict,
    maintenance_result: dict,
) -> dict:
    """
    Combine all module outputs into a single final decision.

    Parameters (all dicts from their respective modules):
        safety_result       – from safety_check.check_safety()
        spoof_result        – from spoof_detector.detect_spoof()
        wear_result         – from wear_detector.predict_wear()
        maintenance_result  – from maintenance_analyzer.analyze_maintenance_log()

    Returns
    -------
    dict with:
        - 'decision'  : str  (one of the four decisions above)
        - 'reasons'   : list[str]  (human-readable explanation of why)
    """
    reasons = []
    decision = "SAFE"

    # ── Rule 1: Hard safety threshold crossed ────────────────────────────────
    if safety_result.get("emergency"):
        decision = "EMERGENCY_STOP"
        for v in safety_result["violations"]:
            reasons.append(
                f"SAFETY THRESHOLD EXCEEDED – {v['joint']} {v['metric']}: "
                f"{v['value']:.1f} > {v['threshold']:.1f}"
            )
        # Short-circuit — no need to check further
        return {"decision": decision, "reasons": reasons}

    # ── Rule 2: Spoof attack detected ────────────────────────────────────────
    if spoof_result.get("spoof_detected"):
        decision = "EMERGENCY_STOP"
        reasons.append(
            f"SPOOF ATTACK DETECTED on joints {spoof_result['suspicious_joints']} "
            f"via {spoof_result['method']}. Cannot trust sensor data."
        )
        for d in spoof_result.get("details", []):
            reasons.append(f"  → {d}")
        return {"decision": decision, "reasons": reasons}

    # ── Rule 3: Both ML and maintenance raise concern → schedule maintenance ─
    wear_flag        = wear_result.get("wear_detected", False)
    maintenance_flag = maintenance_result.get("maintenance_concern", False)

    if wear_flag and maintenance_flag:
        decision = "SCHEDULE_MAINTENANCE"
        reasons.append(
            f"ML model predicts wear (confidence: {wear_result['confidence']:.0%})"
        )
        reasons.append(
            f"Maintenance log independently confirms concern "
            f"(score: {maintenance_result['total_score']}, "
            f"keywords: {maintenance_result['keywords_found']})"
        )
        return {"decision": decision, "reasons": reasons}

    # ── Rule 4: Only one signal raises concern → monitor ─────────────────────
    if wear_flag:
        decision = "MONITOR"
        reasons.append(
            f"ML model predicts wear (confidence: {wear_result['confidence']:.0%}), "
            f"but maintenance log has no strong concern."
        )
        return {"decision": decision, "reasons": reasons}

    if maintenance_flag:
        decision = "MONITOR"
        reasons.append(
            f"Maintenance log raises concern "
            f"(score: {maintenance_result['total_score']}, "
            f"keywords: {maintenance_result['keywords_found']}), "
            f"but ML model sees no wear."
        )
        return {"decision": decision, "reasons": reasons}

    # ── Rule 5: All clear ────────────────────────────────────────────────────
    decision = "SAFE"
    reasons.append("All sensors within safe limits. No wear predicted. No maintenance concerns.")
    return {"decision": decision, "reasons": reasons}
