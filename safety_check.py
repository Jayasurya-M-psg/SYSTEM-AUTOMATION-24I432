"""
safety_check.py
---------------
Requirement 1: Immediate Safety Stop
Rule-based, deterministic — NEVER uses the ML model.
Flags EMERGENCY_STOP if ANY single joint reading crosses a dangerous threshold.
"""

# ─── Safety Thresholds ────────────────────────────────────────────────────────
# These values represent the absolute upper bounds for safe operation.
# Crossing ANY threshold on ANY joint triggers an immediate stop.
SAFETY_THRESHOLDS = {
    "torque_nm":        85.0,   # Nm  – beyond 85 Nm, joint actuator risks structural damage
    "temperature_c":    80.0,   # °C  – above 80°C, motor windings and lubricants degrade rapidly
    "vibration_mm_s":    9.5,   # mm/s – above 9.5 mm/s, fastener fatigue and bearing failure risk
}


def check_safety(readings: list[dict]) -> dict:
    """
    Perform an immediate, rule-based safety check on a list of joint readings.

    Parameters
    ----------
    readings : list of dict
        Each dict must have keys: 'joint', 'torque_nm', 'temperature_c', 'vibration_mm_s'

    Returns
    -------
    dict with keys:
        - 'emergency'  : bool  – True if any threshold is violated
        - 'violations' : list  – details of each violated reading
    """
    violations = []

    for reading in readings:
        joint = reading.get("joint", "unknown")
        for metric, threshold in SAFETY_THRESHOLDS.items():
            value = reading.get(metric)
            if value is None:
                continue
            if value > threshold:
                violations.append({
                    "joint":   joint,
                    "metric":  metric,
                    "value":   value,
                    "threshold": threshold,
                })

    return {
        "emergency":  len(violations) > 0,
        "violations": violations,
    }
