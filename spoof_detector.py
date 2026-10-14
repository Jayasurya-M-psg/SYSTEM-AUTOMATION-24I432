"""
spoof_detector.py
-----------------
Requirement 4: Spoofed-Sensor Defense
Detects if a joint-torque reading has been spoofed to mask a real problem.

Attack model: An attacker feeds an artificially LOW torque value while the arm
is actually under high stress — causing the system to miss a danger signal that
temperature and vibration would otherwise reveal.

Defense strategy: Cross-validate torque against temperature and vibration.
Under normal physics, high mechanical stress raises all three simultaneously.
If torque looks suspiciously low while temperature and vibration are elevated,
the reading is flagged as a potential spoof.
"""

import numpy as np

# ─── Correlation Bounds ────────────────────────────────────────────────────────
# Based on historical data analysis:
#   • High-temperature readings (>= TEMP_HIGH) should correlate with torque >= TORQUE_EXPECTED_MIN
#   • High-vibration readings  (>= VIB_HIGH)  should correlate with torque >= TORQUE_EXPECTED_MIN
#   • If both conditions raise an alarm but torque is low, it's suspicious.

TEMP_HIGH_THRESHOLD    = 65.0   # °C  – "elevated temperature" marker
VIB_HIGH_THRESHOLD     = 6.0    # mm/s – "elevated vibration" marker
TORQUE_EXPECTED_MIN    = 55.0   # Nm  – expected minimum torque when stress signals are high
TORQUE_SUSPICIOUSLY_LOW = 40.0  # Nm  – below this when stressed → likely spoof

# Z-score threshold for statistical outlier detection across joints
ZSCORE_THRESHOLD = 2.5


def _zscore(values: list[float]) -> list[float]:
    """Compute z-scores for a list of values."""
    arr  = np.array(values, dtype=float)
    mean = arr.mean()
    std  = arr.std()
    if std == 0:
        return [0.0] * len(values)
    return list((arr - mean) / std)


def detect_spoof(readings: list[dict]) -> dict:
    """
    Detect potential sensor spoofing across a set of joint readings for one arm.

    Two independent detection methods:
    1. Cross-sensor correlation check:
       If temperature AND vibration are both HIGH but torque is SUSPICIOUSLY LOW,
       flag as spoof — the physics don't add up.

    2. Intra-arm statistical anomaly:
       If one joint's torque is a statistical outlier (low) relative to other joints
       on the same arm while those joints show elevated stress, flag it.

    Parameters
    ----------
    readings : list of dict
        Each dict: {'joint', 'torque_nm', 'temperature_c', 'vibration_mm_s'}

    Returns
    -------
    dict with:
        - 'spoof_detected'  : bool
        - 'suspicious_joints': list[str]
        - 'method'          : str  (which method triggered)
        - 'details'         : list[str]
    """
    suspicious_joints = []
    details = []
    methods_triggered = []

    # ── Method 1: Cross-sensor correlation check ──────────────────────────────
    for r in readings:
        joint = r.get("joint", "unknown")
        torque = r.get("torque_nm", 0)
        temp   = r.get("temperature_c", 0)
        vib    = r.get("vibration_mm_s", 0)

        temp_high = temp >= TEMP_HIGH_THRESHOLD
        vib_high  = vib  >= VIB_HIGH_THRESHOLD
        torque_low = torque <= TORQUE_SUSPICIOUSLY_LOW

        if temp_high and vib_high and torque_low:
            suspicious_joints.append(joint)
            details.append(
                f"[Cross-sensor] {joint}: torque={torque:.1f} Nm (suspiciously low) "
                f"while temp={temp:.1f}°C and vibration={vib:.1f} mm/s are both elevated."
            )
            methods_triggered.append("cross-sensor-correlation")

    # ── Method 2: Statistical outlier across joints ───────────────────────────
    if len(readings) >= 2:
        torques = [r.get("torque_nm", 0) for r in readings]
        temps   = [r.get("temperature_c", 0) for r in readings]
        vibs    = [r.get("vibration_mm_s", 0) for r in readings]

        torque_zscores = _zscore(torques)

        for i, r in enumerate(readings):
            joint = r.get("joint", "unknown")
            tz    = torque_zscores[i]
            # Joint torque is anomalously LOW (z < -threshold) while overall stress is high
            overall_stress = (np.mean(temps) >= 60.0 or np.mean(vibs) >= 5.0)
            if tz < -ZSCORE_THRESHOLD and overall_stress:
                if joint not in suspicious_joints:
                    suspicious_joints.append(joint)
                details.append(
                    f"[Statistical] {joint}: torque z-score={tz:.2f} (anomalously low) "
                    f"vs arm-wide mean torque={np.mean(torques):.1f} Nm under elevated stress."
                )
                methods_triggered.append("statistical-outlier")

    return {
        "spoof_detected":    len(suspicious_joints) > 0,
        "suspicious_joints": suspicious_joints,
        "method":            " + ".join(set(methods_triggered)) if methods_triggered else "none",
        "details":           details,
    }
