"""
main.py
-------
Robotic Arm Joint Health Monitoring System
==========================================
Full pipeline entry point.

Demonstrates all four possible decisions:
  1. SAFE
  2. MONITOR
  3. SCHEDULE_MAINTENANCE
  4. EMERGENCY_STOP

Usage:
    python main.py
    python main.py --arm ARM_07                  # analyze a specific arm from the log
    python main.py --demo all                    # run all four demo scenarios
    python main.py --demo safe
    python main.py --demo monitor
    python main.py --demo schedule
    python main.py --demo emergency
"""

import argparse
import json
import os
import sys

from safety_check       import check_safety
from wear_detector      import train_model, predict_wear
from maintenance_analyzer import analyze_maintenance_log
from spoof_detector     import detect_spoof
from decision_engine    import make_decision

# ─── File paths ───────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH    = os.path.join(BASE_DIR, "historical_data.csv")
LOG_PATH        = os.path.join(BASE_DIR, "maintenance_log.txt")


# ─── Demo Scenarios ───────────────────────────────────────────────────────────
DEMO_SCENARIOS = {

    "safe": {
        "arm_id":      "ARM_04",
        "description": "New arm, low hours, all readings well within limits.",
        "readings": [
            {"joint": "shoulder", "torque_nm": 38.6, "temperature_c": 47.2, "vibration_mm_s": 2.4, "operating_hours": 60},
            {"joint": "elbow",    "torque_nm": 40.1, "temperature_c": 49.5, "vibration_mm_s": 2.8, "operating_hours": 60},
            {"joint": "wrist",    "torque_nm": 25.3, "temperature_c": 41.8, "vibration_mm_s": 1.9, "operating_hours": 60},
        ],
    },

    "monitor": {
        "arm_id":      "ARM_19",
        "description": "High-hour arm; ML detects wear beginning, but log has no specific concerns for ARM_19.",
        "readings": [
            {"joint": "shoulder", "torque_nm": 72.0, "temperature_c": 70.5, "vibration_mm_s": 7.3, "operating_hours": 820},
            {"joint": "elbow",    "torque_nm": 68.5, "temperature_c": 68.2, "vibration_mm_s": 7.0, "operating_hours": 820},
            {"joint": "wrist",    "torque_nm": 53.1, "temperature_c": 64.8, "vibration_mm_s": 6.2, "operating_hours": 820},
        ],
    },

    "schedule": {
        "arm_id":      "ARM_07",
        "description": "High hours; ML detects wear AND maintenance log has concerns about ARM_07.",
        "readings": [
            {"joint": "shoulder", "torque_nm": 73.1, "temperature_c": 70.3, "vibration_mm_s": 7.5, "operating_hours": 900},
            {"joint": "elbow",    "torque_nm": 76.8, "temperature_c": 73.8, "vibration_mm_s": 8.1, "operating_hours": 900},
            {"joint": "wrist",    "torque_nm": 58.2, "temperature_c": 67.5, "vibration_mm_s": 6.9, "operating_hours": 900},
        ],
    },

    "emergency": {
        "arm_id":      "ARM_99",
        "description": "Critical failure — shoulder torque exceeds 85 Nm safety threshold.",
        "readings": [
            {"joint": "shoulder", "torque_nm": 91.3, "temperature_c": 83.7, "vibration_mm_s": 10.2, "operating_hours": 1100},
            {"joint": "elbow",    "torque_nm": 77.4, "temperature_c": 74.2, "vibration_mm_s": 8.3,  "operating_hours": 1100},
            {"joint": "wrist",    "torque_nm": 60.1, "temperature_c": 70.5, "vibration_mm_s": 7.1,  "operating_hours": 1100},
        ],
    },
}


# ─── Core Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(arm_id: str, readings: list[dict]) -> dict:
    """Run the full monitoring pipeline for one arm and return structured results."""

    print(f"\n{'=' * 60}")
    print(f"  ARM: {arm_id}")
    print(f"{'=' * 60}")

    # Step 1 – Immediate safety check (rule-based, never uses ML)
    safety  = check_safety(readings)

    # Step 2 – Predictive wear detection (ML)
    wear    = predict_wear(readings)

    # Step 3 – Maintenance log analysis (keyword NLP)
    maint   = analyze_maintenance_log(LOG_PATH, arm_id)

    # Step 4 – Spoof detection (cross-sensor + statistical)
    spoof   = detect_spoof(readings)

    # Step 5 – Final decision
    outcome = make_decision(safety, spoof, wear, maint)

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n[1] Safety Check      : {'⚠ EMERGENCY' if safety['emergency'] else '✓ OK'}")
    if safety["violations"]:
        for v in safety["violations"]:
            print(f"     ! {v['joint']} {v['metric']} = {v['value']:.1f} (limit {v['threshold']:.1f})")

    print(f"[2] Wear Prediction   : {wear['prediction'].upper()} "
          f"(confidence: {wear['confidence']:.0%})")

    print(f"[3] Maintenance Log   : {'⚠ CONCERN' if maint['maintenance_concern'] else '✓ OK'} "
          f"(score: {maint['total_score']})")
    if maint["matched_entries"]:
        for note in maint["matched_entries"]:
            print(f"     → {note[:80]}{'...' if len(note) > 80 else ''}")

    print(f"[4] Spoof Detection   : {'⚠ SPOOF DETECTED' if spoof['spoof_detected'] else '✓ OK'}")
    if spoof["details"]:
        for d in spoof["details"]:
            print(f"     ! {d}")

    print(f"\n{'─' * 60}")
    print(f"  FINAL DECISION: >>> {outcome['decision']} <<<")
    print(f"{'─' * 60}")
    for r in outcome["reasons"]:
        print(f"  • {r}")
    print()

    return {
        "arm_id":      arm_id,
        "safety":      safety,
        "wear":        wear,
        "maintenance": maint,
        "spoof":       spoof,
        "decision":    outcome,
    }


def run_demo(scenario: str):
    """Run one or all demonstration scenarios."""
    if scenario == "all":
        scenarios = list(DEMO_SCENARIOS.keys())
    elif scenario in DEMO_SCENARIOS:
        scenarios = [scenario]
    else:
        print(f"Unknown scenario '{scenario}'. Choose from: {list(DEMO_SCENARIOS.keys())} or 'all'")
        sys.exit(1)

    for s in scenarios:
        data = DEMO_SCENARIOS[s]
        print(f"\n{'#' * 60}")
        print(f"# DEMO SCENARIO: {s.upper()}")
        print(f"# {data['description']}")
        print(f"{'#' * 60}")
        run_pipeline(data["arm_id"], data["readings"])


def run_arm(arm_id: str):
    """Run pipeline for a real arm from the maintenance log with synthetic readings."""
    # Use schedule scenario readings as a base and adjust arm_id
    data = DEMO_SCENARIOS["schedule"]
    print(f"\nRunning pipeline for {arm_id} with current sensor readings...")
    run_pipeline(arm_id, data["readings"])


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("   Robotic Arm Joint Health Monitoring System")
    print("=" * 62)

    parser = argparse.ArgumentParser(description="Robotic Arm Health Monitor")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", choices=["safe", "monitor", "schedule", "emergency", "all"],
                       help="Run a predefined demo scenario")
    group.add_argument("--arm",  type=str,
                       help="Analyze a specific arm ID (e.g. ARM_07)")
    args = parser.parse_args()

    # Train the ML model once at startup
    print("\n[*] Training wear-detection model on historical dataset...")
    train_model(DATASET_PATH)

    if args.demo:
        run_demo(args.demo)
    elif args.arm:
        run_arm(args.arm)
    else:
        # Default: run all four demo scenarios to show every possible outcome
        print("\n[*] No arguments provided — running all four demo scenarios.\n")
        run_demo("all")


if __name__ == "__main__":
    main()
