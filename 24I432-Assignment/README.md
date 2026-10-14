# Robotic Arm Joint Health Monitoring System

> **Bonus Challenge — Systems Automation**
> Predicting the Past → Day 2 Final Challenge bonus submission.

---

## Overview

This system monitors a fleet of 6-axis robotic arms in real time. Each arm reports sensor data from **shoulder**, **elbow**, and **wrist** joints:

| Sensor | Unit |
|---|---|
| Joint Torque | Nm (Newton-metres) |
| Joint Temperature | °C |
| Vibration | mm/s |

The system processes these readings through **five independent modules** and produces one final decision per arm:

| Decision | Meaning |
|---|---|
| `SAFE` | All signals clear — continue normal operation |
| `MONITOR` | One signal raised concern — increase monitoring frequency |
| `SCHEDULE_MAINTENANCE` | Multiple signals confirm wear — plan maintenance soon |
| `EMERGENCY_STOP` | Danger threshold crossed or spoof detected — halt immediately |

---

## Project Structure

```
robotic-arm-health-monitor/
│
├── main.py                  ← Full pipeline entry point
├── safety_check.py          ← Requirement 1: Immediate safety stop (rule-based)
├── wear_detector.py         ← Requirement 2: Predictive wear detection (ML)
├── maintenance_analyzer.py  ← Requirement 3: Maintenance log NLP analysis
├── spoof_detector.py        ← Requirement 4: Spoofed-sensor defense
├── decision_engine.py       ← Requirement 5: Final decision logic
│
├── historical_data.csv      ← Synthetic dataset (72 labeled readings)
├── maintenance_log.txt      ← Sample maintenance log (12 entries)
│
├── requirements.txt
├── explanation.md           ← Written explanation (half page)
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all four demo scenarios (shows every possible decision)
python main.py

# 3. Run a specific scenario
python main.py --demo safe
python main.py --demo monitor
python main.py --demo schedule
python main.py --demo emergency

# 4. Analyze a specific arm from the maintenance log
python main.py --arm ARM_07
```

---

## The Five Requirements

### 1. Immediate Safety Stop (`safety_check.py`)
A **pure rule-based check** that runs before anything else. No ML involved — not even called if this triggers.

| Metric | Threshold | Rationale |
|---|---|---|
| `torque_nm` | > 85 Nm | Above this, the actuator risks structural damage |
| `temperature_c` | > 80°C | Motor windings and lubricants degrade rapidly |
| `vibration_mm_s` | > 9.5 mm/s | Fastener fatigue and bearing failure become imminent |

If **any single joint** on **any metric** exceeds its threshold → `EMERGENCY_STOP` immediately.

### 2. Predictive Wear Detection (`wear_detector.py`)
A `RandomForestClassifier` trained on `historical_data.csv` (72 labeled readings across 24 arms).

- **Features**: `torque_nm`, `temperature_c`, `vibration_mm_s`, `operating_hours`
- **Labels**: `normal` / `wear`
- Readings per arm are averaged into a single feature vector before prediction.

### 3. Maintenance Notes as a Second Signal (`maintenance_analyzer.py`)
Parses `maintenance_log.txt` and filters entries by arm ID. Each note is scored using a **keyword-weight dictionary** (e.g., `"grinding"` = 3 pts, `"warm"` = 1 pt). A total score ≥ 3 flags a maintenance concern.

This signal is completely independent — it uses **free text**, not sensor numbers.

### 4. Spoofed-Sensor Defense (`spoof_detector.py`)
Two independent detection methods:

1. **Cross-sensor correlation**: If temperature AND vibration are both high (≥65°C, ≥6.0 mm/s) but torque is suspiciously low (≤40 Nm), the physics don't add up → spoof flagged.
2. **Statistical outlier**: If one joint's torque has a z-score < −2.5 relative to the arm's other joints, while overall stress is elevated → spoof flagged.

**What it catches**: An attacker feeding a fake low torque to hide a failing joint.
**What it misses**: A perfectly calibrated spoof that also fakes matching temperature/vibration values, or a slow drift attack below the z-score threshold.

### 5. Final Decision Engine (`decision_engine.py`)
Priority waterfall (highest wins):

```
1. Safety threshold crossed?   → EMERGENCY_STOP
2. Spoof detected?             → EMERGENCY_STOP
3. Wear AND maintenance flag?  → SCHEDULE_MAINTENANCE
4. Wear OR maintenance flag?   → MONITOR
5. All clear?                  → SAFE
```

---

## Demo Scenarios

| Scenario | Arm | Expected Decision | Key Signal |
|---|---|---|---|
| `--demo safe` | ARM_04 | `SAFE` | Low hours, all within limits |
| `--demo monitor` | ARM_19 | `MONITOR` | ML detects wear, no log concerns |
| `--demo schedule` | ARM_07 | `SCHEDULE_MAINTENANCE` | ML + maintenance log both flag it |
| `--demo emergency` | ARM_99 | `EMERGENCY_STOP` | Shoulder torque = 91.3 Nm (> 85) |

---

## Dataset Description

`historical_data.csv` contains **72 labeled readings** across 24 arms × 3 joints:

| Column | Description |
|---|---|
| `reading_id` | Unique reading identifier |
| `arm_id` | Arm identifier (ARM_01 to ARM_24) |
| `joint` | shoulder / elbow / wrist |
| `torque_nm` | Joint torque in Nm |
| `temperature_c` | Joint temperature in °C |
| `vibration_mm_s` | Vibration in mm/s |
| `operating_hours` | Cumulative hours of arm operation |
| `label` | `normal` or `wear` |

Labels follow a realistic pattern: arms with >450 operating hours and elevated sensor readings are labeled `wear`.

---

## Constraints Addressed

| Constraint | How it's met |
|---|---|
| Safety check never depends on ML | `safety_check.py` has zero imports from `wear_detector.py` |
| All four decisions reachable | Each demo scenario produces a different decision |
| You decide everything | Thresholds, architecture, keywords, dataset — all self-designed |
