
import random
import time

# Create a separate log file
LOG_FILE = "five_system_log.txt"


# --------------------------------------------------
# STEP 1: SENSE - Generate synthetic sensor data
# --------------------------------------------------
def generate_sensor_data():
    systems = {
        "Motor System": {
            "temperature": round(random.uniform(55, 95), 2),
            "vibration": round(random.uniform(1, 10), 2)
        },
        "Cooling System": {
            "temperature": round(random.uniform(30, 90), 2),
            "flow_rate": round(random.uniform(10, 50), 2)
        },
        "Hydraulic System": {
            "pressure": round(random.uniform(20, 60), 2),
            "oil_temperature": round(random.uniform(40, 100), 2)
        },
        "Electrical System": {
            "current": round(random.uniform(8, 30), 2),
            "voltage": round(random.uniform(190, 250), 2)
        },
        "Conveyor System": {
            "vibration": round(random.uniform(1, 10), 2),
            "belt_speed": round(random.uniform(20, 100), 2)
        }
    }

    return systems


# --------------------------------------------------
# STEP 2: THINK - Detect faults and calculate scores
# --------------------------------------------------
def evaluate_systems(systems):
    results = {}

    for name, data in systems.items():
        score = 0
        faults = []

        if name == "Motor System":
            if data["temperature"] > 80:
                score += (data["temperature"] - 80) / 5
                faults.append("High motor temperature")

            if data["vibration"] > 7:
                score += (data["vibration"] - 7) * 2
                faults.append("Excessive vibration")

        elif name == "Cooling System":
            if data["temperature"] > 75:
                score += (data["temperature"] - 75) / 5
                faults.append("Cooling temperature too high")

            if data["flow_rate"] < 20:
                score += (20 - data["flow_rate"]) / 3
                faults.append("Low coolant flow")

        elif name == "Hydraulic System":
            if data["pressure"] > 45:
                score += (data["pressure"] - 45) / 3
                faults.append("Excessive hydraulic pressure")

            if data["oil_temperature"] > 85:
                score += (data["oil_temperature"] - 85) / 5
                faults.append("Hydraulic oil overheating")

        elif name == "Electrical System":
            if data["current"] > 22:
                score += (data["current"] - 22) / 2
                faults.append("Excessive electrical current")

            if abs(data["voltage"] - 230) > 15:
                score += (abs(data["voltage"] - 230) - 15) / 3
                faults.append("Abnormal voltage")

        elif name == "Conveyor System":
            if data["vibration"] > 7:
                score += (data["vibration"] - 7) * 2
                faults.append("Excessive conveyor vibration")

            if data["belt_speed"] < 35:
                score += (35 - data["belt_speed"]) / 5
                faults.append("Conveyor belt running slowly")

        results[name] = {
            "score": round(score, 2),
            "faults": faults
        }

    return results


# --------------------------------------------------
# STEP 3: ACT - Select corrective action
# --------------------------------------------------
def take_action(system_name, faults):
    solutions = {
        "Motor System":
            "Inspect motor, improve cooling, and check bearings.",

        "Cooling System":
            "Inspect coolant pump, clear blockages, and restore flow.",

        "Hydraulic System":
            "Inspect pressure valves and check hydraulic oil.",

        "Electrical System":
            "Inspect wiring, check electrical load, and verify voltage.",

        "Conveyor System":
            "Inspect belt alignment, rollers, and conveyor drive."
    }

    print(f"\nHIGHEST FAULT SYSTEM: {system_name}")
    print(f"Detected faults: {', '.join(faults)}")
    print(f"CORRECTIVE ACTION: {solutions[system_name]}")
    print("STATUS: Action selected (simulation only).")


# --------------------------------------------------
# STEP 4: LOG - Write decisions to a separate file
# --------------------------------------------------
def log_decision(systems, results, highest_system):
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        file.write(f"\nTimestamp: {timestamp}\n")
        file.write("-" * 60 + "\n")

        for name, data in systems.items():
            result = results[name]

            status = "FAULT" if result["score"] > 0 else "NORMAL"

            file.write(
                f"System: {name} | "
                f"Readings: {data} | "
                f"Fault Score: {result['score']} | "
                f"Status: {status} | "
                f"Faults: {result['faults']}\n"
            )

        file.write(f"Highest Priority System: {highest_system}\n")
        file.write("=" * 60 + "\n")


# --------------------------------------------------
# STEP 5: MAIN - Run every second
# --------------------------------------------------
def main():
    # Create the new file if it does not already exist
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write("Five-System Fault Detection Log\n")

    print("=" * 65)
    print("       FIVE-SYSTEM FAULT DETECTION ENGINE")
    print("                SENSE - THINK - ACT")
    print("=" * 65)
    print(f"Log file: {LOG_FILE}")
    print("Press Ctrl+C to stop.\n")

    tick = 1

    try:
        while True:
            # SENSE
            systems = generate_sensor_data()

            # THINK
            results = evaluate_systems(systems)

            print(f"\nTICK {tick:03d} - SENSOR READINGS")
            print("-" * 65)

            for name, data in systems.items():
                score = results[name]["score"]
                status = "FAULT" if score > 0 else "NORMAL"

                print(
                    f"{name:20} | "
                    f"Readings: {data} | "
                    f"Score: {score:5.2f} | {status}"
                )

            # Find the highest fault score
            highest_system = max(
                results,
                key=lambda name: results[name]["score"]
            )

            highest_score = results[highest_system]["score"]

            print("\nTHINK - PRIORITY DECISION")

            if highest_score > 0:
                print(
                    f"Highest fault: {highest_system} "
                    f"(Score: {highest_score:.2f})"
                )

                # ACT
                take_action(
                    highest_system,
                    results[highest_system]["faults"]
                )
            else:
                highest_system = "NONE"
                print("All systems are operating normally.")
                print("ACT: Continue normal operation.")

            # LOG
            log_decision(systems, results, highest_system)

            print(f"\nDecision saved to {LOG_FILE}")
            print("=" * 65)

            tick += 1
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nRule engine stopped by user.")
        print(f"All decisions are saved in {LOG_FILE}")


if __name__ == "__main__":
    main()
