
import random
import time

LOG_FILE = "day1_log.txt"
def generate_sensor_data():
    return {
        "temperature": round(random.uniform(50, 95), 2),
        "pressure": round(random.uniform(20, 50), 2),
        "vibration": round(random.uniform(0, 10), 2),
        "current": round(random.uniform(5, 25), 2)
    }
RULES = [
    {
        "name": "HIGH_TEMPERATURE",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 85,
        "action": "STOP_MACHINE"
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 2,
        "condition": lambda r: r["vibration"] > 8,
        "action": "CHECK_MACHINE"
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 3,
        "condition": lambda r: r["pressure"] > 45,
        "action": "REDUCE_PRESSURE"
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 4,
        "condition": lambda r: r["current"] > 22,
        "action": "CHECK_ELECTRICAL_SYSTEM"
    }
]
def evaluate_rules(reading, rules):
    for rule in sorted(rules, key=lambda r: r["priority"]):
        if rule["condition"](reading):
            return rule["name"], rule["action"]

    return "NO_RULE_FIRED", "CONTINUE_NORMAL"

def take_action(action):
    actions = {
        "STOP_MACHINE": "Machine stopped due to high temperature!",
        "CHECK_MACHINE": "Warning! Check machine vibration.",
        "REDUCE_PRESSURE": "Warning! Reduce the machine pressure.",
        "CHECK_ELECTRICAL_SYSTEM": "Warning! Check electrical current.",
        "CONTINUE_NORMAL": "Machine is operating normally."
    }

    return actions.get(action, "Unknown action")


# STEP 5: Log each decision
def log_decision(reading, rule_name, action):
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Reading: {reading} | "
            f"Rule: {rule_name} | Action: {action}\n"
        )
def main():
    print("==========================================")
    print("     INDUSTRIAL RULE ENGINE")
    print("        SENSE - THINK - ACT")
    print("==========================================")
    print("Press Ctrl+C to stop.\n")

    tick = 1

    try:
        while True:
            # SENSE
            reading = generate_sensor_data()

            # THINK
            rule_name, action = evaluate_rules(reading, RULES)

            # ACT
            outcome = take_action(action)

            print(f"[Tick {tick:03d}] SENSE -> {reading}")
            print(f"           THINK -> {rule_name}")
            print(f"           ACT   -> {outcome}")
            print("-" * 65)

            log_decision(reading, rule_name, action)

            tick += 1
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nRule engine stopped by user.")


if __name__ == "__main__":
    main()
