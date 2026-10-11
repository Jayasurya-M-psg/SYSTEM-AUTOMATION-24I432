
import random
import csv
from collections import Counter

LOG_FILE = "day2_log.csv"

# State weights / sample counts
STATES = {
    "NORMAL": 60,
    "WARNING": 10,
    "OVERHEATING": 10,
    "OVERLOAD": 10,
    "CRITICAL": 10
}


# STEP 1: Generate synthetic sensor data for each state
def generate_sensor_data(state):
    if state == "NORMAL":
        temperature = random.uniform(40, 65)
        pressure = random.uniform(25, 40)
        vibration = random.uniform(1, 4)
        current = random.uniform(5, 15)
        rpm = random.randint(1000, 2000)

    elif state == "WARNING":
        temperature = random.uniform(65, 80)
        pressure = random.uniform(40, 48)
        vibration = random.uniform(4, 7)
        current = random.uniform(15, 21)
        rpm = random.randint(2000, 2450)

    elif state == "OVERHEATING":
        temperature = random.uniform(86, 100)
        pressure = random.uniform(25, 40)
        vibration = random.uniform(2, 5)
        current = random.uniform(10, 18)
        rpm = random.randint(1200, 2200)

    elif state == "OVERLOAD":
        temperature = random.uniform(65, 82)
        pressure = random.uniform(40, 50)
        vibration = random.uniform(5, 8)
        current = random.uniform(23, 30)
        rpm = random.randint(2501, 3000)

    else:  # CRITICAL
        temperature = random.uniform(90, 100)
        pressure = random.uniform(52, 60)
        vibration = random.uniform(9, 12)
        current = random.uniform(25, 30)
        rpm = random.randint(2600, 3000)

    return {
        "temperature": round(temperature, 2),
        "pressure": round(pressure, 2),
        "vibration": round(vibration, 2),
        "current": round(current, 2),
        "rpm": rpm,
        "state": state
    }


# STEP 2: Generate 100 samples with the required distribution
def main():
    dataset = []

    for state, count in STATES.items():
        for _ in range(count):
            dataset.append(generate_sensor_data(state))

    # Shuffle rows
    random.shuffle(dataset)

    # STEP 3: Save to CSV without timestamps
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "temperature",
                "pressure",
                "vibration",
                "current",
                "rpm",
                "state"
            ]
        )

        writer.writeheader()
        writer.writerows(dataset)

    # STEP 4: Verify state distribution
    counts = Counter(row["state"] for row in dataset)

    print("=" * 45)
    print("SYNTHETIC SENSOR DATA GENERATED")
    print("=" * 45)
    print(f"Total samples: {len(dataset)}")
    print(f"CSV file: {LOG_FILE}")
    print("\nState distribution:")

    for state in STATES:
        print(f"{state:15} : {counts[state]} samples")

    print("\nNo timestamp column included.")
    print("All samples saved successfully.")


if __name__ == "__main__":
    main()
