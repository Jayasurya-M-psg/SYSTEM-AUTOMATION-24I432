
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)


# =========================================================
# STEP 1: CONFIGURATION
# =========================================================

# Get the directory containing this Python script
BASE_DIR = Path(__file__).resolve().parent

# Your CSV files are inside Day-2/Data
DATA_DIR = BASE_DIR / "Data"

# Dataset to use for training
DATA_FILE = DATA_DIR / "day2_log.csv"

# Location to save the trained model
MODEL_FILE = BASE_DIR / "best_machine_state_model.pkl"

# Sensor features
FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "current",
    "rpm"
]

# Target column
TARGET = "state"


# =========================================================
# STEP 2: LOAD DATASET
# =========================================================

print("=" * 65)
print("MACHINE STATE PREDICTION - MODEL TRAINING")
print("=" * 65)

print("\nPython script:", Path(__file__).resolve())
print("Dataset path:", DATA_FILE)
print("Dataset exists:", DATA_FILE.is_file())

# Stop if the file does not exist
if not DATA_FILE.is_file():
    alternative = DATA_DIR / "day2_data.csv"

    if alternative.is_file():
        raise FileNotFoundError(
            f"\nThe file day2_log.csv was not found.\n"
            f"But day2_data.csv exists at:\n{alternative}\n"
            "If you want to use day2_data.csv, change DATA_FILE "
            "to DATA_DIR / 'day2_data.csv'."
        )

    raise FileNotFoundError(
        f"\nDataset not found at:\n{DATA_FILE}\n\n"
        "Check that day2_log.csv is inside the Day-2/Data folder."
    )

# Load the CSV only once
df = pd.read_csv(DATA_FILE)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")
print("Total rows before cleaning:", len(df))
print("Columns:", df.columns.tolist())


# =========================================================
# STEP 3: VALIDATE AND CLEAN DATA
# =========================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns: {missing_columns}\n"
        f"Columns found: {df.columns.tolist()}\n"
        f"Required columns: {required_columns}"
    )

# Keep only the required columns
df = df[required_columns].copy()

# Convert sensor columns to numeric values
for column in FEATURES:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Clean target labels
df[TARGET] = df[TARGET].astype("string").str.strip()

# Remove rows with missing or invalid values
df = df.dropna(subset=required_columns)
df = df[df[TARGET] != ""]

if df.empty:
    raise ValueError(
        "No valid rows remain after cleaning the dataset."
    )

X = df[FEATURES]
y = df[TARGET].astype(str)

print("\n" + "=" * 65)
print("DATASET INFORMATION")
print("=" * 65)

print("Dataset:", DATA_FILE)
print("Valid samples:", len(df))
print("Number of features:", len(FEATURES))
print("Features:", FEATURES)
print("Number of states:", y.nunique())

print("\nState distribution:")
print(y.value_counts().to_string())


# =========================================================
# STEP 4: CHECK DATA FOR CROSS-VALIDATION
# =========================================================

class_counts = y.value_counts()

if len(class_counts) < 2:
    raise ValueError(
        "The dataset must contain at least two different states."
    )

if class_counts.min() < 5:
    raise ValueError(
        "Each state must have at least 5 samples "
        "for 5-fold cross-validation."
    )

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================================================
# STEP 5: DEFINE MACHINE LEARNING MODELS
# =========================================================

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}


# =========================================================
# STEP 6: COMPARE MODELS USING CROSS-VALIDATION
# =========================================================

print("\n" + "=" * 65)
print("5-FOLD CROSS-VALIDATION")
print("=" * 65)

results = {}

for name, model in models.items():

    # Macro F1 gives equal importance to each class
    f1_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1
    )

    # Calculate accuracy for comparison
    accuracy_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    mean_f1 = f1_scores.mean()
    mean_accuracy = accuracy_scores.mean()

    results[name] = {
        "model": model,
        "macro_f1": mean_f1,
        "accuracy": mean_accuracy
    }

    print(f"\nModel: {name}")
    print(f"Mean Macro F1: {mean_f1 * 100:.2f}%")
    print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%")


# =========================================================
# STEP 7: SELECT THE BEST MODEL
# =========================================================

# Highest mean Macro F1 wins.
# Accuracy breaks ties.
best_name = max(
    results,
    key=lambda name: (
        results[name]["macro_f1"],
        results[name]["accuracy"]
    )
)

best_model = results[best_name]["model"]

print("\n" + "-" * 65)
print("SELECTED MODEL:", best_name)

print(
    f"Mean CV Macro F1: "
    f"{results[best_name]['macro_f1'] * 100:.2f}%"
)

print(
    f"Mean CV Accuracy: "
    f"{results[best_name]['accuracy'] * 100:.2f}%"
)


# =========================================================
# STEP 8: EVALUATE THE SELECTED MODEL
# =========================================================

# Each sample is predicted by a model that did not train on it
y_pred_cv = cross_val_predict(
    best_model,
    X,
    y,
    cv=cv,
    n_jobs=1
)

class_order = sorted(y.unique())

print("\n" + "=" * 65)
print("CROSS-VALIDATED CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        y,
        y_pred_cv,
        labels=class_order,
        zero_division=0
    )
)

print("Confusion matrix (actual rows, predicted columns):")

print(
    confusion_matrix(
        y,
        y_pred_cv,
        labels=class_order
    )
)

print("\nClass order:", class_order)

print(
    f"\nOut-of-fold accuracy: "
    f"{accuracy_score(y, y_pred_cv) * 100:.2f}%"
)

print(
    f"Out-of-fold Macro F1: "
    f"{f1_score(y, y_pred_cv, average='macro') * 100:.2f}%"
)


# =========================================================
# STEP 9: TRAIN AND SAVE THE FINAL MODEL
# =========================================================

print("\nTraining the selected model on all valid samples...")

best_model.fit(X, y)

# Save the model together with its feature information
model_package = {
    "model": best_model,
    "model_name": best_name,
    "features": FEATURES,
    "classes": class_order
}

joblib.dump(model_package, MODEL_FILE)

print("\n" + "=" * 65)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 65)

print("Selected model:", best_name)
print("Saved model:", MODEL_FILE)


# =========================================================
# STEP 10: PREDICT A NEW SENSOR READING
# =========================================================

new_reading = pd.DataFrame(
    [
        {
            "temperature": 95.0,
            "pressure": 55.0,
            "vibration": 10.0,
            "current": 28.0,
            "rpm": 2800.0
        }
    ],
    columns=FEATURES
)

prediction = best_model.predict(new_reading)[0]

print("\n" + "=" * 65)
print("NEW SENSOR READING")
print("=" * 65)

print(new_reading.to_string(index=False))

print("\nPredicted state:", prediction)

# Display probability estimates if supported
if hasattr(best_model, "predict_proba"):

    probabilities = best_model.predict_proba(new_reading)[0]

    print("\nModel probability estimates:")

    sorted_probabilities = sorted(
        zip(best_model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True
    )

    for state, probability in sorted_probabilities:
        print(f"{state:<15}: {probability * 100:.2f}%")

print("\nTraining completed successfully!")
