"""
wear_detector.py
----------------
Requirement 2: Predictive Wear Detection
Trains a RandomForestClassifier on the historical dataset.
Predicts 'wear' or 'normal' for a given set of joint readings.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Features used for prediction
FEATURE_COLS = ["torque_nm", "temperature_c", "vibration_mm_s", "operating_hours"]
LABEL_COL    = "label"

_model   = None   # Cached trained model
_encoder = None   # Label encoder


def _aggregate_readings(readings: list[dict]) -> dict:
    """Aggregate a list of per-joint readings into a single feature vector."""
    torques = [r["torque_nm"]        for r in readings]
    temps   = [r["temperature_c"]    for r in readings]
    vibs    = [r["vibration_mm_s"]   for r in readings]
    hours   = [r.get("operating_hours", 0) for r in readings]
    return {
        "torque_nm":       np.mean(torques),
        "temperature_c":   np.mean(temps),
        "vibration_mm_s":  np.mean(vibs),
        "operating_hours": np.mean(hours),
    }


def train_model(data_path: str = "historical_data.csv") -> dict:
    """
    Load the historical dataset and train the RandomForest wear-detection model.

    Returns a dict with training accuracy and classification report.
    """
    global _model, _encoder

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Encode labels: 'normal' → 0, 'wear' → 1
    _encoder = LabelEncoder()
    df["label_enc"] = _encoder.fit_transform(df[LABEL_COL])

    X = df[FEATURE_COLS].values
    y = df["label_enc"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    _model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    _model.fit(X_train, y_train)

    y_pred = _model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=_encoder.classes_)
    accuracy = (_model.score(X_test, y_test) * 100)

    print(f"[WearDetector] Model trained. Test accuracy: {accuracy:.1f}%")
    print(report)

    return {"accuracy": accuracy, "report": report}


def predict_wear(readings: list[dict]) -> dict:
    """
    Predict whether the arm shows signs of wear based on current sensor readings.

    Parameters
    ----------
    readings : list of dict
        Per-joint sensor readings (same format as safety_check).

    Returns
    -------
    dict with:
        - 'wear_detected'  : bool
        - 'confidence'     : float (0–1 probability of wear class)
        - 'prediction'     : str ('normal' | 'wear')
    """
    global _model, _encoder

    if _model is None:
        train_model()

    features = _aggregate_readings(readings)
    X = np.array([[features[c] for c in FEATURE_COLS]])

    pred_idx  = _model.predict(X)[0]
    proba     = _model.predict_proba(X)[0]
    wear_idx  = list(_encoder.classes_).index("wear")
    confidence = float(proba[wear_idx])

    prediction = _encoder.inverse_transform([pred_idx])[0]

    return {
        "wear_detected": prediction == "wear",
        "confidence":    confidence,
        "prediction":    prediction,
    }
