from pathlib import Path
from datetime import datetime
import os
import re
import hmac

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ============================================================
# PATHS AND SETTINGS
# ============================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["temperature", "pressure", "vibration", "current", "rpm"]
TARGET = "state"
REQUIRED = FEATURES + [TARGET]

DAY1_OPTIONS = [
    PROJECT_DIR / "Day-1" / "day1_log.csv",
    PROJECT_DIR / "Day-1" / "day1_data.csv",
]
DAY2_OPTIONS = [
    PROJECT_DIR / "Day-2" / "Data" / "day2_log.csv",
    PROJECT_DIR / "Day-2" / "Data" / "day2_data.csv",
]

COMBINED_CSV = OUTPUT_DIR / "combined_dataset.csv"
MODEL_PATH = OUTPUT_DIR / "best_machine_state_model.pkl"
EVENT_LOG = OUTPUT_DIR / "event_log.csv"

APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "ChangeMe123!")

st.set_page_config(
    page_title="Combined Execution | Live Machine Monitor",
    page_icon="⚙️",
    layout="wide",
)

# ============================================================
# LOGIN
# ============================================================
def show_login():
    st.title("🔐 Combined Execution Login")
    st.caption("Sign in to open the machine monitoring dashboard.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        valid_user = hmac.compare_digest(username.strip(), APP_USERNAME)
        valid_password = hmac.compare_digest(password, APP_PASSWORD)
        if valid_user and valid_password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username.strip()
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.info("Demo login: username `admin`, password `ChangeMe123!`.")
    st.caption("For a real deployment, use proper authentication and securely managed secrets.")


if not st.session_state.get("authenticated", False):
    show_login()
    st.stop()

# ============================================================
# CSV HELPERS
# Important: source rows are rebuilt in memory on each rerun.
# Prediction history is stored separately and is never overwritten
# when loading the Day-1 and Day-2 source files.
# ============================================================
def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        "temp": "temperature",
        "temperature_c": "temperature",
        "press": "pressure",
        "vibration_level": "vibration",
        "current_a": "current",
        "motor_rpm": "rpm",
        "machine_state": "state",
        "label": "state",
        "status": "state",
    }
    return df.rename(columns={c: aliases.get(c, c) for c in df.columns})


def load_source_file(path, source):
    if not path.is_file():
        return None, f"{source}: file not found at {path}"
    try:
        df = normalize_columns(pd.read_csv(path))
    except Exception as exc:
        return None, f"{source}: CSV read error: {exc}"

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, f"{source}: not used for ML; missing columns {missing}"

    df = df[REQUIRED].copy()
    df["source"] = source
    return df, f"{source}: loaded {len(df)} rows"


def load_training_data():
    chunks, messages = [], []
    for options, source in (
        (DAY1_OPTIONS, "Day-1"),
        (DAY2_OPTIONS, "Day-2"),
    ):
        path = next((p for p in options if p.is_file()), None)
        if path is None:
            messages.append(f"{source}: no expected CSV found.")
            continue
        frame, message = load_source_file(path, source)
        messages.append(message)
        if frame is not None:
            chunks.append(frame)

    if not chunks:
        return pd.DataFrame(columns=REQUIRED + ["source"]), messages

    data = pd.concat(chunks, ignore_index=True)
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[TARGET] = data[TARGET].astype("string").str.strip()
    data = data.dropna(subset=REQUIRED)
    data = data[data[TARGET].notna() & (data[TARGET] != "")]
    # Do not write training data over the history CSV here.
    return data, messages


def save_combined_training_csv(data):
    """Save the current combined Day-1 + Day-2 base dataset."""
    data.to_csv(COMBINED_CSV, index=False)


def append_prediction_history(result):
    """Append prediction outcomes to combined_dataset.csv without losing old rows."""
    output_columns = [
        "temperature", "pressure", "vibration", "current", "rpm",
        "state", "source", "rule_prediction", "ml_prediction",
        "nlp_prediction", "final_decision", "operator_comment",
        "prediction_time", "record_type",
    ]
    row = {
        "temperature": result["sensor"]["temperature"],
        "pressure": result["sensor"]["pressure"],
        "vibration": result["sensor"]["vibration"],
        "current": result["sensor"]["current"],
        "rpm": result["sensor"]["rpm"],
        # Preserve the known source label separately; do not treat prediction as truth.
        "state": result["actual_label"],
        "source": result["source"],
        "rule_prediction": result["rule"],
        "ml_prediction": result["ml"],
        "nlp_prediction": result["nlp"],
        "final_decision": result["decision"],
        "operator_comment": result["comment"],
        "prediction_time": result["timestamp"],
        "record_type": "PREDICTION_HISTORY",
    }

    if COMBINED_CSV.exists():
        try:
            existing = pd.read_csv(COMBINED_CSV)
        except Exception:
            existing = pd.DataFrame(columns=output_columns)
    else:
        existing = pd.DataFrame(columns=output_columns)

    # Add any new columns to old rows while preserving existing values.
    for col in output_columns:
        if col not in existing.columns:
            existing[col] = ""

    existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    existing.to_csv(COMBINED_CSV, index=False)


def append_event_log(result, action):
    event = {
        "timestamp": result["timestamp"],
        **result["sensor"],
        "source": result["source"],
        "actual_label": result["actual_label"],
        "rule_prediction": result["rule"],
        "ml_prediction": result["ml"],
        "nlp_prediction": result["nlp"],
        "final_decision": result["decision"],
        "operator_comment": result["comment"],
        "action": action,
        "authenticated_user": st.session_state.get("username", "user"),
    }
    pd.DataFrame([event]).to_csv(
        EVENT_LOG, mode="a", header=not EVENT_LOG.exists(), index=False
    )


# ============================================================
# MODEL TRAINING
# ============================================================
@st.cache_resource
def train_best_model(data_signature, csv_data_json):
    # Signature invalidates Streamlit cache when training rows change.
    from io import StringIO
    data = pd.read_json(StringIO(csv_data_json), orient="records")
    X, y = data[FEATURES], data[TARGET].astype(str)

    counts = y.value_counts()
    if y.nunique() < 2 or int(counts.min()) < 2:
        raise ValueError("At least two labelled examples per state are required.")

    folds = min(5, int(counts.min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        ),
    }

    scores = {}
    for name, model in candidates.items():
        scores[name] = float(cross_val_score(
            model, X, y, cv=cv, scoring="f1_macro", n_jobs=1
        ).mean())

    best_name = max(scores, key=scores.get)
    best_model = candidates[best_name]
    best_model.fit(X, y)
    package = {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURES,
        "classes": sorted(y.unique()),
    }
    joblib.dump(package, MODEL_PATH)
    return package, scores


# ============================================================
# RULE ENGINE + NLP COMMENT
# ============================================================
def rules_state(sensor):
    flags = []
    if sensor["temperature"] >= 90:
        flags.append("Critical temperature >= 90")
    elif sensor["temperature"] >= 80:
        flags.append("High temperature >= 80")
    if sensor["vibration"] >= 9:
        flags.append("Critical vibration >= 9")
    elif sensor["vibration"] >= 7:
        flags.append("High vibration >= 7")
    if sensor["current"] >= 25:
        flags.append("Current overload >= 25")
    if sensor["pressure"] >= 52 or sensor["pressure"] < 20:
        flags.append("Pressure outside demo range")
    if sensor["rpm"] > 2500:
        flags.append("High RPM > 2500")

    if any("critical" in f.lower() or "overload" in f.lower() for f in flags):
        return "CRITICAL", flags
    if flags:
        return "WARNING", flags
    return "NORMAL", flags


def nlp_status(comment):
    text = re.sub(r"\s+", " ", str(comment).lower()).strip()
    if not text:
        return "UNKNOWN", "No operator comment entered."

    # Ambiguous negative phrases are sent for review rather than declared safe.
    if any(p in text for p in [
        "no smoke", "no fire", "no sparking", "not overheating",
        "smoke not detected", "fire not detected"
    ]):
        return "REVIEW REQUIRED", "Negation detected; verify the comment manually."

    critical = [
        "fire detected", "smoke detected", "emergency", "sparking",
        "burning smell", "critical failure", "machine is on fire",
        "dangerous", "motor is burning",
    ]
    warning = [
        "overheating", "temperature is high", "high temperature",
        "high vibration", "unusual vibration", "strange noise",
        "pressure is abnormal", "leak detected", "overload", "warning",
    ]
    normal = [
        "operating normally", "working fine", "machine is stable",
        "no issue detected", "no problem", "all normal",
    ]

    if any(p in text for p in critical):
        return "CRITICAL", "Critical safety phrase detected."
    if any(p in text for p in warning):
        return "WARNING", "Warning phrase detected."
    if any(p in text for p in normal):
        return "NORMAL", "Normal-status phrase detected."
    return "REVIEW REQUIRED", "Comment not recognized; do not assume the machine is safe."


def final_decision(rule, ml, nlp):
    states = [rule, ml, nlp]
    if "CRITICAL" in states:
        return "CRITICAL", "At least one channel reports CRITICAL."
    if "WARNING" in states:
        return "WARNING", "At least one channel reports WARNING."
    if any(s in states for s in ["REVIEW REQUIRED", "UNKNOWN", "UNAVAILABLE"]):
        return "REVIEW REQUIRED", "A channel is uncertain or unavailable."
    if all(s == "NORMAL" for s in states):
        return "NORMAL", "Rules, ML and NLP all indicate NORMAL."
    return "REVIEW REQUIRED", "Signals disagree."


def print_raw_terminal(result):
    print("\n" + "=" * 72, flush=True)
    print("DASHBOARD PREDICTION | " + result["timestamp"], flush=True)
    print("RAW SENSOR DATA:", result["sensor"], flush=True)
    print("OPERATOR COMMENT:", result["comment"], flush=True)
    print("RULE PREDICTION:", result["rule"], flush=True)
    print("ML PREDICTION:", result["ml"], flush=True)
    print("NLP PREDICTION:", result["nlp"], flush=True)
    print("FINAL DECISION:", result["decision"], flush=True)
    print("=" * 72 + "\n", flush=True)


# ============================================================
# DASHBOARD
# ============================================================
left, right = st.columns([4, 1])
with left:
    st.title("⚙️ Combined Execution — Live Machine Monitor")
    st.caption("Day-1 + Day-2 CSV → Rules + ML + NLP → Decision → Dashboard")
with right:
    st.write(f"Signed in: **{st.session_state.get('username', 'user')}**")
    if st.button("Log out"):
        st.session_state["authenticated"] = False
        st.session_state.pop("username", None)
        st.rerun()

data, load_messages = load_training_data()
with st.expander("Source CSV status", expanded=False):
    for message in load_messages:
        st.write("•", message)

if data.empty:
    st.error(
        "No compatible labelled sensor rows found. Both training CSVs need "
        "temperature, pressure, vibration, current, rpm, and state columns."
    )
    st.stop()

# Initialize combined CSV only if missing. Never reset it on every rerun.
if not COMBINED_CSV.exists():
    save_combined_training_csv(data)

m1, m2, m3 = st.columns(3)
m1.metric("Training sensor rows", len(data))
m2.metric("State labels", data[TARGET].nunique())
m3.metric("Prediction history file", "Ready" if COMBINED_CSV.exists() else "Not created")

with st.expander("Preview training data"):
    st.dataframe(data, use_container_width=True)

# Model training only uses clean original Day-1/Day-2 rows, not appended prediction rows.
package, model_scores, model_error = None, {}, None
try:
    signature = str(hash(pd.util.hash_pandas_object(data, index=True).sum()))
    training_json = data[REQUIRED].to_json(orient="records")
    package, model_scores = train_best_model(signature, training_json)
except Exception as exc:
    model_error = str(exc)
    st.warning(f"ML model unavailable: {model_error}. Rules and NLP remain available.")

st.divider()
st.subheader("1. Sensor reading")
st.caption("This demo uses the next row from the CSV, not a physical live sensor feed.")

if "row_index" not in st.session_state:
    st.session_state["row_index"] = 0

row_col, info_col = st.columns([1, 3])
with row_col:
    if st.button("▶ Predict next CSV row", type="primary", use_container_width=True):
        st.session_state["row_index"] = (st.session_state["row_index"] + 1) % len(data)
        st.session_state["run_prediction"] = True
with info_col:
    st.write(f"CSV row index: {st.session_state['row_index']} of {len(data) - 1}")

st.subheader("2. NLP operator comment")
comment = st.text_input(
    "Enter operator comment",
    value="Motor is overheating and vibration is high",
    placeholder="Examples: smoke detected near motor; machine is operating normally",
)
if st.button("Predict current row + analyze comment", use_container_width=True):
    st.session_state["run_prediction"] = True

if st.session_state.get("run_prediction", False):
    row = data.iloc[st.session_state["row_index"]]
    sensor = {feature: float(row[feature]) for feature in FEATURES}
    actual_label = str(row[TARGET])
    source = str(row.get("source", "CSV"))

    rule, flags = rules_state(sensor)
    nlp, nlp_explanation = nlp_status(comment)
    ml = "UNAVAILABLE"
    probabilities = {}
    selected_model = "Not available"

    if package is not None:
        model = package["model"]
        selected_model = package["model_name"]
        sample = pd.DataFrame([sensor], columns=FEATURES)
        ml = str(model.predict(sample)[0])
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(sample)[0]
            probabilities = dict(zip(model.classes_, p))

    decision, reason = final_decision(rule, ml, nlp)
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "actual_label": actual_label,
        "sensor": sensor,
        "rule": rule,
        "flags": flags,
        "ml": ml,
        "model": selected_model,
        "probabilities": probabilities,
        "nlp": nlp,
        "nlp_explanation": nlp_explanation,
        "decision": decision,
        "reason": reason,
        "comment": comment,
    }

    # Store result in session first, then append it exactly once per click.
    st.session_state["last_result"] = result
    print_raw_terminal(result)
    append_prediction_history(result)
    st.session_state["run_prediction"] = False
    st.success(f"Prediction history appended to: {COMBINED_CSV}")

result = st.session_state.get("last_result")
if result:
    st.divider()
    st.subheader("3. Outcome dashboard")
    a, b, c, d = st.columns(4)
    a.metric("Rule engine", result["rule"])
    b.metric("ML prediction", result["ml"])
    c.metric("NLP status", result["nlp"])
    d.metric("Final decision", result["decision"])

    st.write("**Prediction time:**", result["timestamp"])
    st.write("**CSV source:**", result["source"])
    st.write("**Original dataset label:**", result["actual_label"])
    st.write("**Decision explanation:**", result["reason"])
    st.write("**Rule flags:**", result["flags"] or ["No configured rule threshold exceeded"])
    st.write("**NLP explanation:**", result["nlp_explanation"])
    st.write("**Operator comment:**", result["comment"])

    if result["decision"] == "CRITICAL":
        st.error("CRITICAL — follow the approved safety procedure.")
    elif result["decision"] in ("WARNING", "REVIEW REQUIRED"):
        st.warning("Review required. Do not assume the equipment is safe.")
    else:
        st.success("Rules, ML and NLP indicate NORMAL.")

    st.write("**Raw sensor values**")
    st.dataframe(pd.DataFrame([result["sensor"]]), use_container_width=True)

    if result["probabilities"]:
        st.write("**ML probability estimates**")
        p_df = pd.DataFrame([
            {"State": state, "Probability (%)": round(float(prob) * 100, 2)}
            for state, prob in result["probabilities"].items()
        ]).sort_values("Probability (%)", ascending=False)
        st.dataframe(p_df, use_container_width=True)

    with st.expander("Model comparison (mean cross-validation macro F1)"):
        if model_scores:
            st.dataframe(pd.DataFrame([
                {"Model": name, "Mean Macro F1 (%)": round(score * 100, 2)}
                for name, score in model_scores.items()
            ]).sort_values("Mean Macro F1 (%)", ascending=False), use_container_width=True)
        else:
            st.caption(model_error or "Model comparison not available.")

    st.divider()
    st.subheader("4. Login-protected simulated action")
    st.warning("Demo only: no physical equipment is controlled.")
    confirmed = st.checkbox(
        "I reviewed the result and authorize logging a simulated action.",
        key="action_acknowledgement",
    )
    action = st.selectbox(
        "Simulated action",
        ["Log event only", "Simulate alert", "Simulate stop request"],
    )
    if st.button("Save action to event history"):
        if not confirmed:
            st.error("Tick the acknowledgement checkbox first.")
        else:
            append_event_log(result, action)
            st.success(f"Simulated action saved to {EVENT_LOG}")

st.divider()
st.subheader("5. Prediction history from combined_dataset.csv")
try:
    history = pd.read_csv(COMBINED_CSV)
    st.dataframe(history.tail(30), use_container_width=True)
    st.download_button(
        "Download combined dataset and prediction history",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="combined_dataset.csv",
        mime="text/csv",
    )
except Exception as exc:
    st.warning(f"Could not read combined CSV: {exc}")

if EVENT_LOG.exists():
    with st.expander("Recent simulated action history"):
        st.dataframe(pd.read_csv(EVENT_LOG).tail(20), use_container_width=True)

st.caption(
    "CSV playback is not a physical real-time sensor feed. Demo thresholds and NLP phrases are illustrative. "
    "Prediction rows are appended to combined_dataset.csv; model training uses source CSV rows only."
)
