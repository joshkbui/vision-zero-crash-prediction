"""
Vision Zero Injury Risk Simulator
DATA 4382-002 Capstone 2 | Josh Bui & Danson Vo

Streamlit app for real-time crash injury prediction using XGBoost
with SHAP waterfall explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Zero Injury Risk Simulator",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Model + Encoders ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("models/xgb_model.pkl")
        enc = joblib.load("models/encoders.pkl")
        return model, enc
    except FileNotFoundError:
        return None, None

model, enc = load_artifacts()

if model is None:
    st.title("🚦 Vision Zero Injury Risk Simulator")
    st.error(
        "Model files not found. Run `python scripts/train_model.py` "
        "(or the 'Train model' GitHub Action) to create models/xgb_model.pkl and models/encoders.pkl."
    )
    st.stop()

FEATURE_NAMES = enc["feature_names"]
LE_DICT = enc["le_dict"]
OPTIONS = enc["options"]
THRESHOLD = enc["threshold"]
REFERENCE_YEAR = enc.get("reference_year", 2024)
METRICS = enc.get("metrics", {})


def choose(label, column, key=None):
    """Selectbox whose choices are the most common values seen in training."""
    return st.selectbox(label, OPTIONS[column], key=key or column)


def encode(column, value):
    """Label-encode a value exactly as in training (unseen -> UNKNOWN, else 0)."""
    mapping = LE_DICT[column]
    if value in mapping:
        return mapping[value]
    return mapping.get("UNKNOWN", 0)


# ─── UI ────────────────────────────────────────────────────────────────────────
recall_txt = f"{METRICS['recall']:.0%} injury recall" if "recall" in METRICS else "high injury recall"
st.title("🚦 Vision Zero Injury Risk Simulator")
st.markdown(
    "**Predict crash injury probability** using Montgomery County, MD crash data. "
    f"Built with XGBoost ({recall_txt}) + SHAP explanations."
)
st.markdown("---")

# ─── Sidebar Inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Crash Scenario Inputs")

with st.sidebar:
    st.subheader("Environment")
    weather = choose("Weather", "Weather")
    light = choose("Light Condition", "Light")
    surface = choose("Surface Condition", "Surface Condition")
    traffic_control = choose("Traffic Control", "Traffic Control")
    route_type = choose("Route Type", "Route Type")

    st.subheader("Vehicle")
    vehicle_body = choose("Vehicle Body Type", "Vehicle Body Type")
    vehicle_damage = choose("Vehicle Damage Extent", "Vehicle Damage Extent")
    vehicle_movement = choose("Vehicle Movement", "Vehicle Movement")
    vehicle_dir = choose("Vehicle Going Direction", "Vehicle Going Dir")
    vehicle_impact = choose("Vehicle First Impact Location", "Vehicle First Impact Location")
    driverless = choose("Driverless Vehicle", "Driverless Vehicle")
    parked = choose("Parked Vehicle", "Parked Vehicle")
    vehicle_age = st.slider("Vehicle Age (years)", 0, 30, 5)

    st.subheader("Driver")
    substance = choose("Driver Substance Abuse", "Driver Substance Abuse")
    at_fault = choose("Driver At Fault", "Driver At Fault")
    distracted = choose("Driver Distracted By", "Driver Distracted By")
    non_motorist_substance = choose("Non-Motorist Substance Abuse", "Non-Motorist Substance Abuse")

    st.subheader("Incident")
    collision_type = choose("Collision Type", "Collision Type")
    circumstance = choose("Circumstance", "Circumstance")
    speed_limit = st.slider("Speed Limit (mph)", 5, 75, 35, step=5)

    st.subheader("Time & Location")
    hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    latitude = st.number_input("Latitude", value=39.1547, format="%.4f")
    longitude = st.number_input("Longitude", value=-77.2405, format="%.4f")

# ─── Feature Engineering (matches scripts/train_model.py) ──────────────────────
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
           "Friday": 4, "Saturday": 5, "Sunday": 6}
day_num = day_map[day_of_week]
is_night = 1 if (hour >= 22 or hour <= 5) else 0
is_weekend = 1 if day_num >= 5 else 0

raw_inputs = {
    "Weather": weather,
    "Surface Condition": surface,
    "Light": light,
    "Traffic Control": traffic_control,
    "Collision Type": collision_type,
    "Circumstance": circumstance,
    "Vehicle Movement": vehicle_movement,
    "Speed Limit": speed_limit,
    "Driver Substance Abuse": substance,
    "Driver At Fault": at_fault,
    "Vehicle Damage Extent": vehicle_damage,
    "Vehicle Body Type": vehicle_body,
    "Route Type": route_type,
    "Vehicle Going Dir": vehicle_dir,
    "Driverless Vehicle": driverless,
    "Parked Vehicle": parked,
    "Vehicle First Impact Location": vehicle_impact,
    "Driver Distracted By": distracted,
    "Non-Motorist Substance Abuse": non_motorist_substance,
    "Latitude": latitude,
    "Longitude": longitude,
    "Hour": hour,
    "DayOfWeek": day_num,
    "IsNight": is_night,
    "IsWeekend": is_weekend,
    "Vehicle Age": vehicle_age,
}

encoded = {c: (encode(c, v) if c in LE_DICT else v) for c, v in raw_inputs.items()}
encoded["Speed_x_Night"] = encoded["Speed Limit"] * encoded["IsNight"]
encoded["Substance_x_Night"] = encoded["Driver Substance Abuse"] * encoded["IsNight"]
encoded["VehicleAge_x_Collision"] = encoded["Vehicle Age"] * encoded["Collision Type"]
input_df = pd.DataFrame([encoded])[FEATURE_NAMES].astype(float)

# ─── Prediction ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Crash Scenario Summary")
    summary = pd.DataFrame({"Value": [str(v) for v in raw_inputs.values()]}, index=list(raw_inputs.keys()))
    st.dataframe(summary, height=400)

with col2:
    st.subheader("Injury Risk Prediction")

    if st.button("Predict Injury Risk", type="primary", use_container_width=True):
        try:
            prob = float(model.predict_proba(input_df)[0][1])
            prediction = 1 if prob >= THRESHOLD else 0

            # Risk gauge
            if prob >= 0.75:
                risk_level = "HIGH RISK"
                color = "red"
            elif prob >= 0.50:
                risk_level = "MODERATE RISK"
                color = "orange"
            elif prob >= 0.30:
                risk_level = "LOW-MODERATE RISK"
                color = "orange"
            else:
                risk_level = "LOW RISK"
                color = "green"

            st.markdown(f"### Injury Probability: `{prob:.1%}`")
            st.markdown(f"### Prediction: {'**INJURY LIKELY**' if prediction == 1 else '**No Injury Expected**'}")
            st.markdown(f"### Risk Level: :{color}[{risk_level}]")

            st.progress(prob)
            st.caption(f"Threshold: {THRESHOLD} | Model: XGBoost | {recall_txt.capitalize()}")

            # SHAP explanation
            st.markdown("---")
            st.subheader("SHAP Explanation (Why this prediction?)")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            shap_row = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            base = explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = np.ravel(base)[-1]

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_row,
                    base_values=base,
                    data=input_df.iloc[0].values,
                    feature_names=FEATURE_NAMES,
                ),
                show=False,
                max_display=15,
            )
            st.pyplot(plt.gcf())
            plt.close("all")
            st.caption("Values are label-encoded. Red pushes toward injury; blue pushes toward no injury.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.info("Configure the crash scenario in the sidebar, then click 'Predict Injury Risk'.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*Vision Zero Injury Risk Simulator — DATA 4382-002 Capstone 2 | "
    "Josh Bui & Danson Vo | University of Texas at Arlington*"
)
