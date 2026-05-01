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

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
      try:
                model = joblib.load("models/xgb_model.pkl")
                return model
except FileNotFoundError:
        st.error("Model file not found. Please train the model first (run notebooks/Final_Deployment_Code.ipynb).")
        return None

model = load_model()

# ─── Feature Lists ─────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
      "Weather", "Surface Condition", "Light", "Traffic Control",
      "Collision Type", "Circumstance", "Vehicle Movement",
      "Driver Substance Abuse", "Driver At Fault", "Vehicle Damage Extent",
      "Vehicle Body Type", "Route Type", "Vehicle Going Dir",
      "Driverless Vehicle", "Parked Vehicle", "Vehicle First Impact Location",
      "Driver Distracted By", "Non-Motorist Substance Abuse"
]

NUMERIC_FEATURES = [
      "Speed Limit", "Latitude", "Longitude",
      "Hour", "DayOfWeek", "Vehicle Age"
]

BINARY_FEATURES = ["IsNight", "IsWeekend"]

INTERACTION_FEATURES = ["Speed_x_Night", "Substance_x_Night", "VehicleAge_x_Collision"]

THRESHOLD = 0.40

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("🚦 Vision Zero Injury Risk Simulator")
st.markdown(
      "**Predict crash injury probability** using Montgomery County, MD crash data. "
      "Built with XGBoost (91% injury recall) + SHAP explanations."
)
st.markdown("---")

# ─── Sidebar Inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Crash Scenario Inputs")

with st.sidebar:
      st.subheader("Environment")
      weather = st.selectbox("Weather", ["CLEAR", "CLOUDY", "RAINING", "SNOWING", "FOG", "UNKNOWN"])
      light = st.selectbox("Light Condition", ["DAYLIGHT", "DARK - LIGHTED", "DARK - NOT LIGHTED", "DAWN", "DUSK"])
      surface = st.selectbox("Surface Condition", ["DRY", "WET", "ICE", "SNOW", "SAND/MUD/DIRT", "UNKNOWN"])
      traffic_control = st.selectbox("Traffic Control", ["NO CONTROLS", "TRAFFIC SIGNAL", "STOP SIGN", "YIELD SIGN", "UNKNOWN"])
      route_type = st.selectbox("Route Type", ["MARYLAND (STATE)", "COUNTY", "MUNICIPAL", "INTERSTATE", "US"])

    st.subheader("Vehicle")
    vehicle_body = st.selectbox("Vehicle Body Type", ["PASSENGER CAR", "PICKUP TRUCK", "SUV", "VAN", "MOTORCYCLE", "BUS", "TRUCK", "OTHER"])
    vehicle_damage = st.selectbox("Vehicle Damage Extent", ["NO DAMAGE", "SUPERFICIAL", "FUNCTIONAL", "DISABLING", "DESTROYED"])
    vehicle_movement = st.selectbox("Vehicle Movement", ["GOING STRAIGHT", "TURNING RIGHT", "TURNING LEFT", "SLOWING OR STOPPING", "PARKED", "BACKING", "UNKNOWN"])
    vehicle_dir = st.selectbox("Vehicle Going Direction", ["NORTH", "SOUTH", "EAST", "WEST", "UNKNOWN"])
    vehicle_impact = st.selectbox("Vehicle First Impact Location", ["FRONT", "REAR", "RIGHT SIDE", "LEFT SIDE", "UNKNOWN"])
    driverless = st.selectbox("Driverless Vehicle", ["No", "Yes"])
    parked = st.selectbox("Parked Vehicle", ["No", "Yes"])
    vehicle_age = st.slider("Vehicle Age (years)", 0, 30, 5)

    st.subheader("Driver")
    substance = st.selectbox("Driver Substance Abuse", ["NONE DETECTED", "ALCOHOL PRESENT", "ILLEGAL DRUG DETECTED", "COMBINED SUBSTANCE", "UNKNOWN"])
    at_fault = st.selectbox("Driver At Fault", ["Yes", "No", "Unknown"])
    distracted = st.selectbox("Driver Distracted By", ["NOT DISTRACTED", "BY ELECTRONIC DEVICE", "BY OUTSIDE PERSON/OBJECT", "OTHER ACTION", "UNKNOWN"])
    non_motorist_substance = st.selectbox("Non-Motorist Substance Abuse", ["NONE DETECTED", "ALCOHOL PRESENT", "UNKNOWN"])

    st.subheader("Incident")
    collision_type = st.selectbox("Collision Type", ["SAME DIR REAR END", "ANGLE MEETS LEFT HEAD ON", "SAME DIRECTION SIDESWIPE", "ANGLE MEETS LEFT TURN", "HEAD ON", "SINGLE VEHICLE", "OTHER"])
    circumstance = st.selectbox("Circumstance", ["UNKNOWN", "DRIVER INEXPERIENCE", "DISREGARDED TRAFFIC CONTROL", "FAILURE TO YIELD RIGHT OF WAY", "DRIVING TOO FAST FOR CONDITIONS"])
    speed_limit = st.slider("Speed Limit (mph)", 5, 75, 35, step=5)

    st.subheader("Time & Location")
    hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    latitude = st.number_input("Latitude", value=39.1547, format="%.4f")
    longitude = st.number_input("Longitude", value=-77.2405, format="%.4f")

# ─── Feature Engineering ───────────────────────────────────────────────────────
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                      "Friday": 4, "Saturday": 5, "Sunday": 6}

is_night = 1 if (hour < 6 or hour >= 20) else 0
is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
substance_binary = 0 if substance == "NONE DETECTED" else 1
collision_encoded = hash(collision_type) % 10  # simplified

speed_x_night = speed_limit * is_night
substance_x_night = substance_binary * is_night
vehicle_age_x_collision = vehicle_age * collision_encoded

driverless_binary = 1 if driverless == "Yes" else 0
parked_binary = 1 if parked == "Yes" else 0

# ─── Build Input DataFrame ─────────────────────────────────────────────────────
input_data = {
      "Weather": weather,
      "Surface Condition": surface,
      "Light": light,
      "Traffic Control": traffic_control,
      "Collision Type": collision_type,
      "Circumstance": circumstance,
      "Vehicle Movement": vehicle_movement,
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
      "Speed Limit": speed_limit,
      "Latitude": latitude,
      "Longitude": longitude,
      "Hour": hour,
      "DayOfWeek": day_map[day_of_week],
      "Vehicle Age": vehicle_age,
      "IsNight": is_night,
      "IsWeekend": is_weekend,
      "Speed_x_Night": speed_x_night,
      "Substance_x_Night": substance_x_night,
      "VehicleAge_x_Collision": vehicle_age_x_collision,
}

input_df = pd.DataFrame([input_data])

# ─── Prediction ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
      st.subheader("Crash Scenario Summary")
      st.dataframe(input_df.T.rename(columns={0: "Value"}), height=400)

with col2:
      st.subheader("Injury Risk Prediction")

    if model is not None and st.button("Predict Injury Risk", type="primary", use_container_width=True):
              try:
                            prob = model.predict_proba(input_df)[0][1]
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
                color = "yellow"
else:
                risk_level = "LOW RISK"
                  color = "green"

            st.markdown(f"### Injury Probability: `{prob:.1%}`")
            st.markdown(f"### Prediction: {'**INJURY LIKELY**' if prediction == 1 else '**No Injury Expected**'}")
            st.markdown(f"### Risk Level: :{color}[{risk_level}]")

            st.progress(float(prob))
            st.caption(f"Threshold: {THRESHOLD} | Model: XGBoost | Injury Recall: 91%")

            # SHAP explanation
            st.markdown("---")
            st.subheader("SHAP Explanation (Why this prediction?)")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                              shap.Explanation(
                                                    values=shap_values[0],
                                                    base_values=explainer.expected_value,
                                                    data=input_df.iloc[0],
                                                    feature_names=input_df.columns.tolist()
                              ),
                              show=False,
                              max_display=15
            )
            st.pyplot(fig)
            plt.close()

except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Note: The model requires training data encoders. Run notebooks/Final_Deployment_Code.ipynb to generate the full model artifact.")
elif model is None:
        st.info("Load the trained model to enable predictions.")
else:
        st.info("Configure the crash scenario in the sidebar, then click 'Predict Injury Risk'.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
      "*Vision Zero Injury Risk Simulator — DATA 4382-002 Capstone 2 | "
      "Josh Bui & Danson Vo | University of Texas at Arlington*"
)
