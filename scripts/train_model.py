"""
Retrain the V1 XGBoost injury model and save everything the Streamlit app needs.

Same recipe as notebooks/Final_Deployment_Code.ipynb (V1 FINAL):
label encoding, 3 interaction features, 80/20 stratified split,
SMOTETomek on train only, XGBoost(300 trees, depth 6, lr 0.05, spw 5), threshold 0.40.

Outputs:
  models/xgb_model.pkl  - trained XGBClassifier
  models/encoders.pkl   - feature names, per-column label maps, dropdown options,
                          threshold and test-set metrics

Usage: python scripts/train_model.py path/to/Crash_Reporting_-_Drivers_Data.csv
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

THRESHOLD = 0.40
REFERENCE_YEAR = 2024  # Vehicle Age = 2024 - Vehicle Year, as in the notebook

CATEGORICAL = [
    "Weather", "Surface Condition", "Light", "Traffic Control",
    "Collision Type", "Circumstance", "Vehicle Movement",
    "Driver Substance Abuse", "Driver At Fault", "Vehicle Damage Extent",
    "Vehicle Body Type", "Route Type", "Vehicle Going Dir",
    "Driverless Vehicle", "Parked Vehicle", "Vehicle First Impact Location",
    "Driver Distracted By", "Non-Motorist Substance Abuse",
]
FEATURES = [
    "Weather", "Surface Condition", "Light", "Traffic Control",
    "Collision Type", "Circumstance", "Vehicle Movement",
    "Speed Limit", "Driver Substance Abuse", "Driver At Fault",
    "Vehicle Damage Extent", "Vehicle Body Type", "Route Type",
    "Vehicle Going Dir", "Driverless Vehicle", "Parked Vehicle",
    "Vehicle First Impact Location", "Driver Distracted By",
    "Non-Motorist Substance Abuse", "Latitude", "Longitude",
    "Hour", "DayOfWeek", "IsNight", "IsWeekend", "Vehicle Age",
]
BINARY_MAP = {
    "NO APPARENT INJURY": 0,
    "POSSIBLE INJURY": 1,
    "SUSPECTED MINOR INJURY": 1,
    "SUSPECTED SERIOUS INJURY": 1,
    "FATAL INJURY": 1,
}


def add_interactions(X):
    X = X.copy()
    X["Speed_x_Night"] = X["Speed Limit"] * X["IsNight"]
    X["Substance_x_Night"] = X["Driver Substance Abuse"] * X["IsNight"]
    X["VehicleAge_x_Collision"] = X["Vehicle Age"] * X["Collision Type"]
    return X


def main(csv_path, out_dir="models"):
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df):,} rows")

    # 1. Target
    df["Injury Severity"] = df["Injury Severity"].astype(str).str.upper().str.strip()
    df["Severity_Class"] = df["Injury Severity"].map(BINARY_MAP)
    df = df.dropna(subset=["Severity_Class"])
    df["Severity_Class"] = df["Severity_Class"].astype(int)
    print(df["Severity_Class"].value_counts(normalize=True).round(3))

    # 2. Feature engineering
    dt = pd.to_datetime(df["Crash Date/Time"], errors="coerce", format="mixed")
    df["Hour"] = dt.dt.hour
    df["DayOfWeek"] = dt.dt.dayofweek
    df["IsNight"] = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Vehicle Age"] = REFERENCE_YEAR - pd.to_numeric(df["Vehicle Year"], errors="coerce").replace(0, np.nan)
    for col in ["Speed Limit", "Latitude", "Longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Fill + encode (one map per column, saved for the app)
    data = df[FEATURES + ["Severity_Class"]].copy()
    numeric_medians = {}
    for col in FEATURES:
        if col in CATEGORICAL:
            data[col] = data[col].fillna("UNKNOWN").astype(str)
        else:
            numeric_medians[col] = float(data[col].median())
            data[col] = data[col].fillna(numeric_medians[col])

    le_dict, options = {}, {}
    for col in CATEGORICAL:
        classes = sorted(data[col].unique())  # same ordering as sklearn LabelEncoder
        le_dict[col] = {c: i for i, c in enumerate(classes)}
        # Dropdown choices: the most common values seen in training
        options[col] = data[col].value_counts().head(15).index.tolist()
        data[col] = data[col].map(le_dict[col]).astype(int)

    X, y = data[FEATURES], data["Severity_Class"]

    # 4-5. Split + interactions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = add_interactions(X_train), add_interactions(X_test)
    feature_names = list(X_train.columns)

    # 6. SMOTETomek on train only
    X_train_sm, y_train_sm = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
    print("After SMOTETomek:", pd.Series(y_train_sm).value_counts().to_dict())

    # 7. XGBoost V1
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=5, eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    model.fit(X_train_sm, y_train_sm)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    metrics = {
        "rows": int(len(data)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "recall": round(float(recall_score(y_test, pred)), 3),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 3),
        "auc": round(float(roc_auc_score(y_test, proba)), 3),
        "threshold": THRESHOLD,
    }
    print("Test metrics:", json.dumps(metrics))

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "xgb_model.pkl"))
    joblib.dump(
        {
            "feature_names": feature_names,
            "le_dict": le_dict,
            "options": options,
            "numeric_medians": numeric_medians,
            "threshold": THRESHOLD,
            "reference_year": REFERENCE_YEAR,
            "metrics": metrics,
        },
        os.path.join(out_dir, "encoders.pkl"),
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model + encoders to {out_dir}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/Crash_Reporting_-_Drivers_Data.csv")
