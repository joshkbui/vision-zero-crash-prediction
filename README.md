# Vision Zero Injury Risk Simulator
### Crash Injury Prediction for Montgomery County, MD
**DATA 4382-002 — Capstone 2 | Josh Bui & Danson Vo**

**Live app:** [vision-zero-simulator.streamlit.app](https://vision-zero-simulator.streamlit.app/)

---

## Table of Contents
1. [Project Title](#1-project-title)
2. [Business Problem](#2-business-problem)
3. [Project Overview](#3-project-overview)
4. [Data](#4-data)
5. [Preprocessing](#5-preprocessing)
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
7. [Modeling Approach](#7-modeling-approach)
8. [Model Training](#8-model-training)
9. [Results](#9-results)
10. [Model Interpretation (SHAP)](#10-model-interpretation-shap)
11. [Key Insights](#11-key-insights)
12. [Conclusion](#12-conclusion)
13. [Future Work](#13-future-work)
14. [How to Run](#14-how-to-run)
15. [Repository Structure](#15-repository-structure)
16. [Requirements](#16-requirements)

---

## 1. Project Title

**Vision Zero Injury Risk Simulator**
*Binary Crash Injury Prediction for Montgomery County, MD*

---

## 2. Business Problem

Traffic crashes are a leading cause of preventable death and injury in the United States. Montgomery County, MD processes thousands of crash reports annually. Emergency responders, traffic engineers, and policy makers need a fast, reliable way to assess which crashes are most likely to result in injury — enabling smarter resource allocation, targeted road safety interventions, and data-driven Vision Zero initiatives.

**Goal:** Build a binary classification model that predicts whether a crash results in an injury (1) or no injury (0), using structured crash report data available at the time of the incident.

---

## 3. Project Overview

This project builds a full end-to-end machine learning pipeline:

- **Data source:** Montgomery County Crash Reporting – Drivers Data (~220K rows) from the county open data portal
- **Target variable:** Injury Severity remapped to binary — NO APPARENT INJURY = 0, all other severity levels = 1
- **Best model:** XGBoost (threshold = 0.40) — **90% injury recall**
- **Deployment:** Interactive Streamlit app — *Vision Zero Injury Risk Simulator* — with real-time SHAP waterfall explanations
- **Fairness analysis:** SHAP-based subgroup analysis by lighting condition, weather, and time of day

---

## 4. Data

| Property | Value |
|---|---|
| **Source** | [Montgomery County Open Data Portal](https://data.montgomerycountymd.gov/) |
| **Dataset** | Crash Reporting – Drivers Data |
| **Rows** | 219,907 (after removing rows with no valid injury severity) |
| **Target** | Injury Severity: binary (0 = No Injury, 1 = Injury) |
| **Class imbalance** | ~82% no injury / ~18% injury |

**Features used (29 total):**

| Category | Features |
|---|---|
| Environment | Weather, Surface Condition, Light, Traffic Control, Route Type |
| Vehicle | Vehicle Body Type, Vehicle Damage Extent, Vehicle Movement, Vehicle Going Dir, Vehicle Age, Vehicle First Impact Location, Driverless Vehicle, Parked Vehicle |
| Driver | Driver Substance Abuse, Driver At Fault, Driver Distracted By, Non-Motorist Substance Abuse |
| Incident | Collision Type, Circumstance, Speed Limit |
| Spatial/Temporal | Latitude, Longitude, Hour, DayOfWeek, IsNight, IsWeekend |
| Interaction | Speed_x_Night, Substance_x_Night, VehicleAge_x_Collision |

---

## 5. Preprocessing

- **Missing values:** Median imputation for numeric; "UNKNOWN" for categoricals
- **Encoding:** Label encoding for all categorical features (one mapping per column, saved in `models/encoders.pkl` so the app encodes inputs exactly like training)
- **Target remapping:** NO APPARENT INJURY = 0; all other severity levels = 1
- **Class imbalance:** Addressed with **SMOTETomek** (combined oversampling + Tomek links cleaning) on the training set only
- **Feature engineering:** Hour and DayOfWeek extracted from timestamps; binary flags IsNight (Hour ≥ 22 or Hour ≤ 5) and IsWeekend; Vehicle Age = 2024 − Vehicle Year; interaction terms Speed_x_Night, Substance_x_Night, VehicleAge_x_Collision
- **Train/test split:** 80/20 stratified

---

## 6. Exploratory Data Analysis (EDA)

Key findings from EDA:

- **Time patterns:** Crashes spike during rush hours (7-9 AM, 4-6 PM); nighttime crashes show higher injury rates
- **Substance abuse:** Crashes with driver substance abuse show disproportionately higher injury severity
- **Speed limit:** Higher speed limit zones correlate strongly with injury outcomes
- **Lighting conditions:** Dark/no-streetlight conditions increase injury probability — a key fairness dimension
- **Weather:** Rain and snow show elevated injury rates vs. clear conditions
- **Vehicle damage extent:** Disabling and functional damage are strong predictors
- **Collision type:** Angle and head-on collisions have the highest injury rates

Visualizations available in `images/` (confusion_matrix.png, shap_bar.png, shap_beeswarm.png, fairness_light.png).

---

## 7. Modeling Approach

Four models were trained and compared in the notebook:

| Model | Notes |
|---|---|
| **Logistic Regression** | Baseline — fast, interpretable, limited on non-linear patterns |
| **Random Forest** | Ensemble — good recall, slower inference |
| **LightGBM** | Gradient boosting — fast training, competitive performance |
| **XGBoost** | **Best model** — highest injury recall at optimized threshold |

**Key design choices:**
- Threshold tuned from 0.50 to **0.40** to maximize injury recall (minimize missed injuries)
- `scale_pos_weight = 5` to further address class imbalance alongside SMOTETomek
- Recall prioritized over precision — false negatives (missed injuries) are costlier in safety-critical contexts

---

## 8. Model Training

**XGBoost Final Hyperparameters:**

```python
XGBClassifier(
    n_estimators     = 300,
    max_depth        = 6,
    learning_rate    = 0.05,
    scale_pos_weight = 5,
    eval_metric      = 'logloss',
    random_state     = 42
)
# Prediction threshold: 0.40
```

**Training pipeline (`scripts/train_model.py`):**
1. Load and clean raw crash data
2. Engineer features and interaction terms
3. Label-encode categoricals and save the encodings
4. Apply SMOTETomek resampling (training set only)
5. Train XGBoost with the hyperparameters above
6. Apply threshold = 0.40 on predicted probabilities
7. Evaluate on the held-out test set
8. Save `models/xgb_model.pkl`, `models/encoders.pkl`, and `models/metrics.json`

The deployed model is trained by the **Train model** GitHub Action (`.github/workflows/train-model.yml`), which downloads the latest county data and runs the script. Exploratory work, model comparison, and SHAP plots are in `notebooks/Final_Deployment_Code.ipynb`.

---

## 9. Results

**XGBoost @ threshold = 0.40 (held-out test set, 20%)**

| Metric | Value |
|---|---|
| Injury Recall | **0.90** |
| Injury Precision | 0.26 |
| Injury F1 | 0.41 |
| Accuracy | 0.53 |
| ROC AUC | 0.75 |

**90% recall on the injury class** — the model correctly flags 9 out of 10 injurious crashes.

The trade-off is deliberate: with the low 0.40 threshold, SMOTETomek, and `scale_pos_weight = 5`, the model leans heavily toward predicting injury. It rarely misses an injury crash, but many non-injury crashes are also flagged, which lowers precision and overall accuracy. Current numbers are always in `models/metrics.json`.

See `images/confusion_matrix.png` for visualization.

---

## 10. Model Interpretation (SHAP)

SHAP (SHapley Additive exPlanations) was used for global and local interpretation.

**Global feature importance (`images/shap_bar.png`):**
Top predictors: Vehicle Damage Extent, Speed Limit, Collision Type, Driver Substance Abuse, IsNight, Weather, Speed_x_Night, Vehicle Age.

**SHAP beeswarm plot (`images/shap_beeswarm.png`):**
Shows distribution and direction of feature impacts. High speed limits and severe vehicle damage consistently push predictions toward injury.

**Fairness analysis (`images/fairness_light.png`):**
SHAP mean absolute values computed by lighting condition, weather, and time of day. Feature importance is consistent across subgroups — no evidence of proxy discrimination through environmental conditions.

**Local explanations:**
Every prediction in the Streamlit app includes a SHAP waterfall chart showing which features drove the individual prediction.

---

## 11. Key Insights

- **Speed limit** is the most controllable risk factor — 50+ mph zones substantially increase injury probability
- **Nighttime driving amplifies risk non-linearly** — the Speed_x_Night interaction captures this effect
- **Substance abuse** is a strong predictor even when controlling for other factors
- **Vehicle damage extent** is the strongest retrospective predictor — useful for injury triage
- **Head-on and angle collisions** carry the highest injury probabilities
- **Consistent performance across subgroups** supports use as an equitable risk-scoring tool

---

## 12. Conclusion

This project demonstrates that XGBoost with threshold calibration and SMOTETomek resampling achieves 90% injury recall on real-world crash data. The model is interpretable via SHAP, deployable in a real-time Streamlit app, and performs consistently across environmental subgroups. Its main limitation is precision: favoring recall means many non-injury crashes are flagged as high risk.

The Vision Zero Injury Risk Simulator translates raw crash data into actionable, explainable risk scores — a meaningful contribution to data-driven traffic safety policy.

---

## 13. Future Work

- Improve precision without giving up recall (threshold and resampling tuning, better categorical encoding)
- Harmonize the county's old and new category labels (e.g., "CLEAR" vs. "Clear")
- Incorporate pedestrian/victim data (Montgomery County Non-Motorists dataset)
- Geospatial risk mapping — overlay predictions on the county road network
- Temporal drift monitoring and periodic retraining
- Calibration improvement via Platt scaling or isotonic regression
- Expand to multi-class severity prediction (no injury / possible / injury / fatal)
- API deployment (FastAPI) for integration with county dispatch systems
- Counterfactual explanations ("what would need to change to avoid injury?")

---

## 14. How to Run

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/joshkbui/vision-zero-crash-prediction.git
cd vision-zero-crash-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Retrain the model

**On GitHub:** go to **Actions → Train model → Run workflow**. It downloads the latest data, retrains, and commits the new model files.

**Locally:**

```bash
# Download "Crash Reporting – Drivers Data" as CSV into data/ (see data/README.md), then:
python scripts/train_model.py data/Crash_Reporting_-_Drivers_Data.csv
```

### Data

Download **Crash Reporting – Drivers Data** from the [Montgomery County Open Data Portal](https://data.montgomerycountymd.gov/) and place in `data/`. See `data/README.md` for details.

---

## 15. Repository Structure

```
vision-zero-crash-prediction/
├── README.md                        # Project documentation (this file)
├── app.py                           # Streamlit deployment app
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Ignored files
│
├── .github/workflows/
│   └── train-model.yml              # GitHub Action: download data + retrain
│
├── scripts/
│   └── train_model.py               # Training script used for the deployed model
│
├── data/
│   └── README.md                    # Data download instructions
│
├── models/
│   ├── xgb_model.pkl                # Trained XGBoost model
│   ├── encoders.pkl                 # Label encodings, feature names, dropdown options
│   ├── metrics.json                 # Test-set metrics
│   └── README.md                    # Model artifacts description
│
├── notebooks/
│   └── Final_Deployment_Code.ipynb  # Exploration, model comparison & SHAP analysis
│
├── results/                         # Model evaluation outputs
│
└── images/
    ├── confusion_matrix.png         # Confusion matrix (XGBoost @ threshold=0.40)
    ├── shap_bar.png                 # Global SHAP feature importance
    ├── shap_beeswarm.png            # SHAP beeswarm plot
    └── fairness_light.png           # Fairness analysis by lighting condition
```

---

## 16. Requirements

See `requirements.txt`. The core ML libraries are pinned to the versions used to train the deployed model:

```
scikit-learn==1.9.1
xgboost==3.4.1
imbalanced-learn==0.14.2
shap==0.54.1
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

*Built by Josh Bui & Danson Vo — DATA 4382-002 Capstone 2, University of Texas at Arlington*
