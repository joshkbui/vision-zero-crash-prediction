# Vision Zero Injury Risk Simulator
### Crash Injury Prediction for Montgomery County, MD
**DATA 4382-002 — Capstone 2 | Josh Bui & Danson Vo**

---

## Table of Contents
1. [Project Title](#1-project-title)
2. 2. [Business Problem](#2-business-problem)
   3. 3. [Project Overview](#3-project-overview)
      4. 4. [Data](#4-data)
         5. 5. [Preprocessing](#5-preprocessing)
            6. 6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
               7. 7. [Modeling Approach](#7-modeling-approach)
                  8. 8. [Model Training](#8-model-training)
                     9. 9. [Results](#9-results)
                        10. 10. [Model Interpretation (SHAP)](#10-model-interpretation-shap)
                            11. 11. [Key Insights](#11-key-insights)
                                12. 12. [Conclusion](#12-conclusion)
                                    13. 13. [Future Work](#13-future-work)
                                        14. 14. [How to Run](#14-how-to-run)
                                            15. 15. [Repository Structure](#15-repository-structure)
                                                16. 16. [Requirements](#16-requirements)
                                                   
                                                    17. ---
                                                   
                                                    18. ## 1. Project Title
                                                   
                                                    19. **Vision Zero Injury Risk Simulator**
                                                    20. *Binary Crash Injury Prediction for Montgomery County, MD*
                                                   
                                                    21. ---
                                                   
                                                    22. ## 2. Business Problem
                                                   
                                                    23. Traffic crashes are a leading cause of preventable death and injury in the United States. Montgomery County, MD processes thousands of crash reports annually. Emergency responders, traffic engineers, and policy makers need a fast, reliable way to assess which crashes are most likely to result in injury — enabling smarter resource allocation, targeted road safety interventions, and data-driven Vision Zero initiatives.
                                                   
                                                    24. **Goal:** Build a binary classification model that predicts whether a crash results in an injury (1) or no injury (0), using structured crash report data available at the time of the incident.
                                                   
                                                    25. ---
                                                   
                                                    26. ## 3. Project Overview
                                                   
                                                    27. This project builds a full end-to-end machine learning pipeline:
                                                   
                                                    28. - **Data source:** Montgomery County Crash Reporting – Drivers Data (~201K rows) from the county open data portal
                                                        - - **Target variable:** Injury Severity remapped to binary — NO APPARENT INJURY = 0, all other severity levels = 1
                                                          - - **Best model:** XGBoost (threshold = 0.40) — **91% injury recall**
                                                            - - **Deployment:** Interactive Streamlit app — *Vision Zero Injury Risk Simulator* — with real-time SHAP waterfall explanations
                                                              - - **Fairness analysis:** SHAP-based subgroup analysis by lighting condition, weather, and time of day
                                                               
                                                                - ---

                                                                ## 4. Data

                                                                | Property | Value |
                                                                |---|---|
                                                                | **Source** | [Montgomery County Open Data Portal](https://data.montgomerycountymd.gov/) |
                                                                | **Dataset** | Crash Reporting – Drivers Data |
                                                                | **Rows** | ~201,000 |
                                                                | **Target** | Injury Severity: binary (0 = No Injury, 1 = Injury) |
                                                                | **Class imbalance** | ~60% no injury / ~40% injury |

                                                                **Features used (27 total):**

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

                                                                - **Missing values:** Median imputation for numeric; mode or "UNKNOWN" for categoricals
                                                                - - **Encoding:** Label encoding for all categorical features
                                                                  - - **Target remapping:** NO APPARENT INJURY = 0; all other severity levels = 1
                                                                    - - **Class imbalance:** Addressed with **SMOTETomek** (combined oversampling + Tomek links cleaning)
                                                                      - - **Feature engineering:** Hour and DayOfWeek extracted from timestamps; binary flags IsNight (Hour < 6 or Hour >= 20) and IsWeekend; interaction terms Speed_x_Night, Substance_x_Night, VehicleAge_x_Collision
                                                                        - - **Train/test split:** 80/20 stratified
                                                                         
                                                                          - ---

                                                                          ## 6. Exploratory Data Analysis (EDA)

                                                                          Key findings from EDA:

                                                                          - **Time patterns:** Crashes spike during rush hours (7-9 AM, 4-6 PM); nighttime crashes show higher injury rates
                                                                          - - **Substance abuse:** Crashes with driver substance abuse show disproportionately higher injury severity
                                                                            - - **Speed limit:** Higher speed limit zones correlate strongly with injury outcomes
                                                                              - - **Lighting conditions:** Dark/no-streetlight conditions increase injury probability — a key fairness dimension
                                                                                - - **Weather:** Rain and snow show elevated injury rates vs. clear conditions
                                                                                  - - **Vehicle damage extent:** Disabling and functional damage are strong predictors
                                                                                    - - **Collision type:** Angle and head-on collisions have the highest injury rates
                                                                                     
                                                                                      - Visualizations available in `images/` (confusion_matrix.png, shap_bar.png, shap_beeswarm.png, fairness_light.png).
                                                                                     
                                                                                      - ---

                                                                                      ## 7. Modeling Approach

                                                                                      Four models were trained and compared:

                                                                                      | Model | Notes |
                                                                                      |---|---|
                                                                                      | **Logistic Regression** | Baseline — fast, interpretable, limited on non-linear patterns |
                                                                                      | **Random Forest** | Ensemble — good recall, slower inference |
                                                                                      | **LightGBM** | Gradient boosting — fast training, competitive performance |
                                                                                      | **XGBoost** | **Best model** — highest injury recall at optimized threshold |

                                                                                      **Key design choices:**
                                                                                      - Threshold tuned from 0.50 to **0.40** to maximize injury recall (minimize missed injuries)
                                                                                      - - `scale_pos_weight = 5` to further address class imbalance alongside SMOTETomek
                                                                                        - - Recall prioritized over precision — false negatives (missed injuries) are costlier in safety-critical contexts
                                                                                         
                                                                                          - ---

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

                                                                                          **Training pipeline:**
                                                                                          1. Load and clean raw crash data
                                                                                          2. 2. Engineer features and interaction terms
                                                                                             3. 3. Apply SMOTETomek resampling (training set only)
                                                                                                4. 4. Train XGBoost with tuned hyperparameters
                                                                                                   5. 5. Apply threshold = 0.40 on predicted probabilities
                                                                                                      6. 6. Evaluate on held-out test set
                                                                                                         7. 7. Generate SHAP explanations
                                                                                                            8. 8. Save model (`models/xgb_model.pkl`) and artifacts
                                                                                                              
                                                                                                               9. Full training code: `notebooks/Final_Deployment_Code.ipynb`
                                                                                                              
                                                                                                               10. ---
                                                                                                              
                                                                                                               11. ## 9. Results
                                                                                                              
                                                                                                               12. **XGBoost @ threshold = 0.40 (Test Set)**
                                                                                                              
                                                                                                               13. | Metric | No Injury (0) | Injury (1) | Weighted Avg |
                                                                                                               14. |---|---|---|---|
                                                                                                               15. | Precision | 0.93 | 0.74 | 0.86 |
                                                                                                               16. | Recall | 0.79 | 0.91 | 0.84 |
                                                                                                               17. | F1-Score | 0.85 | 0.82 | 0.84 |
                                                                                                               18. | Accuracy | — | — | **84%** |
                                                                                                              
                                                                                                               19. **91% recall on the injury class** — the model correctly identifies 9 out of 10 injurious crashes.
                                                                                                              
                                                                                                               20. **Model comparison:**
                                                                                                              
                                                                                                               21. | Model | Injury Recall | F1 (Injury) | AUC |
                                                                                                               22. |---|---|---|---|
                                                                                                               23. | Logistic Regression | 0.76 | 0.74 | 0.82 |
                                                                                                               24. | Random Forest | 0.85 | 0.80 | 0.89 |
                                                                                                               25. | LightGBM | 0.88 | 0.81 | 0.91 |
                                                                                                               26. | **XGBoost (threshold=0.40)** | **0.91** | **0.82** | **0.92** |
                                                                                                              
                                                                                                               27. See `images/confusion_matrix.png` for visualization.
                                                                                                              
                                                                                                               28. ---
                                                                                                              
                                                                                                               29. ## 10. Model Interpretation (SHAP)
                                                                                                              
                                                                                                               30. SHAP (SHapley Additive exPlanations) was used for global and local interpretation.
                                                                                                              
                                                                                                               31. **Global feature importance (`images/shap_bar.png`):**
                                                                                                               32. Top predictors: Vehicle Damage Extent, Speed Limit, Collision Type, Driver Substance Abuse, IsNight, Weather, Speed_x_Night, Vehicle Age.
                                                                                                              
                                                                                                               33. **SHAP beeswarm plot (`images/shap_beeswarm.png`):**
                                                                                                               34. Shows distribution and direction of feature impacts. High speed limits and severe vehicle damage consistently push predictions toward injury.
                                                                                                              
                                                                                                               35. **Fairness analysis (`images/fairness_light.png`):**
                                                                                                               36. SHAP mean absolute values computed by lighting condition, weather, and time of day. Feature importance is consistent across subgroups — no evidence of proxy discrimination through environmental conditions.
                                                                                                              
                                                                                                               37. **Local explanations:**
                                                                                                               38. Every prediction in the Streamlit app includes a SHAP waterfall chart showing which features drove the individual prediction.
                                                                                                              
                                                                                                               39. ---
                                                                                                              
                                                                                                               40. ## 11. Key Insights
                                                                                                              
                                                                                                               41. - **Speed limit** is the most controllable risk factor — 50+ mph zones substantially increase injury probability
                                                                                                                   - - **Nighttime driving amplifies risk non-linearly** — the Speed_x_Night interaction captures this effect
                                                                                                                     - - **Substance abuse** is a strong predictor even when controlling for other factors
                                                                                                                       - - **Vehicle damage extent** is the strongest retrospective predictor — useful for injury triage
                                                                                                                         - - **Head-on and angle collisions** carry the highest injury probabilities
                                                                                                                           - - **Consistent performance across subgroups** supports use as an equitable risk-scoring tool
                                                                                                                            
                                                                                                                             - ---
                                                                                                                             
                                                                                                                             ## 12. Conclusion
                                                                                                                             
                                                                                                                             This project demonstrates that XGBoost with threshold calibration and SMOTETomek resampling achieves 91% injury recall on real-world crash data. The model is interpretable via SHAP, deployable in a real-time Streamlit app, and performs consistently across environmental subgroups.
                                                                                                                             
                                                                                                                             The Vision Zero Injury Risk Simulator translates raw crash data into actionable, explainable risk scores — a meaningful contribution to data-driven traffic safety policy.
                                                                                                                             
                                                                                                                             ---
                                                                                                                             
                                                                                                                             ## 13. Future Work
                                                                                                                             
                                                                                                                             - Incorporate pedestrian/victim data (Montgomery County Non-Motorists dataset)
                                                                                                                             - - Geospatial risk mapping — overlay predictions on the county road network
                                                                                                                               - - Temporal drift monitoring and periodic retraining
                                                                                                                                 - - Calibration improvement via Platt scaling or isotonic regression
                                                                                                                                   - - Expand to multi-class severity prediction (no injury / possible / injury / fatal)
                                                                                                                                     - - API deployment (FastAPI) for integration with county dispatch systems
                                                                                                                                       - - Counterfactual explanations ("what would need to change to avoid injury?")
                                                                                                                                        
                                                                                                                                         - ---
                                                                                                                                         
                                                                                                                                         ## 14. How to Run
                                                                                                                                         
                                                                                                                                         ### Prerequisites
                                                                                                                                         - Python 3.9+
                                                                                                                                         - - pip
                                                                                                                                          
                                                                                                                                           - ### Setup
                                                                                                                                          
                                                                                                                                           - ```bash
                                                                                                                                             # Clone the repository
                                                                                                                                             git clone https://github.com/joshkbui/vision-zero-crash-prediction.git
                                                                                                                                             cd vision-zero-crash-prediction

                                                                                                                                             # Install dependencies
                                                                                                                                             pip install -r requirements.txt

                                                                                                                                             # Run the Streamlit app
                                                                                                                                             streamlit run app.py
                                                                                                                                             ```
                                                                                                                                             
                                                                                                                                             ### Retrain the model
                                                                                                                                             
                                                                                                                                             Open and run all cells in `notebooks/Final_Deployment_Code.ipynb`. The notebook will:
                                                                                                                                             1. Load raw crash data (place CSV in `data/`)
                                                                                                                                             2. 2. Preprocess and engineer features
                                                                                                                                                3. 3. Apply SMOTETomek resampling
                                                                                                                                                   4. 4. Train and evaluate all four models
                                                                                                                                                      5. 5. Save XGBoost model to `models/xgb_model.pkl`
                                                                                                                                                         6. 6. Generate SHAP plots to `images/`
                                                                                                                                                           
                                                                                                                                                            7. ### Data
                                                                                                                                                           
                                                                                                                                                            8. Download **Crash Reporting – Drivers Data** from the [Montgomery County Open Data Portal](https://data.montgomerycountymd.gov/) and place in `data/`. See `data/README.md` for details.
                                                                                                                                                           
                                                                                                                                                            9. ---
                                                                                                                                                           
                                                                                                                                                            10. ## 15. Repository Structure
                                                                                                                                                           
                                                                                                                                                            11. ```
                                                                                                                                                                vision-zero-crash-prediction/
                                                                                                                                                                ├── README.md                        # Project documentation (this file)
                                                                                                                                                                ├── app.py                           # Streamlit deployment app
                                                                                                                                                                ├── requirements.txt                 # Python dependencies
                                                                                                                                                                ├── .gitignore                       # Ignored files
                                                                                                                                                                │
                                                                                                                                                                ├── data/
                                                                                                                                                                │   └── README.md                    # Data download instructions
                                                                                                                                                                │
                                                                                                                                                                ├── models/
                                                                                                                                                                │   └── README.md                    # Model artifacts description
                                                                                                                                                                │
                                                                                                                                                                ├── notebooks/
                                                                                                                                                                │   └── Final_Deployment_Code.ipynb  # Full training & evaluation pipeline
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
                                                                                                                                                                
                                                                                                                                                                ```
                                                                                                                                                                pandas>=1.5.0
                                                                                                                                                                numpy>=1.23.0
                                                                                                                                                                scikit-learn>=1.2.0
                                                                                                                                                                xgboost>=1.7.0
                                                                                                                                                                lightgbm>=3.3.0
                                                                                                                                                                imbalanced-learn>=0.10.0
                                                                                                                                                                shap>=0.41.0
                                                                                                                                                                streamlit>=1.20.0
                                                                                                                                                                matplotlib>=3.6.0
                                                                                                                                                                seaborn>=0.12.0
                                                                                                                                                                joblib>=1.2.0
                                                                                                                                                                ```
                                                                                                                                                                
                                                                                                                                                                Install all dependencies:
                                                                                                                                                                ```bash
                                                                                                                                                                pip install -r requirements.txt
                                                                                                                                                                ```
                                                                                                                                                                
                                                                                                                                                                ---
                                                                                                                                                                
                                                                                                                                                                *Built by Josh Bui & Danson Vo — DATA 4382-002 Capstone 2, University of Texas at Arlington*
