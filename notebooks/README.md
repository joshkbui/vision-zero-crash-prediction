# Notebooks

This directory contains the Jupyter notebooks for the Vision Zero Injury Risk Simulator project.

## Main Notebook

**`Final_Deployment_Code.ipynb`** — Full training and evaluation pipeline

This notebook covers:
1. Data loading and exploration (EDA)
2. Feature engineering and preprocessing
3. SMOTETomek resampling for class imbalance
4. Training all four models (Logistic Regression, Random Forest, LightGBM, XGBoost)
5. Hyperparameter evaluation and threshold tuning
6. Model evaluation (classification report, confusion matrix, AUC-ROC)
7. SHAP global and local interpretation
8. Fairness analysis by lighting condition, weather, and time of day
9. Saving the final XGBoost model to `models/xgb_model.pkl`
10. Generating SHAP plots to `images/`

## How to Run

Install dependencies first:
```bash
pip install -r requirements.txt
```

Then launch the notebook:
```bash
jupyter notebook notebooks/Final_Deployment_Code.ipynb
```

Or open in JupyterLab:
```bash
jupyter lab notebooks/Final_Deployment_Code.ipynb
```

**Note:** Place the raw data CSV in `data/` before running. See `data/README.md` for download instructions.
