# Salary Prediction (MLEE)

This project was developed as a part of an evaluative component for the course ***Machine Learning for Electrical Engineering***

Predict average salaries across global cities while accounting for local cost-of-living economics. This project trains multiple regression models and selects the best one based on RMSPE.

- Task: Regression (predict salary_average)
- Metric: Root Mean Square Percentage Error (RMSPE), lower is better

## Overview
- The dataset contains salary information for 9 job roles across 68 countries and 1,000+ cities.
- Cost-of-living indicators add 52 economic features for 1,528 cities (housing, food, transport, healthcare, purchasing power, etc.).
- Goal: Generalize salaries to new cities by learning how economic conditions shift salary structures.

Why it matters:
- Fair global compensation benchmarking
- Comparing salaries across economic regions
- Insight into cross-geographic pay normalization

## Available datasets
- salary.csv (core salaries)
  - Average salary for a job role in a city
  - Wide range from ~$3.6k to ~$161k annually
  - Raw salary alone is not comparable across regions without normalization
- cost_of_living.csv (economic indicators)
  - 52 features per city (prices, costs, purchasing power)
  - Essential to normalize salary expectations by location economics

## Repository layout
- main.py — end-to-end pipeline: load → process → train/eval → predict
- train.csv — training data
- test.csv — test data
- cost_of_living.csv — COL features
- predictions.csv — generated predictions
- README.md — this file

### **Execute the Program**
```bash
python3 -m venv venv
source venv/bin/activate ## macOS or Linux
venv\Scripts\activate ## Windows
pip install -r requirements.txt
python main.py
```

```bash
You should see per-model metrics and the selected best model. Example:
LinearRegression - Model Accuracy: 0.48575536914089634, RMSPE Score: 1.0217105927146999, Time Taken: 6.17ms
KNeighborsRegressor - Model Accuracy: 0.8250720377848828, RMSPE Score: 0.24914674911364862, Time Taken: 72.18ms
DecisionTreeRegressor - Model Accuracy: 0.9576700213575977, RMSPE Score: 0.11580337979737118, Time Taken: 211.69ms
XGBoost - Model Accuracy: 0.9781607771513405, RMSPE Score: 0.1453257740704211, Time Taken: 582.45ms
BaggingRegressor - Model Accuracy: 0.9699182068608801, RMSPE Score: 0.12287670343019172, Time Taken: 1005.55ms
AdaBoostRegressor - Model Accuracy: 0.6907269870996177, RMSPE Score: 0.9019082952107749, Time Taken: 1046.65ms
RandomForest - Model Accuracy: 0.9749635190527897, RMSPE Score: 0.10824543245136602, Time Taken: 1600.07ms
SupportVectorRegressor - Model Accuracy: -0.0017062335286532893, RMSPE Score: 1.6979786522190032, Time Taken: 2022.10ms
GradientBoosting - Model Accuracy: 0.91394353883972, RMSPE Score: 0.3461904286509873, Time Taken: 2032.73ms
ExtraTreesRegressor - Model Accuracy: 0.9728646292551175, RMSPE Score: 0.09759629452239073, Time Taken: 2736.85ms

Best Model: ExtraTreesRegressor with RMSPE: 0.09759629452239073 and Accuracy: 0.9728646292551175 and Time Taken: 2736.85ms

Predictions saved to predictions.csv
```

## Reproducibility
- Deterministic split via random_state=42
- Results may vary by library versions and hardware
- Parallel training via joblib uses all cores (n_jobs=-1)

