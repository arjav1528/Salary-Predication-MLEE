# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from sklearn.calibration import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from datetime import datetime
from joblib import Parallel, delayed


# Define RMSPE function
def rmspe(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    non_zero_indices = y_true != 0
    if not np.any(non_zero_indices):
        raise Exception("All true values are zero, RMSPE is undefined.")
    
    percentage_errors = (y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices]
    rmspe_value = np.sqrt(np.mean(percentage_errors ** 2))
    
    return rmspe_value


# Define data processing function
def process_data(train, cost_of_living):
    train['city'] = train['city'].str.lower()
    train['state'] = train['state'].str.lower()
    train['country'] = train['country'].str.lower()
    cost_of_living['city'] = cost_of_living['city'].str.lower()
    cost_of_living['state'] = cost_of_living['state'].str.lower()
    cost_of_living['country'] = cost_of_living['country'].str.lower()
    
    merged_data = pd.merge(train, cost_of_living, on=['city', 'state', 'country'], how='left')

    merged_data.fillna(merged_data.median(numeric_only=True), inplace=True)

    categorical_cols = ['country', 'state', 'city', 'role']
    for col in categorical_cols:
        le = LabelEncoder()
        merged_data[col] = le.fit_transform(merged_data[col])

    return merged_data


# Define model training and evaluation function
def train_and_evaluate_model(name, model, X_train, y_train, X_val, y_val):
    init = datetime.now()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = model.score(X_val, y_val)
    rmspe_score = rmspe(y_val, y_pred)
    final = datetime.now()
    time_taken_ms = (final - init).total_seconds() * 1000
    print(f"{name} - Model Accuracy: {accuracy}, RMSPE Score: {rmspe_score}, Time Taken: {time_taken_ms:.2f}ms")
    return name, rmspe_score, model, accuracy, time_taken_ms


# Load datasets
cost_of_living = pd.read_csv('cost_of_living.csv', na_values='NA')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Preprocess data
df = process_data(train, cost_of_living)
X = df.drop(['salary_average', 'ID', 'city_id'], axis=1)
y = df['salary_average']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Define models to evaluate
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
    "LinearRegression": skl.linear_model.LinearRegression(),
    "SupportVectorRegressor": skl.svm.SVR(),
    "KNeighborsRegressor": skl.neighbors.KNeighborsRegressor(),
    "DecisionTreeRegressor": skl.tree.DecisionTreeRegressor(random_state=42),
    "AdaBoostRegressor": skl.ensemble.AdaBoostRegressor(random_state=42),
    "BaggingRegressor": skl.ensemble.BaggingRegressor(random_state=42),
    "ExtraTreesRegressor": skl.ensemble.ExtraTreesRegressor(random_state=42),
}


# Train and evaluate models in parallel
results = Parallel(n_jobs=-1)(delayed(train_and_evaluate_model)(name, model, X_train, y_train, X_val, y_val) for name, model in models.items())


# Find the best model based on RMSPE
best_model_name = None
best_rmspe = float('inf')
best_model = None
best_accuracy = 0
best_time_taken = 0
for name, rmspe_score, model, accuracy, time_taken_ms in results:
    if rmspe_score < best_rmspe:
        best_rmspe = rmspe_score
        best_model_name = name
        best_model = model
        best_accuracy = accuracy
        best_time_taken = time_taken_ms

print(f"\n\nBest Model: {best_model_name} with RMSPE: {best_rmspe} and Accuracy: {best_accuracy} and Time Taken: {best_time_taken:.2f}ms")


# Make predictions on the test set using the best model
df = process_data(test, cost_of_living)
test_X = df.drop(['ID', 'city_id'], axis=1)
test_predictions = best_model.predict(test_X)
test['salary_average'] = test_predictions
test[['ID', 'salary_average']].to_csv('predictions.csv', index=False)
print("\n\nPredictions saved to predictions.csv")
