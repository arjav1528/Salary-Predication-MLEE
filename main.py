# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from datetime import datetime
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


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


# Global encoders storage for consistent encoding between train and test
encoders = {}

# Define feature engineering function
def engineer_features(df, cost_of_living, target_col=None, is_training=True, encoders_dict=None):
    """
    Apply comprehensive feature engineering to improve model accuracy.
    """
    if encoders_dict is None:
        encoders_dict = {}
    
    # Normalize location strings
    df = df.copy()
    df['city'] = df['city'].str.lower()
    df['state'] = df['state'].str.lower()
    df['country'] = df['country'].str.lower()
    cost_of_living = cost_of_living.copy()
    cost_of_living['city'] = cost_of_living['city'].str.lower()
    cost_of_living['state'] = cost_of_living['state'].str.lower()
    cost_of_living['country'] = cost_of_living['country'].str.lower()
    
    # Merge with cost of living data
    merged_data = pd.merge(df, cost_of_living, on=['city', 'state', 'country'], how='left')
    
    # Get cost of living columns
    col_cols = [col for col in merged_data.columns if col.startswith('col_')]
    
    # Fill missing values with median for numeric columns
    numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in merged_data.columns:
            median_val = merged_data[col].median()
            if pd.notna(median_val):
                merged_data[col] = merged_data[col].fillna(median_val)
            else:
                merged_data[col] = merged_data[col].fillna(0)
    
    # Feature Engineering 1: Statistical aggregations of cost of living columns
    if col_cols:
        merged_data['col_mean'] = merged_data[col_cols].mean(axis=1)
        merged_data['col_std'] = merged_data[col_cols].std(axis=1)
        merged_data['col_min'] = merged_data[col_cols].min(axis=1)
        merged_data['col_max'] = merged_data[col_cols].max(axis=1)
        merged_data['col_median'] = merged_data[col_cols].median(axis=1)
        merged_data['col_range'] = merged_data['col_max'] - merged_data['col_min']
        merged_data['col_skew'] = merged_data[col_cols].skew(axis=1)
        merged_data['col_non_null_count'] = merged_data[col_cols].notna().sum(axis=1)
        
        # Feature Engineering 2: Key cost of living indicators
        key_cols = ['col_14', 'col_15', 'col_16', 'col_17', 'col_18', 'col_19', 'col_20', 'col_30', 'col_31']
        available_key_cols = [col for col in key_cols if col in merged_data.columns]
        if available_key_cols:
            merged_data['key_col_mean'] = merged_data[available_key_cols].mean(axis=1)
            merged_data['key_col_sum'] = merged_data[available_key_cols].sum(axis=1)
    
    # Feature Engineering 3: Domain-specific features
    merged_data['is_us'] = (merged_data['country'] == 'united states of america').astype(int)
    
    # Calculate high cost threshold from training data or use a default
    if 'col_mean' in merged_data.columns:
        if is_training:
            threshold = merged_data['col_mean'].quantile(0.75)
            encoders_dict['high_cost_threshold'] = threshold
        else:
            threshold = encoders_dict.get('high_cost_threshold', merged_data['col_mean'].quantile(0.75))
        merged_data['is_high_cost_country'] = (merged_data['col_mean'] > threshold).astype(int)
    
    # Feature Engineering 4: Location hierarchy features
    merged_data['location_combo'] = merged_data['country'] + '_' + merged_data['state'] + '_' + merged_data['city']
    
    # Feature Engineering 5: Role-based features
    merged_data['role_length'] = merged_data['role'].str.len()
    merged_data['is_analyst'] = merged_data['role'].str.contains('analyst', case=False, na=False).astype(int)
    merged_data['is_specialist'] = merged_data['role'].str.contains('specialist', case=False, na=False).astype(int)
    
    # Encoding for categorical features
    categorical_cols = ['country', 'state', 'city', 'role', 'location_combo']
    
    # Label encoding (works for both train and test)
    for col in categorical_cols:
        if col in merged_data.columns:
            key = f'{col}_label_encoder'
            if is_training:
                le = LabelEncoder()
                merged_data[f'{col}_label_encoded'] = le.fit_transform(merged_data[col].astype(str))
                encoders_dict[key] = le
            else:
                le = encoders_dict.get(key)
                if le is not None:
                    # Handle unseen categories
                    unique_vals = set(le.classes_)
                    merged_data[f'{col}_label_encoded'] = merged_data[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in unique_vals else -1
                    )
                else:
                    # Fallback: simple hash encoding
                    merged_data[f'{col}_label_encoded'] = merged_data[col].astype(str).apply(
                        lambda x: hash(x) % 10000
                    )
    
    # Target encoding (only for training, or use mean encoding for test)
    if is_training and target_col is not None and target_col in merged_data.columns:
        for col in categorical_cols:
            if col in merged_data.columns:
                te = TargetEncoder(smooth="auto")
                merged_data[f'{col}_target_encoded'] = te.fit_transform(
                    merged_data[[col]], merged_data[target_col]
                )
                encoders_dict[f'{col}_target_encoder'] = te
    else:
        # For test data, use stored target encoders or fallback to label encoding
        for col in categorical_cols:
            if col in merged_data.columns:
                te = encoders_dict.get(f'{col}_target_encoder')
                if te is not None:
                    merged_data[f'{col}_target_encoded'] = te.transform(merged_data[[col]])
                else:
                    # Fallback: use label encoded value as proxy
                    if f'{col}_label_encoded' in merged_data.columns:
                        merged_data[f'{col}_target_encoded'] = merged_data[f'{col}_label_encoded']
                    else:
                        merged_data[f'{col}_target_encoded'] = 0
    
    # Feature Engineering 7: Interaction features between role and cost metrics
    if 'role_label_encoded' in merged_data.columns and col_cols:
        merged_data['role_col_mean_interaction'] = merged_data['role_label_encoded'] * merged_data['col_mean']
        merged_data['role_col_median_interaction'] = merged_data['role_label_encoded'] * merged_data['col_median']
        if 'key_col_mean' in merged_data.columns:
            merged_data['role_key_col_interaction'] = merged_data['role_label_encoded'] * merged_data['key_col_mean']
    
    # Feature Engineering 8: Location and cost interactions
    if 'country_label_encoded' in merged_data.columns:
        merged_data['country_col_mean_interaction'] = merged_data['country_label_encoded'] * merged_data['col_mean']
        merged_data['country_col_median_interaction'] = merged_data['country_label_encoded'] * merged_data['col_median']
    
    # Feature Engineering 9: Polynomial features for important numeric columns
    if 'col_mean' in merged_data.columns:
        merged_data['col_mean_squared'] = merged_data['col_mean'] ** 2
        merged_data['col_mean_sqrt'] = np.sqrt(np.abs(merged_data['col_mean']))
        merged_data['col_mean_log'] = np.log1p(np.abs(merged_data['col_mean']))
    
    if 'col_median' in merged_data.columns:
        merged_data['col_median_squared'] = merged_data['col_median'] ** 2
    
    # Feature Engineering 10: Ratio features
    if 'col_max' in merged_data.columns and 'col_min' in merged_data.columns:
        merged_data['col_max_min_ratio'] = merged_data['col_max'] / (merged_data['col_min'] + 1e-6)
    
    if 'col_mean' in merged_data.columns and 'col_std' in merged_data.columns:
        merged_data['col_cv'] = merged_data['col_std'] / (merged_data['col_mean'] + 1e-6)  # Coefficient of variation
    
    # Feature Engineering 11: Percentile-based features
    if col_cols and is_training:
        # Calculate percentiles for each cost column
        percentile_cols = []
        for col in col_cols[:20]:  # Use first 20 to avoid too many features
            if col in merged_data.columns:
                for p in [25, 50, 75, 90]:
                    percentile_val = merged_data[col].quantile(p / 100.0)
                    encoders_dict[f'{col}_p{p}'] = percentile_val
                    merged_data[f'{col}_above_p{p}'] = (merged_data[col] > percentile_val).astype(int)
                    percentile_cols.append(f'{col}_above_p{p}')
    elif col_cols:
        # Apply stored percentiles for test data
        for col in col_cols[:20]:
            if col in merged_data.columns:
                for p in [25, 50, 75, 90]:
                    percentile_val = encoders_dict.get(f'{col}_p{p}', merged_data[col].quantile(p / 100.0))
                    merged_data[f'{col}_above_p{p}'] = (merged_data[col] > percentile_val).astype(int)
    
    # Feature Engineering 12: Clustering-based features
    if col_cols and len(col_cols) > 5:
        # Use key cost columns for clustering
        cluster_cols = [c for c in col_cols[:15] if c in merged_data.columns]
        if len(cluster_cols) >= 5:
            cluster_data = merged_data[cluster_cols].fillna(0)
            if is_training:
                # Fit KMeans on training data
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                merged_data['cost_cluster'] = kmeans.fit_predict(cluster_data)
                encoders_dict['kmeans'] = kmeans
            else:
                # Predict clusters for test data
                kmeans = encoders_dict.get('kmeans')
                if kmeans is not None:
                    merged_data['cost_cluster'] = kmeans.predict(cluster_data)
                else:
                    merged_data['cost_cluster'] = 0
    
    # Feature Engineering 13: More role-specific features
    if 'role' in merged_data.columns:
        merged_data['role_has_finance'] = merged_data['role'].str.contains('finance|treasury|budget', case=False, na=False).astype(int)
        merged_data['role_has_hr'] = merged_data['role'].str.contains('human|hr|specialist', case=False, na=False).astype(int)
        merged_data['role_has_tech'] = merged_data['role'].str.contains('automation|analyst', case=False, na=False).astype(int)
        merged_data['role_has_supply'] = merged_data['role'].str.contains('supply|procurement', case=False, na=False).astype(int)
    
    # Feature Engineering 14: Location-based aggregations
    if is_training and target_col is not None and target_col in merged_data.columns:
        # Calculate mean salary by location (for target encoding alternative)
        for loc_col in ['country', 'state', 'city']:
            if loc_col in merged_data.columns:
                location_means = merged_data.groupby(loc_col)[target_col].mean().to_dict()
                encoders_dict[f'{loc_col}_salary_mean'] = location_means
                merged_data[f'{loc_col}_avg_salary'] = merged_data[loc_col].map(location_means).fillna(merged_data[target_col].mean())
    else:
        # Apply stored location means
        for loc_col in ['country', 'state', 'city']:
            if loc_col in merged_data.columns:
                location_means = encoders_dict.get(f'{loc_col}_salary_mean', {})
                global_mean = encoders_dict.get('global_salary_mean', 0)
                merged_data[f'{loc_col}_avg_salary'] = merged_data[loc_col].map(location_means).fillna(global_mean)
    
    if is_training and target_col is not None and target_col in merged_data.columns:
        encoders_dict['global_salary_mean'] = merged_data[target_col].mean()
    
    # Feature Engineering 15: More sophisticated interactions
    if 'role_label_encoded' in merged_data.columns and 'country_label_encoded' in merged_data.columns:
        merged_data['role_country_interaction'] = merged_data['role_label_encoded'] * merged_data['country_label_encoded']
    
    if 'role_label_encoded' in merged_data.columns and 'state_label_encoded' in merged_data.columns:
        merged_data['role_state_interaction'] = merged_data['role_label_encoded'] * merged_data['state_label_encoded']
    
    if 'country_label_encoded' in merged_data.columns and 'col_mean' in merged_data.columns:
        merged_data['country_col_mean_squared'] = merged_data['country_label_encoded'] * (merged_data['col_mean'] ** 2)
    
    # Feature Engineering 16: Individual important cost columns (top ones)
    if col_cols:
        # Use columns that are likely most important (housing, food, etc.)
        important_cols = ['col_14', 'col_15', 'col_16', 'col_17', 'col_18', 'col_19', 'col_20', 'col_30', 'col_31']
        for col in important_cols:
            if col in merged_data.columns:
                # Create normalized versions
                if is_training:
                    col_mean = merged_data[col].mean()
                    col_std = merged_data[col].std()
                    encoders_dict[f'{col}_mean'] = col_mean
                    encoders_dict[f'{col}_std'] = col_std
                    if col_std > 0:
                        merged_data[f'{col}_normalized'] = (merged_data[col] - col_mean) / col_std
                    else:
                        merged_data[f'{col}_normalized'] = 0
                else:
                    col_mean = encoders_dict.get(f'{col}_mean', merged_data[col].mean())
                    col_std = encoders_dict.get(f'{col}_std', merged_data[col].std())
                    if col_std > 0:
                        merged_data[f'{col}_normalized'] = (merged_data[col] - col_mean) / col_std
                    else:
                        merged_data[f'{col}_normalized'] = 0
    
    # Feature Engineering 17: Cross-features between all location levels
    if 'country_label_encoded' in merged_data.columns and 'state_label_encoded' in merged_data.columns:
        merged_data['country_state_interaction'] = merged_data['country_label_encoded'] * merged_data['state_label_encoded']
    
    if 'state_label_encoded' in merged_data.columns and 'city_label_encoded' in merged_data.columns:
        merged_data['state_city_interaction'] = merged_data['state_label_encoded'] * merged_data['city_label_encoded']
    
    # Feature Engineering 18: Advanced polynomial and transformation features
    if 'col_mean' in merged_data.columns:
        merged_data['col_mean_cubed'] = np.sign(merged_data['col_mean']) * (np.abs(merged_data['col_mean']) ** (1/3))
        merged_data['col_mean_exp'] = np.exp(np.clip(merged_data['col_mean'] / 1000, -10, 10))
    
    if 'col_median' in merged_data.columns and 'col_mean' in merged_data.columns:
        merged_data['col_median_mean_diff'] = merged_data['col_median'] - merged_data['col_mean']
        merged_data['col_median_mean_ratio'] = merged_data['col_median'] / (merged_data['col_mean'] + 1e-6)
    
    return merged_data, encoders_dict


# Define data processing function (simplified wrapper)
def process_data(train, cost_of_living, target_col='salary_average', is_training=True, encoders_dict=None):
    return engineer_features(train, cost_of_living, target_col=target_col, is_training=is_training, encoders_dict=encoders_dict)


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
df, encoders = process_data(train, cost_of_living, target_col='salary_average', is_training=True, encoders_dict={})
X = df.drop(['salary_average', 'ID', 'city_id'], axis=1, errors='ignore')
# Also drop original categorical columns if they exist (keep encoded versions)
cols_to_drop = ['country', 'state', 'city', 'role', 'location_combo']
for col in cols_to_drop:
    if col in X.columns:
        X = X.drop(col, axis=1)
y = df['salary_average']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Selection: Remove low-importance features
print(f"\nOriginal number of features: {X_train.shape[1]}")
# Use RandomForest to identify important features
rf_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_selector.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_selector.feature_importances_
}).sort_values('importance', ascending=False)

# Keep top 80% of features or features with importance > 0.0001
min_importance = max(0.0001, feature_importances['importance'].quantile(0.2))
important_features = feature_importances[feature_importances['importance'] >= min_importance]['feature'].tolist()

print(f"Selected {len(important_features)} important features (from {X_train.shape[1]})")
print(f"Top 10 most important features: {feature_importances.head(10)['feature'].tolist()}")

# Apply feature selection
X_train_selected = X_train[important_features].copy()
X_val_selected = X_val[important_features].copy()
X_train = X_train_selected
X_val = X_val_selected


# Define models to evaluate with improved hyperparameters
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, 
                                         min_samples_leaf=2, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=7,
                                                  min_samples_split=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, min_child_weight=3,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
    "ExtraTreesRegressor": skl.ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20,
                                                             min_samples_split=5, min_samples_leaf=2,
                                                             random_state=42, n_jobs=-1),
    "BaggingRegressor": skl.ensemble.BaggingRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "AdaBoostRegressor": skl.ensemble.AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
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

# Create an ensemble of top 3 models
print("\n\nCreating ensemble model from top performers...")
top_models = sorted(results, key=lambda x: x[1])[:3]  # Top 3 by RMSPE
print(f"Top 3 models for ensemble: {[name for name, _, _, _, _ in top_models]}")

# Create voting ensemble
ensemble_models = [(name, model) for name, _, model, _, _ in top_models]
voting_ensemble = VotingRegressor(estimators=ensemble_models)

# Train ensemble
init = datetime.now()
voting_ensemble.fit(X_train, y_train)
y_pred_ensemble = voting_ensemble.predict(X_val)
ensemble_accuracy = voting_ensemble.score(X_val, y_val)
ensemble_rmspe = rmspe(y_val, y_pred_ensemble)
final = datetime.now()
ensemble_time = (final - init).total_seconds() * 1000

print(f"Ensemble Model - Accuracy: {ensemble_accuracy}, RMSPE Score: {ensemble_rmspe}, Time Taken: {ensemble_time:.2f}ms")

# Use ensemble if it's better
if ensemble_rmspe < best_rmspe:
    print(f"\nEnsemble model is better! Using ensemble instead of {best_model_name}")
    best_model = voting_ensemble
    best_model_name = "Ensemble"
    best_rmspe = ensemble_rmspe
    best_accuracy = ensemble_accuracy
else:
    print(f"\nBest single model ({best_model_name}) is better than ensemble. Using single model.")


# Make predictions on the test set using the best model
print("\n\nProcessing test data...")

# Process test data using the encoders from training
test_df_processed, _ = engineer_features(test, cost_of_living, target_col=None, is_training=False, encoders_dict=encoders)

# Align columns between train and test
train_cols = set(X_train.columns)
test_cols = set(test_df_processed.columns)

# Keep only columns that exist in both
common_cols = sorted(list(train_cols & test_cols))
missing_cols = train_cols - test_cols

if missing_cols:
    print(f"Warning: {len(missing_cols)} columns missing in test data. Filling with 0.")
    for col in missing_cols:
        test_df_processed[col] = 0

# Select only the columns that were used for training (after feature selection)
# Make sure all important features exist in test data
missing_important = set(important_features) - set(test_df_processed.columns)
if missing_important:
    print(f"Warning: {len(missing_important)} important features missing in test data. Filling with 0.")
    for col in missing_important:
        test_df_processed[col] = 0

test_X = test_df_processed[important_features].copy()

# Ensure same column order as training
test_X = test_X[X_train.columns]

test_predictions = best_model.predict(test_X)
test['salary_average'] = test_predictions
test[['ID', 'salary_average']].to_csv('predictions.csv', index=False)
print("\n\nPredictions saved to predictions.csv")
