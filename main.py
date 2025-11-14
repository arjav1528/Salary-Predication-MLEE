# Advanced Salary Prediction Model - Optimized for RMSPE
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from datetime import datetime
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


# RMSPE metric
def rmspe(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero):
        return float('inf')
    pct_errors = (y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]
    return np.sqrt(np.mean(pct_errors ** 2))


# Advanced feature engineering
def create_features(df, cost_of_living, target=None, encoders=None):
    if encoders is None:
        encoders = {}
    
    # Normalize strings
    df = df.copy()
    for col in ['city', 'state', 'country']:
        df[col] = df[col].str.lower().str.strip()
    
    cost_of_living = cost_of_living.copy()
    for col in ['city', 'state', 'country']:
        cost_of_living[col] = cost_of_living[col].str.lower().str.strip()
    
    # Merge
    data = pd.merge(df, cost_of_living, on=['city', 'state', 'country'], how='left')
    
    # Get all cost columns
    cost_cols = [c for c in data.columns if c.startswith('col_')]
    
    # Fill missing cost values with median
    for col in cost_cols:
        if col in data.columns:
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val if pd.notna(median_val) else 0)
    
    # Statistical aggregations
    if cost_cols:
        data['cost_mean'] = data[cost_cols].mean(axis=1)
        data['cost_median'] = data[cost_cols].median(axis=1)
        data['cost_std'] = data[cost_cols].std(axis=1).fillna(0)
        data['cost_max'] = data[cost_cols].max(axis=1)
        data['cost_min'] = data[cost_cols].min(axis=1)
        data['cost_range'] = data['cost_max'] - data['cost_min']
        data['cost_q25'] = data[cost_cols].quantile(0.25, axis=1)
        data['cost_q75'] = data[cost_cols].quantile(0.75, axis=1)
        data['cost_iqr'] = data['cost_q75'] - data['cost_q25']
        data['cost_skew'] = data[cost_cols].skew(axis=1).fillna(0)
        data['cost_non_null'] = data[cost_cols].notna().sum(axis=1)
    
    # Key cost columns (important ones)
    key_cols = ['col_14', 'col_15', 'col_16', 'col_17', 'col_18', 'col_19', 'col_20', 'col_30', 'col_31']
    available_key = [c for c in key_cols if c in data.columns]
    if available_key:
        data['key_cost_mean'] = data[available_key].mean(axis=1)
        data['key_cost_std'] = data[available_key].std(axis=1).fillna(0)
        data['key_cost_sum'] = data[available_key].sum(axis=1)
    
    # Use top individual cost columns (normalized)
    top_cost_cols = cost_cols[:15]  # Use first 15 cost columns
    for col in top_cost_cols:
        if col in data.columns:
            col_name = col.replace('col_', 'cost_')
            if f'{col_name}_norm' not in encoders:
                mean_val = data[col].mean()
                std_val = data[col].std()
                encoders[f'{col_name}_norm'] = (mean_val, std_val)
                if std_val > 0:
                    data[f'{col_name}_norm'] = (data[col] - mean_val) / std_val
                else:
                    data[f'{col_name}_norm'] = 0
            else:
                mean_val, std_val = encoders[f'{col_name}_norm']
                if std_val > 0:
                    data[f'{col_name}_norm'] = (data[col] - mean_val) / std_val
                else:
                    data[f'{col_name}_norm'] = 0
    
    # Location features
    data['is_us'] = (data['country'] == 'united states of america').astype(int)
    
    # Role features
    data['is_analyst'] = data['role'].str.contains('analyst', case=False, na=False).astype(int)
    data['is_specialist'] = data['role'].str.contains('specialist', case=False, na=False).astype(int)
    data['is_finance'] = data['role'].str.contains('finance|treasury|budget', case=False, na=False).astype(int)
    data['role_length'] = data['role'].str.len()
    
    # Encode categoricals
    cat_cols = ['country', 'state', 'city', 'role']
    
    # Label encoding
    for col in cat_cols:
        if col in data.columns:
            key = f'{col}_le'
            if key not in encoders:
                le = LabelEncoder()
                data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
                encoders[key] = le
            else:
                le = encoders[key]
                unique_vals = set(le.classes_)
                data[f'{col}_encoded'] = data[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in unique_vals else -1
                )
    
    # Target encoding
    if target is not None and target in data.columns:
        valid_mask = data[target].notna()
        if valid_mask.sum() > 0:
            for col in cat_cols:
                if col in data.columns:
                    key = f'{col}_te'
                    if key not in encoders:
                        te = TargetEncoder(smooth="auto")
                        data.loc[valid_mask, f'{col}_target'] = te.fit_transform(
                            data.loc[valid_mask, [col]], data.loc[valid_mask, target]
                        )
                        data[f'{col}_target'] = data[f'{col}_target'].fillna(data[target].mean())
                        encoders[key] = te
                    else:
                        te = encoders[key]
                        data[f'{col}_target'] = te.transform(data[[col]])
    
    # Location-based salary means (CRITICAL FEATURE)
    if target is not None and target in data.columns:
        valid_target = data[target].dropna()
        if len(valid_target) > 0:
            for loc in ['country', 'state', 'city']:
                if loc in data.columns:
                    key = f'{loc}_mean'
                    if key not in encoders:
                        means = data.groupby(loc)[target].mean().to_dict()
                        encoders[key] = means
                        data[f'{loc}_salary_mean'] = data[loc].map(means)
                    else:
                        means = encoders[key]
                        global_mean = encoders.get('global_mean', valid_target.mean())
                        data[f'{loc}_salary_mean'] = data[loc].map(means).fillna(global_mean)
            
            if 'global_mean' not in encoders:
                encoders['global_mean'] = valid_target.mean()
    
    # Advanced interactions
    if 'role_encoded' in data.columns:
        if 'cost_mean' in data.columns:
            data['role_cost_mean'] = data['role_encoded'] * data['cost_mean']
            data['role_cost_median'] = data['role_encoded'] * data['cost_median']
        if 'country_encoded' in data.columns:
            data['role_country'] = data['role_encoded'] * data['country_encoded']
        if 'state_encoded' in data.columns:
            data['role_state'] = data['role_encoded'] * data['state_encoded']
    
    if 'country_encoded' in data.columns:
        if 'cost_mean' in data.columns:
            data['country_cost_mean'] = data['country_encoded'] * data['cost_mean']
            data['country_cost_median'] = data['country_encoded'] * data['cost_median']
        if 'state_encoded' in data.columns:
            data['country_state'] = data['country_encoded'] * data['state_encoded']
    
    if 'state_encoded' in data.columns and 'cost_mean' in data.columns:
        data['state_cost_mean'] = data['state_encoded'] * data['cost_mean']
    
    # Polynomial features
    if 'cost_mean' in data.columns:
        data['cost_mean_sq'] = data['cost_mean'] ** 2
        data['cost_mean_sqrt'] = np.sqrt(np.abs(data['cost_mean']))
        data['cost_mean_log'] = np.log1p(np.abs(data['cost_mean']))
        data['cost_mean_cubed'] = np.sign(data['cost_mean']) * (np.abs(data['cost_mean']) ** (1/3))
    
    if 'cost_median' in data.columns:
        data['cost_median_sq'] = data['cost_median'] ** 2
        data['cost_median_log'] = np.log1p(np.abs(data['cost_median']))
    
    # Ratio features
    if 'cost_max' in data.columns and 'cost_min' in data.columns:
        data['cost_max_min_ratio'] = data['cost_max'] / (data['cost_min'] + 1e-6)
    
    if 'cost_mean' in data.columns and 'cost_std' in data.columns:
        data['cost_cv'] = data['cost_std'] / (data['cost_mean'] + 1e-6)
    
    if 'cost_median' in data.columns and 'cost_mean' in data.columns:
        data['cost_median_mean_ratio'] = data['cost_median'] / (data['cost_mean'] + 1e-6)
        data['cost_median_mean_diff'] = data['cost_median'] - data['cost_mean']
    
    return data, encoders


# Model training with RMSPE
def train_model(name, model, X_train, y_train, X_val, y_val):
    start = datetime.now()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = model.score(X_val, y_val)
    rmspe_score = rmspe(y_val, y_pred)
    time_ms = (datetime.now() - start).total_seconds() * 1000
    print(f"{name:20s} - Acc: {accuracy:.6f}, RMSPE: {rmspe_score:.6f}, Time: {time_ms:.0f}ms")
    return name, rmspe_score, model, accuracy, time_ms


# Load data
print("="*70)
print("ADVANCED SALARY PREDICTION MODEL - OPTIMIZED FOR RMSPE")
print("="*70)
print("\nLoading data...")
cost_of_living = pd.read_csv('cost_of_living.csv', na_values='NA')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Create features
print("Creating advanced features...")
train_data, encoders = create_features(train, cost_of_living, target='salary_average', encoders={})

# Prepare features
X = train_data.drop(['salary_average', 'ID', 'city_id'], axis=1, errors='ignore')
for col in ['country', 'state', 'city', 'role']:
    if col in X.columns:
        X = X.drop(col, axis=1)

y = train_data['salary_average']

# Clean data
valid_mask = y.notna() & np.isfinite(y)
X = X[valid_mask].copy()
y = y[valid_mask].copy()
X = X.fillna(0)
for col in X.columns:
    if X[col].dtype in [np.float64, np.float32]:
        X[col] = X[col].replace([np.inf, -np.inf], 0)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Validation: {X_val.shape[0]} samples\n")

# Feature selection - keep top features using multiple models
print("Selecting important features...")
rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_selector = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)

rf_selector.fit(X_train, y_train)
xgb_selector.fit(X_train, y_train)

# Combine importances from both models
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'rf_importance': rf_selector.feature_importances_,
    'xgb_importance': xgb_selector.feature_importances_
})
feature_importance['combined_importance'] = (
    feature_importance['rf_importance'] * 0.5 + 
    feature_importance['xgb_importance'] * 0.5
)
feature_importance = feature_importance.sort_values('combined_importance', ascending=False)

# Keep top 90% of features (less aggressive selection)
min_importance = feature_importance['combined_importance'].quantile(0.10)
important_features = feature_importance[feature_importance['combined_importance'] >= min_importance]['feature'].tolist()

print(f"Selected {len(important_features)} features from {X_train.shape[1]}")
print(f"Top 5 features: {feature_importance.head(5)['feature'].tolist()}")

X_train = X_train[important_features].copy()
X_val = X_val[important_features].copy()

# Optimized models with better hyperparameters
print("\nTraining optimized models...")
models = {
    "XGBoost": XGBRegressor(
        n_estimators=600, learning_rate=0.025, max_depth=9,
        min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
        gamma=0.15, reg_alpha=0.15, reg_lambda=1.5,
        random_state=42, n_jobs=-1, tree_method='hist'
    ),
    "XGBoost2": XGBRegressor(
        n_estimators=700, learning_rate=0.02, max_depth=8,
        min_child_weight=6, subsample=0.88, colsample_bytree=0.88,
        gamma=0.2, reg_alpha=0.2, reg_lambda=2.0,
        random_state=123, n_jobs=-1, tree_method='hist'
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.025, max_depth=9,
        min_samples_split=3, min_samples_leaf=1,
        subsample=0.9, max_features='sqrt',
        random_state=42
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=600, max_depth=28, min_samples_split=3,
        min_samples_leaf=1, max_features='sqrt',
        random_state=42, n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=600, max_depth=28, min_samples_split=3,
        min_samples_leaf=1, max_features='sqrt',
        random_state=42, n_jobs=-1
    ),
}

# Train all models
results = Parallel(n_jobs=-1)(
    delayed(train_model)(name, model, X_train, y_train, X_val, y_val)
    for name, model in models.items()
)

# Find best
best_name, best_rmspe, best_model, best_acc, best_time = min(results, key=lambda x: x[1])
print(f"\n{'='*70}")
print(f"Best Single Model: {best_name}")
print(f"  Accuracy: {best_acc:.6f}")
print(f"  RMSPE:    {best_rmspe:.6f}")
print(f"{'='*70}")

# Create stacking ensemble (more powerful than voting)
print("\nCreating STACKING ensemble...")
top_4 = sorted(results, key=lambda x: x[1])[:4]  # Use top 4 models
base_models = [(name, model) for name, _, model, _, _ in top_4]

# Meta-learner with regularization
meta_learner = Ridge(alpha=0.5)

# Stacking ensemble with more CV folds
stacking_ensemble = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=7,  # More folds for better generalization
    n_jobs=-1,
    passthrough=True  # Include original features
)

start = datetime.now()
stacking_ensemble.fit(X_train, y_train)
y_pred_stack = stacking_ensemble.predict(X_val)
stack_acc = stacking_ensemble.score(X_val, y_val)
stack_rmspe = rmspe(y_val, y_pred_stack)
stack_time = (datetime.now() - start).total_seconds() * 1000

print(f"Stacking Ensemble:")
print(f"  Accuracy: {stack_acc:.6f}")
print(f"  RMSPE:    {stack_rmspe:.6f}")
print(f"  Time:     {stack_time:.0f}ms")

# Also try voting ensemble
voting_ensemble = VotingRegressor(estimators=base_models)
voting_ensemble.fit(X_train, y_train)
y_pred_vote = voting_ensemble.predict(X_val)
vote_acc = voting_ensemble.score(X_val, y_val)
vote_rmspe = rmspe(y_val, y_pred_vote)

print(f"\nVoting Ensemble:")
print(f"  Accuracy: {vote_acc:.6f}")
print(f"  RMSPE:    {vote_rmspe:.6f}")

# Choose best - prefer stacking for better generalization
all_results = [
    (best_name, best_rmspe, best_model),
    ("Stacking", stack_rmspe, stacking_ensemble),
    ("Voting", vote_rmspe, voting_ensemble)
]

# Prefer stacking if it's close (within 5% of best), as it generalizes better
best_single_rmspe = best_rmspe
if stack_rmspe <= best_single_rmspe * 1.05:  # Within 5% of best
    final_name, final_rmspe, final_model = "Stacking", stack_rmspe, stacking_ensemble
    print(f"\n✓ Using Stacking Ensemble (better generalization)")
else:
    final_name, final_rmspe, final_model = min(all_results, key=lambda x: x[1])
    print(f"\n✓ Using {final_name} (best performance)")

print(f"\n{'='*70}")
print(f"✓ FINAL MODEL: {final_name}")
print(f"✓ Validation RMSPE: {final_rmspe:.6f}")
print(f"{'='*70}")

# Predict on test
print("\nProcessing test data...")
test_data, _ = create_features(test, cost_of_living, target=None, encoders=encoders)

test_X = test_data.drop(['ID', 'city_id'], axis=1, errors='ignore')
for col in ['country', 'state', 'city', 'role']:
    if col in test_X.columns:
        test_X = test_X.drop(col, axis=1)

# Align features
missing = set(important_features) - set(test_X.columns)
if missing:
    print(f"Filling {len(missing)} missing features with 0")
    for col in missing:
        test_X[col] = 0

test_X = test_X[important_features].copy()
test_X = test_X[X_train.columns]

# Predict
print("Making predictions...")
predictions = final_model.predict(test_X)

# Post-process: ensure no negative or extreme values
# Use more conservative bounds based on training data
min_salary = y.min() * 0.5
max_salary = y.max() * 1.5
predictions = np.clip(predictions, min_salary, max_salary)

test['salary_average'] = predictions
test[['ID', 'salary_average']].to_csv('predictions.csv', index=False)

print(f"\n{'='*70}")
print(f"✓ Predictions saved to predictions.csv")
print(f"✓ Final model: {final_name}")
print(f"✓ Validation RMSPE: {final_rmspe:.6f}")
print(f"{'='*70}")
