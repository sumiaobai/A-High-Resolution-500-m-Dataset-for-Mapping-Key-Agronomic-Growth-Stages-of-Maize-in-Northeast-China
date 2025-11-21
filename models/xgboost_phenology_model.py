import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump, load
from xgboost import XGBRegressor
from scipy.stats import uniform, randint

# ---------------- Basic Configuration ----------------
np.random.seed(42)

# TODO: Set your growth stage and file paths here
GROWTH_STAGE = ""  # TODO: Set your growth stage (e.g., 'maturity')
INPUT_DATA_PATH = ""  # TODO: Set your input Excel file path
MODEL_SAVE_PATH = ""  # TODO: Set your model save path
RESULT_SAVE_PATH = ""  # TODO: Set your result save path

# Check if paths are set
if not GROWTH_STAGE or not INPUT_DATA_PATH or not MODEL_SAVE_PATH or not RESULT_SAVE_PATH:
    print("Please set GROWTH_STAGE, INPUT_DATA_PATH, MODEL_SAVE_PATH, and RESULT_SAVE_PATH at the top of the script")
    exit(1)

# Read data
try:
    data = pd.read_excel(INPUT_DATA_PATH, sheet_name='Sheet1').dropna()
    print(f"Successfully loaded data with {len(data)} rows")
except Exception as e:
    print(f"Error loading data from {INPUT_DATA_PATH}: {e}")
    exit(1)

# Column selection (consistent with inference code)
feature_columns = data.columns[6:21]   # 15 feature columns (must match inference code)
target_column = 'light_difference'  # TODO: Confirm target column name
non_feature_columns = data.columns[:6]  # Keep for result export

# Extract arrays
X = data[feature_columns].values
y = data[target_column].values.astype(float)
nf = data[non_feature_columns].reset_index(drop=True)

# Split data (avoid data leakage)
X_train_val_orig, X_test_orig, y_train_val, y_test, nf_train_val, nf_test = train_test_split(
    X, y, nf, test_size=0.2, random_state=42
)

# No scaling: use original features directly
X_train_val = X_train_val_orig
X_test = X_test_orig

# ---------------- Hyperparameter Search Space ----------------
param_dist = {
    "n_estimators": randint(200, 1200),
    "learning_rate": uniform(0.01, 0.29),
    "max_depth": randint(3, 12),
    "min_child_weight": randint(1, 10),
    "subsample": uniform(0.5, 0.5),         # 0.5~1.0
    "colsample_bytree": uniform(0.5, 0.5),  # 0.5~1.0
    "colsample_bynode": uniform(0.5, 0.5),
    "gamma": uniform(0.0, 5.0),
    "reg_alpha": uniform(0.0, 1.0),
    "reg_lambda": uniform(0.0, 1.0),
}

# ---------------- Base Model and CV ----------------
base_model = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",   # CPU acceleration; use "gpu_hist" if GPU available
    random_state=42,
    n_jobs=1              # Avoid parallelization conflict with CV
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=100,
    scoring="neg_mean_squared_error",
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=8              # Parallelize at CV level
)

# ---------------- Perform Search ----------------
print("Starting hyperparameter search...")
random_search.fit(X_train_val, y_train_val)
print("Best parameter combination:", random_search.best_params_)

# ---------------- Optimal Model + Early Stopping ----------------
best_params = random_search.best_params_
best_model = XGBRegressor(
    **best_params,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    random_state=42,
    n_jobs=1,
    early_stopping_rounds=50  # Early stopping in constructor to avoid warnings
)

# Training (no early_stopping_rounds in fit method)
eval_set = [(X_train_val, y_train_val), (X_test, y_test)]
best_model.fit(
    X_train_val, y_train_val,
    eval_set=eval_set,
    verbose=False
)

# ---------------- Save and Load Model ----------------
model_path = MODEL_SAVE_PATH.format(stage=GROWTH_STAGE)
os.makedirs(os.path.dirname(model_path), exist_ok=True)
dump(best_model, model_path)
print(f"Model saved to: {model_path}")

# Load model to verify it works
loaded_model = load(model_path)

# ---------------- Prediction and Evaluation ----------------
y_pred_train = loaded_model.predict(X_train_val)
y_pred_test = loaded_model.predict(X_test)

def evaluate_model(y_true, y_pred, name="Training Set"):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        name: Dataset name for display
        
    Returns:
        Dictionary of evaluation metrics
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    # MAPE with epsilon to avoid division by zero
    eps = 1e-6
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))))
    
    print(f"XGBoost {name} RMSE: {rmse:.4f}")
    print(f"XGBoost {name} MAE : {mae:.4f}")
    print(f"XGBoost {name} R2  : {r2:.4f}")
    print(f"XGBoost {name} MAPE: {mape:.4f}")
    
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

# Evaluate on both training and test sets
train_metrics = evaluate_model(y_train_val, y_pred_train, "Training Set")
test_metrics = evaluate_model(y_test, y_pred_test, "Test Set")

# ---------------- Save Results (with original features and predictions) ----------------
def save_results_to_excel(orig_features, non_features, actual, predicted, feature_cols, path):
    """
    Save prediction results to Excel file with original features and predictions.
    
    Args:
        orig_features: Original feature values
        non_features: Non-feature columns (metadata)
        actual: Actual target values
        predicted: Predicted target values
        feature_cols: Feature column names
        path: Output file path
        
    Returns:
        None
    """
    df_feat = pd.DataFrame(orig_features, columns=feature_cols)
    df_feat["Actual"] = actual
    df_feat["Predicted"] = predicted
    df_feat["Residual"] = df_feat["Actual"] - df_feat["Predicted"]
    out = pd.concat([non_features.reset_index(drop=True), df_feat], axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_excel(path, index=False)
    print(f"Results saved to: {path}")

# Save training and test results
train_results_path = RESULT_SAVE_PATH.format(stage=GROWTH_STAGE, suffix="train")
test_results_path = RESULT_SAVE_PATH.format(stage=GROWTH_STAGE, suffix="test")

save_results_to_excel(X_train_val_orig, nf_train_val, y_train_val, y_pred_train, feature_columns, train_results_path)
save_results_to_excel(X_test_orig, nf_test, y_test, y_pred_test, feature_columns, test_results_path)

# ---------------- Optional: Export Feature Importance ----------------
try:
    fi = pd.Series(best_model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    fi_path = RESULT_SAVE_PATH.format(stage=GROWTH_STAGE, suffix="feature_importance")
    fi.to_excel(fi_path, header=["importance"])
    print(f"Feature importance saved to: {fi_path}")
    
    # Print top 10 most important features
    print("\nTop 10 most important features:")
    for i, (feature, importance) in enumerate(fi.head(10).items()):
        print(f"{i+1}. {feature}: {importance:.4f}")
        
except Exception as e:
    print(f"Failed to export feature importance: {e}")

print(f"\nProcessing completed for growth stage: {GROWTH_STAGE}")
