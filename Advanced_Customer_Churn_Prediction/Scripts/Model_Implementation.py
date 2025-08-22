import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
import joblib
import os # Ensure os is imported for path operations

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Create a directory for artifacts if it doesn't exist ---
output_dir = 'artifacts'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory '{output_dir}' ensured.")

print("--- Step 2: Model Implementation with Hyperparameter Tuning and Advanced Models ---")

# --- Load processed data from data_preprocessing.py ---
try:
    X_train = joblib.load(os.path.join(output_dir, 'X_train_resampled.pkl'))
    y_train = joblib.load(os.path.join(output_dir, 'y_train_resampled.pkl'))
    X_test = joblib.load(os.path.join(output_dir, 'X_test.pkl'))
    y_test = joblib.load(os.path.join(output_dir, 'y_test.pkl'))
    print("\nProcessed data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Required data files not found in '{output_dir}'. Please run data_preprocessing.py first.")
    exit()

# --- 2.1 Hyperparameter Tuning for Logistic Regression ---
print("\nPerforming GridSearchCV for Logistic Regression...")
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
# FIX: Changed n_jobs=-1 to n_jobs=1 to avoid multiprocessing issues on Windows
grid_search_lr = GridSearchCV(LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
                              param_grid_lr, cv=5, scoring='f1', n_jobs=1, verbose=1) # Changed n_jobs to 1
grid_search_lr.fit(X_train, y_train)
best_lr_model = grid_search_lr.best_estimator_
print(f"Best parameters for Logistic Regression: {grid_search_lr.best_params_}")
print(f"Best F1-score on CV for Logistic Regression: {grid_search_lr.best_score_:.4f}")

# --- 2.2 Hyperparameter Tuning for Random Forest Classifier ---
print("\nPerforming RandomizedSearchCV for Random Forest Classifier...")
param_dist_rf = {
    'n_estimators': [100, 200, 300, 400],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
# FIX: Changed n_jobs=-1 to n_jobs=1 to avoid multiprocessing issues on Windows
random_search_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                                      param_dist_rf, n_iter=50, cv=5, scoring='f1', random_state=42, n_jobs=1, verbose=1) # Changed n_jobs to 1
random_search_rf.fit(X_train, y_train)
best_rf_model = random_search_rf.best_estimator_
print(f"Best parameters for Random Forest Classifier: {random_search_rf.best_params_}")
print(f"Best F1-score on CV for Random Forest Classifier: {random_search_rf.best_score_:.4f}")


# --- 2.3 Implementation of XGBoost Classifier ---
print("\nTraining XGBoost Classifier...")
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              use_label_encoder=False, random_state=42,
                              n_estimators=200, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train, y_train)
print("XGBoost Classifier Trained.")

print("\n--- Model Implementation Complete ---")

# --- Save trained models for the next step ---
joblib.dump(best_lr_model, os.path.join(output_dir, 'best_lr_model.pkl'))
joblib.dump(best_rf_model, os.path.join(output_dir, 'best_rf_model.pkl'))
joblib.dump(xgb_model, os.path.join(output_dir, 'xgb_model.pkl'))

print(f"\nTuned models saved to '{output_dir}'.")
