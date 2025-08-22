import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import warnings
import math
import xgboost as xgb
import shap
from fairlearn.metrics import MetricFrame, demographic_parity_ratio

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Install necessary libraries if not already installed ---
# This command installs the fairlearn library in your Colab environment.
# Uncomment the line below and run this cell if you get a ModuleNotFoundError for 'fairlearn'.
pip install fairlearn

print("--- Step 1: Data Preprocessing (Advanced and Robust) ---")

# 1. Load Dataset
df = pd.read_csv('Telco-Customer-Churn.csv')
print("\n1. Dataset Loaded. Initial 5 rows:")
print(df.head())

# Initial Data Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)
df.drop('customerID', axis=1, inplace=True)
for col in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']:
    if col in df.columns:
        df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("\nBasic data cleaning completed: TotalCharges handled, customerID dropped, service values standardized, Churn mapped.")
print(f"Number of missing values in 'TotalCharges' after handling: {df['TotalCharges'].isnull().sum()}")


# --- Advanced Exploratory Data Analysis (EDA) ---
print("\n--- Advanced EDA: Visualizing Data Distributions and Relationships ---")

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Distribution Analysis for Numerical Features (split by Churn)
plt.figure(figsize=(18, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(data=df, x=col, hue='Churn', kde=True, palette='viridis', alpha=0.7)
    plt.title(f'Distribution of {col} by Churn')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig('numerical_distributions_by_churn.png')
plt.show()
print("Saved 'numerical_distributions_by_churn.png' showing distributions of numerical features by churn status.")

# Categorical Feature Analysis (Churn Rate per Category)
categorical_cols_for_eda = df.select_dtypes(include='object').columns

num_categorical_cols = len(categorical_cols_for_eda)
num_cols_per_row = 3
num_rows = math.ceil(num_categorical_cols / num_cols_per_row)

plt.figure(figsize=(20, 5 * num_rows))
for i, col in enumerate(categorical_cols_for_eda):
    plt.subplot(num_rows, num_cols_per_row, i + 1)
    churn_rate = df.groupby(col)['Churn'].mean().reset_index()
    sns.barplot(data=churn_rate, x=col, y='Churn', palette='magma')
    plt.title(f'Churn Rate by {col}')
    plt.ylabel('Churn Rate')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('categorical_churn_rates.png')
plt.show()
print("Saved 'categorical_churn_rates.png' showing churn rates for categorical features.")

# Correlation Matrix for Numerical Features
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols + ['Churn']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features and Churn')
plt.tight_layout()
plt.savefig('numerical_correlation_matrix.png')
plt.show()
print("Saved 'numerical_correlation_matrix.png' showing correlation matrix.")


# --- Feature Engineering ---
print("\n--- Feature Engineering: Creating New Informative Features ---")

df['MonthlyCharges_x_Tenure'] = df['MonthlyCharges'] * df['tenure']
print("Created 'MonthlyCharges_x_Tenure' interaction feature.")

df['TenureGroup'] = pd.cut(df['tenure'],
                           bins=[0, 12, 24, 48, 60, 72],
                           labels=['0-12M', '13-24M', '25-48M', '49-60M', '61-72M'],
                           right=False).astype(str)
print("Created 'TenureGroup' by binning 'tenure' and ensured it's a string type.")


# --- Outlier Detection and Treatment ---
print("\n--- Outlier Treatment: Capping Outliers ---")

for col in ['MonthlyCharges', 'TotalCharges']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    print(f"Outliers for '{col}' capped using IQR method. Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")


# --- One-Hot Encoding ---
print("\n--- One-Hot Encoding: Converting Categorical to Numerical ---")

categorical_cols_after_fe = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=categorical_cols_after_fe, drop_first=True)
print("\nCategorical columns (including new engineered features) encoded using One-Hot Encoding.")
print(f"Shape of dataset after encoding: {df_encoded.shape}")

if not df_encoded.select_dtypes(include='object').empty:
    print("\nERROR: Object dtypes still found in df_encoded. Please inspect columns:")
    print(df_encoded.select_dtypes(include='object').columns)
    raise ValueError("DataFrame still contains non-numeric columns after encoding.")


# Store original df for fairness assessment later (before scaling and SMOTE)
df_for_fairness = df_encoded.copy()


# Prepare features (X) and target (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']


# --- Train-Test Split ---
print(f"\n--- Train-Test Split: Dividing Data for Training and Testing ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Shape of X_train (Training Features): {X_train.shape}")
print(f"Shape of X_test (Testing Features): {X_test.shape}")
print(f"Shape of y_train (Training Target): {y_train.shape}")
print(f"Shape of y_test (Testing Target): {y_test.shape}")
print(f"Churn distribution in y_train:\n{y_train.value_counts(normalize=True)}")
print(f"Churn distribution in y_test:\n{y_test.value_counts(normalize=True)}")


# --- Feature Scaling ---
print("\n--- Feature Scaling: Standardizing Numerical Features ---")
numerical_features_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyCharges_x_Tenure']

scaler = StandardScaler()

# Fit scaler only on training data and transform both train and test
X_train[numerical_features_to_scale] = scaler.fit_transform(X_train[numerical_features_to_scale])
X_test[numerical_features_to_scale] = scaler.transform(X_test[numerical_features_to_scale])

print("Numerical features (tenure, MonthlyCharges, TotalCharges, MonthlyCharges_x_Tenure) have been scaled.")
print("First 5 rows of X_train after scaling:")
print(X_train.head())


# --- Handling Imbalanced Data (SMOTE) ---
print("\n--- Handling Imbalanced Data: Applying SMOTE ---")
if not X_train.select_dtypes(include='object').empty:
    print("\nERROR: X_train still contains object dtypes right before SMOTE. Problematic columns:")
    print(X_train.select_dtypes(include='object').columns)
    raise ValueError("X_train contains non-numeric columns. SMOTE requires all numerical input.")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Shape of X_train before SMOTE: {X_train.shape}")
print(f"Shape of X_train after SMOTE: {X_train_resampled.shape}")
print(f"Original Churn distribution in y_train:\n{y_train.value_counts(normalize=True)}")
print(f"Churn distribution in y_train after SMOTE:\n{y_train_resampled.value_counts(normalize=True)}")
print("SMOTE applied to balance the training dataset.")

print("\n--- Advanced Data Preprocessing Complete ---")

# Re-assign X_train and y_train to the resampled versions for subsequent steps
X_train = X_train_resampled
y_train = y_train_resampled


print("\n--- Step 2: Model Implementation with Hyperparameter Tuning and Advanced Models ---")

# --- 2.1 Hyperparameter Tuning for Logistic Regression ---
print("\nPerforming GridSearchCV for Logistic Regression...")
# Define parameter grid
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
# Use GridSearchCV with 5-fold cross-validation and F1-score as metric
grid_search_lr = GridSearchCV(LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000), # increased max_iter for convergence
                              param_grid_lr, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search_lr.fit(X_train, y_train)
best_lr_model = grid_search_lr.best_estimator_
print(f"Best parameters for Logistic Regression: {grid_search_lr.best_params_}")
print(f"Best F1-score on CV for Logistic Regression: {grid_search_lr.best_score_:.4f}")

# --- 2.2 Hyperparameter Tuning for Random Forest Classifier ---
print("\nPerforming RandomizedSearchCV for Random Forest Classifier...")
# Define parameter distribution
param_dist_rf = {
    'n_estimators': [100, 200, 300, 400],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None], # None means nodes are expanded until all leaves are pure or contain less than min_samples_split samples
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
# Use RandomizedSearchCV with 5-fold cross-validation, F1-score, and 50 iterations
random_search_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                                      param_dist_rf, n_iter=50, cv=5, scoring='f1', random_state=42, n_jobs=-1, verbose=1)
random_search_rf.fit(X_train, y_train)
best_rf_model = random_search_rf.best_estimator_
print(f"Best parameters for Random Forest Classifier: {random_search_rf.best_params_}")
print(f"Best F1-score on CV for Random Forest Classifier: {random_search_rf.best_score_:.4f}")


# --- 2.3 Implementation of XGBoost Classifier ---
print("\nTraining XGBoost Classifier...")
# Initialize XGBoost Classifier with appropriate parameters for classification
# 'objective': 'binary:logistic' for binary classification
# 'eval_metric': 'logloss' for evaluation during training
# 'use_label_encoder': False is deprecated warning, so set to False
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              use_label_encoder=False, random_state=42,
                              n_estimators=200, learning_rate=0.1, max_depth=5) # Example parameters
# You might consider tuning these XGBoost parameters as well with RandomizedSearchCV for optimal performance
xgb_model.fit(X_train, y_train)
print("XGBoost Classifier Trained.")

print("\n--- Model Implementation Complete ---")


print("\n--- Step 3: Comprehensive Evaluation (Tuned Models & Advanced Metrics) ---")

# --- Predictions for Evaluation ---
y_pred_lr_tuned = best_lr_model.predict(X_test)
y_prob_lr_tuned = best_lr_model.predict_proba(X_test)[:, 1]

y_pred_rf_tuned = best_rf_model.predict(X_test)
y_prob_rf_tuned = best_rf_model.predict_proba(X_test)[:, 1]

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# --- 3.1 Evaluate Performance for Tuned Models ---
print("\n--- Evaluation for Tuned Logistic Regression ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr_tuned):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr_tuned):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr_tuned):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr_tuned):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob_lr_tuned):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_tuned))

print("\n--- Evaluation for Tuned Random Forest Classifier ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_tuned):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf_tuned):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf_tuned):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf_tuned):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob_rf_tuned):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf_tuned))

print("\n--- Evaluation for XGBoost Classifier ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob_xgb):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))


# --- 3.2 Compare Models using ROC Curve ---
print("\nGenerating ROC Curve for all models comparison...")
fpr_lr_tuned, tpr_lr_tuned, _ = roc_curve(y_test, y_prob_lr_tuned)
fpr_rf_tuned, tpr_rf_tuned, _ = roc_curve(y_test, y_prob_rf_tuned)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

plt.figure(figsize=(10, 7))
plt.plot(fpr_lr_tuned, tpr_lr_tuned, label=f'Tuned Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr_tuned):.2f})')
plt.plot(fpr_rf_tuned, tpr_rf_tuned, label=f'Tuned Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf_tuned):.2f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, y_prob_xgb):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Churn Prediction Models (Tuned & Advanced)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve_all_models.png')
plt.show()
print("ROC Curve plot for all models saved as 'roc_curve_all_models.png'.")


# --- 3.3 Model Interpretability (SHAP Values) ---
print("\n--- Model Interpretability: SHAP Values for XGBoost (Best Model Example) ---")
# Using XGBoost as an example for SHAP, as it often performs well and is complex
# You can change this to `best_rf_model` if Random Forest is performing better for you.
explainer = shap.TreeExplainer(xgb_model)
# SHAP values for the test set (or a subset for faster computation)
# Ensure X_test has column names for SHAP plotting
X_test_display = pd.DataFrame(X_test, columns=X.columns) # Convert to DataFrame if it's a numpy array after scaling

# Calculate SHAP values for the positive class (churn=1)
shap_values = explainer.shap_values(X_test)

# Summary plot (bar plot of mean absolute SHAP values)
shap.summary_plot(shap_values, X_test_display, plot_type="bar", show=False)
plt.title('SHAP Feature Importance for Churn Prediction (XGBoost)')
plt.tight_layout()
plt.savefig('shap_feature_importance_xgb.png')
plt.show()
print("SHAP feature importance plot saved as 'shap_feature_importance_xgb.png'.")

# Dependence plots (for individual feature impact) - Example for 'tenure'
# This helps understand the relationship between a feature and the model's output
# shap.dependence_plot("tenure", shap_values, X_test_display, show=False)
# plt.title('SHAP Dependence Plot for Tenure')
# plt.tight_layout()
# plt.savefig('shap_dependence_tenure.png')
# plt.show()
# print("SHAP dependence plot for 'tenure' saved as 'shap_dependence_tenure.png'.")


# --- 3.4 Fairness Assessment ---
print("\n--- Fairness Assessment using Fairlearn ---")
# Recreate the original DataFrame structure to align with X_test for sensitive features
# We need the original (or a copy before extensive processing like SMOTE on X_train) to get
# the sensitive feature columns that were one-hot encoded.
# For evaluation on X_test, we need the original sensitive features corresponding to X_test indices.

# Get original df (before train-test split and SMOTE on X_train) with encoded features
# df_for_fairness contains the encoded features for the entire dataset
# We need to extract the sensitive features from the original X_test which corresponds
# to the test split from df_encoded
sensitive_features_df_test = df_for_fairness.loc[X_test.index, ['gender_Male', 'SeniorCitizen']] # Assuming these are the encoded sensitive features

# Evaluate fairness for 'gender_Male'
if 'gender_Male' in sensitive_features_df_test.columns:
    print("\nFairness evaluation for 'gender_Male':")
    # Accuracy by group
    mf_accuracy_gender = MetricFrame(metrics=accuracy_score,
                                     y_true=y_test,
                                     y_pred=y_pred_xgb, # Using XGBoost predictions
                                     sensitive_features=sensitive_features_df_test['gender_Male'])
    print("Accuracy by Gender (0=Female, 1=Male):\n", mf_accuracy_gender.by_group)

    # Demographic Parity Ratio (ratio of positive prediction rate for favored group to un-favored group)
    # Ideally close to 1
    demographic_parity_gender = demographic_parity_ratio(
        y_true=y_test,
        y_pred=y_pred_xgb, # Using XGBoost predictions
        sensitive_features=sensitive_features_df_test['gender_Male']
    ) # Removed control_features=None
    print(f"Demographic Parity Ratio (Gender): {demographic_parity_gender:.4f} (Closer to 1 is fairer)")

    # Equal Opportunity (Recall by group) - measures if true positive rate is equal across groups
    mf_recall_gender = MetricFrame(metrics=recall_score,
                                   y_true=y_test,
                                   y_pred=y_pred_xgb,
                                   sensitive_features=sensitive_features_df_test['gender_Male'])
    print("Recall by Gender:\n", mf_recall_gender.by_group)

# Evaluate fairness for 'SeniorCitizen'
if 'SeniorCitizen' in sensitive_features_df_test.columns:
    print("\nFairness evaluation for 'SeniorCitizen':")
    # Accuracy by group
    mf_accuracy_senior = MetricFrame(metrics=accuracy_score,
                                     y_true=y_test,
                                     y_pred=y_pred_xgb, # Using XGBoost predictions
                                     sensitive_features=sensitive_features_df_test['SeniorCitizen'])
    print("Accuracy by SeniorCitizen (0=No, 1=Yes):\n", mf_accuracy_senior.by_group)

    # Demographic Parity Ratio
    demographic_parity_senior = demographic_parity_ratio(
        y_true=y_test,
        y_pred=y_pred_xgb,
        sensitive_features=sensitive_features_df_test['SeniorCitizen']
    ) # Removed control_features=None
    print(f"Demographic Parity Ratio (SeniorCitizen): {demographic_parity_senior:.4f} (Closer to 1 is fairer)")

    # Equal Opportunity (Recall by group)
    mf_recall_senior = MetricFrame(metrics=recall_score,
                                   y_true=y_test,
                                   y_pred=y_pred_xgb,
                                   sensitive_features=sensitive_features_df_test['SeniorCitizen'])
    print("Recall by SeniorCitizen:\n", mf_recall_senior.by_group)


print("\n--- Comprehensive Evaluation Complete ---")
