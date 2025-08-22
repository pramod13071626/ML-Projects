import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import warnings
import joblib
import shap
from fairlearn.metrics import MetricFrame, demographic_parity_ratio

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Create a directory for artifacts if it doesn't exist ---
import os
output_dir = 'artifacts'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory '{output_dir}' ensured.")

print("\n--- Step 3: Comprehensive Evaluation (Tuned Models & Advanced Metrics) ---")

# --- Load data and models from previous steps ---
try:
    X_test = joblib.load(os.path.join(output_dir, 'X_test.pkl'))
    y_test = joblib.load(os.path.join(output_dir, 'y_test.pkl'))
    best_lr_model = joblib.load(os.path.join(output_dir, 'best_lr_model.pkl'))
    best_rf_model = joblib.load(os.path.join(output_dir, 'best_rf_model.pkl'))
    xgb_model = joblib.load(os.path.join(output_dir, 'xgb_model.pkl'))
    X_columns = joblib.load(os.path.join(output_dir, 'X_columns.pkl')) # Needed for SHAP plot column names
    df_for_fairness = joblib.load(os.path.join(output_dir, 'df_for_fairness.pkl')) # Needed for sensitive features

    # Re-assemble the test set data for plotting purposes (including original numerical values)
    # This requires merging X_test (scaled) with the non-scaled original features for plotting.
    # We will use df_for_fairness and y_test to get the original unscaled values.
    # It's crucial that X_test.index corresponds to df_for_fairness.index
    df_test_original_features = df_for_fairness.loc[X_test.index, ['tenure', 'MonthlyCharges', 'Churn']].copy()
    # The 'Churn' column is already numeric (0 or 1) from data_preprocessing.py
    df_test_original_features['Churn_Label'] = df_test_original_features['Churn'].map({0: 'No Churn', 1: 'Churn'})

    print("\nTest data, models, column names, and fairness dataframe loaded successfully.")
except FileNotFoundError:
    print(f"Error: Required files not found in '{output_dir}'. Please run data_preprocessing.py and model_implementation.py first.")
    exit()

# --- Predictions for Evaluation ---
y_pred_lr_tuned = best_lr_model.predict(X_test)
y_prob_lr_tuned = best_lr_model.predict_proba(X_test)[:, 1]

y_pred_rf_tuned = best_rf_model.predict(X_test)
y_prob_rf_tuned = best_rf_model.predict_proba(X_test)[:, 1]

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# --- 3.1 Evaluate Performance for Tuned Models ---
print("\n--- Evaluation for Tuned Logistic Regression ---")
lr_accuracy = accuracy_score(y_test, y_pred_lr_tuned)
lr_precision = precision_score(y_test, y_pred_lr_tuned)
lr_recall = recall_score(y_test, y_pred_lr_tuned)
lr_f1 = f1_score(y_test, y_pred_lr_tuned)
lr_roc_auc = roc_auc_score(y_test, y_prob_lr_tuned)
print(f"Accuracy: {lr_accuracy * 100:.2f}%")
print(f"Precision: {lr_precision * 100:.2f}%")
print(f"Recall: {lr_recall * 100:.2f}%")
print(f"F1-Score: {lr_f1 * 100:.2f}%")
print(f"ROC AUC Score: {lr_roc_auc * 100:.2f}%")
cm_lr = confusion_matrix(y_test, y_pred_lr_tuned)
print("Confusion Matrix:\n", cm_lr)

print("\n--- Evaluation for Tuned Random Forest Classifier ---")
rf_accuracy = accuracy_score(y_test, y_pred_rf_tuned)
rf_precision = precision_score(y_test, y_pred_rf_tuned)
rf_recall = recall_score(y_test, y_pred_rf_tuned)
rf_f1 = f1_score(y_test, y_pred_rf_tuned)
rf_roc_auc = roc_auc_score(y_test, y_prob_rf_tuned)
print(f"Accuracy: {rf_accuracy * 100:.2f}%")
print(f"Precision: {rf_precision * 100:.2f}%")
print(f"Recall: {rf_recall * 100:.2f}%")
print(f"F1-Score: {rf_f1 * 100:.2f}%")
print(f"ROC AUC Score: {rf_roc_auc * 100:.2f}%")
cm_rf = confusion_matrix(y_test, y_pred_rf_tuned)
print("Confusion Matrix:\n", cm_rf)

print("\n--- Evaluation for XGBoost Classifier ---")
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_roc_auc = roc_auc_score(y_test, y_prob_xgb)
print(f"Accuracy: {xgb_accuracy * 100:.2f}%")
print(f"Precision: {xgb_precision * 100:.2f}%")
print(f"Recall: {xgb_recall * 100:.2f}%")
print(f"F1-Score: {xgb_f1 * 100:.2f}%")
print(f"ROC AUC Score: {xgb_roc_auc * 100:.2f}%")
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print("Confusion Matrix:\n", cm_xgb)


# --- Consolidated Evaluation Metrics Table ---
print("\n--- Consolidated Model Evaluation Metrics ---")
metrics_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [lr_accuracy * 100, rf_accuracy * 100, xgb_accuracy * 100],
    'Precision': [lr_precision * 100, rf_precision * 100, xgb_precision * 100],
    'Recall': [lr_recall * 100, rf_recall * 100, xgb_recall * 100],
    'F1-Score': [lr_f1 * 100, rf_f1 * 100, xgb_f1 * 100],
    'ROC AUC': [lr_roc_auc * 100, rf_roc_auc * 100, xgb_roc_auc * 100]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.iloc[:, 1:] = metrics_df.iloc[:, 1:].round(2)
print(metrics_df.to_string(index=False))
print("\nMetrics table generated.")


# --- Visualizing All Metrics ---
print("\n--- Visualizing All Evaluation Metrics ---")
metrics_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(14, 10))
ax = sns.barplot(data=metrics_melted, x='Metric', y='Score', hue='Model', palette='viridis', zorder=2)
plt.title('Comparison of Model Performance Across Metrics (%)')
plt.ylabel('Score (%)')
plt.xlabel('Metric')
plt.ylim(0, 100)
plt.legend(title='Model')
plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%', label_type='edge', padding=3, fontsize=9, color='black', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_performance_comparison_percentage.png'))
plt.show()
print(f"Model performance comparison plot saved as '{output_dir}/model_performance_comparison_percentage.png'.")


# --- 3.2 Compare Models using ROC Curve ---
print("\nGenerating ROC Curve for all models comparison...")
fpr_lr_tuned, tpr_lr_tuned, _ = roc_curve(y_test, y_prob_lr_tuned)
fpr_rf_tuned, tpr_rf_tuned, _ = roc_curve(y_test, y_prob_rf_tuned)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

plt.figure(figsize=(10, 7))
plt.plot(fpr_lr_tuned, tpr_lr_tuned, label=f'Tuned Logistic Regression (AUC = {lr_roc_auc * 100:.2f}%)')
plt.plot(fpr_rf_tuned, tpr_rf_tuned, label=f'Tuned Random Forest (AUC = {rf_roc_auc * 100:.2f}%)')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_roc_auc * 100:.2f}%)')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Churn Prediction Models (Tuned & Advanced)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve_all_models.png'))
plt.show()
print(f"ROC Curve plot for all models saved as '{output_dir}/roc_curve_all_models.png'.")


# --- NEW: Consolidated Confusion Matrix Table ---
print("\n--- Consolidated Confusion Matrix Table ---")

# Extract individual components of confusion matrices
tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
tn_xgb, fp_xgb, fn_xgb, tp_xgb = cm_xgb.ravel()

confusion_matrix_data = {
    'Algorithm': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'True Positive (Class 1)': [tp_lr, tp_rf, tp_xgb],
    'False Negative (Class 1)': [fn_lr, fn_rf, fn_xgb],
    'False Positive (Class 0)': [fp_lr, fp_rf, fp_xgb],
    'True Negative (Class 0)': [tn_lr, tn_rf, tn_xgb]
}
cm_df = pd.DataFrame(confusion_matrix_data)
print(cm_df.to_string(index=False))
print("\nConsolidated Confusion Matrix Table generated.")


# --- NEW: Precision-Recall (PR) Curves ---
print("\n--- Generating Precision-Recall Curves ---")
plt.figure(figsize=(10, 7))

# Logistic Regression
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_prob_lr_tuned)
ap_lr = average_precision_score(y_test, y_prob_lr_tuned)
plt.plot(recall_lr, precision_lr, label=f'Tuned Logistic Regression (AP = {ap_lr * 100:.2f}%)') # Multiplied by 100 and added %

# Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf_tuned)
ap_rf = average_precision_score(y_test, y_prob_rf_tuned)
plt.plot(recall_rf, precision_rf, label=f'Tuned Random Forest (AP = {ap_rf * 100:.2f}%)') # Multiplied by 100 and added %

# XGBoost
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
ap_xgb = average_precision_score(y_test, y_prob_xgb)
plt.plot(recall_xgb, precision_xgb, label=f'XGBoost (AP = {ap_xgb * 100:.2f}%)') # Multiplied by 100 and added %

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Churn Prediction Models')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'))
plt.show()
print(f"Precision-Recall Curves plot saved as '{output_dir}/precision_recall_curves.png'.")


# --- NEW: Probability Calibration Plots (Reliability Diagrams) ---
print("\n--- Generating Probability Calibration Plots (Reliability Diagrams) ---")
plt.figure(figsize=(10, 7))
plt.plot([0, 100], [0, 100], 'k:', label='Perfectly calibrated') # Diagonal line for perfect calibration (0-100 range)

# Logistic Regression
fraction_of_positives_lr, mean_predicted_value_lr = calibration_curve(y_test, y_prob_lr_tuned, n_bins=10)
plt.plot(mean_predicted_value_lr * 100, fraction_of_positives_lr * 100, "s-", # Multiplied by 100
         label=f"Logistic Regression (Avg. Prob = {np.mean(y_prob_lr_tuned) * 100:.2f}%)") # Multiplied by 100 and added %

# Random Forest
fraction_of_positives_rf, mean_predicted_value_rf = calibration_curve(y_test, y_prob_rf_tuned, n_bins=10)
plt.plot(mean_predicted_value_rf * 100, fraction_of_positives_rf * 100, "s-", # Multiplied by 100
         label=f"Random Forest (Avg. Prob = {np.mean(y_prob_rf_tuned) * 100:.2f}%)") # Multiplied by 100 and added %

# XGBoost
fraction_of_positives_xgb, mean_predicted_value_xgb = calibration_curve(y_test, y_prob_xgb, n_bins=10)
plt.plot(mean_predicted_value_xgb * 100, fraction_of_positives_xgb * 100, "s-", # Multiplied by 100
         label=f"XGBoost (Avg. Prob = {np.mean(y_prob_xgb) * 100:.2f}%)") # Multiplied by 100 and added %

plt.xlabel('Mean predicted value (%)') # Updated label
plt.ylabel('Fraction of positives (%)') # Updated label
plt.title('Probability Calibration Curves (Reliability Diagram)')
plt.legend(loc="lower right")
plt.grid(True)
plt.xlim(0, 100) # Set x-axis limit to 0-100
plt.ylim(0, 100) # Set y-axis limit to 0-100
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'calibration_curves.png'))
plt.show()
print(f"Probability Calibration Plots saved as '{output_dir}/calibration_curves.png'.")


# --- NEW: Churn Rate Heatmap by Contract and Internet Service ---
print("\n--- Visualizing Churn Rate by Contract & Internet Service ---")

# Ensure 'Contract' and 'InternetService' are in df_for_fairness and are the correct types
# Also ensure 'Churn' is available, which it is in df_for_fairness
# Check for relevant one-hot encoded columns from data_preprocessing.py
required_cols_contract = ['Contract_One year', 'Contract_Two year']
required_cols_internet = ['InternetService_Fiber optic', 'InternetService_No']

if all(col in df_for_fairness.columns for col in required_cols_contract + required_cols_internet):

    temp_df = df_for_fairness.copy()
    temp_df['Contract_Original'] = 'Month-to-month'
    temp_df.loc[temp_df['Contract_One year'] == 1, 'Contract_Original'] = 'One year'
    temp_df.loc[temp_df['Contract_Two year'] == 1, 'Contract_Original'] = 'Two year'

    temp_df['InternetService_Original'] = 'DSL' # Default if neither Fiber Optic nor No Internet is 1
    temp_df.loc[temp_df['InternetService_Fiber optic'] == 1, 'InternetService_Original'] = 'Fiber optic'
    temp_df.loc[temp_df['InternetService_No'] == 1, 'InternetService_Original'] = 'No'

    # Calculate churn rate for each combination
    churn_rate_pivot = temp_df.pivot_table(index='Contract_Original',
                                           columns='InternetService_Original',
                                           values='Churn',
                                           aggfunc='mean')

    plt.figure(figsize=(10, 7))
    sns.heatmap(churn_rate_pivot * 100, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Churn Rate (%)'})
    plt.title('Churn Rate (%) by Contract Type and Internet Service')
    plt.xlabel('Internet Service')
    plt.ylabel('Contract Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'churn_rate_contract_internet_heatmap.png'))
    plt.show()
    print(f"Churn Rate Heatmap by Contract & Internet Service saved as '{output_dir}/churn_rate_contract_internet_heatmap.png'.")
else:
    print("\nSkipping Churn Rate Heatmap: Required one-hot encoded columns for Contract or InternetService not found in df_for_fairness. Please check data_preprocessing.py output.")


# --- NEW: Scatter Plot - Tenure vs Monthly Charges colored by Churn with Density Contours ---
print("\n--- Visualizing Customer Segments: Tenure vs Monthly Charges with Density Contours ---")

plt.figure(figsize=(12, 8))
# Scatter plot for individual points
sns.scatterplot(
    data=df_test_original_features,
    x='tenure',
    y='MonthlyCharges',
    hue='Churn_Label',
    palette={'No Churn': 'blue', 'Churn': 'red'},
    alpha=0.6,
    s=50,
    edgecolor='w',
    linewidth=0.5,
    zorder=2 # Ensure scatter points are above contours
)

# Add KDE contours for 'No Churn'
sns.kdeplot(
    data=df_test_original_features[df_test_original_features['Churn_Label'] == 'No Churn'],
    x='tenure',
    y='MonthlyCharges',
    color='blue',
    levels=3, # Number of contour lines
    linewidths=1.5,
    alpha=0.5,
    zorder=1 # Ensure contours are below scatter points
)

# Add KDE contours for 'Churn'
sns.kdeplot(
    data=df_test_original_features[df_test_original_features['Churn_Label'] == 'Churn'],
    x='tenure',
    y='MonthlyCharges',
    color='red',
    levels=3,
    linewidths=1.5,
    alpha=0.5,
    zorder=1
)

plt.title('Customer Churn Behavior by Tenure and Monthly Charges with Density Contours')
plt.xlabel('Tenure (Months)')
plt.ylabel('Monthly Charges ($)')
plt.legend(title='Churn Status')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'tenure_monthlycharges_scatter_density.png'))
plt.show()
print(f"Scatter plot of Tenure vs Monthly Charges with Density Contours saved as '{output_dir}/tenure_monthlycharges_scatter_density.png'.")


# --- NEW: Threshold Optimization Plot for XGBoost (Example) ---
print("\n--- Generating Threshold Optimization Plot for XGBoost ---")

thresholds = np.linspace(0, 1, 100) # 100 thresholds from 0 to 1
precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []

for t in thresholds:
    y_pred_threshold = (y_prob_xgb >= t).astype(int)
    precision_scores.append(precision_score(y_test, y_pred_threshold, zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred_threshold, zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred_threshold, zero_division=0))
    accuracy_scores.append(accuracy_score(y_test, y_pred_threshold))

plt.figure(figsize=(12, 7))
# Multiplied by 100 for percentage display on plot
plt.plot(thresholds * 100, np.array(precision_scores) * 100, label='Precision (%)', color='green')
plt.plot(thresholds * 100, np.array(recall_scores) * 100, label='Recall (%)', color='red')
plt.plot(thresholds * 100, np.array(f1_scores) * 100, label='F1-Score (%)', color='purple')
plt.plot(thresholds * 100, np.array(accuracy_scores) * 100, label='Accuracy (%)', color='blue', linestyle='--')

plt.xlabel('Probability Threshold (%)') # Updated label
plt.ylabel('Score (%)') # Updated label
plt.title('Metric Scores vs. Probability Threshold (XGBoost)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 100) # Set x-axis limit to 0-100
plt.ylim(0, 100) # Set y-axis limit to 0-100
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'threshold_optimization_xgb.png'))
plt.show()
print(f"Threshold Optimization Plot for XGBoost saved as '{output_dir}/threshold_optimization_xgb.png'.")


# --- 3.3 Model Interpretability (SHAP Values) ---
print("\n--- Model Interpretability: SHAP Values for XGBoost (Best Model Example) ---")
explainer = shap.TreeExplainer(xgb_model)
X_test_display = pd.DataFrame(X_test, columns=X_columns)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test_display, plot_type="bar", show=False)
plt.title('SHAP Feature Importance for Churn Prediction (XGBoost)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_feature_importance_xgb.png'))
plt.show()
print(f"SHAP feature importance plot saved as '{output_dir}/shap_feature_importance_xgb.png'.")


# --- 3.4 Fairness Assessment ---
print("\n--- Fairness Assessment using Fairlearn ---")
sensitive_features_df_test = df_for_fairness.loc[X_test.index, ['gender_Male', 'SeniorCitizen']]

# Evaluate fairness for 'gender_Male'
if 'gender_Male' in sensitive_features_df_test.columns:
    print("\nFairness evaluation for 'gender_Male':")
    mf_accuracy_gender = MetricFrame(metrics=accuracy_score,
                                     y_true=y_test,
                                     y_pred=y_pred_xgb,
                                     sensitive_features=sensitive_features_df_test['gender_Male'])
    print("Accuracy by Gender (0=Female, 1=Male):\n", mf_accuracy_gender.by_group.apply(lambda x: f"{x * 100:.2f}%"))

    demographic_parity_gender = demographic_parity_ratio(
        y_true=y_test,
        y_pred=y_pred_xgb,
        sensitive_features=sensitive_features_df_test['gender_Male']
    )
    print(f"Demographic Parity Ratio (Gender): {demographic_parity_gender:.4f} (Closer to 1 is fairer)")

    mf_recall_gender = MetricFrame(metrics=recall_score,
                                   y_true=y_test,
                                   y_pred=y_pred_xgb,
                                   sensitive_features=sensitive_features_df_test['gender_Male'])
    print("Recall by Gender:\n", mf_recall_gender.by_group.apply(lambda x: f"{x * 100:.2f}%"))

    # Visualizing Fairness for Gender
    plt.figure(figsize=(10, 6))
    ax_gender_acc = mf_accuracy_gender.by_group.apply(lambda x: x * 100).plot(kind='bar', color=['blue', 'orange'], ax=plt.gca(), zorder=2) # Scale for plot
    plt.title('Accuracy by Gender (%)')
    plt.ylabel('Accuracy Score (%)')
    plt.xlabel('Gender (0=Female, 1=Male)')
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
    for container in ax_gender_acc.containers: # Add labels to bars
        ax_gender_acc.bar_label(container, fmt='%.2f%%', label_type='edge', padding=3, fontsize=9, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fairness_accuracy_gender.png'))
    plt.show()
    print(f"Fairness Accuracy by Gender plot saved as '{output_dir}/fairness_accuracy_gender.png'.")

    plt.figure(figsize=(10, 6))
    ax_gender_recall = mf_recall_gender.by_group.apply(lambda x: x * 100).plot(kind='bar', color=['blue', 'orange'], ax=plt.gca(), zorder=2) # Scale for plot
    plt.title('Recall by Gender (%)')
    plt.ylabel('Recall Score (%)')
    plt.xlabel('Gender (0=Female, 1=Male)')
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
    for container in ax_gender_recall.containers: # Add labels to bars
        ax_gender_recall.bar_label(container, fmt='%.2f%%', label_type='edge', padding=3, fontsize=9, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fairness_recall_gender.png'))
    plt.show()
    print(f"Fairness Recall by Gender plot saved as '{output_dir}/fairness_recall_gender.png'.")


# Evaluate fairness for 'SeniorCitizen'
if 'SeniorCitizen' in sensitive_features_df_test.columns:
    print("\nFairness evaluation for 'SeniorCitizen':")
    mf_accuracy_senior = MetricFrame(metrics=accuracy_score,
                                     y_true=y_test,
                                     y_pred=y_pred_xgb,
                                     sensitive_features=sensitive_features_df_test['SeniorCitizen'])
    print("Accuracy by SeniorCitizen (0=No, 1=Yes):\n", mf_accuracy_senior.by_group.apply(lambda x: f"{x * 100:.2f}%"))

    demographic_parity_senior = demographic_parity_ratio(
        y_true=y_test,
        y_pred=y_pred_xgb,
        sensitive_features=sensitive_features_df_test['SeniorCitizen']
    )
    print(f"Demographic Parity Ratio (SeniorCitizen): {demographic_parity_senior:.4f} (Closer to 1 is fairer)")

    mf_recall_senior = MetricFrame(metrics=recall_score,
                                   y_true=y_test,
                                   y_pred=y_pred_xgb,
                                   sensitive_features=sensitive_features_df_test['SeniorCitizen'])
    print("Recall by SeniorCitizen:\n", mf_recall_senior.by_group.apply(lambda x: f"{x * 100:.2f}%"))

    # Visualizing Fairness for Senior Citizen
    plt.figure(figsize=(10, 6))
    ax_senior_acc = mf_accuracy_senior.by_group.apply(lambda x: x * 100).plot(kind='bar', color=['green', 'purple'], ax=plt.gca(), zorder=2) # Scale for plot
    plt.title('Accuracy by Senior Citizen Status (%)')
    plt.ylabel('Accuracy Score (%)')
    plt.xlabel('Senior Citizen (0=No, 1=Yes)')
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
    for container in ax_senior_acc.containers: # Add labels to bars
        ax_senior_acc.bar_label(container, fmt='%.2f%%', label_type='edge', padding=3, fontsize=9, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fairness_accuracy_seniorcitizen.png'))
    plt.show()
    print(f"Fairness Accuracy by Senior Citizen plot saved as '{output_dir}/fairness_accuracy_seniorcitizen.png'.")

    plt.figure(figsize=(10, 6))
    ax_senior_recall = mf_recall_senior.by_group.apply(lambda x: x * 100).plot(kind='bar', color=['green', 'purple'], ax=plt.gca(), zorder=2) # Scale for plot
    plt.title('Recall by Senior Citizen Status (%)')
    plt.ylabel('Recall Score (%)')
    plt.xlabel('Senior Citizen (0=No, 1=Yes)')
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
    for container in ax_senior_recall.containers: # Add labels to bars
        ax_senior_recall.bar_label(container, fmt='%.2f%%', label_type='edge', padding=3, fontsize=9, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fairness_recall_seniorcitizen.png'))
    plt.show()
    print(f"Fairness Recall by Senior Citizen plot saved as '{output_dir}/fairness_recall_seniorcitizen.png'.")


print("\n--- Comprehensive Evaluation Complete ---")
