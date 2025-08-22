import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
import math
import joblib
import os

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Create a directory for artifacts if it doesn't exist (absolute path) ---
output_dir = os.path.join(os.getcwd(), 'artifacts')
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory '{output_dir}' ensured.")
print("Current Working Directory:", os.getcwd())

print("--- Step 1: Data Preprocessing (Advanced and Robust) ---")

# 1. Load Dataset
df = pd.read_csv('Data/Raw/Telco-Customer-Churn_raw.csv')
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

# --- Advanced EDA ---
print("\n--- Advanced EDA: Visualizing Data Distributions and Relationships ---")
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Distribution Analysis
plt.figure(figsize=(18, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 3, i + 1)
    sns.histplot(data=df, x=col, hue='Churn', kde=True, palette='viridis', alpha=0.7)
    plt.title(f'Distribution of {col} by Churn')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'numerical_distributions_by_churn.png'))
plt.show()
print("Saved 'numerical_distributions_by_churn.png'.")

# Categorical Feature Analysis
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
plt.savefig(os.path.join(output_dir, 'categorical_churn_rates.png'))
plt.show()
print("Saved 'categorical_churn_rates.png'.")

# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols + ['Churn']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features and Churn')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'numerical_correlation_matrix.png'))
plt.show()
print("Saved 'numerical_correlation_matrix.png'.")

# --- Feature Engineering ---
print("\n--- Feature Engineering: Creating New Informative Features ---")
df['MonthlyCharges_x_Tenure'] = df['MonthlyCharges'] * df['tenure']
print("Created 'MonthlyCharges_x_Tenure' interaction feature.")

df['TenureGroup'] = pd.cut(df['tenure'],
                           bins=[0, 12, 24, 48, 60, 72],
                           labels=['0-12M', '13-24M', '25-48M', '49-60M', '61-72M'],
                           right=False).astype(str)
print("Created 'TenureGroup'.")

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
    print(f"Capped '{col}' outliers. Bounds: {lower_bound:.2f}-{upper_bound:.2f}")

# --- One-Hot Encoding ---
print("\n--- One-Hot Encoding ---")
categorical_cols_after_fe = df.select_dtypes(include='object').columns
df_encoded = pd.get_dummies(df, columns=categorical_cols_after_fe, drop_first=True)
print(f"Dataset shape after encoding: {df_encoded.shape}")

processed_csv_path = os.path.join(output_dir, 'telco_customer_churn_processed.csv')
df_encoded.to_csv(processed_csv_path, index=False)
print(f"Processed CSV saved: '{processed_csv_path}'")

# Verification
if os.path.exists(processed_csv_path):
    print(f"CSV exists. Size: {os.path.getsize(processed_csv_path)/1024:.2f} KB")
else:
    print("ERROR: CSV not found!")

# Store for fairness
df_for_fairness = df_encoded.copy()

# Features and Target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# --- Train-Test Split ---
print("\n--- Train-Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# --- Feature Scaling ---
numerical_features_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyCharges_x_Tenure']
scaler = StandardScaler()
X_train[numerical_features_to_scale] = scaler.fit_transform(X_train[numerical_features_to_scale])
X_test[numerical_features_to_scale] = scaler.transform(X_test[numerical_features_to_scale])
print("Numerical features scaled.")

# --- Handling Imbalanced Data (SMOTE) ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("SMOTE applied to balance dataset.")

# --- Save processed objects ---
joblib.dump(X_train_resampled, os.path.join(output_dir, 'X_train_resampled.pkl'))
joblib.dump(y_train_resampled, os.path.join(output_dir, 'y_train_resampled.pkl'))
joblib.dump(X_test, os.path.join(output_dir, 'X_test.pkl'))
joblib.dump(y_test, os.path.join(output_dir, 'y_test.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
joblib.dump(X.columns, os.path.join(output_dir, 'X_columns.pkl'))
joblib.dump(df_for_fairness, os.path.join(output_dir, 'df_for_fairness.pkl'))

# Verification
saved_joblib_files = [f for f in os.listdir(output_dir) if f.endswith('.pkl')]
print(f"Joblib files saved in '{output_dir}': {saved_joblib_files}")

print("\n--- Preprocessing Complete ---")
