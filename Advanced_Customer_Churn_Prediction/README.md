# 📉 Advanced Customer Churn Prediction

This project predicts customer churn for a telecom dataset using advanced machine learning models. It includes end-to-end data preprocessing, exploratory analysis, feature engineering, model building, interpretability, and fairness evaluation.

🔗 **Project Folder**: [Click here to view in repository](https://github.com/pramod13071626/ML-Projects/tree/main/Advanced_Customer_Churn_Prediction)

---

## 📌 Objective

To build a robust ML pipeline that can predict whether a customer is likely to churn, while ensuring:
- High predictive performance
- Explainability of results
- Fairness across sensitive features (e.g., gender, senior citizen status)

---

## 📊 Workflow

### 1. Data Collection & Preprocessing
- Dataset: **Telco Customer Churn**  
- Steps:
  - Handle missing values
  - Convert `TotalCharges` to numeric
  - Drop irrelevant columns (`customerID`)
  - Standardize categorical values
  - Apply one-hot encoding
  - Scale features using StandardScaler
  - Handle class imbalance with **SMOTE**

### 2. Exploratory Data Analysis (EDA)
- Churn distribution and class imbalance visualization
- Categorical churn rates (e.g., gender, contract type)
- Correlation heatmaps for numerical features
- Distribution plots for tenure, monthly charges, etc.

### 3. Feature Engineering
- Interaction feature: `MonthlyCharges x Tenure`
- Tenure grouping into categories
- Outlier detection & treatment (IQR method)

### 4. Model Building
- Algorithms used:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- Hyperparameter tuning with **GridSearchCV / RandomizedSearchCV**
- Cross-validation for reliable performance

### 5. Model Evaluation
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC AUC
- ROC curve comparison across models

### 6. Model Interpretability
- **SHAP values** for feature importance
- Key drivers of churn identified (e.g., contract type, tenure, charges)

### 7. Fairness Assessment
- Evaluated fairness across sensitive attributes using **Fairlearn**
- Ensured equitable performance across customer demographics

---

## 📂 Folder Structure

Advanced_Customer_Churn_Prediction/
│
├── Data/
│ ├── Raw/ # Original dataset
│ └── Processed/ # Cleaned dataset
│
├── Scripts/ # Python scripts
│ ├── Data_Processing.py
│ ├── Model_Implementation.py
│ ├── Evaluation.py
│ └── optional_file.py
│
├── Reports/
│ ├── Docs/ # Documentation & outputs
│ └── Figures/ # Graphs, plots, ROC curves
│
├── artifacts/ # Saved models & preprocessed files
│ ├── xgb_model.pkl
│ ├── best_rf_model.pkl
│ ├── scaler.pkl
│ └── X_train_resampled.pkl, etc.
│
├── requirements.txt # Project dependencies
└── README.md # Project overview.
