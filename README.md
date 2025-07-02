# Telecom Churn Prediction (MLOps Pipeline)

This project predicts customer churn using ML models and Streamlit UI, with plans to integrate MLflow, Airflow, and GitHub Actions for a production-grade pipeline.

## Folder Structure
- `data/`: Raw and processed datasets
- `scripts/`: ETL, training, and prediction scripts
- `models/`: Serialized model artifacts
- `notebooks/`: EDA and experimentation

## Setup
```bash
python -m venv venv
pip install -r requirements.txt

---
Missing or invalid TotalCharges
Some entries were blank or non-numeric.
➤ Will convert to numeric using errors='coerce' and handle nulls appropriately (drop or impute).

Churn is imbalanced
Majority of customers did not churn (No > Yes).
➤ Will consider stratified train-test split; resampling is optional.

Churn higher in Month-to-month contracts
Indicates a high-risk customer group.
➤ Will encode the Contract feature and use it for modeling.

Higher churn in customers with no Tech Support
Lack of support impacts customer retention.
➤ Will retain TechSupport as a key categorical feature.

High MonthlyCharges & low tenure linked to churn
New, high-paying users are more likely to leave early.
➤ Will include MonthlyCharges and tenure as important features.

Call Drop Rate and Data Usage show variation
These may correlate with churn or satisfaction.
➤ Will scale/normalize these numerical features before training.

Categorical text features like PaymentMethod, InternetService
Need to be machine-readable.
➤ Will apply one-hot or label encoding.

Complaint-related features (resolved/unresolved/total)
Offer service quality signals.
➤ Will retain all complaint columns as numeric inputs.

customerID is a unique identifier
Not useful for prediction.
➤ Will drop this column before model training.

