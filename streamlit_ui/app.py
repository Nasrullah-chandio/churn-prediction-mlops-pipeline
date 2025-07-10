import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from streamlit_ui.utils import transform_for_model

import streamlit as st
import pandas as pd
import joblib
import json

# === Paths ===
MODEL_PATH = "artifacts/model_trainer/model.joblib"
SCALER_PATH = "artifacts/model_trainer/scaler.joblib"
COLUMNS_PATH = "artifacts/model_trainer/columns.json"
RAW_DATA_PATH = "artifacts/feature_engineering/inference_data.csv"


# === Load model artifacts ===
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(COLUMNS_PATH, "r") as f:
        columns = json.load(f)
    return model, scaler, columns

model, scaler, columns = load_model()

# === Page config ===
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("🤖 Customer Churn Prediction")

# === Mode Switch ===
mode = st.radio("Choose Mode:", ["Predict", "Bulk Predict"], horizontal=True)

def get(row, colname):
    return str(row[colname].values[0]) if colname in row.columns else "N/A"

# === Mode 1: Predict by Customer ID ===
if mode == "Predict":
    st.markdown("Enter a **Customer ID** to view subscriber details and churn prediction.")
    customer_id = st.text_input("🔍 Enter Customer ID", placeholder="e.g. 7590-VHVEG")

    if customer_id:
        df = pd.read_csv(RAW_DATA_PATH)

        if customer_id not in df["customerID"].values:
            st.error("❌ Customer ID not found.")
        else:
            row = df[df["customerID"] == customer_id].copy()
            X_model = transform_for_model(row, columns)
            X_scaled = scaler.transform(X_model)
            churn_proba = model.predict_proba(X_scaled)[0][1]

            # === Display Info ===
            st.markdown("## 🧑‍💼 Subscriber Details")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("### 👤 Customer Demographics")
                html = f"""
                <div style='font-family: monospace;'>
                    <div><span style='display:inline-block; width: 180px;'>Customer ID:</span> {customer_id}</div>
                    <div><span style='display:inline-block; width: 180px;'>Gender:</span> {get(row, 'gender')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Senior Citizen:</span> {get(row, 'SeniorCitizen')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Partner:</span> {get(row, 'Partner')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Dependents:</span> {get(row, 'Dependents')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Tenure (months):</span> {get(row, 'tenure')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Contract:</span> {get(row, 'Contract')}</div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

            with col2:
                st.markdown("### 🛠️ Services Subscribed")
                html = f"""
                <div style='font-family: monospace;'>
                    <div><span style='display:inline-block; width: 180px;'>Phone Service:</span> {get(row, 'PhoneService')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Multiple Lines:</span> {get(row, 'MultipleLines')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Internet Service:</span> {get(row, 'InternetService')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Tech Support:</span> {get(row, 'TechSupport')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Streaming TV:</span> {get(row, 'StreamingTV')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Streaming Movies:</span> {get(row, 'StreamingMovies')}</div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

            with col3:
                st.markdown("### 🌐 Internet Related Services")
                html = f"""
                <div style='font-family: monospace;'>
                    <div><span style='display:inline-block; width: 180px;'>Online Security:</span> {get(row, 'OnlineSecurity')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Online Backup:</span> {get(row, 'OnlineBackup')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Device Protection:</span> {get(row, 'DeviceProtection')}</div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

            with col4:
                st.markdown("### 💳 Billing and Payment")
                html = f"""
                <div style='font-family: monospace;'>
                    <div><span style='display:inline-block; width: 180px;'>Paperless Billing:</span> {get(row, 'PaperlessBilling')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Payment Method:</span> {get(row, 'PaymentMethod')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Monthly Charges:</span> {get(row, 'MonthlyCharges')}</div>
                    <div><span style='display:inline-block; width: 180px;'>Total Charges:</span> {get(row, 'TotalCharges')}</div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🎯 Prediction!")
            if churn_proba >= 0.5:
                st.error(f"⚠️ The customer is **likely to churn** with a probability of **{churn_proba:.2%}**.")
            else:
                st.success(f"😊 The customer will **stay** with a likelihood of **{1 - churn_proba:.2%}**.")

# === Mode 2: Bulk Prediction ===
elif mode == "Bulk Predict":
    st.subheader("📁 Upload CSV for Bulk Churn Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            transformed_df = bulk_df.copy()
            X_model = transform_for_model(transformed_df, columns)
            X_scaled = scaler.transform(X_model)
            churn_probas = model.predict_proba(X_scaled)[:, 1]

            result_df = bulk_df.copy()
            result_df["churn_probability"] = churn_probas

            st.success("✅ Predictions added successfully!")
            st.markdown("### 🔍 Preview")
            st.dataframe(result_df.head(10))

            # CSV download
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results CSV", data=csv, file_name="bulk_churn_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")
