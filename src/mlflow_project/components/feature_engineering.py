import pandas as pd
import numpy as np
from pathlib import Path
from mlflow_project.config.configuration import ConfigurationManager

class FeatureEngineering:
    def __init__(self, config):
        self.config = config

    def run(self):
        print("📥 Reading preprocessed data...")
        df_preprocessed = pd.read_csv(self.config.preprocessed_data_path)

        df = df_preprocessed.copy()
        df["Churn_Label"] = df["Churn"].map({'Yes': 1, 'No': 0})
        df.drop(columns=["Churn"], errors="ignore", inplace=True)

        bool_cols = df.select_dtypes(include=["bool"]).columns
        df[bool_cols] = df[bool_cols].astype(int)

     # Derived features
        df["complaints_per_tenure"] = df["total_complaints"] / (df["tenure"] + 1)
        df["unresolved_ratio"] = df["unresolved_complaints"] / (df["total_complaints"] + 1)
        df["drop_rate_per_tenure"] = df["call_drop_rate"] / (df["tenure"] + 1)
        df["data_usage_per_tenure"] = df["data_usage_gb"] / (df["tenure"] + 1)

    # Save unencoded version for Streamlit
        df.to_csv(self.config.inference_data_path, index=False)
        print(f"✅ Saved inference data to: {self.config.inference_data_path}")

    # Drop ID before model
        df_model = df.drop(columns=["customerID"])

    # Identify categorical columns (object/string types)
        cat_cols = df_model.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

        df_model.to_csv(self.config.feature_data_path, index=False)
        print(f"✅ Saved model training data to: {self.config.feature_data_path}")
