import json
import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def log_into_mlflow(self):
        # Load data
        df = pd.read_csv(self.config.data_path)
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]

        # Match column order from training
        with open(self.config.columns_path, "r") as f:
            original_columns = json.load(f)
        X = X[original_columns]

        # Split test data
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        # Load model and scaler
        model = joblib.load(self.config.model_path)
        scaler = joblib.load(self.config.scaler_path)
        X_test_scaled = scaler.transform(X_test)

        # Predict and calculate metrics
        y_pred = model.predict(X_test_scaled)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        # Save metrics to JSON
        Path(self.config.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Log to MLflow
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

        print("✅ Evaluation metrics saved and logged to MLflow.")
