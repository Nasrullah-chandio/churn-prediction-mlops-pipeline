from pathlib import Path

# Path-based constants
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
SCHEMA_FILE_PATH = Path("schema.yaml")

# Static constants
ARTIFACTS_DIR = "artifacts"
MODEL_NAME = "xgb_churn_model"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "ChurnPredictionExperiment"
