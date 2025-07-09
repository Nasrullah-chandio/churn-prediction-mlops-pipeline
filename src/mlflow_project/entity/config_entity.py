from dataclasses import dataclass
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    root_dir: Path
    bss_data_path: Path
    network_data_path: Path
    complaints_data_path: Path
    ingested_dir: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    data_path: Path
    status_file: Path
    schema_file: Path

# ✅ NEW: Data Preprocessing Config
@dataclass
class DataPreprocessingConfig:
    root_dir: Path
    bss_data_path: Path
    network_data_path: Path
    complaints_data_path: Path
    preprocessed_data_path: Path


@dataclass
class FeatureEngineeringConfig:
    root_dir: str
    input_data_path: str
    feature_data_path: str
    inference_data_path: str
    preprocessed_data_path: str
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass(frozen=True)
class ModelTrainerConfig:
    data_path: Path
    model_path: Path
    scaler_path: Path
    columns_path: Path
    target_column: str
    test_size: float
    random_state: int

@dataclass
class ModelEvaluationConfig:
    data_path: Path
    model_path: Path
    scaler_path: Path
    columns_path: Path           # ✅ Add this line
    mlflow_uri: str
    metrics_path: Path
    target_column: str
    test_size: float
    random_state: int
