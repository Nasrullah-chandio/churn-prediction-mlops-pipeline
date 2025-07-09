from mlflow_project.constants import *
from mlflow_project.utils.common import read_yaml, create_directories
from mlflow_project.entity.config_entity import DataIngestionConfig
from mlflow_project.entity.config_entity import DataValidationConfig
from mlflow_project.entity.config_entity import DataPreprocessingConfig
from mlflow_project.entity.config_entity import FeatureEngineeringConfig
from mlflow_project.entity.config_entity import ModelTrainerConfig
from mlflow_project.entity.config_entity import ModelEvaluationConfig
from mlflow_project.entity.config_entity import ModelEvaluationConfig

from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([Path(config.root_dir)])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            bss_data_path=Path(config.bss_data_path),
            network_data_path=Path(config.network_data_path),
            complaints_data_path=Path(config.complaints_data_path),
            ingested_dir=Path(config.ingested_dir)
        )
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.config.schema

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            status_file=Path(config.status_file),
            schema_file=Path(schema)
        )

    

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        create_directories([config.root_dir])

        return DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            bss_data_path=Path(config.bss_data_path),
            network_data_path=Path(config.network_data_path),
            complaints_data_path=Path(config.complaints_data_path),
            preprocessed_data_path=Path(config.preprocessed_data_path)
        )


    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        config = self.config.feature_engineering

        create_directories([config.root_dir])

        return FeatureEngineeringConfig(
            root_dir=config.root_dir,
            input_data_path=config.input_data_path,
            feature_data_path=config.feature_data_path,
            inference_data_path=config.inference_data_path,
            preprocessed_data_path=config.preprocessed_data_path
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([Path(config.root_dir)])
        return ModelTrainerConfig(
            data_path=Path(config.data_path),
            model_path=Path(config.model_path),
            scaler_path=Path(config.scaler_path),
            columns_path=Path(config.columns_path),       
            test_size=config.test_size,
            random_state=config.random_state,
            target_column=config.target_column   
        )


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([Path(config.root_dir)])
        return ModelEvaluationConfig(
            data_path=Path(config.data_path),
            model_path=Path(config.model_path),
            scaler_path=Path(config.scaler_path),
            columns_path=Path(config.columns_path),
            metrics_path=Path(config.metrics_path),
            target_column=config.target_column,
            mlflow_uri=config.mlflow_uri,
            test_size=config.test_size,              
            random_state=config.random_state         
        )