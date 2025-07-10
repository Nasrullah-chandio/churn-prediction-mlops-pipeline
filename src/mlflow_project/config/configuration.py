from mlflow_project.constants import *
from mlflow_project.utils.common import read_yaml, create_directories
from mlflow_project.entity.config_entity import DataIngestionConfig
from mlflow_project.entity.config_entity import DataValidationConfig
from mlflow_project.entity.config_entity import DataPreprocessingConfig
from mlflow_project.entity.config_entity import FeatureEngineeringConfig
from mlflow_project.entity.config_entity import ModelTrainerConfig
from mlflow_project.entity.config_entity import ModelEvaluationConfig
from mlflow_project.entity.config_entity import ModelEvaluationConfig

import os
from mlflow_project.constants import *
from mlflow_project.utils.common import read_yaml, create_directories
from mlflow_project.entity.config_entity import ModelTrainerConfig
from pathlib import Path
from box.exceptions import BoxKeyError





class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(Path(config_filepath))
        self.params = read_yaml(Path(params_filepath))

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
        try:
            config = self.params.model_trainer
        except BoxKeyError as e:
            raise ValueError(f"Missing key in params.yaml: {e}")

        create_directories([Path(config.root_dir)])

        return ModelTrainerConfig(
            root_dir=config.root_dir,
            model_name=config.model_name,
            target_column=config.target_column,
            data_path=os.path.join(self.config.artifacts_root, "feature_engineering", "feature_engineered_data.csv"),
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            max_depth=config.max_depth,
            random_state=config.random_state,
            model_path=os.path.join(config.root_dir, "model.joblib"),
            scaler_path=os.path.join(config.root_dir, "scaler.joblib"),
            columns_path=os.path.join(config.root_dir, "columns.json")
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