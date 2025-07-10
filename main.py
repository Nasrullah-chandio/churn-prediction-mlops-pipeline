from src.mlflow_project.pipeline.stage_01_data_ingestion import main as stage_01_data_ingestion
from src.mlflow_project.pipeline.stage_02_data_validation import main as stage_02_data_validation
from src.mlflow_project.pipeline.stage_03_data_preprocessing import main as stage_03_data_preprocessing
from src.mlflow_project.pipeline.stage_04_feature_engineering import main as stage_04_feature_engineering
from src.mlflow_project.pipeline.stage_05_model_trainer import ModelTrainerPipeline
from src.mlflow_project.pipeline.stage_06_model_evaluation import ModelEvaluationTrainingPipeline

if __name__ == "__main__":
    # Stage 01 - Data Ingestion
    stage_01_data_ingestion()

    # Stage 02 - Data Validation
    stage_02_data_validation()

    # Stage 03 - Data Preprocessing
    stage_03_data_preprocessing()

    # Stage 04 - Feature Engineering
    stage_04_feature_engineering()

    # Stage 05 - Model Training
    stage_05 = ModelTrainerPipeline()
    stage_05.main()

    # Stage 06 - Model Evaluation
    stage_06 = ModelEvaluationTrainingPipeline()
    stage_06.main()
