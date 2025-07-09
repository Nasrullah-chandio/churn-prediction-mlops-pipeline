# src/mlflow_project/pipeline/stage_01_data_ingestion.py

from mlflow_project.config.configuration import ConfigurationManager
from mlflow_project.components.data_ingestion import DataIngestion
from mlflow_project.utils.common import logger

STAGE_NAME = "Data Ingestion Stage"

def main():
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.initiate_data_ingestion()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
