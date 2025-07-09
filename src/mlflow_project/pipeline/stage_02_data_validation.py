from mlflow_project.config.configuration import ConfigurationManager
from mlflow_project.components.data_validation import DataValidation
from mlflow_project import logger

STAGE_NAME = "Data Validation Stage"

def main():
    config = ConfigurationManager()
    data_validation_config = config.get_data_validation_config()
    data_validation = DataValidation(config=data_validation_config)
    data_validation.validate_all_columns()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
