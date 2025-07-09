from mlflow_project.config.configuration import ConfigurationManager
from mlflow_project.components.data_preprocessing import DataPreprocessing
from mlflow_project.utils.common import logger


def main():
    config = ConfigurationManager()
    preprocessing_config = config.get_data_preprocessing_config()
    
    logger.info(">>>>> Data Preprocessing Stage started <<<<<")
    try:
        data_preprocessor = DataPreprocessing(config=preprocessing_config)
        data_preprocessor.preprocess_and_merge_data()
        logger.info(">>>>> Data Preprocessing Stage completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    main()
