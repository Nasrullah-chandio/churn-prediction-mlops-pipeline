from mlflow_project.config.configuration import ConfigurationManager
from mlflow_project.components.feature_engineering import FeatureEngineering
from mlflow_project.utils.common import logger

STAGE_NAME = "Feature Engineering Stage"

def main():
    config = ConfigurationManager().get_feature_engineering_config()
    feature_engineer = FeatureEngineering(config)
    feature_engineer.run()

if __name__ == "__main__":
    try:
        logger.info(f">>>>> {STAGE_NAME} started <<<<<")
        main()
        logger.info(f">>>>> {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
