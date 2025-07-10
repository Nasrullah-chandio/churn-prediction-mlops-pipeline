# src/mlflow_project/pipeline/stage_06_model_evaluation.py

from mlflow_project.config.configuration import ConfigurationManager
from mlflow_project.components.model_evaluation import ModelEvaluation
from mlflow_project import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_eval = ModelEvaluation(config=model_evaluation_config)
        model_eval.log_into_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f"\\n{'='*20} {STAGE_NAME} started {'='*20}")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f"{'='*20} {STAGE_NAME} completed {'='*20}\n")
    except Exception as e:
        logger.exception(e)
        raise e
