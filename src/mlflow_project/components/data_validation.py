import os
import pandas as pd
import yaml
from mlflow_project.utils.common import read_yaml, save_text
from mlflow_project.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            df = pd.read_csv(self.config.data_path)
            schema = read_yaml(self.config.schema_file)

            all_cols = list(df.columns)
            expected_cols = list(schema['bss_data']['columns'].keys())

            validation_status = all(col in all_cols for col in expected_cols)

            self.config.status_file.parent.mkdir(parents=True, exist_ok=True)
            save_text(str(validation_status), self.config.status_file)            
            return validation_status
        except Exception as e:
            raise e
