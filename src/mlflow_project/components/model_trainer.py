import json
import pandas as pd
from lightgbm import LGBMClassifier
import joblib
from sklearn.preprocessing import StandardScaler

from mlflow_project import logger
from mlflow_project.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load data
        data = pd.read_csv(self.config.data_path)

        # Split features and target
        X = data.drop([self.config.target_column], axis=1)
        y = data[[self.config.target_column]]

        # Save column names BEFORE converting to NumPy
        original_columns = X.columns

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save the scaler
        joblib.dump(scaler, self.config.scaler_path)

        # Save the column names
        with open(self.config.columns_path, "w") as f:
            json.dump(list(original_columns), f)

        # Train the LightGBM model
        model = LGBMClassifier(
            n_estimators=self.config.n_estimators,
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth
        )
        model.fit(X_scaled, y.values.ravel())

        # Save the trained model
        joblib.dump(model, self.config.model_path)

        logger.info(f"Model and scaler saved to: {self.config.root_dir}")
