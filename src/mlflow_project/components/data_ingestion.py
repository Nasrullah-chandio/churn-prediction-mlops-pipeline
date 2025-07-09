import os
import pandas as pd
from mlflow_project.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        # Read the 3 source CSVs
        bss_df = pd.read_csv(self.config.bss_data_path)
        network_df = pd.read_csv(self.config.network_data_path)
        complaints_df = pd.read_csv(self.config.complaints_data_path)

        # Create the ingested directory if it doesn't exist
        os.makedirs(self.config.ingested_dir, exist_ok=True)

        # Save raw files into the ingested directory (optional for logging/debugging)
        bss_df.to_csv(os.path.join(self.config.ingested_dir, "bss_data.csv"), index=False)
        network_df.to_csv(os.path.join(self.config.ingested_dir, "network_data.csv"), index=False)
        complaints_df.to_csv(os.path.join(self.config.ingested_dir, "complaints_data.csv"), index=False)

        return bss_df, network_df, complaints_df
