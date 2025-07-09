import pandas as pd
from mlflow_project.entity.config_entity import DataPreprocessingConfig
from mlflow_project.utils.common import create_directories
import os

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def preprocess_and_merge_data(self):
        # Read input files
        df_bss = pd.read_csv(self.config.bss_data_path)
        df_complaints = pd.read_csv(self.config.complaints_data_path)
        df_network = pd.read_csv(self.config.network_data_path)

        # 1. BSS preprocessing
        df_bss['TotalCharges'] = pd.to_numeric(df_bss['TotalCharges'], errors='coerce')
        df_bss['Churn_Label'] = df_bss['Churn'].map({'Yes': 1, 'No': 0})

        # 2. Complaints preprocessing
        df_complaints.rename(columns={'date': 'complaint_date', 'issue': 'complaint_type'}, inplace=True)
        agg_complaints = df_complaints.groupby('customerID').agg(
            total_complaints=('complaint_id', 'count'),
            resolved_complaints=('status', lambda x: (x == 'Resolved').sum()),
            unresolved_complaints=('status', lambda x: (x != 'Resolved').sum())
        ).reset_index()

        # 3. Network preprocessing
        df_network['call_drop_rate'] = df_network['dropped_calls'] / df_network['total_calls']
        df_network['data_usage_gb'] = df_network['data_volume_MB'] / 1024
        agg_network = df_network.groupby('customerID').agg({
            'call_drop_rate': 'mean',
            'data_usage_gb': 'mean',
            'throughput_Mbps': 'mean'
        }).reset_index()

        # 4. Merge all
        df_merged = df_bss.merge(agg_complaints, on='customerID', how='left')
        df_merged = df_merged.merge(agg_network, on='customerID', how='left')

        # Save final merged dataset
        create_directories([os.path.dirname(self.config.preprocessed_data_path)])
        df_merged.to_csv(self.config.preprocessed_data_path, index=False)
