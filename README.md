<h1 align="left">Telecom Churn Prediction</h1>

<h2>Table of Contents</h2>

- [Overview](#overview)
- [Objective](#objective)
- [Dataset](#data)
- [MlFlow Integration](#mlflow)
- [Data Version Control (DVC)](#data_versioning)
- [Deployementn & CI/CD](#deployement)
- [Running Locally](#execution)
- [Result](#result)


<a id="overview"></a>
<h2>Overview</h2>
<p align="justify">
This repository presents a complete end-to-end machine learning pipeline designed to predict customer churn. It encompasses all major stages of a production-ready workflow — from data ingestion and validation to transformation, model training, and evaluation. The goal is to build an accurate predictive model that identifies customers at risk of churning, enabling businesses to implement timely retention strategies.
</p>

<a id="objective"></a>
<h2>Objective</h2>
<p align="justify">
The objective of this project is to build a scalable and modular churn prediction system that can help businesses identify customers who are likely to discontinue their services. By leveraging historical data and machine learning techniques, the system aims to generate accurate churn predictions and provide actionable insights for retention efforts. The solution is designed with a production-ready pipeline in mind, incorporating best practices like data versioning (DVC), experiment tracking (MLflow), and a user-friendly interface (Streamlit) for both single and bulk predictions.
</p>

<a id="data"></a>
<h2>Dataset</h2>
<p align="justify">
Three datasets were used in this project from different sources each from Business domain, complaints platform and network systems. Business data was downloaded from kaggle using the link https://www.kaggle.com/datasets/blastchar/telco-customer-churn while complaints and network datasets were randomly generated using simulation
</p>

<a id="mlflow"></a>
<h2>MlFlow Integration</h2>
<p align="justify">
MLflow was integrated into the project using a locally hosted tracking server defined via MLFLOW_TRACKING_URI. This setup enabled seamless experiment tracking, comparison of model runs, and monitoring of performance metrics—all within a self-contained environment.</p>

<a id="data_versioning"></a>
<h2>Data Versioning with DVC</h2>
<p align="justify">
To implement Data Version Control (DVC) in this project, a structured dvc.yaml file was created to define each pipeline stage. This file orchestrates the workflow by specifying:

cmd: Python script to execute

deps: Dependencies such as scripts, config files, or input data

outs: Outputs generated from that stage (e.g., processed data, models)

params: Training parameters used by the model (defined in params.yaml)

metrics: Evaluation metrics logged for model performance tracking

Below is an overview of the DVC pipeline stages used in this churn prediction project:

data_ingestion
Runs stage_01_data_ingestion.py to load the raw dataset from the local source. Dependencies include the ingestion script, configuration manager, and the raw CSV file. The ingested data is stored in the artifacts/data_ingestion folder.

data_validation
Executes stage_02_data_validation.py to verify schema integrity and data quality. It uses the schema file, validation logic, and previously ingested data as dependencies, and outputs a status.txt file indicating success or failure.

data_preprocessing
Applies cleaning logic through stage_03_data_preprocessing.py including null handling and datatype corrections. The cleaned dataset is saved for further processing.

data_feature_engineering
Enhances the dataset in stage_04_data_feature_engineering.py by creating derived features (e.g., usage ratios, churn flags). The final dataset is stored in artifacts/feature_engineering.

model_trainer
Trains a LightGBM model using stage_05_model_trainer.py. Dependencies include the engineered features and training parameters (from params.yaml). Outputs include the serialized model (model.joblib), feature columns, and scaler objects.

model_evaluation
Assesses model performance via stage_06_model_evaluation.py using metrics like accuracy, F1-score, and classification report. The results are saved in metrics.json.

This modular structure ensures reproducibility and makes it easier to rerun only the affected stages when changes are made—greatly boosting development efficiency and experiment tracking.</p>

<a id="deployement"></a>
<h2>Deployement & CI/CD</h2>
<p align="justify">
Deployment and CI/CD integration are planned as future enhancements to this project. The goal is to automate the training pipeline using GitHub Actions and enable seamless model deployment via cloud-based services such as AWS, Azure, or GCP. A Streamlit-based UI is already available locally, and future updates will package the application into a production-ready Docker container, with options to deploy as a web service. Additionally, CI/CD workflows will be integrated to automate testing, model retraining, and versioned deployment upon code commits.</a>. 

<a id="execution"></a>
<h2>Running Locally</h2>
<p align="justify">

### STEP 01 - Clone the repository

```bash
git clone https://github.com/Nasrullah-chandio/churn-prediction-mlops-pipeline.git
```

### STEP 02 - Create a virtual environment 

**Windows** (cmd) <br>

```bash
cd cd churn-prediction-mlops-pipeline
pip install virtualenv
python -m virtualenv venv
```

or

```bash
python3 -m venv venv
```

**macOS/Linux** <br>

```bash
cd End-to-End-Customer-Churn-Prediction-using-MLflow-and-DVC
pip install virtualenv
python -m virtualenv venv
```

### STEP 03 - Activate environment <br>

**Windows** (cmd)

```bash
venv\scripts\activate
```

**macOS/Linux**

```bash
. venv/bin/activate
```

or

```bash
source venv/bin/activate
```

### STEP 04 - Install the Requirements

Windows/macOS/Linux <br>

```bash
pip install -r requirements.txt
```



### STEP 05 - Run app.py

```bash
python streamlit-ui/app.py
```

Now,

```bash
Open the url: http://127.0.0.1:5000/ 
```

<br />

<a id="result"></a>
<h2>Result</h2>
<p align="justify">
The following screenshots demonstrate the functionality of the Streamlit-based user interface for churn prediction. The first image shows a single-customer prediction form, while the second captures bulk churn predictions from uploaded data. These results highlight the practical utility of the model and its front-end integration for business use.
</p>

<p align="center">
  <img src="screenshots/single_test.png" alt="Single Prediction" width="700"/>
</p>
<p align="center">
  <img src="screenshots/bulk%20test.png"" alt="Bulk Prediction" width="700"/>
</p>
