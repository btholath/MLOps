Project Overview
The project represents a complete ETL and Machine Learning pipeline — from raw data ingestion to model deployment in the cloud — using modular components and configuration-driven architecture.

📁 Project Structure (Visualized)
MongoDB → Ingestion → Validation → Transformation → Training → Evaluation → Pusher → Cloud (AWS/Azure)

🔧 Components
Component Description
Data Ingestion Extracts data from MongoDB or other sources
Data Validation Validates raw data for schema, nulls, types
Data Transformation Cleans, encodes, and transforms data into ML-ready format
Model Training Trains the ML model on transformed dataset
Model Evaluation Evaluates model performance to determine if it's deployable
Model Pusher Deploys the approved model to AWS, Azure, or other cloud targets

🔄 ETL Pipeline Explained
🔹 1. Extract
Sources:
CSV datasets (local)
REST APIs (free or paid)
AWS S3 buckets
Internal databases (MongoDB, MySQL, etc.)
Tools: Python scripts or ingestion connectors
Goal: Retrieve raw data from structured/unstructured sources.

🔸 2. Transform
Steps:
Basic Preprocessing (null handling, type casting)
Cleaning raw data (duplicates, outliers)
Convert to intermediate format: JSON
Tools: Python (Pandas, NumPy, custom modules)
Output: Cleaned JSON ready for modeling

🔻 3. Load
Destinations:
MongoDB (hosted via Atlas)
AWS DynamoDB
MySQL
S3 Buckets
Goal: Store transformed data or trained models for future use or deployment.

☁️ Model Deployment Flow
If model passes evaluation metrics, it is pushed via the Model Pusher Component
Target: Cloud providers like AWS, Azure
Artifacts: Saved model binaries, evaluation reports

🧱 Modules

1. Data Ingestion
   Sources: MongoDB, CSVs, APIs, S3
   Extracts raw data into pipeline

2. Data Validation
   Checks for schema mismatch, nulls, and data integrity

3. Data Transformation
   Cleans and preprocesses data
   Converts raw input into machine learning-ready format

4. Model Training
   Trains model using transformed data
   Saves training artifacts and logs

5. Model Evaluation
   Validates model accuracy, F1, AUC, etc.
   Decides whether the model is production-ready

6. Model Deployment
   Deploys validated models to cloud platforms (AWS, Azure)
   Outputs saved as Model Pusher Artifacts

🔄 ETL Pipeline Steps
Extract
Input: CSV files, APIs, S3 buckets, internal databases
Output: Raw dataset for transformation
Transform
Cleansing, encoding, feature engineering
Output: JSON-formatted, clean dataset

Load
Targets: MongoDB Atlas, AWS DynamoDB, MySQL, S3

Purpose: Persist transformed data or model outputs

☁️ Cloud Integration
Supports deployment to:

AWS

Azure

Can be extended to GCP and other environments

📦 Artifacts
Stored at each stage (Ingestion, Transformation, Training, Evaluation, Deployment)

Used for auditing, retraining, and monitoring

🚀 Future Enhancements
Add CI/CD for model retraining

Add monitoring with Prometheus + Grafana

Support for real-time streaming (Kafka, Kinesis)

The document outlines the components and processes involved in a data pipeline for model training and evaluation.
Data Ingestion Config ​

Configures data ingestion parameters such as collection name, data directories, and file paths for training and testing datasets. ​
Includes a train-test split ratio for data preparation.

Data Validation

Validates data integrity by checking for missing numeric columns in training and testing datasets.
Generates validation status reports and drift reports to monitor data quality.

Data Transformation

Preprocesses data by dropping target columns and handling missing values using techniques like Simple Imputer.
Outputs transformed data arrays for training and testing, along with a saved preprocessing object.

Model Trainer

Initiates model training using numpy arrays for training and testing data.
Evaluates models based on expected accuracy and selects the best-performing model.
Saves the trained model and associated metrics for future use.

Deployment

Using GitHub Actions
/home/bijut/aws_apps/MLOps/01_etl_pipeline/.github/main.yml


## GitHub setup
(.venv) bijut@b:~/aws_apps/MLOps$ git init
(.venv) bijut@b:~/aws_apps/MLOps$ git remote add origin git@github.com:btholath/MLOps.git
(.venv) bijut@b:~/aws_apps/MLOps$ git branch -M main
(.venv) bijut@b:~/aws_apps/MLOps$ git add .
(.venv) bijut@b:~/aws_apps/MLOps$ git commit -m "version 1.0"
(.venv) bijut@b:~/aws_apps/MLOps$ git push -u origin main

## Python virtual environment
bijut@b:~/aws_apps$ source .venv/bin/activate
(.venv) bijut@b:~/aws_apps$ cd MLOps/01_etl_pipeline/
(.venv) bijut@b:~/aws_apps/MLOps/01_etl_pipeline$ pip install --upgrade pip
(.venv) bijut@b:~/aws_apps/MLOps/01_etl_pipeline$ pip install -r requirements.txt
