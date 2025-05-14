"""
ML Pipeline Runner Script
-------------------------
This is the orchestrator script to run the entire ML pipeline.
Steps performed:
1. Initialize configuration
2. Run data ingestion to fetch and save raw/train/test data
3. Validate schema and check data drift
4. Transform the data using KNN imputation
5. Train multiple models, evaluate them, and select the best one
6. Track metrics and artifacts using MLflow on DagsHub

Audience:
- Data scientists and engineers running ML experiments
- MLOps teams for integration and monitoring
"""

from etl_pipeline.security.components.data_ingestion import DataIngestion
from etl_pipeline.security.components.data_validation import DataValidation

from etl_pipeline.security.exception.exception import NetworkSecurityException
from etl_pipeline.security.logging.logger_config import logging
from etl_pipeline.security.entity.config_entity import DataIngestionConfig, DataValidationConfig
from etl_pipeline.security.entity.config_entity import TrainingPipelineConfig


import sys

if __name__ == '__main__':
    try:
        # Initialize the overall pipeline config
        trainingpipelineconfig = TrainingPipelineConfig()

        # Step 1: Data Ingestion
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        logging.info("Data Ingestion Artifact")
        print(data_ingestion_artifact)

        # Step 2: Data Validation
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Initiate the data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("data Validation Completed")
        print(data_validation_artifact)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
