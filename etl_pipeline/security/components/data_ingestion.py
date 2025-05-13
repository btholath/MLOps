"""
Data Ingestion Script
----------------------
This script is part of an end-to-end machine learning pipeline. It is responsible for:
- Extracting data from a MongoDB collection
- Exporting the data into a feature store (a raw CSV file)
- Splitting the data into training and testing datasets

This module uses environment variables for MongoDB credentials, structured configuration for paths,
and logs key operations.

Intended Audience:
- Data Scientists: To understand and customize data loading for ML pipelines
- Developers: To integrate with the rest of the ML system
"""

import os
import sys

# Add parent directory to Python path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from etl_pipeline.security.exception.exception import NetworkSecurityException
from etl_pipeline.security.logging.logger_config import logging
## configuration of the Data Ingestion Config
from etl_pipeline.security.entity.config_entity import DataIngestionConfig
from etl_pipeline.security.entity.artifact_entity import DataIngestionArtifact

import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    """
        Handles the full data ingestion process including:
        - Connecting to MongoDB
        - Exporting data into CSV (feature store)
        - Splitting into train and test files
    """
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self):
        """
        Connect to MongoDB and export the specified collection into a pandas DataFrame.
        Returns:
            pd.DataFrame: Cleaned data frame without MongoDB ObjectIDs
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise NetworkSecurityException

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        """
        Save the entire dataset into the feature store path as a CSV file.
        Args:
            dataframe (pd.DataFrame): Full dataset to save
        Returns:
            pd.DataFrame: The same dataframe passed in
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Split the full dataset into train and test sets based on the configured ratio.
        Saves both to disk.
        Args:
            dataframe (pd.DataFrame): The dataset to split
        """
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info(f"Exported train and test file path.")


        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        """
        Main method to initiate the data ingestion process.
        Returns:
            DataIngestionArtifact: Paths to the train and test files
        """
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:
            raise NetworkSecurityException