"""
Data Transformation Script
---------------------------
This script handles transforming raw training and testing datasets to make them suitable for model training.
It includes:
- Missing value imputation using KNN
- Train/test data transformation
- Saving the transformed data as NumPy arrays and the preprocessing pipeline

Target Audience:
- Data Scientists: Preprocessing setup and training/test transformations
- ML Engineers: Artifact creation, pipeline consistency
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from etl_pipeline.security.constants.training_pipeline import TARGET_COLUMN
from etl_pipeline.security.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from etl_pipeline.security.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from etl_pipeline.security.entity.config_entity import DataTransformationConfig
from etl_pipeline.security.exception.exception import NetworkSecurityException
from etl_pipeline.security.logging.logger_config import logging
from etl_pipeline.security.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    """
    Orchestrates the data transformation process:
    - Reads validated datasets
    - Applies KNN imputation to handle missing values
    - Combines input and target features into NumPy arrays
    - Saves preprocessing pipeline and transformed data

    Handles data transformation process including preprocessing, transforming,
    and saving the data for machine learning pipelines.

    This class is responsible for applying preprocessing steps such as imputation
    and feature transformations on the dataset. It reads raw input data, applies
    transformations using a pipeline, and saves the processed data along with
    the preprocessing object. Data transformation artifacts, including the
    transformed dataset paths and the preprocessor object file path, are
    prepared and returned.

    :ivar data_validation_artifact: Contains validation information of the input
        train and test data files.
    :type data_validation_artifact: DataValidationArtifact
    :ivar data_transformation_config: Contains transformation configuration details
        such as file paths to save transformed outputs and objects.
    :type data_transformation_config: DataTransformationConfig
    """
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Reads CSV file from a specified path.
        Args:
            file_path (str): Path to CSV
        Returns:
            pd.DataFrame: Loaded dataset

        Reads data from a CSV file and returns it as a pandas DataFrame.

        This method attempts to read the specified file at the provided path using pandas'
        `read_csv` function. If the file cannot be opened or read for any reason, it raises
        a custom `NetworkSecurityException` carrying the original exception information.

        :param file_path: Path to the CSV file to be read.
        :type file_path: str

        :return: A pandas DataFrame containing the data from the file.
        :rtype: pd.DataFrame

        :raises NetworkSecurityException: If reading the file fails for any reason.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(cls) -> Pipeline:
        """
        Creates a data transformation pipeline.
        Uses KNN imputer initialized with params from config.

        Returns:
            Pipeline: scikit-learn pipeline with KNNImputer

        Creates and returns a data transformation pipeline object for preprocessing.

        This method initializes a KNNImputer with specified parameters and incorporates it into
        a pipeline. The pipeline can then be used for handling missing values in datasets during
        transformation processes.

        :raises NetworkSecurityException: If an exception occurs during KNNImputer initialization
            or pipeline creation.
        :return: A scikit-learn pipeline that contains the data transformation steps.
        :rtype: Pipeline
        """
        logging.info(
            "Entered get_data_trnasformer_object method of Trnasformation class"
        )
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(
                f"Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
            processor: Pipeline = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Main transformation process:
        - Loads validated train and test data
        - Separates input/target features
        - Applies preprocessing
        - Saves NumPy arrays and pipeline object

        Returns:
            DataTransformationArtifact: Metadata containing file paths

        Initiates the data transformation phase by transforming training and testing datasets.

        Detailed Summary:
        This function is responsible for performing the data transformation process, which
        includes reading the data, preparing input and target features, applying preprocessing,
        generating transformed feature sets, and saving the transformed datasets and
        preprocessor object for future use. The process involves dropping target column
        from feature sets, replacing specific target values (-1 with 0), fitting a preprocessing
        object, transforming data, and saving the outputs.

        :return: An artifact of type DataTransformationArtifact containing paths to the
                 transformed training data, transformed testing data, and the preprocessor
                 object used for transformation.
        :rtype: DataTransformationArtifact

        :raises NetworkSecurityException: If any error occurs during the transformation process,
                                          the exception is raised with details of the encountered
                                          error and propagated to the caller for further handling.
        """
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Prepare training data
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            # Prepare test data
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)

            # Fit on training input and transform both train/test
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Combine inputs and targets
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Save transformed arrays and pipeline object
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr, )
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object, )

            save_object("final_model/preprocessor.pkl", preprocessor_object, )

            # preparing artifacts
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact



        except Exception as e:
            raise NetworkSecurityException(e, sys)
