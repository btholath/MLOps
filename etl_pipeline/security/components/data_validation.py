"""
Data Validation Script
-----------------------
This script is part of an ML pipeline. It performs validation tasks such as:
- Verifying that input data contains expected columns
- Checking for data drift using statistical tests (KS test)
- Writing data drift reports and validated datasets

Target Audience:
- Data Scientists: To validate data quality and statistical consistency
- ML Engineers: To automate and log validation steps in the pipeline
"""

from etl_pipeline.security.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from etl_pipeline.security.entity.config_entity import DataValidationConfig
from etl_pipeline.security.exception.exception import NetworkSecurityException
from etl_pipeline.security.logging.logger_config import logging
from etl_pipeline.security.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os, sys
from etl_pipeline.security.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    """
    Validates raw data based on:
    - Schema compliance (column count)
    - Statistical similarity (data drift detection)
    - Saves a drift report and validated data files

    Handles the validation of data within the data ingestion system.

    The `DataValidation` class provides functionalities to validate the
    ingested data by ensuring compliance with predefined schema constraints,
    including the number of columns and potential data drift detection.
    It supports reading data files, comparing datasets for consistency,
    and generating validation reports.

    :ivar data_ingestion_artifact: Contains information regarding the ingested
        data artifacts like file paths of train and test datasets.
    :type data_ingestion_artifact: DataIngestionArtifact
    :ivar data_validation_config: Configuration details required for data
        validation, including schema and report file paths.
    :type data_validation_config: DataValidationConfig
    :ivar _schema_config: The schema definition loaded from the schema file that
        defines data requirements, such as column constraints.
    :type _schema_config: dict
    """
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Read CSV data from the provided file path.
        Returns:
            pd.DataFrame: Parsed DataFrame
        Reads data from a CSV file and returns it as a pandas DataFrame.

        This static method reads the content of a CSV file using pandas and
        returns the data in DataFrame format. If any exception occurs during
        the reading process, it raises a custom exception with an error
        message and system information.

        :param file_path:"""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate the DataFrame against the expected number of columns.
        Returns:
            bool: True if column count matches schema

        Validates whether the number of columns in the provided dataframe matches the
        expected number of columns as defined in the schema configuration. Logs the
        required and actual number of columns for transparency. Returns `True` if the
        dataframe has the correct number of columns, otherwise `False`.

        :param dataframe: The pandas DataFrame to validate.
        :type dataframe: pd.DataFrame
        :return: A boolean indicating whether the number of columns matches the schema
                 configuration.
        :rtype: bool
        :raises NetworkSecurityException: If any exception occurs during the validation
                                          process.
        """
        try:
            number_of_columns = len(self._schema_config)
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        """
        Detect data drift using the Kolmogorov-Smirnov test.
        Writes a YAML report with drift status for each feature.

        Args:
            base_df (DataFrame): Baseline data (typically train set)
            current_df (DataFrame): Comparison data (typically test set)
            threshold (float): p-value threshold for identifying drift

        Returns:
            bool: True if no significant drift found, False otherwise

        Detects dataset drift between the base dataset and the current dataset using the Kolmogorov-Smirnov
        test for distribution comparison. This method evaluates whether the data from two compared datasets
        come from the same distribution for each column. If the detected p-value for any column is less than
        the threshold, it indicates data drift.

        The drift detection results, including p-values and drift status for each column, are stored in a
        drift report YAML file at the specified path in the data validation configuration.

        Features:
        - Compares each column in the datasets using the two-sample Kolmogorov-Smirnov test.
        - Determines drift based on a defined threshold for the p-value.
        - Saves drift detection results in a YAML file including details of detected drift status and p-values.
        - Ensures directory creation for the drift report file path if it does not exist.

        :param base_df: Base dataset used for reference with the original data distribution. Must have the
            same schema as `current_df`.
        :type base_df: pandas.DataFrame
        :param current_df: Current dataset to be compared against the base dataset for drift detection. Must
            have the same schema as `base_df`.
        :type current_df: pandas.DataFrame
        :param threshold: The p-value threshold for determining drift. If the p-value for a column is less
            than the threshold, drift is considered to have occurred. Defaults to 0.05.
        :type threshold: float
        :return: Boolean value indicating the overall drift status. Returns `True` if no drift is detected
            in any column (all p-values are above the threshold); otherwise, returns `False`.
        :rtype: bool
        :raises NetworkSecurityException: Raised if any exception occurs during the drift detection process.
        """
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found

                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Perform full validation pipeline:
        - Load train/test data
        - Validate schema compliance
        - Detect data drift
        - Write validated CSVs

        Returns:
            DataValidationArtifact: metadata output of validation process

        Initiates data validation process that includes reading train and test datasets,
        validating their columns, detecting dataset drift, and saving validated datasets
        to specified paths. A `DataValidationArtifact` object is created and returned
        to summarize the validation results.

        :raises NetworkSecurityException: Raised if any exception occurs during the
            data validation process.

        :return: A `DataValidationArtifact` object containing the validation results,
            such as validation status, file paths for validated datasets, and drift
            report path.
        :rtype: DataValidationArtifact
        """
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            ## read the data from train and test
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            ## validate number of columns

            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"Train dataframe does not contain all columns.\n"
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"Test dataframe does not contain all columns.\n"

                ## lets check datadrift
            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True

            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)



