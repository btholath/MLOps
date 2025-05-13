"""
Model Training Script
----------------------
This module is responsible for training classification models on the transformed data.
It includes:
- Loading transformed training and testing datasets
- Training multiple ML classifiers with hyperparameter tuning
- Selecting and logging the best model using MLflow on DagsHub
- Saving the final model and tracking experiment metadata

Target Audience:
- Data Scientists: Understand model comparison and evaluation metrics
- MLOps Engineers: Enable reproducibility, tracking, and deployment
"""

import os
import sys

from etl_pipeline.security.exception.exception import NetworkSecurityException
from etl_pipeline.security.logging.logger_config import logging

from etl_pipeline.security.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from etl_pipeline.security.entity.config_entity import ModelTrainerConfig

from etl_pipeline.security.utils.ml_utils.model.estimator import NetworkModel
from etl_pipeline.security.utils.main_utils.utils import save_object, load_object
from etl_pipeline.security.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from etl_pipeline.security.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
from urllib.parse import urlparse

import dagshub

# dagshub.init(repo_owner='bijutholath', repo_name='MLOps', mlflow=True)

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/krishnaik06/networksecurity.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "krishnaik06"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "7104284f1bb44ece21e0e2adb4e36a250ae3251f"


class ModelTrainer:
    """
    Orchestrates the model training and evaluation workflow.
    It compares multiple classifiers and selects the best based on F1-score.
    Also logs the experiment via MLflow on DagsHub.

    Handles training machine learning models using various algorithms, tracking experiments with MLflow,
    and saving trained models for subsequent use.

    This class is designed to automate the process of model training, evaluation, experiment tracking, and
    artifacts generation. It leverages a variety of machine learning algorithms and evaluates them
    based on predefined performance metrics. The best-performing model is identified, tracked via MLflow,
    and saved for deployment. This module plays a critical role in standardizing the training pipeline.

    :ivar model_trainer_config: Configuration for model training, including paths and parameters.
    :type model_trainer_config: ModelTrainerConfig
    :ivar data_transformation_artifact: Artifact containing data transformation results, such as processed data paths.
    :type data_transformation_artifact: DataTransformationArtifact
    """
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classificationmetric):
        """
        Logs experiment metrics and model into MLflow.
        Args:
            best_model: The model to log
            classificationmetric: Object containing F1, precision, recall

        Logs model training metrics and registers the model in MLFlow model registry.

        This function takes a trained machine learning model and its associated
        classification metrics, logs the metrics in the MLFlow tracking server,
        and registers the model in the MLFlow model registry. If the tracking
        server storage type is not "file", the model is registered with a specified
        name. Otherwise, only logging is performed.

        :param best_model: The trained machine learning model to be logged and
            optionally registered with the MLFlow model registry.
        :type best_model: Any
        :param classificationmetric: An object containing classification metrics
            such as f1_score, precision_score, and recall_score to be logged
            in MLFlow.
        :type classificationmetric: Any
        :return: None
        """
        mlflow.set_registry_uri("https://dagshub.com/krishnaik06/networksecurity.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            mlflow.sklearn.log_model(best_model, "model")
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model)
            else:
                mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, X_train, y_train, x_test, y_test):
        """
        Trains multiple classification models and selects the best one.
        Logs training and testing metrics and saves the model.

        Returns:
            ModelTrainerArtifact: metadata with paths and scores

        Trains the machine learning models provided in the models dictionary using the supplied training
        and testing datasets. The function evaluates each model with the specified hyperparameters,
        logs the best model and its performance metrics using MLflow, and saves the pre-trained
        model for future usage.

        :param X_train: Training dataset features.
        :type X_train: pandas.DataFrame or numpy.ndarray
        :param y_train: Training dataset target values.
        :type y_train: pandas.Series or numpy.ndarray
        :param x_test: Testing dataset features for evaluation purposes.
        :type x_test: pandas.DataFrame or numpy.ndarray
        :param y_test: Testing dataset target values for evaluation purposes.
        :type y_test: pandas.Series or numpy.ndarray
        :return: Artifact containing the trained model's file path, and performance metrics for both
                 training and testing datasets.
        :rtype: ModelTrainerArtifact
        """
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }
        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest": {
                # 'criterion':['gini', 'entropy', 'log_loss'],

                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8, 16, 32, 128, 256]
            },
            "Gradient Boosting": {
                # 'loss':['log_loss', 'exponential'],
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                'learning_rate': [.1, .01, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }

        }
        model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=x_test, y_test=y_test,
                                             models=models, param=params)

        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred = best_model.predict(X_train)

        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        ## Track the experiements with mlflow
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=NetworkModel)
        # model pusher
        save_object("final_model/model.pkl", best_model)

        ## Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
            )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Entry point for model training. Loads data, trains, tracks, and returns artifact.
        Returns:
            ModelTrainerArtifact

        Initiates the model training process by loading the transformed training and
        testing data, splitting them into features and target variables, and invoking
        the model training functionality.

        :return: An artifact containing information about the trained model.
        :rtype: ModelTrainerArtifact

        :raises NetworkSecurityException: Raised when an exception occurs during the
            initiation of model training.
        """
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact


        except Exception as e:
            raise NetworkSecurityException(e, sys)