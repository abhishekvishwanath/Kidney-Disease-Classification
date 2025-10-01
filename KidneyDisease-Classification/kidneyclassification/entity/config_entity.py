import os
import sys
from datetime import datetime
from kidneyclassification import constants
from kidneyclassification.exception.exception import CustomException


class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")):
        try:
            self.timestamp = timestamp
            self.pipeline_name = constants.PIPELINE_NAME
            self.artifact_name = constants.ARTIFACT_DIR
            self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        except Exception as e:
            raise CustomException(e, sys)


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_ingestion_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                constants.DATA_INGESTION_DIR_NAME,
            )
            # dataset root with subfolders per class
            self.dataset_root = constants.DEFAULT_DATASET_ROOT
            self.feature_store_file_path = os.path.join(
                self.data_ingestion_dir,
                constants.DATA_INGESTION_FEATURE_STORE_DIR,
                constants.FILE_NAME,
            )
            self.train_file_path = os.path.join(
                self.data_ingestion_dir,
                constants.DATA_INGESTION_INGESTED_DIR,
                constants.TRAIN_FILE_NAME,
            )
            self.test_file_path = os.path.join(
                self.data_ingestion_dir,
                constants.DATA_INGESTION_INGESTED_DIR,
                constants.TEST_FILE_NAME,
            )
            self.test_size = constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        except Exception as e:
            raise CustomException(e, sys)


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir, constants.DATA_TRANSFORMATION_DIR
        )
        self.transformed_object_file_path = os.path.join(
            self.data_transformation_dir, constants.DATA_TRANSFORMATION_TRANSFORMER_OBJECT
        )
        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir,
            constants.DATA_TRANSFORMATION_TRANSFORMED_DIR,
            constants.TRAIN_FILE_NAME.replace(".csv", "_data.npz"),
        )
        self.transformed_test_file_path = os.path.join(
            self.data_transformation_dir,
            constants.DATA_TRANSFORMATION_TRANSFORMED_DIR,
            constants.TEST_FILE_NAME.replace(".csv", "_data.npz"),
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config
        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir, constants.MODEL_TRAINER_DIR
        )
        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir, constants.MODEL_TRAINER_TRAINED_MODEL_NAME
        )
        self.expected_score = constants.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = (
            constants.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD
        )
