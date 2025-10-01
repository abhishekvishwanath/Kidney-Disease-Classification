from kidneyclassification.components.data_ingestion import DataIngestion
from kidneyclassification.components.data_transformation import DataTransformation
from kidneyclassification.components.model_trainer import ModelTrainer
from kidneyclassification.exception.exception import CustomException
from kidneyclassification.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from kidneyclassification.logging.logger import logging
import sys

if __name__ == "__main__":
    try:
        logging.info("Starting Kidney Disease Classification ML Pipeline")

        # Data Ingestion
        logging.info("Starting data ingestion stage")
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed: {data_ingestion_artifact}")

        # Data Transformation
        logging.info("Starting data transformation stage")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_ingestion_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data transformation completed: {data_transformation_artifact}")

        # Model Training
        logging.info("Starting model training stage")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info(f"Model training completed: {model_trainer_artifact}")

        logging.info("Kidney Disease Classification ML Pipeline completed successfully!")

    except Exception as e:
        raise CustomException(e, sys)
