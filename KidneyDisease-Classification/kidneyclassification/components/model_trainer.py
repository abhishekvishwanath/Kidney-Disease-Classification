import os
import sys
import numpy as np
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import mlflow
import mlflow.tensorflow

from kidneyclassification.exception.exception import CustomException
from kidneyclassification.logging.logger import logging
from kidneyclassification.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from kidneyclassification.entity.config_entity import ModelTrainerConfig
from kidneyclassification.utils.main_utils import load_object, save_object
from kidneyclassification.utils.ml_utils import get_classification_metrics
from kidneyclassification.constants import (
    EPOCHS,
    LEARNING_RATE,
)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.random_state = 42
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)
        except Exception as e:
            raise CustomException(e, sys)

    def _load_npz(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            data = np.load(path)
            X = data['X']
            y = data['y']
            return X, y
        except Exception as e:
            raise CustomException(e, sys)

    def _build_cnn(self, input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
        try:
            model = models.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(num_classes, activation='softmax'),
            ])
            opt = optimizers.Adam(learning_rate=LEARNING_RATE)
            model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        try:
            # Load preprocessor to get class mapping and image size
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            num_classes = len(preprocessor.class_to_index)
            input_shape = preprocessor.image_size

            logging.info(f"Building CNN with input_shape={input_shape}, num_classes={num_classes}")
            model = self._build_cnn(input_shape=input_shape, num_classes=num_classes)

            logging.info("Starting CNN training with MLflow autologging")
            # Enable MLflow autologging for TensorFlow/Keras
            try:
                mlflow.tensorflow.autolog()
            except Exception as e:
                logging.warning(f"Failed to enable MLflow autologging: {e}")

            with mlflow.start_run():
                history = model.fit(
                    X_train,
                    y_train,
                    validation_split=0.1,
                    epochs=EPOCHS,
                    shuffle=True,
                    verbose=1,
                )

            logging.info("Evaluating model on test set")
            test_probs = model.predict(X_test, verbose=0)
            y_pred = np.argmax(test_probs, axis=1)

            # Compute metrics (macro average for multi-class)
            test_scores = get_classification_metrics(y_test, y_pred, average="macro")

            # Compute train metrics on training predictions
            train_probs = model.predict(X_train, verbose=0)
            y_train_pred = np.argmax(train_probs, axis=1)
            train_scores = get_classification_metrics(y_train, y_train_pred, average="macro")

            # Save model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            model.save(self.model_trainer_config.trained_model_file_path)
            # Also save under final_model for parity with original project
            os.makedirs("final_model", exist_ok=True)
            model.save(os.path.join("final_model", "model.h5"))

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_scores,
                test_metric_artifact=test_scores,
            )
            logging.info(f"Model training completed with Test F1 (macro): {test_scores.f1_score:.4f}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model trainer")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info("Loading transformed training and test data")
            X_train, y_train = self._load_npz(train_file_path)
            X_test, y_test = self._load_npz(test_file_path)

            return self.train_model(X_train, y_train, X_test, y_test)
        except Exception as e:
            raise CustomException(e, sys)
