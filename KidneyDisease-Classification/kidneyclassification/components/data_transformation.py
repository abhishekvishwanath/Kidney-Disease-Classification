import os
import sys
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Tuple

from kidneyclassification.exception.exception import CustomException
from kidneyclassification.logging.logger import logging
from kidneyclassification.entity.artifacts_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from kidneyclassification.entity.config_entity import DataTransformationConfig
from kidneyclassification.constants import (
    TARGET_COLUMN,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
)
from kidneyclassification.utils.main_utils import save_object


class SimplePreprocessor:
    """
    Lightweight preprocessor to carry class mappings and image size.
    """

    def __init__(self, class_to_index: Dict[str, int], image_size: Tuple[int, int, int]):
        self.class_to_index = class_to_index
        self.image_size = image_size


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

    def _read_csv(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def _build_class_mapping(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, int]:
        try:
            classes = sorted(set(train_df[TARGET_COLUMN].unique()).union(set(test_df[TARGET_COLUMN].unique())))
            class_to_index = {c: i for i, c in enumerate(classes)}
            logging.info(f"Class mapping: {class_to_index}")
            return class_to_index
        except Exception as e:
            raise CustomException(e, sys)

    def _load_and_preprocess_images(self, df: pd.DataFrame, class_to_index: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        try:
            n = len(df)
            X = np.zeros((n, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.float32)
            y = np.zeros((n,), dtype=np.int64)
            for i, row in enumerate(df.itertuples(index=False)):
                path = getattr(row, 'path')
                label_name = getattr(row, TARGET_COLUMN)
                label_idx = class_to_index[label_name]
                # Load image
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize 0-1
                X[i] = arr
                y[i] = label_idx
                if i % 500 == 0:
                    logging.info(f"Processed {i}/{n} images...")
            return X, y
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = self._read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = self._read_csv(self.data_ingestion_artifact.test_file_path)

            # Ensure required columns exist
            if 'path' not in train_df.columns or TARGET_COLUMN not in train_df.columns:
                raise CustomException(f"Train CSV must contain 'path' and '{TARGET_COLUMN}' columns", sys)
            if 'path' not in test_df.columns or TARGET_COLUMN not in test_df.columns:
                raise CustomException(f"Test CSV must contain 'path' and '{TARGET_COLUMN}' columns", sys)

            class_to_index = self._build_class_mapping(train_df, test_df)

            X_train, y_train = self._load_and_preprocess_images(train_df, class_to_index)
            X_test, y_test = self._load_and_preprocess_images(test_df, class_to_index)

            # Save datasets as NPZ
            transformed_dir = os.path.dirname(self.data_transformation_config.transformed_train_file_path)
            os.makedirs(transformed_dir, exist_ok=True)
            np.savez_compressed(self.data_transformation_config.transformed_train_file_path, X=X_train, y=y_train)
            np.savez_compressed(self.data_transformation_config.transformed_test_file_path, X=X_test, y=y_test)
            logging.info(
                f"Saved transformed train to {self.data_transformation_config.transformed_train_file_path} and test to {self.data_transformation_config.transformed_test_file_path}"
            )

            # Save preprocessor with class mapping and image size
            preprocessor = SimplePreprocessor(
                class_to_index=class_to_index,
                image_size=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
            )
            # Save to final_model and to artifact path to mirror original project
            save_object(os.path.join("final_model", "preprocessor.pkl"), preprocessor)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
