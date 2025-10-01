import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple

from kidneyclassification.exception.exception import CustomException
from kidneyclassification.logging.logger import logging
from kidneyclassification.entity.config_entity import DataIngestionConfig
from kidneyclassification.entity.artifacts_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def _scan_image_dataset(self, root_dir: str) -> pd.DataFrame:
        try:
            logging.info(f"Scanning image dataset at: {root_dir}")
            if not os.path.isdir(root_dir):
                raise FileNotFoundError(f"Dataset root not found: {root_dir}")

            records = []
            class_names: List[str] = []
            for class_name in sorted(os.listdir(root_dir)):
                class_path = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                class_names.append(class_name)
                for fname in os.listdir(class_path):
                    fpath = os.path.join(class_path, fname)
                    if not os.path.isfile(fpath):
                        continue
                    # basic image extension check
                    if os.path.splitext(fpath)[1].lower() not in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".webp",
                    ]:
                        continue
                    records.append({"path": fpath, "label": class_name})

            if not records:
                raise ValueError("No image files found in dataset root. Ensure subfolders per class contain images.")

            df = pd.DataFrame(records)
            logging.info(f"Found {len(df)} images across classes: {class_names}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_to_feature_store(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Exporting image index to feature store CSV")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            df.to_csv(self.data_ingestion_config.feature_store_file_path, index=False)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def split_data_as_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logging.info("Stratified train/test split on image index")
            train_df, test_df = train_test_split(
                df,
                test_size=self.data_ingestion_config.test_size,
                random_state=42,
                stratify=df["label"],
            )
            train_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            test_dir = os.path.dirname(self.data_ingestion_config.test_file_path)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            train_df.to_csv(self.data_ingestion_config.train_file_path, index=False)
            test_df.to_csv(self.data_ingestion_config.test_file_path, index=False)
            logging.info(
                f"Saved train index to {self.data_ingestion_config.train_file_path} and test index to {self.data_ingestion_config.test_file_path}"
            )
            return train_df, test_df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            df = self._scan_image_dataset(self.data_ingestion_config.dataset_root)
            df = self.export_data_to_feature_store(df)
            self.split_data_as_train_test(df)
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path,
            )
            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
