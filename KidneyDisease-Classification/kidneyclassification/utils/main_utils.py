import os
import sys
import numpy as np
import pickle
import yaml
from kidneyclassification.exception.exception import CustomException
from kidneyclassification.logging.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        logging.info(f"Reading YAML file: {file_path}")
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CustomException(e, sys)


def write_yaml_file(file_path: str, data: dict) -> None:
    try:
        logging.info(f"Writing YAML file: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(data, file)
        logging.info(f"YAML written: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def save_numpy_array_data(data: np.ndarray, file_path: str) -> None:
    try:
        logging.info(f"Saving numpy array to: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            np.save(file, data)
        logging.info(f"Numpy array saved: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        logging.info(f"Loading numpy array from: {file_path}")
        with open(file_path, 'rb') as file:
            return np.load(file)
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info(f"Saving object to: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Object saved: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    try:
        logging.info(f"Loading object from: {file_path}")
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        logging.info(f"Object loaded: {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)
