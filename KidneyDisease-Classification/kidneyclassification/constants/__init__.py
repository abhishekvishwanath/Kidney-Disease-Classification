import os

# Project constants mirroring the Network Security project
PIPELINE_NAME = "Kidney Disease Classification"
ARTIFACT_DIR = "artifact"

# Data file names (ingestion will create these CSVs with image paths and labels)
FILE_NAME = "image_index.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TARGET_COLUMN = "label"

# Ingestion configs
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION = 0.2

# Root dataset directory where class-subfolders reside
# You can change this to point to your dataset
# Default assumes sibling Data folder next to this project and inner folder holds class subfolders
DEFAULT_DATASET_ROOT = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "..",
        "Data",
        "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
        "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
    )
)

# Data transformation constants
DATA_TRANSFORMATION_DIR = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR = "transformed"
DATA_TRANSFORMATION_TRANSFORMER_OBJECT = "preprocessor.pkl"

# Image processing params
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Model trainer constants
MODEL_TRAINER_DIR = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_NAME = "kidney_cnn.h5"
MODEL_TRAINER_EXPECTED_SCORE = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD = 0.1

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_STATE = 42
