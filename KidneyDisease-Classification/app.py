import os
import sys
from io import BytesIO
from typing import Dict

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from PIL import Image
import numpy as np
import tensorflow as tf

from kidneyclassification.logging.logger import logging
from kidneyclassification.exception.exception import CustomException
from kidneyclassification.components.data_ingestion import DataIngestion
from kidneyclassification.components.data_transformation import DataTransformation
from kidneyclassification.components.model_trainer import ModelTrainer
from kidneyclassification.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from kidneyclassification.utils.main_utils import load_object
from kidneyclassification.constants import IMAGE_HEIGHT, IMAGE_WIDTH


templates = Jinja2Templates(directory='./templates')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/', tags=['home'])
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/docs_redirect', tags=['auth'])
async def docs_redirect():
    return RedirectResponse(url='/docs')


@app.get('/train', tags=['train'])
async def train():
    try:
        logging.info("Starting training pipeline via API...")
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion = DataIngestion(DataIngestionConfig(training_pipeline_config))
        di_artifact = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation(di_artifact, DataTransformationConfig(training_pipeline_config))
        dt_artifact = data_transformation.initiate_data_transformation()

        model_trainer = ModelTrainer(ModelTrainerConfig(training_pipeline_config), dt_artifact)
        mt_artifact = model_trainer.initiate_model_trainer()
        return {"status": "success", "artifact": str(mt_artifact)}
    except Exception as e:
        raise CustomException(e, sys)


def _prepare_image(file_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(file_bytes)) as img:
            img = img.convert('RGB')
            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        raise CustomException(e, sys)


@app.post('/predict', tags=['predict'])
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        X = _prepare_image(contents)

        preprocessor = load_object(os.path.join('final_model', 'preprocessor.pkl'))
        model = tf.keras.models.load_model(os.path.join('final_model', 'model.h5'))
        probs = model.predict(X, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        # Map index to class name
        index_to_class: Dict[int, str] = {v: k for k, v in preprocessor.class_to_index.items()}
        pred_class = index_to_class.get(pred_idx, str(pred_idx))

        # Top-5 like display (or all classes if <=5)
        classes_sorted = sorted([(index_to_class[i], float(p)) for i, p in enumerate(probs)], key=lambda x: x[1], reverse=True)
        top = classes_sorted[: min(5, len(classes_sorted))]

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "pred_class": pred_class,
                "confidences": top,
            },
        )
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app_run(app, host='0.0.0.0', port=8000)
