# Kidney Disease Classification (DLOps)

This project mirrors the structure and logic of your existing Network Security MLOps project, adapted for CNN-based kidney disease image classification.

- Same folder structure, modular components, configs, logging, and scripts.
- Data ingestion loads images from a root folder with subfolders per class.
- Data transformation prepares TensorFlow `tf.data.Dataset` pipelines with augmentation and normalization.
- Model trainer builds and trains a TensorFlow/Keras CNN.
- Artifacts, logging, and configuration-driven runs are preserved.

## Quickstart

1. Place your dataset under `Data/` so that it contains subfolders per class:

```
Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/
  Normal/
  Cyst/
  Tumor/
  Stone/
```

2. Run the training pipeline:

```
python main.py
```

3. Trained model and preprocessor will be saved under `final_model/` and pipeline artifacts under `artifact/`.

## Project Structure

This mirrors your Network Security project:

- `kidneyclassification/components/` – `data_ingestion.py`, `data_transformation.py`, `model_trainer.py`
- `kidneyclassification/entity/` – `config_entity.py`, `artifacts_entity.py`
- `kidneyclassification/constants/` – global constants
- `kidneyclassification/logging/` – logger setup
- `kidneyclassification/exception/` – custom exception
- `kidneyclassification/utils/` – common utils (YAML, numpy, pickle)
- `main.py` – Orchestrates the pipeline
- `requirements.txt` – Dependencies

Adjust constants in `kidneyclassification/constants/__init__.py` if needed (e.g., image size, batch size, split ratios).
