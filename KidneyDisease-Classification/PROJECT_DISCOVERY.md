# PHASE 1: PROJECT DISCOVERY - Kidney Disease Classification DLOps Project

## 📋 PROJECT STRUCTURE OVERVIEW

### **Repository Layout**
```
KidneyDisease-Classification/
├── main.py                          # Training pipeline orchestrator
├── app.py                           # FastAPI inference server
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup configuration
├── README.md                        # Project documentation
│
├── kidneyclassification/           # Main package
│   ├── components/                  # Pipeline components
│   │   ├── data_ingestion.py       # Image dataset scanning & train/test split
│   │   ├── data_transformation.py  # Image preprocessing & normalization
│   │   └── model_trainer.py        # CNN model training with MLflow
│   ├── entity/                      # Configuration & artifacts entities
│   │   ├── config_entity.py        # Pipeline configuration classes
│   │   └── artifacts_entity.py     # Artifact data classes
│   ├── constants/                   # Global constants
│   │   └── __init__.py             # Image size, hyperparameters, paths
│   ├── exception/                   # Custom exception handling
│   │   └── exception.py            # CustomException class
│   ├── logging/                     # Logging configuration
│   │   └── logger.py               # File-based logging setup
│   └── utils/                       # Utility functions
│       ├── main_utils.py           # File I/O, pickle, numpy utilities
│       └── ml_utils.py             # Classification metrics calculation
│
├── Data/                            # Dataset (sibling directory)
│   └── CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/
│       ├── Normal/                  # 5,077 images
│       ├── Cyst/                    # 3,709 images
│       ├── Stone/                   # 1,377 images
│       └── Tumor/                   # 2,283 images
│
├── artifacts/                       # Pipeline artifacts (timestamped)
├── final_model/                     # Saved models & preprocessor
│   ├── kidney_cnn.h5               # Trained CNN model (existing)
│   └── preprocessor.pkl            # Preprocessor with class mapping
├── logs/                           # Training logs
└── templates/                      # FastAPI HTML templates
    ├── index.html
    └── result.html
```

---

## 🏥 DATASET STRUCTURE

**Dataset Type**: Image Classification (Medical CT Scans)  
**Location**: `../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/`

**Classes** (4 classes):
- `Normal` - 5,077 images
- `Cyst` - 3,709 images
- `Stone` - 1,377 images
- `Tumor` - 2,283 images

**Total**: ~12,446 images

**Split Strategy**: Stratified train/test split (80/20) with random_state=42

---

## 🧠 MODEL ARCHITECTURE

**Type**: Custom Convolutional Neural Network (CNN) using TensorFlow/Keras

**Architecture**:
```
Input: (224, 224, 3) RGB images
├── Conv2D(32 filters, 3x3) + ReLU + Padding
├── MaxPooling2D(2x2)
├── Conv2D(64 filters, 3x3) + ReLU + Padding
├── MaxPooling2D(2x2)
├── Conv2D(128 filters, 3x3) + ReLU + Padding
├── MaxPooling2D(2x2)
├── Dropout(0.3)
├── Flatten
├── Dense(256) + ReLU
├── Dropout(0.4)
└── Dense(4) + Softmax → 4-class classification
```

**Training Configuration**:
- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: `Adam` with `learning_rate=1e-3` (0.001)
- **Metrics**: `accuracy`
- **Batch Size**: 32
- **Epochs**: 10
- **Validation Split**: 10% (internal validation during training)
- **Image Size**: 224x224x3 (normalized to [0, 1])
- **Random State**: 42 (for reproducibility)

---

## 📜 TRAINING SCRIPTS

### **Main Training Pipeline** (`main.py`)
- Orchestrates end-to-end pipeline:
  1. **Data Ingestion** → Scans image dataset, creates train/test CSV splits
  2. **Data Transformation** → Preprocesses images, saves NPZ arrays + preprocessor
  3. **Model Training** → Builds CNN, trains with MLflow tracking, saves model

### **Component Scripts**:
- `components/data_ingestion.py`: `DataIngestion` class
- `components/data_transformation.py`: `DataTransformation` class  
- `components/model_trainer.py`: `ModelTrainer` class

---

## 🔮 INFERENCE SCRIPTS

### **FastAPI Application** (`app.py`)
- **Endpoints**:
  - `GET /` - API status
  - `GET /health` - Health check
  - `GET /train` - Trigger training pipeline via API
  - `POST /predict` - Image prediction endpoint

- **Preprocessing**: Same as training (resize to 224x224, normalize [0,1])
- **Model Loading**: Loads from `final_model/model.h5` or `final_model/kidney_cnn.h5`
- **Output**: Predicted class + confidence scores for all classes

---

## 📊 EXPERIMENT TRACKING

**Tool**: MLflow (`mlflow.tensorflow.autolog()`)

**Tracked Metrics**:
- Training/validation loss & accuracy (auto-logged by MLflow)
- Custom metrics: Precision, Recall, F1-score (macro average)
- Model artifacts saved to `artifacts/` with timestamps

**Logging**:
- File-based logging to `logs/` directory
- Timestamped log files (YYYY-MM-DD_HH-MM-SS.log)

---

## 🔄 CI/CD & PIPELINE COMPONENTS

**Orchestration**: 
- Modular pipeline with config entities
- Timestamped artifact directories for versioning
- Reproducible runs with fixed random seeds

**Not Present**:
- GitHub Actions / CI/CD pipelines
- Docker containerization (not in codebase)
- Kubernetes deployment configs

---

## 🚀 DEPLOYMENT COMPONENTS

**Type**: FastAPI REST API

**Features**:
- CORS middleware enabled
- File upload for image prediction
- HTML template rendering (templates/)
- Health check endpoint

**Deployment Readiness**:
- ✅ API server ready (`uvicorn`)
- ⚠️ Docker not configured
- ⚠️ No environment variable management (.env)
- ⚠️ No production WSGI configuration

---

## 📦 KEY DEPENDENCIES

- `tensorflow>=2.12` - Deep learning framework
- `numpy>=1.23` - Numerical operations
- `pandas>=1.5` - Data handling
- `scikit-learn>=1.2` - Train/test split & metrics
- `Pillow>=9.5` - Image processing
- `mlflow>=2.10` - Experiment tracking
- `fastapi>=0.110` - API framework
- `uvicorn>=0.23` - ASGI server
- `python-multipart>=0.0.9` - File upload support
- `Jinja2>=3.1` - Template rendering

---

## 🔑 DLOps FEATURES IDENTIFIED

✅ **Data Versioning**: Feature store CSV + timestamped artifacts  
✅ **Model Versioning**: Timestamped artifact directories  
✅ **Experiment Tracking**: MLflow integration  
✅ **Reproducibility**: Fixed random seeds, deterministic operations  
✅ **Pipeline Orchestration**: Modular component-based design  
⚠️ **Monitoring**: Logging present, but no production monitoring  
⚠️ **Scalability**: Single-machine training, API can scale horizontally

---

## 📝 SUMMARY

**Project Type**: End-to-end DLOps pipeline for medical image classification  
**Domain**: Healthcare AI - Kidney Disease Detection (CT Scans)  
**ML Task**: Multi-class image classification (4 classes)  
**Model**: Custom CNN with TensorFlow/Keras  
**Infrastructure**: Python-based pipeline + FastAPI inference server  
**Tracking**: MLflow + file-based logging

**Status**: Project structure is complete and production-ready architecture is in place. Model exists at `final_model/kidney_cnn.h5`. Ready for training validation and inference testing.
