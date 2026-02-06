# PHASE 6: ARCHITECTURE DIAGRAM - Kidney Disease Classification DLOps Pipeline

## 🏗️ END-TO-END SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KIDNEY DISEASE CLASSIFICATION                        │
│                            DLOps PIPELINE ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════

1. DATA INGESTION LAYER
═══════════════════════════════════════════════════════════════════════════════

    [CT Scan Dataset]
           │
           │ 12,446 images
           │ 4 classes: Normal, Cyst, Stone, Tumor
           ▼
    ┌─────────────────────────────────────────────────────┐
    │         DataIngestion Component                     │
    │  ────────────────────────────────────────────      │
    │  • Scan image dataset (class-subfolders)            │
    │  • Build image index DataFrame                      │
    │  • Create feature store CSV                         │
    │  • Stratified train/test split (80/20)              │
    │  • Save train.csv, test.csv                         │
    └─────────────────────────────────────────────────────┘
           │
           │ DataIngestionArtifact
           ▼
    [artifact/{timestamp}/data_ingestion/]
           ├── feature_store/image_index.csv
           ├── ingested/train.csv (9,956 samples)
           └── ingested/test.csv (2,490 samples)

═══════════════════════════════════════════════════════════════════════════════

2. DATA TRANSFORMATION LAYER
═══════════════════════════════════════════════════════════════════════════════

    [train.csv, test.csv]
           │
           ▼
    ┌─────────────────────────────────────────────────────┐
    │      DataTransformation Component                   │
    │  ────────────────────────────────────────────      │
    │  • Build class mapping (Cyst:0, Normal:1, etc.)    │
    │  • Load images from paths                          │
    │  • Resize to 224x224 (or 128x128 for existing)     │
    │  • Normalize pixels: /255.0 → [0, 1]               │
    │  • Convert to numpy arrays (float32)               │
    │  • Save as NPZ files (compressed)                  │
    │  • Create preprocessor with class mapping           │
    └─────────────────────────────────────────────────────┘
           │
           │ DataTransformationArtifact
           ▼
    [artifact/{timestamp}/data_transformation/]
           ├── preprocessor.pkl (class mapping + image size)
           ├── transformed/train_data.npz (X_train, y_train)
           └── transformed/test_data.npz (X_test, y_test)

═══════════════════════════════════════════════════════════════════════════════

3. MODEL TRAINING PIPELINE
═══════════════════════════════════════════════════════════════════════════════

    [train_data.npz, test_data.npz]
           │
           ▼
    ┌─────────────────────────────────────────────────────┐
    │           ModelTrainer Component                    │
    │  ────────────────────────────────────────────      │
    │  • Load preprocessor (get input_shape, num_classes)│
    │  • Build CNN architecture                          │
    │     ├── Conv2D(32) → MaxPool                       │
    │     ├── Conv2D(64) → MaxPool                       │
    │     ├── Conv2D(128) → MaxPool                      │
    │     ├── Dropout(0.3)                               │
    │     ├── Dense(256) → Dropout(0.4)                  │
    │     └── Dense(4) → Softmax                         │
    │  • Compile: Adam(lr=1e-3), sparse_categorical...  │
    │  • Train: 10 epochs, batch_size=32                 │
    │  • MLflow autologging (metrics, params, artifacts) │
    │  • Evaluate on test set                            │
    │  • Calculate metrics (Precision, Recall, F1)       │
    │  • Save model + preprocessor                        │
    └─────────────────────────────────────────────────────┘
           │
           │ ModelTrainerArtifact
           ▼
    [artifact/{timestamp}/model_trainer/]
           └── kidney_cnn.h5 (97 MB)

    [final_model/] (latest)
           ├── kidney_cnn.h5 (production model)
           └── preprocessor.pkl (class mapping)

    [MLflow Tracking]
           ├── Runs: metrics, parameters, artifacts
           ├── Backend: SQLite database
           └── Location: ./mlruns/

═══════════════════════════════════════════════════════════════════════════════

4. MODEL REGISTRY & VERSIONING
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────┐
    │            Model Versioning Strategy                │
    │  ────────────────────────────────────────────      │
    │  • Timestamped artifacts: artifact/{timestamp}/    │
    │  • Latest model: final_model/kidney_cnn.h5         │
    │  • Preprocessor versioning: preprocessor.pkl       │
    │  • MLflow run tracking: run_id, experiment_id      │
    │  • Artifact metadata: model size, metrics          │
    └─────────────────────────────────────────────────────┘

    [Model Registry (Future Enhancement)]
           ├── MLflow Model Registry (staging/production)
           ├── Model promotion workflow
           └── A/B testing capabilities

═══════════════════════════════════════════════════════════════════════════════

5. INFERENCE SERVICE
═══════════════════════════════════════════════════════════════════════════════

    [FastAPI Application - app.py]
           │
           ├── GET / → API status
           ├── GET /health → Health check
           ├── GET /train → Trigger training pipeline
           │
           └── POST /predict → Image classification
                  │
                  ▼
           ┌─────────────────────────────────────────────────┐
           │           Inference Pipeline                     │
           │  ────────────────────────────────────────      │
           │  1. Receive image bytes (UploadFile)            │
           │  2. Load preprocessor (class mapping)           │
           │  3. Preprocess image:                           │
           │     • Convert to RGB (PIL)                      │
           │     • Resize to 224x224 (or 128x128)            │
           │     • Normalize: /255.0 → [0, 1]                │
           │     • Expand dimensions: (1, H, W, 3)           │
           │  4. Load model (final_model/kidney_cnn.h5)      │
           │  5. Predict: model.predict(X)                   │
           │  6. Get class probabilities                     │
           │  7. Map index → class name                      │
           │  8. Return: pred_class, confidence, top-5       │
           └─────────────────────────────────────────────────┘
                  │
                  ▼
           [JSON Response]
           {
             "pred_class": "Normal",
             "confidences": [
               ["Normal", 1.0000],
               ["Cyst", 0.0000],
               ["Stone", 0.0000],
               ["Tumor", 0.0000]
             ]
           }

    [Uvicorn ASGI Server]
           │
           └── Host: 0.0.0.0, Port: 8000
                  │
                  ▼
           [Production Deployment]
           ├── Docker containerization (future)
           ├── Kubernetes deployment (future)
           └── Load balancer (future)

═══════════════════════════════════════════════════════════════════════════════

6. EXPERIMENT TRACKING & MONITORING
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────┐
    │            MLflow Tracking Backend                  │
    │  ────────────────────────────────────────────      │
    │  • Automatic metric logging (loss, accuracy)        │
    │  • Parameter tracking (epochs, batch_size, lr)      │
    │  • Artifact logging (model files, plots)            │
    │  • Run comparison and visualization                 │
    │  • Local SQLite database                            │
    └─────────────────────────────────────────────────────┘
           │
           ▼
    [mlruns/] directory
           ├── experiments/
           │   └── {experiment_id}/
           │       └── runs/
           │           └── {run_id}/
           │               ├── metrics/
           │               ├── params/
           │               └── artifacts/
           └── 0/ (default experiment)

    ┌─────────────────────────────────────────────────────┐
    │              File-Based Logging                     │
    │  ────────────────────────────────────────────      │
    │  • Application logs: logs/{timestamp}.log           │
    │  • Timestamped log files                            │
    │  • INFO, ERROR, WARNING levels                      │
    └─────────────────────────────────────────────────────┘

    [Future: Production Monitoring]
           ├── Prometheus metrics (API latency, throughput)
           ├── Grafana dashboards (visualization)
           ├── Evidently AI (data drift detection)
           └── Alerting (model degradation alerts)

═══════════════════════════════════════════════════════════════════════════════

7. CI/CD FLOW (FUTURE ENHANCEMENT)
═══════════════════════════════════════════════════════════════════════════════

    [GitHub Repository]
           │
           ├── Code changes
           │
           ▼
    [GitHub Actions CI/CD]
           │
           ├── Run tests (unit, integration)
           ├── Lint code (pylint, black)
           ├── Build Docker image
           ├── Run training pipeline (optional)
           ├── Evaluate model metrics
           └── Deploy to staging/production
                  │
                  ▼
           [Kubernetes Cluster]
                  ├── Deployment pods
                  ├── Service load balancer
                  └── Ingress controller

═══════════════════════════════════════════════════════════════════════════════

8. DEPLOYMENT LAYER
═══════════════════════════════════════════════════════════════════════════════

    [Current: Local Development]
           │
           └── python app.py → uvicorn (localhost:8000)

    [Future: Production Deployment]
           │
           ├── Docker Container
           │   └── Dockerfile with model baked in
           │
           ├── Kubernetes Deployment
           │   ├── Deployment: replicas, resource limits
           │   ├── Service: LoadBalancer/NodePort
           │   └── ConfigMap: environment variables
           │
           ├── Cloud Deployment (AWS/GCP/Azure)
           │   ├── AWS: ECS/EKS + S3 (models)
           │   ├── GCP: Cloud Run / GKE + GCS
           │   └── Azure: AKS + Blob Storage
           │
           └── Edge Deployment (Future)
               └── TensorFlow Lite / ONNX Runtime

═══════════════════════════════════════════════════════════════════════════════

9. DATA FLOW SUMMARY
═══════════════════════════════════════════════════════════════════════════════

    RAW DATA → INGESTION → TRANSFORMATION → TRAINING → MODEL → INFERENCE → API

    ┌────────┐    ┌──────────┐    ┌──────────────┐    ┌────────┐    ┌──────┐
    │ Images │ → │ Ingest   │ → │ Transform    │ → │ Train  │ → │ Model│
    │        │    │ (CSVs)   │    │ (NPZ arrays) │    │ (CNN)  │    │      │
    └────────┘    └──────────┘    └──────────────┘    └────────┘    └──────┘
                                                                         │
                                                                         ▼
                                                                  ┌──────────┐
                                                                  │ Inference│
                                                                  │ (FastAPI)│
                                                                  └──────────┘

═══════════════════════════════════════════════════════════════════════════════

10. KEY COMPONENTS & TECHNOLOGIES
═══════════════════════════════════════════════════════════════════════════════

    • Python 3.13
    • TensorFlow/Keras 2.20.0
    • FastAPI 0.128.0
    • MLflow 3.8.1
    • NumPy, Pandas, Pillow
    • scikit-learn
    • Uvicorn (ASGI server)

    Pipeline Components:
    • kidneyclassification/components/data_ingestion.py
    • kidneyclassification/components/data_transformation.py
    • kidneyclassification/components/model_trainer.py

    Entity Classes:
    • kidneyclassification/entity/config_entity.py
    • kidneyclassification/entity/artifacts_entity.py

    Utilities:
    • kidneyclassification/utils/main_utils.py
    • kidneyclassification/utils/ml_utils.py

═══════════════════════════════════════════════════════════════════════════════

**Architecture Status**: ✅ **DOCUMENTED - PRODUCTION-READY FOUNDATION**

---

**Documentation Date**: January 18, 2026  
**Architecture Version**: 1.0  
**Status**: End-to-end pipeline documented and validated
