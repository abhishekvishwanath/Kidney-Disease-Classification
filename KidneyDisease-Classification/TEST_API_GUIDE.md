# 🧪 FastAPI Testing Guide - Kidney Disease Classification

## 🚀 Server Status

The FastAPI server is running on: **http://localhost:8000**

---

## 📋 How to Test with Sample Image

### **Method 1: Using Swagger UI (Recommended)**

1. **Open Swagger UI**:
   - Go to: **http://localhost:8000/docs**
   - This opens an interactive API documentation interface

2. **Test the `/predict` endpoint**:
   - Scroll down to find the **`POST /predict`** endpoint
   - Click on it to expand
   - Click the **"Try it out"** button

3. **Upload a sample image**:
   - Click **"Choose File"** button
   - Select a sample image from: `../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/`
   - Example paths:
     - `../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Normal/Normal- (2158).jpg`
     - `../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Tumor/Tumor- (1).jpg`
     - `../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Cyst/...`
     - `../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Stone/...`

4. **Execute the request**:
   - Click the **"Execute"** button
   - View the response below with:
     - Predicted class name
     - Confidence scores for all classes

---

### **Method 2: Using cURL (Command Line)**

Run this command from the project root directory:

```bash
# Test with a Normal kidney image
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Normal/Normal- (2158).jpg"

# Test with a Tumor image
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Tumor/Tumor- (1).jpg"

# Test with a Cyst image
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Cyst/Cyst- (1).jpg"

# Test with a Stone image
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Stone/Stone- (1).jpg"
```

---

### **Method 3: Using Python Requests**

Create a test script `test_api.py`:

```python
import requests

# Test with a sample image
image_path = "../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Normal/Normal- (2158).jpg"

with open(image_path, 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)

print("Status Code:", response.status_code)
print("Response:", response.text)
```

Run:
```bash
python test_api.py
```

---

## 🔍 API Endpoints

### **1. Health Check**
- **URL**: `http://localhost:8000/health`
- **Method**: GET
- **Response**: `{"status": "healthy"}`

### **2. API Status**
- **URL**: `http://localhost:8000/`
- **Method**: GET
- **Response**: `{"status": "ok", "message": "Kidney Disease Classification API"}`

### **3. Predict (Main Endpoint)**
- **URL**: `http://localhost:8000/predict`
- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Parameters**: 
  - `file`: Image file (JPG, PNG, JPEG)
- **Response**: HTML page with prediction results or JSON with:
  - `pred_class`: Predicted class name (Normal, Cyst, Stone, or Tumor)
  - `confidences`: List of [class_name, probability] tuples

### **4. Swagger Documentation**
- **URL**: `http://localhost:8000/docs`
- **Method**: GET
- **Description**: Interactive API documentation

### **5. ReDoc Documentation**
- **URL**: `http://localhost:8000/redoc`
- **Method**: GET
- **Description**: Alternative API documentation

---

## 📊 Expected Response Format

When you call `/predict`, you'll receive:

**HTML Response** (default):
- Shows predicted class
- Shows confidence scores for all classes
- Formatted in a table

**JSON Response** (if API returns JSON):
```json
{
  "pred_class": "Normal",
  "confidences": [
    ["Normal", 1.0000],
    ["Cyst", 0.0000],
    ["Stone", 0.0000],
    ["Tumor", 0.0000]
  ]
}
```

---

## 🎯 Sample Images for Testing

You can use any image from these directories:

1. **Normal**: `../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Normal/`
2. **Tumor**: `../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Tumor/`
3. **Cyst**: `../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Cyst/`
4. **Stone**: `../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Stone/`

Each directory contains multiple JPG images that you can use for testing.

---

## ⚠️ Troubleshooting

### **Server not running?**
```bash
cd KidneyDisease-Classification
source ../venv/bin/activate
python app.py
```

### **Port 8000 already in use?**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
# Edit app.py: change port=8000 to port=8001
```

### **Model not found error?**
- Ensure `final_model/kidney_cnn.h5` exists
- Model file size should be ~97 MB

### **Preprocessor not found?**
- Ensure `final_model/preprocessor.pkl` exists
- Run the training pipeline if missing

---

## ✅ Quick Test Checklist

- [ ] Server is running (`curl http://localhost:8000/health`)
- [ ] Swagger UI opens (`http://localhost:8000/docs`)
- [ ] Sample image exists in dataset directory
- [ ] Model file exists: `final_model/kidney_cnn.h5`
- [ ] Preprocessor exists: `final_model/preprocessor.pkl`

---

**Happy Testing! 🚀**
