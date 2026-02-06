"""
Test script to evaluate model and run inference on sample images
"""
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from kidneyclassification.utils.main_utils import load_object
from kidneyclassification.utils.ml_utils import get_classification_metrics
from kidneyclassification.constants import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

# The existing model was trained with 128x128 images, not 224x224
# Override constants for testing the existing model
MODEL_IMAGE_SIZE = 128  # Existing model uses 128x128

def load_model_and_preprocessor():
    """Load trained model and preprocessor"""
    model_path = 'final_model/kidney_cnn.h5'
    preprocessor_path = 'final_model/preprocessor.pkl'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None, None
    
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    preprocessor = None
    if os.path.exists(preprocessor_path):
        print(f"Loading preprocessor from {preprocessor_path}")
        preprocessor = load_object(preprocessor_path)
    else:
        print(f"Preprocessor not found at {preprocessor_path}")
    
    return model, preprocessor

def evaluate_model_on_test_set(model, test_data_path=None):
    """Evaluate model on test set if available"""
    # Try to find latest test data
    if test_data_path is None:
        # Look for latest test data in artifacts
        artifacts_dir = "artifacts"
        if os.path.exists(artifacts_dir):
            for root, dirs, files in os.walk(artifacts_dir):
                if "test_data.npz" in files:
                    test_data_path = os.path.join(root, "test_data.npz")
                    break
    
    if test_data_path and os.path.exists(test_data_path):
        print(f"\nEvaluating model on test set: {test_data_path}")
        data = np.load(test_data_path)
        X_test = data['X']
        y_test = data['y']
        
        print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
        
        # Predict
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        metrics = get_classification_metrics(y_test, y_pred, average="macro")
        
        print(f"\n=== Test Set Evaluation Metrics ===")
        print(f"Precision (macro): {metrics.precision:.4f}")
        print(f"Recall (macro): {metrics.recall:.4f}")
        print(f"F1-Score (macro): {metrics.f1_score:.4f}")
        
        # Calculate accuracy
        accuracy = np.mean(y_test == y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        return metrics, accuracy, y_test, y_pred
    else:
        print("Test set not found. Skipping evaluation.")
        return None, None, None, None

def test_inference_on_sample_image(model, preprocessor, image_path):
    """Test inference on a single image"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    print(f"\n=== Testing Inference on Sample Image ===")
    print(f"Image path: {image_path}")
    
    # Load and preprocess image (same as training)
    # Note: Existing model uses 128x128, not 224x224
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    
    # Expand dimensions for batch
    X = np.expand_dims(arr, axis=0)
    
    # Predict
    probs = model.predict(X, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    
    # Map index to class name
    if preprocessor and hasattr(preprocessor, 'class_to_index'):
        index_to_class = {v: k for k, v in preprocessor.class_to_index.items()}
        pred_class = index_to_class.get(pred_idx, f"class_{pred_idx}")
    else:
        pred_class = f"class_{pred_idx}"
    
    print(f"\nPrediction Results:")
    print(f"  Predicted Class: {pred_class}")
    print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"\nAll Class Probabilities:")
    for i, prob in enumerate(probs):
        class_name = index_to_class.get(i, f"class_{i}") if (preprocessor and hasattr(preprocessor, 'class_to_index')) else f"class_{i}"
        print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    return pred_class, confidence, probs

def find_sample_image():
    """Find a sample image from the dataset"""
    dataset_root = "../Data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    
    # Try each class
    for class_name in ["Normal", "Cyst", "Stone", "Tumor"]:
        class_dir = os.path.join(dataset_root, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                return os.path.join(class_dir, images[0]), class_name
    
    return None, None

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL EVALUATION AND INFERENCE TEST")
    print("=" * 60)
    
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor()
    if model is None:
        sys.exit(1)
    
    print(f"\nModel Summary:")
    model.summary()
    
    # Evaluate on test set
    metrics, accuracy, y_test, y_pred = evaluate_model_on_test_set(model)
    
    # Test inference on sample image
    sample_image_path, true_class = find_sample_image()
    if sample_image_path:
        pred_class, confidence, probs = test_inference_on_sample_image(
            model, preprocessor, sample_image_path
        )
        print(f"\nSample Image True Class: {true_class}")
        print(f"Sample Image Predicted Class: {pred_class}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
