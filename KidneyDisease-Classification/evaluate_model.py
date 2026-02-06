"""Evaluate model on test set with proper image resizing"""
import numpy as np
import tensorflow as tf
from PIL import Image
from kidneyclassification.utils.ml_utils import get_classification_metrics

# Model expects 128x128 images, but test data is 224x224
MODEL_IMAGE_SIZE = 128

def resize_test_data(X_test):
    """Resize test data from 224x224 to 128x128"""
    X_resized = np.zeros((len(X_test), MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3), dtype=np.float32)
    for i in range(len(X_test)):
        img_array = (X_test[i] * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img = img.resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))
        X_resized[i] = np.array(img, dtype=np.float32) / 255.0
    return X_resized

def main():
    print("=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = tf.keras.models.load_model('final_model/kidney_cnn.h5')
    
    # Load test data
    test_path = 'artifact/01_18_2026_12_32_23/data_transformation/transformed/test_data.npz'
    print(f"Loading test data from {test_path}")
    data = np.load(test_path)
    X_test = data['X']
    y_test = data['y']
    
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    print(f"Note: Test data is {X_test.shape[1]}x{X_test.shape[2]}, resizing to {MODEL_IMAGE_SIZE}x{MODEL_IMAGE_SIZE}")
    
    # Resize test data
    print("Resizing test images...")
    X_test_resized = resize_test_data(X_test)
    
    # Predict
    print("Running predictions...")
    y_pred_probs = model.predict(X_test_resized, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = get_classification_metrics(y_test, y_pred, average="macro")
    accuracy = np.mean(y_test == y_pred)
    
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision (macro): {metrics.precision:.4f}")
    print(f"Recall (macro): {metrics.recall:.4f}")
    print(f"F1-Score (macro): {metrics.f1_score:.4f}")
    
    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print("Rows=Actual, Columns=Predicted")
    print("\n     ", " ".join([f"{name:>6}" for name in class_names]))
    for i, name in enumerate(class_names):
        print(f"{name:>6} ", " ".join([f"{cm[i,j]:>6}" for j in range(len(class_names))]))
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
