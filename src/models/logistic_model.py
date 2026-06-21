from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import mnist


def load_mnist_data_flat() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST and flatten images to 784 pixels (28x28 = 784)"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Flatten from (28, 28) to (784,) and normalize
    x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0
    
    return x_train_flat, y_train, x_test_flat, y_test


def train_and_save_logistic_model(output_dir: Path) -> None:
    """Train LogisticRegression on MNIST and save it"""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "digit_logistic_model.pkl"

    # Load flattened data
    x_train_flat, y_train, x_test_flat, y_test = load_mnist_data_flat()

    # Create and train model
    print("Training Logistic Regression model on MNIST...")
    model = LogisticRegression(max_iter=100, random_state=42)
    model.fit(x_train_flat, y_train)

    # Test accuracy
    accuracy = model.score(x_test_flat, y_test)
    print(f"Test accuracy: {accuracy:.4f}")  # Should show  be 0.92

    # Save model
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")


def load_logistic_model(model_path: Optional[str] = None):
    """Load saved LogisticRegression model"""
    if model_path is None:
        model_path = Path(__file__).resolve().parent / "digit_logistic_model.pkl"
    
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    return joblib.load(path)


def predict_digit(model, image_array: np.ndarray) -> int:
    """Predict digit from image using LogisticRegression model"""
    # Handle different image formats
    if image_array.ndim == 2:
        # Grayscale image (28, 28)
        image_array = image_array.reshape(1, -1)
    elif image_array.ndim == 3:
        # Image with channel (28, 28, 1)
        image_array = image_array.reshape(1, -1)
    
    # Normalize (0-1 range)
    image = image_array.astype(np.float32) / 255.0
    
    # Predict
    prediction = model.predict(image)
    return int(prediction[0])


def main() -> None:
    """Train LogisticRegression model"""
    output_dir = Path(__file__).resolve().parents[1]  # src/models/ path
    train_and_save_logistic_model(output_dir)


if __name__ == "__main__":
    main()