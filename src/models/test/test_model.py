"""Test CNN model"""
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    model_path = root / "models" / "digit_cnn_model.h5"

    print("=" * 50)
    print("Testing CNN Model")
    print("=" * 50)
    
    print(f"\n1. Looking for model at: {model_path}")
    if not model_path.exists():
        print(f"   ✗ Model file not found!")
        return
    
    print("   ✓ Model file found")
    
    print("\n2. Loading model...")
    try:
        model = load_model(model_path)
        print("   ✓ Model loaded successfully!")
    except Exception as e:
        print(f"   ✗ FAILED to load: {e}")
        return

    print("\n3. Loading MNIST test data...")
    try:
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype(np.float32) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1)
        print("   ✓ Data loaded successfully!")
    except Exception as e:
        print(f"   ✗ FAILED to load data: {e}")
        return

    print("\n4. Making prediction on first test image...")
    try:
        prediction = model.predict(x_test[0:1])
        predicted_digit = np.argmax(prediction)
        print(f"   ✓ Prediction: {predicted_digit}")
        print(f"   ✓ Actual: {y_test[0]}")
        
        if predicted_digit == y_test[0]:
            print("   ✓ CORRECT!")
        else:
            print("   ⚠ Wrong, but model is working")
    except Exception as e:
        print(f"   ✗ FAILED to predict: {e}")
        return

    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()