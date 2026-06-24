"""Test LogisticRegression model"""
from pathlib import Path
import numpy as np
import joblib


def test_logistic_model():
    """Test that LogisticRegression model works"""
    
    print("=" * 50)
    print("Testing LogisticRegression Model")
    print("=" * 50)
    
    # Load model
    print("\n1. Loading model...")
    try:
        model_path = Path(__file__).resolve().parents[2] / "models" / "digit_logistic_model.pkl"
        print(f"   Looking for model at: {model_path}")
        
        if not model_path.exists():
            print(f"   [FAIL] Model file not found at: {model_path}")
            return False
        
        model = joblib.load(model_path)
        print("   [OK] Model loaded successfully!")
    except Exception as e:
        print(f"   [FAIL] Failed to load model: {e}")
        return False
    
    # Create test image
    print("\n2. Creating test image...")
    try:
        dummy_image = np.zeros((28, 28))
        print(f"   [OK] Test image created (shape: {dummy_image.shape})")
    except Exception as e:
        print(f"   [FAIL] Failed to create image: {e}")
        return False
    
    # Make prediction
    print("\n3. Making prediction...")
    try:
        # Flatten image
        image_flat = dummy_image.reshape(1, -1) / 255.0
        result = model.predict(image_flat)
        result = int(result[0])
        print(f"   [OK] Prediction result: {result}")
        
        # Check if result is valid (should be 0-9)
        if 0 <= result <= 9:
            print(f"   [OK] Result is valid (digit {result})")
        else:
            print(f"   [FAIL] Invalid result (should be 0-9, got {result})")
            return False
            
    except Exception as e:
        print(f"   [FAIL] Failed to predict: {e}")
        return False
    
    # Success!
    print("\n" + "=" * 50)
    print("[OK] ALL TESTS PASSED!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    test_logistic_model()
