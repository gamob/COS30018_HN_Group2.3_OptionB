
from src.models.logistic_model import predict_digit, load_logistic_model
import numpy as np


def test_logistic_model():

    
    print("=" * 50)
    print("Testing LogisticRegression Model")
    print("=" * 50)
    
    # Load model
    print("\n1. Loading model...")
    try:
        model = load_logistic_model()
        print("   ✓ Model loaded successfully!")
    except Exception as e:
        print(f"   ✗ FAILED to load model: {e}")
        return False
    
    # Create test image
    print("\n2. Creating test image...")
    try:
        dummy_image = np.zeros((28, 28))
        print(f"   ✓ Test image created (shape: {dummy_image.shape})")
    except Exception as e:
        print(f"   ✗ FAILED to create image: {e}")
        return False
    
    # Make prediction
    print("\n3. Making prediction...")
    try:
        result = predict_digit(model, dummy_image)
        print(f"   ✓ Prediction result: {result}")
        
        # Check if result is valid (should be 0-9)
        if 0 <= result <= 9:
            print(f"   ✓ Result is valid (digit {result})")
        else:
            print(f"   ✗ Invalid result (should be 0-9, got {result})")
            return False
            
    except Exception as e:
        print(f"   ✗ FAILED to predict: {e}")
        return False
    
    # Success!
    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    test_logistic_model()