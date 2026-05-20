from pathlib import Path

from keras.models import load_model
from keras.datasets import mnist
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    model_path = root / "models" / "digit_cnn_model.h5"

    model = load_model(model_path)
    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = x_test.astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)

    prediction = model.predict(x_test[0:1])
    predicted_digit = np.argmax(prediction)

    print("Predicted:", predicted_digit)
    print("Actual:", y_test[0])


if __name__ == "__main__":
    main()
