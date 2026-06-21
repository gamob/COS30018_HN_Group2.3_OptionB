from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
except Exception:
    try:
        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    except Exception: 
        mnist = None
        Sequential = None
        Conv2D = None
        MaxPooling2D = None
        Flatten = None
        Dense = None

# If you use the newest version of Tensorflow v2.4 you need to import these below command since Keras intergrated into Tensorflow
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential

MODEL_FILENAME = "digit_cnn_model.h5"


def build_model() -> Sequential:
    if Sequential is None:
        raise ImportError("Keras/TensorFlow is not installed. Install with 'pip install tensorflow' to build the model.")

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_mnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mnist is None:
        raise ImportError("Keras/TensorFlow is not installed. Install with 'pip install tensorflow' to load MNIST data.")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return x_train, y_train, x_test, y_test


def train_and_save_model(output_dir: Path, epochs: int = 5) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME

    x_train, y_train, x_test, y_test = load_mnist_data()
    model = build_model()

    print("Training model on MNIST...")
    model.fit(x_train, y_train, epochs=epochs, validation_split=0.1)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    model.save(model_path)
    print(f"Saved trained model to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN model on MNIST and save it to src/models.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parents[1]), help="Directory to save the trained model")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_and_save_model(output_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()
