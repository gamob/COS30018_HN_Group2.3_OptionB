from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, RandomRotation, RandomTranslation, RandomZoom
    from keras.models import Sequential
    from keras.datasets import mnist
except Exception:
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, RandomRotation, RandomTranslation, RandomZoom
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.datasets import mnist
    except Exception:
        Sequential = None
        Conv2D = None
        MaxPooling2D = None
        Flatten = None
        Dense = None
        mnist = None

from src.preprocessing.preprocessing import preprocess_image

MODEL_FILENAME = "letter_cnn_model.h5"
DIGIT_MODEL_FILENAME = "digit_cnn_model.h5"
MAPPING_FILENAME = "letter_class_mapping.json"


def build_model() -> Sequential:
    if Sequential is None:
        raise ImportError("Keras/TensorFlow is not installed. Install with 'pip install tensorflow' to build the model.")

    model = Sequential([
        Input(shape=(28, 28, 1)),
        RandomRotation(0.06),
        RandomTranslation(0.08, 0.08),
        RandomZoom(0.08),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.20),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.35),
        Dense(26, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_digit_model() -> Sequential:
    """Build the 10-class CNN used for MNIST-style digit recognition."""
    if Sequential is None:
        raise ImportError("Keras/TensorFlow is not installed. Install with 'pip install tensorflow' to build the model.")

    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _load_image_array(image_path: Path) -> np.ndarray:
    try:
        array = preprocess_image(
            image_path,
            size=(28, 28),
            method="otsu",
            blur_ksize=5,
            adaptive_params=(15, 7),
            thresh=128,
            invert=False,
            normalize=True,
            margin=4,
        )
    except Exception:
        image = Image.open(image_path).convert("L")
        array = np.array(image.resize((28, 28), Image.LANCZOS), dtype=np.float32) / 255.0

    return array.reshape(28, 28, 1)


def load_letter_data(
    data_dir: Path,
    max_train_per_class: int = 200,
    max_valid_per_class: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    if not train_dir.exists() or not valid_dir.exists():
        raise FileNotFoundError(f"Expected letter dataset folders at {data_dir}")

    def _load_split(split_dir: Path, max_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        image_arrays = []
        labels = []
        class_names = sorted([path.name.upper() for path in split_dir.iterdir() if path.is_dir()])
        for class_name in class_names:
            class_dir = split_dir / class_name
            image_paths = sorted(class_dir.glob("*.png"))
            if max_per_class > 0:
                image_paths = image_paths[:max_per_class]
            for image_path in image_paths:
                image_arrays.append(_load_image_array(image_path))
                labels.append(class_name)

        if not image_arrays:
            raise ValueError(f"No images found in {split_dir}")

        return np.stack(image_arrays, axis=0), np.array(labels, dtype=object)

    x_train, y_train = _load_split(train_dir, max_train_per_class)
    x_valid, y_valid = _load_split(valid_dir, max_valid_per_class)

    class_names = sorted([path.name.upper() for path in train_dir.iterdir() if path.is_dir()])
    class_to_index = {name: index for index, name in enumerate(class_names)}

    y_train_indices = np.array([class_to_index[label] for label in y_train], dtype=np.int32)
    y_valid_indices = np.array([class_to_index[label] for label in y_valid], dtype=np.int32)
    return x_train, y_train_indices, x_valid, y_valid_indices, class_names


def train_and_save_letter_model(
    output_dir: Path,
    epochs: int = 15,
    data_dir: Optional[Path] = None,
    max_train_per_class: int = 2000,
    max_valid_per_class: int = 300,
    batch_size: int = 128,
    model_filename: str = MODEL_FILENAME,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / model_filename
    mapping_path = output_dir / MAPPING_FILENAME

    dataset_dir = data_dir if data_dir is not None else Path(__file__).resolve().parents[3] / "data" / "letters"
    x_train, y_train, x_valid, y_valid, class_names = load_letter_data(
        dataset_dir,
        max_train_per_class=max_train_per_class,
        max_valid_per_class=max_valid_per_class,
    )
    model = build_model()

    print(f"Training letter CNN on dataset in {dataset_dir}...")
    checkpoint_path = model_path.with_suffix(".best.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_valid, y_valid),
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(x_valid, y_valid)
    print(f"Validation accuracy: {accuracy:.4f}")

    mapping = {str(index): label for index, label in enumerate(class_names)}
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2)

    model.save(model_path)
    print(f"Saved trained model to {model_path}")
    print(f"Saved class mapping to {mapping_path}")


def load_mnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mnist is None:
        raise ImportError("Keras/TensorFlow is not installed. Install with 'pip install tensorflow' to load MNIST.")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32).reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.astype(np.float32).reshape(-1, 28, 28, 1) / 255.0
    return x_train, y_train, x_test, y_test


def train_and_save_digit_model(output_dir: Path, epochs: int = 5) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / DIGIT_MODEL_FILENAME
    x_train, y_train, x_test, y_test = load_mnist_data()
    model = build_digit_model()
    print("Training digit CNN on MNIST...")
    model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, batch_size=128, verbose=1)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    model.save(model_path)
    print(f"Saved trained digit model to {model_path}")


def train_and_save_model(output_dir: Path, epochs: int = 5) -> None:
    """Backward-compatible alias for the original digit training command."""
    train_and_save_digit_model(output_dir, epochs=epochs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN model on handwritten letters and save it to src/models.")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum number of training epochs")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parents[1]), help="Directory to save the trained model")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parents[3] / "data" / "letters"), help="Path to the letters dataset")
    parser.add_argument("--max-train-per-class", type=int, default=2000, help="0 uses every training image")
    parser.add_argument("--max-valid-per-class", type=int, default=300, help="0 uses every validation image")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-filename", default=MODEL_FILENAME)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_and_save_letter_model(
        output_dir,
        epochs=args.epochs,
        data_dir=Path(args.data_dir),
        max_train_per_class=args.max_train_per_class,
        max_valid_per_class=args.max_valid_per_class,
        batch_size=args.batch_size,
        model_filename=args.model_filename,
    )


if __name__ == "__main__":
    main()
