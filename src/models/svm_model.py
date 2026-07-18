"""Support Vector Machine baseline for MNIST digit recognition.

The model accepts the same 28 x 28 grayscale digit images used by the CNN and
logistic-regression models.  It is deliberately kept as a separate baseline so
that model-comparison results can be reported clearly in the assignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

MODEL_FILENAME = "digit_svm_model.pkl"
METRICS_FILENAME = "digit_svm_metrics.json"
RANDOM_STATE = 42


def load_mnist_data_flat() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST, normalise pixel values, and flatten each image to 784 features."""
    try:
        from tensorflow.keras.datasets import mnist
    except ImportError as error:
        raise ImportError(
            "TensorFlow is required to download/load MNIST. Install the project "
            "requirements before training the SVM."
        ) from error

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (
        x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0,
        y_train,
        x_test.reshape(len(x_test), -1).astype(np.float32) / 255.0,
        y_test,
    )


def stratified_sample(
    features: np.ndarray, labels: np.ndarray, max_samples: Optional[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return a reproducible class-balanced subset, or all data when not needed."""
    if max_samples is None or max_samples >= len(features):
        return features, labels
    if max_samples < len(np.unique(labels)):
        raise ValueError("max_samples must include at least one example for each digit.")

    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=max_samples, random_state=RANDOM_STATE
    )
    indices, _ = next(splitter.split(features, labels))
    return features[indices], labels[indices]


def build_svm_model(C: float = 10.0, gamma: str | float = "scale") -> Pipeline:
    """Build an RBF SVM suitable for normalised MNIST pixel features."""
    return Pipeline(
        [("svm", SVC(kernel="rbf", C=C, gamma=gamma, cache_size=512))]
    )


def train_and_save_svm_model(
    output_dir: Path,
    max_train_samples: Optional[int] = 15000,
    C: float = 10.0,
    gamma: str | float = "scale",
) -> dict[str, object]:
    """Train, evaluate, and save an SVM plus machine-readable evaluation metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = load_mnist_data_flat()
    x_train, y_train = stratified_sample(x_train, y_train, max_train_samples)

    print(f"Training RBF SVM on {len(x_train):,} stratified MNIST examples...")
    model = build_svm_model(C=C, gamma=gamma)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    metrics: dict[str, object] = {
        "model": "SVC (RBF kernel)",
        "training_examples": int(len(x_train)),
        "test_examples": int(len(x_test)),
        "C": C,
        "gamma": gamma,
        "random_state": RANDOM_STATE,
        "test_accuracy": accuracy,
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }

    model_path = output_dir / MODEL_FILENAME
    metrics_path = output_dir / METRICS_FILENAME
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved SVM model to {model_path}")
    print(f"Saved evaluation metrics to {metrics_path}")
    return metrics


def load_svm_model(model_path: Optional[str | Path] = None):
    """Load a saved SVM model from the default project location or a supplied path."""
    path = Path(model_path) if model_path else Path(__file__).resolve().parent / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"SVM model file not found: {path}")
    return joblib.load(path)


def predict_digit(model, image_array: np.ndarray) -> int:
    """Predict one digit from a 28 x 28 image in either 0-255 or 0-1 range."""
    image = np.asarray(image_array, dtype=np.float32)
    if image.ndim == 3:
        image = image.mean(axis=-1) if image.shape[-1] != 1 else image[..., 0]
    if image.shape != (28, 28):
        raise ValueError(f"Expected a 28x28 digit image, received shape {image.shape}.")
    if image.max() > 1.0:
        image /= 255.0
    return int(model.predict(image.reshape(1, -1))[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RBF SVM on MNIST.")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--max-train-samples", type=int, default=15000,
        help="Stratified training subset size; use 0 to train on all 60,000 images.",
    )
    parser.add_argument("--C", type=float, default=10.0, help="SVM regularisation parameter.")
    parser.add_argument("--gamma", default="scale", help="RBF kernel gamma value or 'scale'.")
    args = parser.parse_args()
    gamma: str | float = float(args.gamma) if args.gamma != "scale" else args.gamma
    train_and_save_svm_model(
        Path(args.output_dir),
        max_train_samples=None if args.max_train_samples == 0 else args.max_train_samples,
        C=args.C,
        gamma=gamma,
    )


if __name__ == "__main__":
    main()
