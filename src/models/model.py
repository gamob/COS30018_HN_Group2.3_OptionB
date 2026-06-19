from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
try:
    from keras.models import load_model
except Exception:
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        def load_model(*args, **kwargs):
            raise ImportError(
                "Keras or TensorFlow is not installed. Install with 'pip install tensorflow' to use model loading."
            )

MODEL_FILENAME = "digit_cnn_model.h5"


def get_default_model_path() -> Path:
    return Path(__file__).resolve().parent / MODEL_FILENAME


def load_digit_cnn_model(model_path: Optional[str] = None):
    path = Path(model_path) if model_path else get_default_model_path()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return load_model(path)


def predict_digit(model, image_array: np.ndarray) -> int:
    if image_array.ndim == 2:
        image_array = image_array[..., np.newaxis]
    if image_array.ndim == 3 and image_array.shape[-1] != 1:
        image_array = image_array.mean(axis=-1, keepdims=True)

    image = image_array.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    image = np.expand_dims(image, axis=0)
    outputs = model.predict(image)
    return int(np.argmax(outputs, axis=1)[0])
