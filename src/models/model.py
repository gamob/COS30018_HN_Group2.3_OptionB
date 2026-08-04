from __future__ import annotations

import json
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

LETTER_MODEL_FILENAME = "letter_cnn_model.h5"
DIGIT_MODEL_FILENAME = "digit_cnn_model.h5"
# Backward-compatible name used by the original digit-only code and notebook.
MODEL_FILENAME = DIGIT_MODEL_FILENAME
LETTER_MAPPING_FILENAME = "letter_class_mapping.json"
LETTER_CLASS_NAMES = [chr(ord("A") + i) for i in range(26)]
_letter_mapping_cache: Optional[list[str]] = None


def get_default_letter_model_path() -> Path:
    return Path(__file__).resolve().parent / LETTER_MODEL_FILENAME


def get_default_digit_model_path() -> Path:
    return Path(__file__).resolve().parent / DIGIT_MODEL_FILENAME


def get_default_model_path() -> Path:
    """Return the default digit CNN path used by the original project API."""
    return get_default_digit_model_path()


def get_default_letter_mapping_path() -> Path:
    return Path(__file__).resolve().parent / LETTER_MAPPING_FILENAME


def _load_letter_class_mapping(mapping_path: Optional[str] = None) -> list[str]:
    global _letter_mapping_cache

    path = Path(mapping_path) if mapping_path else get_default_letter_mapping_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            raw_mapping = json.load(handle)
        if isinstance(raw_mapping, dict):
            mapping = [str(raw_mapping.get(str(index), LETTER_CLASS_NAMES[index])) for index in range(26)]
        else:
            mapping = [str(label) for label in raw_mapping]
    else:
        mapping = LETTER_CLASS_NAMES.copy()

    _letter_mapping_cache = mapping
    return mapping


def _resolve_class_mapping(class_mapping=None) -> list[str]:
    if class_mapping is None:
        if _letter_mapping_cache is not None:
            return _letter_mapping_cache
        return _load_letter_class_mapping()

    if isinstance(class_mapping, dict):
        return [str(class_mapping.get(str(index), LETTER_CLASS_NAMES[index])) for index in range(26)]
    return [str(label) for label in class_mapping]


def load_letter_cnn_model(model_path: Optional[str] = None, mapping_path: Optional[str] = None):
    path = Path(model_path) if model_path else get_default_letter_model_path()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    _load_letter_class_mapping(mapping_path)
    return load_model(path)


def load_digit_cnn_model(model_path: Optional[str] = None):
    path = Path(model_path) if model_path else get_default_digit_model_path()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return load_model(path)


def predict_letter(model, image_array: np.ndarray, class_mapping=None) -> str:
    if image_array.ndim == 2:
        image_array = image_array[..., np.newaxis]
    if image_array.ndim == 3 and image_array.shape[-1] != 1:
        image_array = image_array.mean(axis=-1, keepdims=True)

    image = image_array.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    image = np.expand_dims(image, axis=0)
    outputs = model.predict(image, verbose=0)
    class_index = int(np.argmax(outputs, axis=1)[0])
    mapping = _resolve_class_mapping(class_mapping)
    if class_index < 0 or class_index >= len(mapping):
        class_index = max(0, min(class_index, len(mapping) - 1))
    return str(mapping[class_index]).upper()


def predict_digit(model, image_array: np.ndarray) -> int:
    if image_array.ndim == 2:
        image_array = image_array[..., np.newaxis]
    if image_array.ndim == 3 and image_array.shape[-1] != 1:
        image_array = image_array.mean(axis=-1, keepdims=True)

    image = image_array.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    outputs = model.predict(np.expand_dims(image, axis=0), verbose=0)
    return int(np.argmax(outputs, axis=1)[0])