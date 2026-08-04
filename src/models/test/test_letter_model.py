from pathlib import Path

import numpy as np
from PIL import Image

from src.models.model import load_letter_cnn_model, predict_letter


def test_predict_letter_returns_alpha_label_for_sample_image() -> None:
    root = Path(__file__).resolve().parents[2]
    model_path = root / "models" / "letter_cnn_model.h5"
    if not model_path.exists():
        raise AssertionError("Letter CNN model has not been trained yet")

    model = load_letter_cnn_model(str(model_path))
    image_path = root.parent / "data" / "letters" / "test" / "A" / "000000.png"
    if not image_path.exists():
        image_path = root.parent / "data" / "letters" / "train" / "A" / "A_001.png"

    image = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)

    prediction = predict_letter(model, image)

    assert isinstance(prediction, str)
    assert prediction.isalpha()
    assert prediction.upper() == prediction
