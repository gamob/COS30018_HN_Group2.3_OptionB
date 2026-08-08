import numpy as np

from src.gui.app import ALPHANUMERIC_CNN_OPTION, MODEL_OPTIONS
from src.models.model import ALPHANUMERIC_CLASS_NAMES, predict_alphanumeric


class _FakeModel:
    def __init__(self, class_index: int) -> None:
        self.class_index = class_index

    def predict(self, image, verbose=0):
        output = np.zeros((1, 36), dtype=np.float32)
        output[0, self.class_index] = 1.0
        return output


def test_alphanumeric_class_order_is_stable() -> None:
    assert ALPHANUMERIC_CLASS_NAMES == list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_predict_alphanumeric_maps_digit_and_letter_outputs() -> None:
    image = np.zeros((28, 28), dtype=np.uint8)
    assert predict_alphanumeric(_FakeModel(7), image) == "7"
    assert predict_alphanumeric(_FakeModel(10), image) == "A"
    assert predict_alphanumeric(_FakeModel(35), image) == "Z"


def test_gui_exposes_alphanumeric_cnn_option() -> None:
    assert ALPHANUMERIC_CNN_OPTION in MODEL_OPTIONS
