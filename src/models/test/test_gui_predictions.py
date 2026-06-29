import numpy as np

from src.gui.app import postprocess_prediction_for_ambiguous_digits


def test_postprocess_prediction_for_ambiguous_digits():
    bottom_heavy = np.zeros((28, 28), dtype=np.float32)
    bottom_heavy[20:, :] = 1.0
    assert postprocess_prediction_for_ambiguous_digits(9, bottom_heavy) == 6

    top_heavy = np.zeros((28, 28), dtype=np.float32)
    top_heavy[:8, :] = 1.0
    assert postprocess_prediction_for_ambiguous_digits(6, top_heavy) == 9
