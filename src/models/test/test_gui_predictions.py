import cv2
import numpy as np

from PIL import Image

from src.gui.app import (
    correct_english_text,
    join_letter_predictions,
    postprocess_letter_prediction,
    postprocess_prediction_for_ambiguous_digits,
)


def test_postprocess_prediction_for_ambiguous_digits():
    bottom_heavy = np.zeros((28, 28), dtype=np.float32)
    bottom_heavy[20:, :] = 1.0
    assert postprocess_prediction_for_ambiguous_digits(9, bottom_heavy) == 6

    top_heavy = np.zeros((28, 28), dtype=np.float32)
    top_heavy[:8, :] = 1.0
    assert postprocess_prediction_for_ambiguous_digits(6, top_heavy) == 9


def test_join_letter_predictions_restores_word_spaces():
    thumbnail = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
    results = [
        (thumbnail, "T", 0, 10),
        (thumbnail, "H", 13, 10),
        (thumbnail, "I", 26, 10),
        (thumbnail, "S", 39, 10),
        (thumbnail, "I", 60, 10),
        (thumbnail, "S", 73, 10),
    ]

    assert join_letter_predictions(results, word_gap_ratio=0.45) == "THIS IS"


def test_postprocess_dotted_short_j_as_i():
    dotted_i = np.zeros((50, 20), dtype=np.uint8)
    dotted_i[15:45, 8:12] = 255
    dotted_i[2:8, 7:13] = 255

    assert postprocess_letter_prediction("J", dotted_i, 50, 100, 95.0, 50.0, 50.0) == "I"
    assert postprocess_letter_prediction("L", dotted_i, 50, 100, 95.0, 50.0, 50.0) == "I"
    assert postprocess_letter_prediction("J", dotted_i, 50, 130, 95.0, 50.0, 50.0) == "J"


def test_postprocess_closed_u_as_a():
    closed_a = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(closed_a, (20, 20), 14, 255, 5)

    assert postprocess_letter_prediction("U", closed_a, 0, 40, 40.0, 0.0, 40.0) == "A"


def test_english_correction_fixes_one_character_word_error():
    assert correct_english_text("EAVTH IS FLAT") == "EARTH IS FLAT"
