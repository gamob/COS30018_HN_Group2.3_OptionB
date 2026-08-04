import numpy as np
import pytest

from src.preprocessing.preprocessing import (
    morphological_clean,
    normalize_polarity,
    preprocess_image,
)


def _letter_image(stroke_width: int) -> np.ndarray:
    image = np.zeros((64, 64), dtype=np.uint8)
    image[8:56, 30:30 + stroke_width] = 255
    image[8:8 + stroke_width, 18:46] = 255
    image[32:32 + stroke_width, 18:46] = 255
    return image


@pytest.mark.parametrize("stroke_width", [1, 5])
def test_opposite_polarities_produce_the_same_model_input(stroke_width: int) -> None:
    white_on_black = _letter_image(stroke_width)
    black_on_white = 255 - white_on_black

    expected = preprocess_image(white_on_black, blur_ksize=1)
    actual = preprocess_image(black_on_white, blur_ksize=1)

    np.testing.assert_array_equal(actual, expected)
    assert np.count_nonzero(actual) > 0


def test_normalize_polarity_accepts_normalized_float_input() -> None:
    white_on_black = _letter_image(1)
    normalized, inverted = normalize_polarity((255 - white_on_black) / 255.0)

    assert inverted is True
    np.testing.assert_array_equal(normalized, white_on_black)


def test_polarity_uses_dominant_background_instead_of_dark_border() -> None:
    framed_black_on_white = np.zeros((64, 64), dtype=np.uint8)
    framed_black_on_white[4:60, 4:60] = 255
    framed_black_on_white[14:50, 29:35] = 0

    normalized, inverted = normalize_polarity(framed_black_on_white)

    assert inverted is True
    assert normalized[32, 32] == 255
    assert normalized[20, 20] == 0


def test_manual_invert_is_applied_after_automatic_normalization() -> None:
    black_on_white = 255 - _letter_image(3)

    automatic = preprocess_image(black_on_white, blur_ksize=1, invert=False)
    overridden = preprocess_image(black_on_white, blur_ksize=1, invert=True)

    assert automatic[14, 14] > overridden[14, 14]


def test_morphological_clean_preserves_a_thin_stroke() -> None:
    thin_stroke = np.zeros((28, 28), dtype=np.uint8)
    thin_stroke[4:24, 14] = 255

    cleaned = morphological_clean(thin_stroke)

    assert np.count_nonzero(cleaned) >= np.count_nonzero(thin_stroke)
