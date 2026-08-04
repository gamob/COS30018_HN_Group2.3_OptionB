import cv2
import numpy as np
import pytest
from pathlib import Path

from PIL import Image

from src.preprocessing.preprocessing import preprocess_image_steps
from src.segmentation.segmentation import segment_letters


def test_segment_letters_orders_separate_characters_left_to_right() -> None:
    image = np.full((60, 120), 255, dtype=np.uint8)
    cv2.rectangle(image, (12, 10), (21, 50), 0, -1)
    cv2.rectangle(image, (12, 27), (39, 34), 0, -1)
    cv2.rectangle(image, (31, 10), (40, 50), 0, -1)
    cv2.rectangle(image, (78, 10), (87, 50), 0, -1)

    letters = segment_letters(image)

    assert len(letters) == 2
    assert letters[0].shape[1] > letters[1].shape[1]


def test_segment_letters_attaches_dot_to_lowercase_i() -> None:
    image = np.full((60, 80), 255, dtype=np.uint8)
    cv2.rectangle(image, (35, 22), (42, 52), 0, -1)
    cv2.circle(image, (39, 10), 4, 0, -1)

    letters = segment_letters(image)

    assert len(letters) == 1
    assert letters[0].shape[0] >= 40


def test_segment_letters_splits_an_unusually_wide_touching_pair() -> None:
    image = np.full((70, 230), 255, dtype=np.uint8)
    # A normal-width reference character.
    cv2.rectangle(image, (0, 10), (35, 18), 0, -1)
    cv2.rectangle(image, (5, 10), (14, 60), 0, -1)
    # Simplified h and i connected by a thin stroke.
    cv2.rectangle(image, (55, 10), (64, 60), 0, -1)
    cv2.rectangle(image, (64, 35), (90, 44), 0, -1)
    cv2.rectangle(image, (82, 30), (91, 60), 0, -1)
    cv2.rectangle(image, (98, 28), (107, 60), 0, -1)
    cv2.circle(image, (103, 15), 5, 0, -1)
    cv2.line(image, (91, 44), (98, 44), 0, 1)
    # Another normal-width reference character.
    cv2.rectangle(image, (160, 10), (190, 60), 0, -1)

    letters = segment_letters(image)

    assert len(letters) == 4


def test_sentence_samples_have_expected_character_counts() -> None:
    samples = sorted((Path(__file__).resolve().parents[3] / "data" / "letter_sentences").glob("*.png"))
    expected_counts = [14, 12, 11]

    if not samples:
        pytest.skip("Manual sentence regression samples are not present")
    assert len(samples) >= len(expected_counts)
    for path, expected in zip(samples, expected_counts):
        steps = preprocess_image_steps(Image.open(path), normalize=False)
        assert len(segment_letters(steps["cleaned"])) == expected
