import cv2
import numpy as np

from src.segmentation.segmentation import segment_digits


def test_segment_digits_does_not_split_large_wide_digit_by_pixel_width() -> None:
    image = np.zeros((300, 600), dtype=np.uint8)
    cv2.ellipse(image, (300, 150), (90, 115), 0, 0, 360, 255, 8)

    segments = segment_digits(image)

    assert len(segments) == 1


def test_segment_digits_attaches_detached_base_to_digit_body() -> None:
    image = np.zeros((240, 420), dtype=np.uint8)
    cv2.line(image, (90, 50), (90, 185), 255, 7)
    cv2.line(image, (70, 205), (135, 205), 255, 7)
    cv2.ellipse(image, (285, 130), (55, 80), 0, 0, 360, 255, 7)

    segments = segment_digits(image)

    assert len(segments) == 2
    assert segments[0].shape[1] >= 60
