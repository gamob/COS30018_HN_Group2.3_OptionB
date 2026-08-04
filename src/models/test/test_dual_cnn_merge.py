from pathlib import Path

from PIL import Image

from src.gui.app import (
    LETTER_CNN_OPTION,
    LOGISTIC_OPTION,
    MODEL_OPTIONS,
    NUMBER_CNN_OPTION,
    SVM_OPTION,
    compose_number_image,
    get_selected_image,
    join_digit_predictions,
)
from src.models.model import (
    DIGIT_MODEL_FILENAME,
    MODEL_FILENAME,
    get_default_digit_model_path,
    get_default_model_path,
)


def _save_solid_image(path: Path, size: tuple[int, int], color: str) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_model_options_preserve_both_cnns_and_digit_baselines() -> None:
    assert MODEL_OPTIONS[:2] == [LETTER_CNN_OPTION, NUMBER_CNN_OPTION]
    assert LOGISTIC_OPTION in MODEL_OPTIONS
    assert SVM_OPTION in MODEL_OPTIONS
    assert len(MODEL_OPTIONS) == len(set(MODEL_OPTIONS))


def test_compose_number_image_preserves_geometry_and_left_to_right_order(
    tmp_path: Path,
) -> None:
    narrow_red = _save_solid_image(tmp_path / "first.png", (10, 20), "red")
    wide_blue = _save_solid_image(tmp_path / "second.png", (20, 10), "blue")

    composed = compose_number_image([str(narrow_red), str(wide_blue)])

    # Both inputs are scaled to height 20. The second image becomes 40px wide;
    # gap and outer padding are each max(4, 20 // 6) == 4px.
    assert composed.size == (62, 28)
    assert composed.getpixel((9, 14)) == (255, 0, 0)
    assert composed.getpixel((38, 14)) == (0, 0, 255)
    assert composed.getpixel((16, 14)) == (255, 255, 255)


def test_get_selected_image_supports_composed_digit_folder_mode(
    tmp_path: Path,
) -> None:
    first = _save_solid_image(tmp_path / "1.png", (8, 16), "black")
    second = _save_solid_image(tmp_path / "0.png", (8, 16), "black")

    selected = get_selected_image(
        "Create number from digit folder",
        uploaded_file=None,
        folder_path=str(tmp_path),
        folder_images=[first, second],
        folder_choice=None,
        selected_digit_paths=[str(first), str(second)],
    )

    assert selected is not None
    image, source, preprocess_input = selected
    assert isinstance(image, Image.Image)
    assert source == "Created number: 1.png + 0.png"
    assert preprocess_input is image
    assert image.size == compose_number_image([str(first), str(second)]).size


def test_join_digit_predictions_keeps_order_zero_and_skips_failures() -> None:
    thumbnail = Image.new("L", (4, 4), 0)
    results = [
        (thumbnail, "Prediction: 1", 1),
        (thumbnail, "Prediction unavailable", None),
        (thumbnail, "Prediction: 0", 0),
        (thumbnail, "Prediction unavailable", -1),
        (thumbnail, "Prediction: 7", 7),
    ]

    assert join_digit_predictions(results) == "107"


def test_legacy_default_model_path_still_targets_digit_cnn() -> None:
    assert DIGIT_MODEL_FILENAME == "digit_cnn_model.h5"
    assert MODEL_FILENAME == DIGIT_MODEL_FILENAME
    assert get_default_model_path() == get_default_digit_model_path()
    assert get_default_model_path().name == "digit_cnn_model.h5"
