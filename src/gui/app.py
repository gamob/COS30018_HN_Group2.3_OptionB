import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_requests").setLevel(logging.ERROR)

import cv2
import numpy as np
try:
    import streamlit as st
except Exception:
    class _DummyDecorator:
        def __call__(self, *args, **kwargs):
            def _wrap(func):
                return func
            return _wrap

    class _DummyStreamlit:
        cache_resource = _DummyDecorator()
        cache_data = _DummyDecorator()

    st = _DummyStreamlit()
from PIL import Image

# Allow imports from the project root when running this script via Streamlit
# Set ROOT_DIR to the repository root (two levels up from this file)
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.preprocessing.preprocessing import center_and_resize, normalize_array, preprocess_image_steps, preprocess_image
from src.models.model import load_digit_cnn_model, load_letter_cnn_model, predict_digit, predict_letter
from src.models.logistic_model import load_logistic_model as load_digit_logistic_model
from src.models.logistic_model import predict_digit as predict_logistic_digit
from src.models.svm_model import load_svm_model as load_digit_svm_model
from src.models.svm_model import predict_digit as predict_svm_digit
from src.segmentation.segmentation import segment_digits, segment_letter_regions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LETTER_CNN_OPTION = "Letter CNN model"
NUMBER_CNN_OPTION = "Number CNN model"
LOGISTIC_OPTION = "Logistic model"
SVM_OPTION = "RBF SVM model"
MOCK_OPTION = "Mock model"
MODEL_OPTIONS = [LETTER_CNN_OPTION, NUMBER_CNN_OPTION, LOGISTIC_OPTION, SVM_OPTION, MOCK_OPTION]
DIGIT_MODEL_OPTIONS = {NUMBER_CNN_OPTION, LOGISTIC_OPTION, SVM_OPTION, MOCK_OPTION}

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None


def find_images_from_folder(folder_path: str) -> List[Path]:
    path = Path(folder_path)
    if not path.exists() or not path.is_dir():
        return []
    images = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    return images


def source_name(source: Any) -> str:
    """Return a display-safe basename for a path or uploaded file."""
    name = getattr(source, "name", None) or str(source)
    return Path(name).name


def load_pil_image(source: Any) -> Image.Image:
    if hasattr(source, "seek"):
        source.seek(0)
    image = Image.open(source)
    image.load()
    return image


def compose_number_image(digit_paths: List[Any]) -> Image.Image:
    """Compose selected digit images into one left-to-right number image."""
    if not digit_paths:
        raise ValueError("Choose at least one digit image to create a number.")

    flattened: List[Image.Image] = []
    for path in digit_paths:
        source = load_pil_image(path)
        if "A" in source.getbands():
            rgba = source.convert("RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            flattened.append(Image.alpha_composite(white, rgba).convert("RGB"))
        else:
            flattened.append(source.convert("RGB"))

    target_height = max(image.height for image in flattened)
    resized: List[Image.Image] = []
    for image in flattened:
        width = max(1, round(image.width * target_height / image.height))
        resized.append(image.resize((width, target_height), Image.Resampling.LANCZOS))

    gap = max(4, target_height // 6)
    padding = gap
    total_width = sum(image.width for image in resized) + gap * (len(resized) - 1) + padding * 2
    canvas = Image.new("RGB", (total_width, target_height + padding * 2), "white")
    x = padding
    for image in resized:
        canvas.paste(image, (x, padding))
        x += image.width + gap
    return canvas


def run_preprocessing(
    image_path: str,
    method: str,
    thresh: int,
    blur_ksize: int,
    adaptive_block: int,
    adaptive_c: int,
    invert: bool,
) -> dict:
    if preprocess_image_steps is None:
        raise RuntimeError("Preprocessing module is not available.")

    return preprocess_image_steps(
        image_path,
        size=(28, 28),
        method=method,
        blur_ksize=blur_ksize,
        adaptive_params=(adaptive_block, adaptive_c),
        thresh=thresh,
        invert=invert,
        normalize=False,
        margin=4,
    )


@st.cache_resource
def load_letter_trained_model() -> Optional[Tuple[object, Path]]:
    model_path = ROOT_DIR / "src" / "models" / "letter_cnn_model.h5"
    if not model_path.exists():
        return None

    model = load_letter_cnn_model(model_path)
    return model, model_path


@st.cache_resource
def load_digit_trained_model() -> Optional[Tuple[object, Path]]:
    model_path = ROOT_DIR / "src" / "models" / "digit_cnn_model.h5"
    if not model_path.exists():
        return None
    model = load_digit_cnn_model(model_path)
    return model, model_path


# Backward-compatible name used by earlier tests/integrations.
load_trained_model = load_letter_trained_model


@st.cache_resource
def load_logistic_model() -> Optional[Tuple[object, Path]]:
    model_path = ROOT_DIR / "src" / "models" / "digit_logistic_model.pkl"
    if not model_path.exists():
        return None

    try:
        model = load_digit_logistic_model(model_path)
    except Exception:
        return None
    return model, model_path


@st.cache_resource
def load_svm_model() -> Optional[Tuple[object, Path]]:
    model_path = ROOT_DIR / "src" / "models" / "digit_svm_model.pkl"
    if not model_path.exists():
        return None

    try:
        model = load_digit_svm_model(model_path)
    except Exception:
        return None
    return model, model_path


def postprocess_prediction_for_ambiguous_digits(prediction: int, image_array: np.ndarray) -> int:
    if prediction not in (6, 9):
        return prediction

    arr = np.asarray(image_array)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.mean(axis=-1)

    if arr.ndim != 2:
        return prediction

    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    mask = arr > 0.5
    if not np.any(mask):
        return prediction

    h, _ = arr.shape
    upper = mask[: h // 2]
    lower = mask[h // 2 :]
    upper_density = float(np.count_nonzero(upper)) / max(1, np.count_nonzero(mask))
    lower_density = float(np.count_nonzero(lower)) / max(1, np.count_nonzero(mask))

    if prediction == 6 and upper_density > lower_density + 0.1:
        return 9
    if prediction == 9 and lower_density > upper_density + 0.1:
        return 6
    return prediction


def postprocess_letter_prediction(
    prediction: str,
    letter_image: np.ndarray,
    top: int,
    bottom: int,
    baseline: float,
    typical_top: float,
    typical_height: float,
) -> str:
    """Resolve a few shape ambiguities using sentence-level geometry."""
    binary = (np.asarray(letter_image) > 0).astype(np.uint8) * 255
    component_count, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    has_detached_mark = component_count > 2

    if prediction == "L" and top >= typical_top - typical_height * 0.10:
        return "I"
    if prediction == "J" and has_detached_mark and bottom <= baseline + typical_height * 0.20:
        return "I"

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    has_enclosed_hole = hierarchy is not None and any(item[3] >= 0 for item in hierarchy[0])
    if prediction == "U" and has_enclosed_hole:
        return "A"
    return prediction


@st.cache_data
def predict_with_model(processed_array: np.ndarray, model_option: str) -> Tuple[object, Optional[float]]:
    if model_option == LETTER_CNN_OPTION:
        model_data = load_letter_trained_model()
        if model_data is None:
            raise RuntimeError("No trained letter CNN model checkpoint was found.")

        model, _ = model_data
        image = processed_array.astype(np.float32)
        if image.ndim == 2:
            image = image[..., np.newaxis]
        if image.max() > 1.0:
            image = image / 255.0

        prediction = predict_letter(model, image)
        confidence = None
        return prediction, confidence

    if model_option == NUMBER_CNN_OPTION:
        model_data = load_digit_trained_model()
        if model_data is None:
            raise RuntimeError("No trained digit CNN model checkpoint was found.")

        model, _ = model_data
        image = processed_array.astype(np.float32)
        if image.ndim == 2:
            image = image[..., np.newaxis]
        if image.max() > 1.0:
            image = image / 255.0
        outputs = model.predict(np.expand_dims(image, axis=0), verbose=0)
        confidence = float(np.max(outputs[0]))
        prediction = int(np.argmax(outputs[0]))
        prediction = postprocess_prediction_for_ambiguous_digits(prediction, processed_array)
        return prediction, confidence

    if model_option == LOGISTIC_OPTION:
        model_data = load_logistic_model()
        if model_data is None:
            raise RuntimeError("No trained Logistic model checkpoint was found.")

        model, _ = model_data
        image = processed_array
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        if image.dtype != np.uint8:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)

        prediction = predict_logistic_digit(model, image)
        prediction = postprocess_prediction_for_ambiguous_digits(prediction, image)
        return prediction, None

    if model_option == SVM_OPTION:
        model_data = load_svm_model()
        if model_data is None:
            raise RuntimeError("No trained RBF SVM model checkpoint was found.")

        model, _ = model_data
        prediction = predict_svm_digit(model, processed_array)
        prediction = postprocess_prediction_for_ambiguous_digits(prediction, processed_array)
        return prediction, None

    raise RuntimeError(f"Unsupported model option: {model_option}")


def get_segmented_digit_predictions(
    clean_image: np.ndarray,
    preprocess_method: str,
    thresh: int,
    blur_ksize: int,
    adaptive_block: int,
    adaptive_c: int,
    invert: bool,
    model_option: str,
) -> list[tuple[Image.Image, str, Optional[int]]]:
    digits = segment_digits(clean_image)
    results = []
    if not digits:
        return results

    for idx, digit in enumerate(digits, start=1):
        thumbnail_array = center_and_resize(digit.astype(np.uint8), size=(96, 96), margin=10)
        thumbnail = Image.fromarray(thumbnail_array)
        if model_option == MOCK_OPTION:
            label = "Mock: 5"
            prediction = 5
        else:
            prediction = None
            try:
                # Segmentation already returns white foreground on black.
                # Re-thresholding this tight crop can erase or invert a digit.
                processed_digit = center_and_resize(digit, size=(28, 28), margin=4)
                if model_option == NUMBER_CNN_OPTION:
                    processed_digit = normalize_array(processed_digit)
                prediction, _ = predict_with_model(processed_digit, model_option)
                label = f"Prediction: {prediction}"
            except Exception:
                label = "Prediction unavailable"

        results.append((thumbnail, label, prediction))
    return results


def join_digit_predictions(
    segmented_results: list[tuple[Image.Image, str, Optional[int]]],
) -> str:
    """Join successful left-to-right digit predictions into one number."""
    return "".join(
        str(prediction)
        for _, _, prediction in segmented_results
        if prediction is not None and prediction >= 0
    )


def get_segmented_letter_predictions(
    clean_image: np.ndarray,
    preprocess_method: str,
    thresh: int,
    blur_ksize: int,
    adaptive_block: int,
    adaptive_c: int,
) -> list[tuple[Image.Image, str, int, int]]:
    """Segment a word and predict each character from left to right."""
    results = []
    segments = segment_letter_regions(clean_image)
    if not segments:
        return results
    baseline = float(np.median([segment.y + segment.height for segment in segments]))
    typical_top = float(np.median([segment.y for segment in segments]))
    typical_height = float(np.median([segment.height for segment in segments]))

    for segment in segments:
        letter = segment.image
        padding = max(2, int(round(max(letter.shape) * 0.15)))
        padded = np.pad(letter, padding, mode="constant", constant_values=0)
        processed = preprocess_image(
            padded,
            size=(28, 28),
            method=preprocess_method,
            blur_ksize=blur_ksize,
            adaptive_params=(adaptive_block, adaptive_c),
            thresh=thresh,
            invert=False,
            normalize=True,
            margin=4,
        )
        prediction, _ = predict_with_model(processed, LETTER_CNN_OPTION)
        prediction = postprocess_letter_prediction(
            str(prediction),
            letter,
            segment.y,
            segment.y + segment.height,
            baseline,
            typical_top,
            typical_height,
        )
        thumbnail = center_and_resize(letter.astype(np.uint8), size=(96, 96), margin=10)
        results.append((Image.fromarray(thumbnail), str(prediction), segment.x, segment.width))
    return results


def join_letter_predictions(
    letter_results: list[tuple[Image.Image, str, int, int]],
    word_gap_ratio: float = 0.45,
) -> str:
    """Join ordered letter predictions and restore large horizontal gaps."""
    if not letter_results:
        return ""

    typical_width = float(np.median([width for _, _, _, width in letter_results]))
    gap_threshold = max(2.0, typical_width * word_gap_ratio)
    text = letter_results[0][1]
    for previous, current in zip(letter_results, letter_results[1:]):
        previous_right = previous[2] + previous[3]
        gap = current[2] - previous_right
        if gap > gap_threshold:
            text += " "
        text += current[1]
    return text


@st.cache_resource
def load_spell_checker():
    return SpellChecker(distance=1) if SpellChecker is not None else None


def correct_english_text(text: str) -> str:
    """Conservatively correct unknown English words by at most one edit."""
    checker = load_spell_checker()
    if checker is None:
        return text

    corrected_words = []
    for word in text.split(" "):
        if len(word) < 3 or word.lower() in checker:
            corrected_words.append(word)
            continue
        correction = checker.correction(word.lower())
        corrected_words.append(correction.upper() if correction else word)
    return " ".join(corrected_words)


def render_thumbnail_cards(
    items: list[tuple[Image.Image, str]],
    item_name: str,
    cards_per_row: int = 10,
) -> None:
    """Render uniform character cards using Streamlit's native components."""
    for row_start in range(0, len(items), cards_per_row):
        row = items[row_start:row_start + cards_per_row]
        # Always keep the same column width. A short final row leaves its
        # unused columns empty instead of stretching the remaining cards.
        columns = st.columns(cards_per_row)
        for offset, (thumbnail, prediction) in enumerate(row):
            item_number = row_start + offset + 1
            with columns[offset].container(border=True):
                st.image(thumbnail, width="stretch")
                st.markdown(
                    f"<div style='text-align:center;font-size:1.25rem;font-weight:700'>"
                    f"{prediction}</div><div style='text-align:center;font-size:.75rem;opacity:.65'>"
                    f"{item_name} {item_number}</div>",
                    unsafe_allow_html=True,
                )


def make_preview_thumbnail(image_source: Any, size: tuple[int, int] = (320, 180)) -> Image.Image:
    """Fit a preprocessing step into a uniform landscape preview canvas."""
    array = np.asarray(image_source)
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if array.size and float(array.max()) <= 1.0 else 1.0
        array = (array * scale).clip(0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.clip(0, 255).astype(np.uint8)

    image = Image.fromarray(array)
    if image.mode not in ("L", "RGB", "RGBA"):
        image = image.convert("L")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new(image.mode, size, 0)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def get_selected_image(
    input_mode: str,
    uploaded_file,
    folder_path: str,
    folder_images: List[Path],
    folder_choice: Optional[str],
    selected_digit_paths: Optional[List[Any]] = None,
) -> Optional[Tuple[Image.Image, str, object]]:
    if input_mode == "Upload image file" and uploaded_file is not None:
        try:
            # Preserve alpha until preprocessing can flatten transparent
            # drawing exports onto a white background.
            image = Image.open(uploaded_file)
            image.load()
            return image, uploaded_file.name, image
        except Exception:
            return None

    if input_mode == "Load from folder" and folder_choice:
        try:
            image = load_pil_image(folder_choice)
            return image, source_name(folder_choice), image
        except Exception:
            return None

    if input_mode == "Load from folder" and folder_images:
        try:
            image = load_pil_image(folder_images[0])
            return image, source_name(folder_images[0]), image
        except Exception:
            return None

    if input_mode == "Create number from digit folder" and selected_digit_paths:
        try:
            image = compose_number_image(selected_digit_paths)
            source_names = " + ".join(source_name(path) for path in selected_digit_paths)
            return image, f"Created number: {source_names}", image
        except Exception:
            return None

    return None


def main() -> None:
    st.set_page_config(page_title="HNRS GUI", layout="wide")
    st.markdown(
        """
        <style>
        /* Preserve the 100%-zoom composition instead of stretching every
           panel across the extra CSS viewport created by browser zoom-out. */
        [data-testid="stMainBlockContainer"],
        .stMainBlockContainer {
            width: min(calc(100% - 2rem), 112rem) !important;
            max-width: 112rem !important;
            margin-inline: auto !important;
            box-sizing: border-box;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("Handwritten Letter and Number Recognition System")
    st.write(
        "Use the sidebar to upload an image, load one from a folder, or compose a number "
        "from individual digit images. Select the letter or number CNN, then press `Run prediction`."
    )

    with st.sidebar:
        st.markdown("<h3 style='margin:0; padding:0;'>Input & Controls</h3>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:-0.75rem'></div>", unsafe_allow_html=True)
        input_mode = st.radio(
            "Select input method:",
            ["Upload image file", "Load from folder", "Create number from digit folder"],
        )

        uploaded_file = None
        folder_path = ""
        folder_images: List[Any] = []
        folder_choice = None
        selected_digit_paths: List[Any] = []

        if input_mode == "Upload image file":
            uploaded_file = st.file_uploader(
                "Upload a handwritten letter or number image",
                type=[ext.strip(".") for ext in sorted(IMAGE_EXTENSIONS)],
            )
        elif input_mode == "Load from folder":
            folder_images = st.file_uploader(
                "Choose image folder",
                type=[ext.strip(".") for ext in sorted(IMAGE_EXTENSIONS)],
                accept_multiple_files="directory",
                key="image_folder_picker",
            ) or []
            if folder_images:
                selected_index = st.selectbox(
                    "Choose an image from the folder",
                    range(len(folder_images)),
                    format_func=lambda index: source_name(folder_images[index]),
                )
                folder_choice = folder_images[selected_index]
        else:
            folder_images = st.file_uploader(
                "Choose folder containing individual digit images",
                type=[ext.strip(".") for ext in sorted(IMAGE_EXTENSIONS)],
                accept_multiple_files="directory",
                key="digit_folder_picker",
            ) or []
            if folder_images:
                digit_count = st.number_input(
                    "Digits in the number",
                    min_value=1,
                    max_value=12,
                    value=3,
                    step=1,
                )
                st.caption("Choose one image for each position, from left to right.")
                for position in range(int(digit_count)):
                    selected_index = st.selectbox(
                        f"Position {position + 1}",
                        range(len(folder_images)),
                        format_func=lambda index, images=folder_images: source_name(images[index]),
                        key=f"created_number_digit_{position}",
                    )
                    selected_digit_paths.append(folder_images[selected_index])

        st.markdown("<div style='margin-top:-0.4rem'></div>", unsafe_allow_html=True)
        st.subheader("Model & Prediction")
        model_option = st.selectbox(
            "Model to use",
            MODEL_OPTIONS,
        )
        should_predict = st.button("Run prediction", type="primary")
        st.markdown("<div style='margin-top:-0.2rem'></div>", unsafe_allow_html=True)
        with st.expander("Preprocessing settings", expanded=False):
            preprocess_method = st.selectbox("Binarization method", ["otsu", "simple", "adaptive"], index=0)

            thresh = st.slider("Threshold (simple)", min_value=0, max_value=255, value=128)
            blur_ksize = st.slider("Blur kernel size (Otsu)", min_value=1, max_value=21, value=5, step=2)
            adaptive_block = st.slider("Adaptive block size", min_value=3, max_value=51, value=15, step=2)
            adaptive_c = st.slider("Adaptive C value", min_value=0, max_value=25, value=7)
            word_gap_ratio = st.slider(
                "Word gap sensitivity",
                min_value=0.20,
                max_value=1.00,
                value=0.70,
                step=0.05,
                help="Lower values detect spaces more easily; higher values require larger gaps.",
            )
            dictionary_correction = st.checkbox("English dictionary correction", value=True)
            invert = st.checkbox("Invert foreground/background", value=False)

    current_image_data = get_selected_image(
        input_mode,
        uploaded_file,
        folder_path,
        folder_images,
        folder_choice,
        selected_digit_paths,
    )
    if current_image_data is None:
        st.info("Please upload an image, select a folder image, or choose digit images to compose a number.")
        return

    image, image_source, preprocess_input = current_image_data
    input_column, result_column = st.columns([1, 2], vertical_alignment="top")
    with input_column:
        st.subheader("Input image")
        st.image(image, caption=f"Source: {image_source}", width=320)
    result_panel = result_column.container()
    with result_panel:
        st.subheader("Prediction result")
        if not should_predict:
            st.info("Click 'Run prediction' in the sidebar to see a result.")

    if should_predict:
        if input_mode == "Create number from digit folder" and model_option == LETTER_CNN_OPTION:
            with result_panel:
                st.error("Composed digit folders require a digit model. Select Number CNN, Logistic, RBF SVM, or Mock.")
            return

        if model_option == LETTER_CNN_OPTION:
            model_data = load_letter_trained_model()
            if model_data is None:
                st.warning("No trained letter CNN model found in the repository. Falling back to the mock model.")
                model_option = MOCK_OPTION
            else:
                st.success(f"Loaded letter CNN model from {model_data[1]}")
        elif model_option == NUMBER_CNN_OPTION:
            model_data = load_digit_trained_model()
            if model_data is None:
                st.warning("No trained number CNN model found in the repository. Falling back to the mock model.")
                model_option = MOCK_OPTION
            else:
                st.success(f"Loaded number CNN model from {model_data[1]}")
        elif model_option == LOGISTIC_OPTION:
            model_data = load_logistic_model()
            if model_data is None:
                st.warning("No trained Logistic model found in the repository. Falling back to the mock model.")
                model_option = MOCK_OPTION
            else:
                st.success(f"Loaded Logistic model from {model_data[1]}")
        elif model_option == SVM_OPTION:
            model_data = load_svm_model()
            if model_data is None:
                st.warning("No trained RBF SVM model found in the repository. Falling back to the mock model.")
                model_option = MOCK_OPTION
            else:
                st.success(f"Loaded RBF SVM model from {model_data[1]}")

    try:
        with st.expander("Preprocessing preview"):
            steps = run_preprocessing(
                preprocess_input,
                method=preprocess_method,
                thresh=thresh,
                blur_ksize=blur_ksize,
                adaptive_block=adaptive_block,
                adaptive_c=adaptive_c,
                invert=invert,
            )
            previews = [
                ("Grayscale", steps["grayscale"]),
                ("Binary", steps["binary"]),
                ("Cleaned", steps.get("cleaned", steps["binary"])),
                ("Centered", steps["centered"]),
                ("Final 28x28", steps["final"]),
            ]
            for column, (caption, preview) in zip(st.columns(5), previews):
                column.image(
                    make_preview_thumbnail(preview),
                    caption=caption,
                    clamp=True,
                    width="stretch",
                )
    except Exception as error:
        with st.expander("Preprocessing preview", expanded=True):
            st.warning(f"Preprocessing preview is unavailable: {error}")
        if should_predict:
            with result_panel:
                st.error("Prediction cannot run until preprocessing succeeds.")
        return

    if not should_predict:
        return

    letter_results: list[tuple[Image.Image, str, int, int]] = []
    segmented_results: list[tuple[Image.Image, str, Optional[int]]] = []

    if model_option == MOCK_OPTION:
        with result_panel:
            st.markdown("**Predicted result**")
            st.code("5", language=None, wrap_lines=True)
            st.info("This is a mock result. Add a trained model checkpoint to use real predictions.")
    else:
        try:
            if model_option == LETTER_CNN_OPTION:
                letter_results = get_segmented_letter_predictions(
                    steps["cleaned"],
                    preprocess_method,
                    thresh,
                    blur_ksize,
                    adaptive_block,
                    adaptive_c,
                )
                if not letter_results:
                    raise RuntimeError("No letters were detected in the image.")
                raw_prediction = join_letter_predictions(letter_results, word_gap_ratio)
                prediction = correct_english_text(raw_prediction) if dictionary_correction else raw_prediction
                confidence = None
                with result_panel:
                    st.code(prediction, language=None, wrap_lines=True)
                    if prediction != raw_prediction:
                        st.caption(f"Raw character recognition: {raw_prediction}")
                    st.caption(f"Detected {len(letter_results)} character(s)")
            else:
                segmented_results = get_segmented_digit_predictions(
                    steps["cleaned"],
                    preprocess_method,
                    thresh,
                    blur_ksize,
                    adaptive_block,
                    adaptive_c,
                    invert,
                    model_option,
                )
                predicted_number = join_digit_predictions(segmented_results)
                if predicted_number:
                    prediction = predicted_number
                    confidence = None
                else:
                    prediction, confidence = predict_with_model(steps["final"], model_option)
                with result_panel:
                    st.markdown("**Predicted result**")
                    st.code(str(prediction), language=None, wrap_lines=True)
                    if confidence is not None:
                        st.caption(f"Confidence: {confidence * 100:.1f}%")
                    if predicted_number:
                        st.caption(f"Detected {len(predicted_number)} digit(s)")
        except Exception as error:
            with result_panel:
                st.error(f"Prediction failed: {error}")
                st.info("The app can still show the image and preprocessing preview even without a model checkpoint.")

    if model_option == LETTER_CNN_OPTION and letter_results:
        with st.expander("Segmented letter thumbnails", expanded=True):
            render_thumbnail_cards(
                [(thumbnail, letter) for thumbnail, letter, _, _ in letter_results],
                "Character",
            )

    if model_option in DIGIT_MODEL_OPTIONS:
        if model_option == MOCK_OPTION and not segmented_results:
            try:
                segmented_results = get_segmented_digit_predictions(
                    steps["cleaned"],
                    preprocess_method,
                    thresh,
                    blur_ksize,
                    adaptive_block,
                    adaptive_c,
                    invert,
                    model_option,
                )
            except Exception:
                segmented_results = []
        with st.expander("Segmented digit thumbnails"):
            if not segmented_results:
                st.info("No segmented digits were found in the image.")
            else:
                render_thumbnail_cards(
                    [
                        (thumb, str(prediction) if prediction is not None else label)
                        for thumb, label, prediction in segmented_results
                    ],
                    "Digit",
                )
    


def _ensure_streamlit_runner() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            raise RuntimeError
    except Exception:
        raise SystemExit(
            "This application must be launched with Streamlit.\n"
            "Run: streamlit run src/gui/app.py"
        )


if __name__ == "__main__":
    _ensure_streamlit_runner()
    main()
