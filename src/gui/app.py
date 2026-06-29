import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_requests").setLevel(logging.ERROR)

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

from src.preprocessing.preprocessing import preprocess_image_steps, preprocess_image
from src.models.model import load_digit_cnn_model
from src.models.logistic_model import load_logistic_model as load_digit_logistic_model
from src.models.logistic_model import predict_digit as predict_logistic_digit
from src.segmentation.segmentation import segment_digits

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MODEL_OPTIONS = ["Mock model", "Keras CNN model", "Logistic model"]


def find_images_from_folder(folder_path: str) -> List[Path]:
    path = Path(folder_path)
    if not path.exists() or not path.is_dir():
        return []
    images = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    return images


def load_pil_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


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
def load_trained_model() -> Optional[Tuple[object, Path]]:
    model_path = ROOT_DIR / "src" / "models" / "digit_cnn_model.h5"
    if not model_path.exists():
        return None

    model = load_digit_cnn_model(model_path)
    return model, model_path


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


@st.cache_data
def predict_with_model(processed_array: np.ndarray, model_option: str) -> Tuple[int, Optional[float]]:
    if model_option == "Keras CNN model":
        model_data = load_trained_model()
        if model_data is None:
            raise RuntimeError("No trained Keras model checkpoint was found.")

        model, _ = model_data
        image = processed_array.astype(np.float32)
        if image.ndim == 2:
            image = image[..., np.newaxis]
        if image.max() > 1.0:
            image = image / 255.0

        image = np.expand_dims(image, axis=0)
        outputs = model.predict(image)
        probs = outputs[0]
        prediction = int(np.argmax(probs))
        confidence = float(np.max(probs))
        prediction = postprocess_prediction_for_ambiguous_digits(prediction, image)
        return prediction, confidence

    if model_option == "Logistic model":
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
        thumbnail = Image.fromarray(digit.astype(np.uint8))
        if model_option == "Mock model":
            label = "Mock: 5"
            prediction = 5
        else:
            prediction = None
            try:
                normalize = model_option == "Keras CNN model"
                processed_digit = preprocess_image(
                    digit,
                    size=(28, 28),
                    method=preprocess_method,
                    blur_ksize=blur_ksize,
                    adaptive_params=(adaptive_block, adaptive_c),
                    thresh=thresh,
                    invert=invert,
                    normalize=normalize,
                    margin=4,
                )
                prediction, _ = predict_with_model(processed_digit, model_option)
                label = f"Prediction: {prediction}"
            except Exception:
                label = "Prediction unavailable"

        results.append((thumbnail, label, prediction))
    return results


def get_selected_image(
    input_mode: str,
    uploaded_file,
    folder_path: str,
    folder_images: List[Path],
    folder_choice: Optional[str],
) -> Optional[Tuple[Image.Image, str, object]]:
    if input_mode == "Upload image file" and uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            return image, uploaded_file.name, image
        except Exception:
            return None

    if input_mode == "Load from folder" and folder_choice:
        try:
            image = load_pil_image(folder_choice)
            return image, folder_choice, folder_choice
        except Exception:
            return None

    if input_mode == "Load from folder" and folder_images:
        try:
            image = load_pil_image(str(folder_images[0]))
            return image, str(folder_images[0]), str(folder_images[0])
        except Exception:
            return None

    return None


def main() -> None:
    st.set_page_config(page_title="HNRS GUI", layout="wide")
    
    st.title("Handwritten Number Recognition System")
    st.write(
        "Use the sidebar to upload a digit image or pick a folder, then press `Predict`. "
        "The app shows the original image, preprocessing preview, and a mock or model prediction."
    )

    with st.sidebar:
        st.header("Input & Controls")
        input_mode = st.radio("Select input method:", ["Upload image file", "Load from folder"])

        uploaded_file = None
        folder_path = ""
        folder_images: List[Path] = []
        folder_choice = None

        if input_mode == "Upload image file":
            uploaded_file = st.file_uploader(
                "Upload a handwritten digit image",
                type=[ext.strip(".") for ext in sorted(IMAGE_EXTENSIONS)],
            )
        else:
            folder_path = st.text_input("Folder path", value="")
            if folder_path:
                folder_images = find_images_from_folder(folder_path)
                if folder_images:
                    folder_choice = st.selectbox(
                        "Choose an image from the folder",
                        [str(path) for path in folder_images],
                    )
                else:
                    st.warning("No image files found in the entered folder.")

        st.markdown("---")
        st.subheader("Preprocessing settings")
        preprocess_method = st.selectbox("Binarization method", ["otsu", "simple", "adaptive"], index=0)
        thresh = st.slider("Threshold (simple)", min_value=0, max_value=255, value=128)
        blur_ksize = st.slider("Blur kernel size (Otsu)", min_value=1, max_value=21, value=5, step=2)
        adaptive_block = st.slider("Adaptive block size", min_value=3, max_value=51, value=15, step=2)
        adaptive_c = st.slider("Adaptive C value", min_value=0, max_value=25, value=7)
        invert = st.checkbox("Invert foreground/background", value=False)

        st.markdown("---")
        model_option = st.selectbox(
            "Model to use",
            MODEL_OPTIONS,
        )

    current_image_data = get_selected_image(input_mode, uploaded_file, folder_path, folder_images, folder_choice)
    if current_image_data is None:
        st.info("Please upload an image or select a valid folder image to continue.")
        return

    image, image_source, preprocess_input = current_image_data
    st.subheader("Input image")
    st.image(image, caption=f"Source: {image_source}", use_column_width=True)

    with st.expander("Preprocessing preview"):
        try:
            steps = run_preprocessing(
                preprocess_input,
                method=preprocess_method,
                thresh=thresh,
                blur_ksize=blur_ksize,
                adaptive_block=adaptive_block,
                adaptive_c=adaptive_c,
                invert=invert,
            )
            cols = st.columns(5)
            cols[0].image(steps["grayscale"], caption="Grayscale", clamp=True)
            cols[1].image(steps["binary"], caption="Binary", clamp=True)
            cols[2].image(steps.get("cleaned", steps["binary"]), caption="Cleaned", clamp=True)
            cols[3].image(steps["centered"], caption="Centered", clamp=True)
            cols[4].image(steps["final"], caption="Final 28x28", clamp=True)
        except Exception as error:
            st.warning(f"Preprocessing preview is unavailable: {error}")

    should_predict = st.sidebar.button("Run prediction")
    if not should_predict:
        st.info("Click 'Run prediction' in the sidebar to see a result.")
        return

    result_title = "Prediction result"
    if model_option == "Keras CNN model":
        model_data = load_trained_model()
        if model_data is None:
            st.warning("No trained Keras model found in the repository. Falling back to the mock model.")
            model_option = "Mock model"
        else:
            checkpoint_path = model_data[1]
            st.success(f"Loaded Keras model from {checkpoint_path}")
    elif model_option == "Logistic model":
        model_data = load_logistic_model()
        if model_data is None:
            st.warning("No trained Logistic model found in the repository. Falling back to the mock model.")
            model_option = "Mock model"
        else:
            checkpoint_path = model_data[1]
            st.success(f"Loaded Logistic model from {checkpoint_path}")

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

    if model_option == "Mock model":
        st.subheader(result_title)
        st.success("I think this is a 5")
        st.info("This is a mock result. Add a trained model checkpoint to use real predictions.")
    else:
        try:
            prediction, confidence = predict_with_model(steps["final"], model_option)
            st.subheader(result_title)
            digit_predictions = [pred for _, _, pred in segmented_results if pred is not None and pred >= 0]
            predicted_number = "".join(str(pred) for pred in digit_predictions) if digit_predictions else str(prediction)
            st.success(f"Predicted result: {predicted_number}")
            if confidence is not None:
                st.caption(f"Confidence: {confidence * 100:.1f}%")
            else:
                st.caption("Model confidence unavailable")
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.info("The app can still show the image and preprocessing preview even without a model checkpoint.")

    with st.expander("Segmented digit thumbnails"):
        if not segmented_results:
            st.info("No segmented digits were found in the image.")
        else:
            cols = st.columns(len(segmented_results))
            for idx, (thumb, label, _) in enumerate(segmented_results, start=1):
                with cols[idx - 1]:
                    st.image(thumb, caption=f"Digit {idx}", use_column_width=True)
                    st.caption(label)
    


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
