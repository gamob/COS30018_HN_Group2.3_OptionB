import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# Allow imports from src/ when running this script via Streamlit
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

try:
    from preprocessing.preprocessing import preprocess_image, preprocess_image_steps
except Exception:
    preprocess_image = None
    preprocess_image_steps = None

try:
    import torch
    from models.model import SimpleCNN
except Exception:
    torch = None
    SimpleCNN = None

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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
    if torch is None or SimpleCNN is None:
        return None

    # Search folders in the source tree for a .pth file.
    candidate_files = list(ROOT_DIR.rglob("*.pth"))
    if not candidate_files:
        return None

    checkpoint_path = candidate_files[0]
    model = SimpleCNN(num_classes=10)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint_path


@st.cache_data
def predict_with_model(processed_array: np.ndarray) -> Tuple[int, float]:
    if torch is None or SimpleCNN is None:
        raise RuntimeError("PyTorch is required for model prediction.")

    model_data = load_trained_model()
    if model_data is None:
        raise RuntimeError("No trained model checkpoint was found.")

    model, _ = model_data
    tensor = torch.from_numpy(processed_array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence = float(probs.max().item())
        prediction = int(probs.argmax(dim=1).item())

    return prediction, confidence


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
            ["Mock model", "SimpleCNN (if available)"],
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
            cols = st.columns(4)
            cols[0].image(steps["grayscale"], caption="Grayscale", clamp=True)
            cols[1].image(steps["binary"], caption="Binary", clamp=True)
            cols[2].image(steps["centered"], caption="Centered", clamp=True)
            cols[3].image(steps["final"], caption="Final 28x28", clamp=True)
        except Exception as error:
            st.warning(f"Preprocessing preview is unavailable: {error}")

    should_predict = st.sidebar.button("Run prediction")
    if not should_predict:
        st.info("Click 'Run prediction' in the sidebar to see a result.")
        return

    result_title = "Prediction result"
    if model_option == "SimpleCNN (if available)":
        if torch is None or SimpleCNN is None:
            st.warning("PyTorch or model code is unavailable. Falling back to the mock model.")
            model_option = "Mock model"
        else:
            model_data = load_trained_model()
            if model_data is None:
                st.warning("No trained model checkpoint found in the repository. Falling back to the mock model.")
                model_option = "Mock model"
            else:
                checkpoint_path = model_data[1]
                st.success(f"Loaded trained model from {checkpoint_path}")

    if model_option == "Mock model":
        st.subheader(result_title)
        st.success("I think this is a 5")
        st.info("This is a mock result. Add a trained model checkpoint to use real predictions.")
    else:
        try:
            prediction, confidence = predict_with_model(steps["final"].astype(np.float32) / 255.0)
            st.subheader(result_title)
            st.metric("Predicted digit", prediction, f"Confidence: {confidence * 100:.1f}%")
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.info("The app can still show the image and preprocessing preview even without a model checkpoint.")


if __name__ == "__main__":
    main()
