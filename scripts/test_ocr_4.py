#!/usr/bin/env python3
"""Test one of the project's CNN OCR models on every image in a folder.

Examples:
  python scripts/test_ocr.py
  python scripts/test_ocr.py --test --model letter
  python scripts/test_ocr.py --test --model number
  python scripts/test_ocr.py --test --model alphanumeric
  python scripts/test_ocr.py --model-file src/models/letter_cnn_model.h5

When --model is omitted, the script asks which model to use. It always asks
for the image directory; there is deliberately no default image directory.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import pytesseract
except Exception:
    pytesseract = None


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.preprocessing import center_and_resize, normalize_array
from src.segmentation.segmentation import binarize_image, segment_digits, segment_letter_regions


MODELS_DIR = PROJECT_ROOT / "src" / "models"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "unseen_data" / "Case#3" / "HandWritten_Alphabet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "desired_output.txt"
DEFAULT_MODEL_KEY = "letter"
MAX_IMAGE_COUNT = 1000
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    filename: str
    labels: tuple[str, ...]
    segmentation: str

    @property
    def path(self) -> pathlib.Path:
        return MODELS_DIR / self.filename


MODEL_SPECS = {
    "letter": ModelSpec(
        key="letter",
        display_name="Letter CNN",
        filename="letter_cnn_model.h5",
        labels=tuple(chr(ord("A") + index) for index in range(26)),
        segmentation="letters",
    ),
    "number": ModelSpec(
        key="number",
        display_name="Number CNN",
        filename="digit_cnn_model.h5",
        labels=tuple(str(index) for index in range(10)),
        segmentation="digits",
    ),
    "alphanumeric": ModelSpec(
        key="alphanumeric",
        display_name="Alphanumeric CNN",
        filename="alphanumeric_cnn_model.h5",
        labels=tuple(str(index) for index in range(10))
        + tuple(chr(ord("A") + index) for index in range(26)),
        segmentation="letters",
    ),
}

MODEL_ALIASES = {
    "1": "letter",
    "letter": "letter",
    "letter-cnn": "letter",
    "letter_cnn": "letter",
    "letter_cnn_model.h5": "letter",
    "2": "number",
    "number": "number",
    "number-cnn": "number",
    "number_cnn": "number",
    "digit": "number",
    "digit-cnn": "number",
    "digit_cnn": "number",
    "digit_cnn_model.h5": "number",
    "3": "alphanumeric",
    "alphanumeric": "alphanumeric",
    "alphanumeric-cnn": "alphanumeric",
    "alphanumeric_cnn": "alphanumeric",
    "alphanumeric_cnn_model.h5": "alphanumeric",
}


def normalize_model_choice(value: str) -> str | None:
    normalized = str(value).strip().lower().replace(" ", "-")
    return MODEL_ALIASES.get(normalized)


def choose_model_interactively() -> ModelSpec:
    print("Select model:")
    print("  1. Letter CNN (A-Z)")
    print("  2. Number CNN (0-9)")
    print("  3. Alphanumeric CNN (0-9, A-Z)")
    while True:
        try:
            choice = input("Model [1-3]: ").strip()
        except EOFError:
            print("Error: a model selection is required.", file=sys.stderr)
            raise SystemExit(2)
        key = normalize_model_choice(choice)
        if key:
            return MODEL_SPECS[key]
        print("Please choose 1, 2, or 3.", file=sys.stderr)


def infer_spec_from_model_path(model_path: pathlib.Path) -> ModelSpec | None:
    key = normalize_model_choice(model_path.name)
    return MODEL_SPECS[key] if key else None


def resolve_model(args: argparse.Namespace) -> tuple[ModelSpec, pathlib.Path]:
    if args.model:
        key = normalize_model_choice(args.model)
        if key:
            spec = MODEL_SPECS[key]
            return spec, spec.path

        possible_path = pathlib.Path(args.model).expanduser()
        if possible_path.is_file():
            spec = infer_spec_from_model_path(possible_path)
            if spec:
                return spec, possible_path
        print(
            "Error: --model must be letter, number, alphanumeric, or a recognized model path.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.model_file:
        model_path = pathlib.Path(args.model_file).expanduser()
        spec = infer_spec_from_model_path(model_path)
        if not spec:
            print(
                "Error: --model-file must point to a Letter, Number, or Alphanumeric CNN.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return spec, model_path

    spec = MODEL_SPECS[DEFAULT_MODEL_KEY]
    return spec, spec.path


def load_cnn_model(model_path: pathlib.Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        import tensorflow as tf
    except Exception as error:
        raise RuntimeError("TensorFlow is not installed.") from error
    return tf.keras.models.load_model(model_path, compile=False)


def image_to_grayscale(image: Image.Image) -> np.ndarray:
    """Convert to grayscale without losing strokes stored in PNG transparency."""
    if image.mode == "RGBA":
        rgba = np.asarray(image)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]
        if np.ptp(alpha) > 0 and np.ptp(rgb) == 0:
            return alpha.astype(np.uint8)
    return np.asarray(image.convert("L"), dtype=np.uint8)


def segment_image(gray: np.ndarray, spec: ModelSpec) -> list[np.ndarray]:
    if spec.segmentation == "digits":
        segments = segment_digits(gray)
    else:
        segments = [region.image for region in segment_letter_regions(gray)]

    if segments:
        return segments

    binary = binarize_image(gray)
    return [binary] if np.any(binary) else []


def preprocess_character(segment: np.ndarray) -> np.ndarray:
    centered = center_and_resize(segment.astype(np.uint8), size=(28, 28), margin=4)
    return normalize_array(centered).reshape(1, 28, 28, 1)


def predict_image(model, spec: ModelSpec, image: Image.Image) -> tuple[str | None, float | None]:
    segments = segment_image(image_to_grayscale(image), spec)
    if not segments:
        return None, None

    characters: list[str] = []
    confidences: list[float] = []
    for segment in segments:
        outputs = np.asarray(model.predict(preprocess_character(segment), verbose=0))
        if outputs.ndim != 2 or outputs.shape[0] != 1:
            raise ValueError(f"Unexpected model output shape: {outputs.shape}")
        if outputs.shape[1] != len(spec.labels):
            raise ValueError(
                f"{spec.display_name} should have {len(spec.labels)} outputs, "
                f"but the loaded model has {outputs.shape[1]}."
            )
        class_index = int(np.argmax(outputs[0]))
        characters.append(spec.labels[class_index])
        confidences.append(float(outputs[0, class_index]))

    return "".join(characters), float(np.mean(confidences))


def tesseract_ocr(image: Image.Image) -> tuple[str | None, float | None]:
    if pytesseract is None:
        return None, None
    try:
        data = pytesseract.image_to_data(
            image,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
        words: list[str] = []
        confidences: list[float] = []
        for text, raw_confidence in zip(data["text"], data["conf"]):
            text = str(text).strip()
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                continue
            if text and confidence >= 0:
                words.append(text)
                confidences.append(confidence / 100.0)
        confidence = float(np.mean(confidences)) if confidences else None
        return " ".join(words), confidence
    except Exception:
        return None, None


def list_image_files(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path.relative_to(directory)).lower(),
    )


def expected_label(image_path: pathlib.Path) -> str | None:
    parent = image_path.parent.name.strip()
    if not parent:
        return None
    return parent.upper()


def evaluate_status(prediction: str | None, expected: str | None) -> str:
    if not prediction:
        return "CANNOT GUESS"
    if expected is None:
        return "WRONG"
    return "CORRECT" if prediction.strip() == expected else "WRONG"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test a CNN OCR model on an image directory.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run Case 3 handwritten alphabetical evaluation mode.",
    )
    parser.add_argument(
        "--model",
        help="Model to test: letter, number, alphanumeric, or a recognized model path.",
    )
    parser.add_argument(
        "--model-file",
        help="Backward-compatible explicit CNN model path.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output result file.")
    parser.add_argument("--no-tesseract", action="store_true", help="Disable Tesseract fallback.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec, model_path = resolve_model(args)
    images_dir = DEFAULT_IMAGES_DIR

    try:
        model = load_cnn_model(model_path)
    except Exception as error:
        print(f"Error: could not load {spec.display_name}: {error}", file=sys.stderr)
        raise SystemExit(2)

    print(f"Using model: {model_path} ({spec.display_name})")
    image_files = list_image_files(images_dir)
    image_files = image_files[:MAX_IMAGE_COUNT]
    if not image_files:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(3)

    lines: list[str] = []
    correct = 0
    failed = 0

    for image_path in image_files:
        relative_name = str(image_path.relative_to(images_dir))
        prediction = None
        confidence = None
        try:
            with Image.open(image_path) as image:
                prediction, confidence = predict_image(model, spec, image)
                if not prediction and not args.no_tesseract:
                    prediction, confidence = tesseract_ocr(image)
        except Exception as error:
            print(f"Warning: prediction failed for {relative_name}: {error}", file=sys.stderr)

        expected = expected_label(image_path)
        status = evaluate_status(prediction, expected)
        if status == "CORRECT":
            correct += 1
        else:
            failed += 1

        display_prediction = str(prediction or "").replace("\n", " ").strip()
        display_confidence = f"{confidence * 100:.2f}%" if confidence is not None else "N/A"
        line = (
            f'Testing {relative_name} ... Predicted: "{display_prediction}" '
            f"Confidence: {display_confidence}    --> [{status}]"
        )
        print(line)
        lines.append(line)

    total = len(image_files)
    success_rate = correct / total * 100 if total else 0.0
    summary = [
        "",
        "=" * 50,
        "SUMMARY:",
        f"  Model                : {spec.display_name}",
        f"  Image Limit          : {MAX_IMAGE_COUNT}",
        f"  Total Tested         : {total}",
        f"  Correct              : {correct}",
        f"  Failed/Wrong         : {failed}",
        f"  Success Rate         : {success_rate:.1f}%",
        "=" * 50,
    ]
    for line in summary:
        print(line)
        lines.append(line)

    output_path = pathlib.Path(args.output)
    try:
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote results to {output_path}")
    except Exception as error:
        print(f"Warning: could not write output file: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
