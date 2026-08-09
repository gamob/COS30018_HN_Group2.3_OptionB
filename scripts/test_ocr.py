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
import csv
import datetime
import pathlib
import re
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
OUTPUT_RESULTS_DIR = PROJECT_ROOT / "documents" / "model-evaluation" / "Prediction_Results"
DEFAULT_OUTPUT_FILENAME = "test_ocr_results.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def ensure_output_directory(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sanitize_filename_part(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip(" _")
    return value[:200] or "output"


def build_output_path(spec: ModelSpec, images_dir: pathlib.Path) -> pathlib.Path:
    try:
        relative_images_dir = images_dir.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        relative_images_dir = images_dir.resolve()
    model_part = sanitize_filename_part(spec.key)
    data_part = sanitize_filename_part(str(relative_images_dir))
    filename = f"test_ocr_{model_part}_{data_part}.txt"
    return OUTPUT_RESULTS_DIR / filename


def build_versioned_output_path(path: pathlib.Path) -> pathlib.Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


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

    spec = choose_model_interactively()
    return spec, spec.path


def ask_for_image_directory() -> pathlib.Path:
    while True:
        try:
            value = input("Image directory: ").strip()
        except EOFError:
            print("Error: an image directory is required.", file=sys.stderr)
            raise SystemExit(2)
        if not value:
            print("Please enter an image directory.", file=sys.stderr)
            continue
        path = pathlib.Path(value).expanduser()
        if path.is_dir():
            return path
        print(f"Directory not found: {path}", file=sys.stderr)


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

    # Run one Keras prediction per source image instead of one prediction per
    # segmented character. This preserves ordering while avoiding repeated
    # model-call overhead for multi-character strings.
    batch = np.concatenate(
        [preprocess_character(segment) for segment in segments],
        axis=0,
    )
    outputs = np.asarray(model.predict(batch, verbose=0))
    if outputs.ndim != 2 or outputs.shape[0] != len(segments):
        raise ValueError(
            f"Expected one model output per segment ({len(segments)}), "
            f"but received shape {outputs.shape}."
        )
    if outputs.shape[1] != len(spec.labels):
        raise ValueError(
            f"{spec.display_name} should have {len(spec.labels)} outputs, "
            f"but the loaded model has {outputs.shape[1]}."
        )

    class_indices = np.argmax(outputs, axis=1)
    characters = [spec.labels[int(index)] for index in class_indices]
    confidences = outputs[np.arange(len(class_indices)), class_indices]

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


def discover_labels_csv(
    images_dir: pathlib.Path,
    explicit_path: str | None,
) -> pathlib.Path | None:
    """Find an explicit manifest or a labels.csv in an ancestor directory."""
    if explicit_path:
        path = pathlib.Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Labels CSV not found: {path}")
        return path

    current = images_dir.resolve()
    for directory in (current, *current.parents):
        candidate = directory / "labels.csv"
        if candidate.is_file():
            return candidate
        if directory == PROJECT_ROOT:
            break
    return None


def load_csv_labels(labels_path: pathlib.Path) -> dict[pathlib.Path, str]:
    """Map absolute image paths to their expected strings from labels.csv."""
    labels: dict[pathlib.Path, str] = {}
    with labels_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "text"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{labels_path} must contain the columns: filename, text"
            )
        for row_number, row in enumerate(reader, start=2):
            filename = str(row.get("filename", "")).strip()
            expected = str(row.get("text", "")).strip().upper()
            if not filename or not expected:
                raise ValueError(f"Missing filename or text at CSV row {row_number}")
            image_path = (labels_path.parent / pathlib.Path(filename)).resolve()
            if image_path in labels:
                raise ValueError(f"Duplicate filename at CSV row {row_number}: {filename}")
            labels[image_path] = expected
    return labels


def expected_label(
    image_path: pathlib.Path,
    root: pathlib.Path,
    csv_labels: dict[pathlib.Path, str] | None = None,
) -> str | None:
    if csv_labels is not None:
        return csv_labels.get(image_path.resolve())
    relative = image_path.relative_to(root)
    if len(relative.parts) < 2:
        return None
    return relative.parts[-2].strip().upper()


def levenshtein_distance(expected: str, prediction: str) -> int:
    """Return the number of insertions, deletions, and substitutions needed."""
    if len(expected) < len(prediction):
        expected, prediction = prediction, expected
    previous = list(range(len(prediction) + 1))
    for row, expected_character in enumerate(expected, start=1):
        current = [row]
        for column, prediction_character in enumerate(prediction, start=1):
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            substitution = previous[column - 1] + (
                expected_character != prediction_character
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def evaluate_status(
    prediction: str | None,
    expected: str | None,
    edit_distance: int | None = None,
) -> str:
    if not prediction or expected is None:
        return "CANNOT PREDICT"
    distance = (
        edit_distance
        if edit_distance is not None
        else levenshtein_distance(expected, prediction.strip().upper())
    )
    if distance == 0:
        return "CORRECT"
    if distance == 1:
        return "NEAR MATCH"
    return "WRONG"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test a CNN OCR model on an image directory.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode (accepted for compatibility; the image folder is still requested).",
    )
    parser.add_argument(
        "--model",
        help="Model to test: letter, number, alphanumeric, or a recognized model path.",
    )
    parser.add_argument(
        "--model-file",
        help="Backward-compatible explicit CNN model path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output result file. A versioned copy is also saved to the same directory. "
            "When omitted, the file name includes the model and image folder."
        ),
    )
    parser.add_argument(
        "--labels-csv",
        help=(
            "CSV manifest with filename and text columns. When omitted, labels.csv "
            "is discovered automatically in the image directory or its ancestors."
        ),
    )
    parser.add_argument("--no-tesseract", action="store_true", help="Disable Tesseract fallback.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec, model_path = resolve_model(args)
    images_dir = ask_for_image_directory()

    try:
        model = load_cnn_model(model_path)
    except Exception as error:
        print(f"Error: could not load {spec.display_name}: {error}", file=sys.stderr)
        raise SystemExit(2)

    print(f"Using model: {model_path} ({spec.display_name})")
    try:
        labels_path = discover_labels_csv(images_dir, args.labels_csv)
        csv_labels = load_csv_labels(labels_path) if labels_path else None
    except Exception as error:
        print(f"Error: could not load labels CSV: {error}", file=sys.stderr)
        raise SystemExit(2)
    if labels_path:
        print(f"Using labels: {labels_path} ({len(csv_labels or {})} entries)")
    else:
        print("Labels CSV not found; using parent folder names as labels.")

    image_files = list_image_files(images_dir)
    if not image_files:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(3)

    lines: list[str] = [
        f"Model: {spec.display_name} ({model_path})",
        f"Image directory: {images_dir}",
        f"Output directory: {OUTPUT_RESULTS_DIR}",
    ]
    correct = 0
    near_match = 0
    failed = 0
    total_edit_distance = 0
    total_expected_characters = 0
    evaluated_images = 0

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

        expected = expected_label(image_path, images_dir, csv_labels)
        normalized_prediction = str(prediction or "").replace("\n", " ").strip().upper()
        edit_distance = None
        correct_characters = None
        if expected is not None:
            edit_distance = levenshtein_distance(expected, normalized_prediction)
            correct_characters = max(0, len(expected) - edit_distance)
            total_edit_distance += edit_distance
            total_expected_characters += len(expected)
            evaluated_images += 1

        status = evaluate_status(prediction, expected, edit_distance)
        if status == "CORRECT":
            correct += 1
        elif status == "NEAR MATCH":
            near_match += 1
        else:
            failed += 1

        display_expected = expected if expected is not None else "N/A"
        if expected is not None and correct_characters is not None:
            character_percentage = (
                correct_characters / len(expected) * 100 if expected else 0.0
            )
            display_characters = f"{correct_characters}/{len(expected)}"
            display_percentage = f"{character_percentage:.2f}%"
        else:
            display_characters = "N/A"
            display_percentage = "N/A"
        line = (
            f"Testing {relative_name} . . . {display_expected} → {normalized_prediction} | "
            f"{display_characters} · {display_percentage} | [{status}]"
        )
        print(line)
        lines.append(line)

    total = len(image_files)
    character_error_rate = (
        total_edit_distance / total_expected_characters
        if total_expected_characters
        else 0.0
    )
    character_accuracy = max(0.0, 1.0 - character_error_rate)
    average_edit_distance = (
        total_edit_distance / evaluated_images if evaluated_images else 0.0
    )
    success_rate = correct / total * 100 if total else 0.0
    near_and_correct_rate = (near_match + correct) / total * 100 if total else 0.0
    summary = [
        "",
        "-" * 50,
        "SUMMARY:",
        f"  Model                     : {spec.display_name}",
        f"  Total Tested              : {total}",
        "",
        f"  Character Accuracy        : {character_accuracy * 100:.2f}%",
        f"  Character Error Rate      : {character_error_rate * 100:.2f}%",
        f"  Average Edit Distance     : {average_edit_distance:.2f}",
        "",
        f"  Failed/Wrong              : {failed}",
        f"  Near Match                : {near_match}",
        f"  Correct                   : {correct}",
        f"  Success Rate              : {success_rate:.2f}%",
        "",
        f"  Near Match + Correct Rate : {near_and_correct_rate:.2f}%",
        "-" * 50,
    ]
    for line in summary:
        print(line)
        lines.append(line)

    output_path = pathlib.Path(args.output) if args.output else build_output_path(spec, images_dir)
    ensure_output_directory(output_path)
    versioned_output_path = build_versioned_output_path(output_path)
    try:
        output_text = "\n".join(lines) + "\n"
        output_path.write_text(output_text, encoding="utf-8")
        versioned_output_path.write_text(output_text, encoding="utf-8")
        print(f"\nWrote results to {output_path}")
        print(f"Wrote versioned results to {versioned_output_path}")
    except Exception as error:
        print(f"Warning: could not write output file: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
