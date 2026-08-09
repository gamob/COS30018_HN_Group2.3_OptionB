#!/usr/bin/env python3
"""Test the project's SVM or Logistic Regression OCR model on an image folder.

Examples:
  python scripts/test_ocr_4.py
  python scripts/test_ocr_4.py --model svm
  python scripts/test_ocr_4.py --model logistic

When ``--model`` is omitted, the script asks which model to use. It always asks
for the image directory; there is deliberately no default image directory.
Labels are read from ``labels.csv`` when one is available. Otherwise, the
script uses each image's parent-folder name as its expected label.
"""

from __future__ import annotations

import argparse
import csv
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

from src.models.logistic_model import load_logistic_model
from src.models.svm_model import load_svm_model
from src.preprocessing.preprocessing import center_and_resize, normalize_array
from src.segmentation.segmentation import binarize_image, segment_digits


MODELS_DIR = PROJECT_ROOT / "src" / "models"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "desired_output.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    filename: str

    @property
    def path(self) -> pathlib.Path:
        return MODELS_DIR / self.filename


MODEL_SPECS = {
    "svm": ModelSpec(
        key="svm",
        display_name="RBF SVM",
        filename="digit_svm_model.pkl",
    ),
    "logistic": ModelSpec(
        key="logistic",
        display_name="Logistic Regression",
        filename="digit_logistic_model.pkl",
    ),
}

MODEL_ALIASES = {
    "1": "svm",
    "svm": "svm",
    "rbf-svm": "svm",
    "rbf_svm": "svm",
    "digit_svm_model.pkl": "svm",
    "2": "logistic",
    "logistic": "logistic",
    "logistic-regression": "logistic",
    "logistic_regression": "logistic",
    "digit_logistic_model.pkl": "logistic",
}


def normalize_model_choice(value: str) -> str | None:
    normalized = str(value).strip().lower().replace(" ", "-")
    return MODEL_ALIASES.get(normalized)


def choose_model_interactively() -> ModelSpec:
    print("Select model:")
    print("  1. RBF SVM")
    print("  2. Logistic Regression")
    while True:
        try:
            choice = input("Model [1-2]: ").strip()
        except EOFError:
            print("Error: a model selection is required.", file=sys.stderr)
            raise SystemExit(2)
        key = normalize_model_choice(choice)
        if key:
            return MODEL_SPECS[key]
        print("Please choose 1 or 2.", file=sys.stderr)


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
            "Error: --model must be svm, logistic, or a recognized model path.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.model_file:
        model_path = pathlib.Path(args.model_file).expanduser()
        spec = infer_spec_from_model_path(model_path)
        if not spec:
            print(
                "Error: --model-file must point to the SVM or Logistic model.",
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


def load_selected_model(spec: ModelSpec, model_path: pathlib.Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if spec.key == "svm":
        return load_svm_model(model_path)
    return load_logistic_model(str(model_path))


def image_to_grayscale(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.uint8)


def segment_image(gray: np.ndarray) -> list[np.ndarray]:
    segments = segment_digits(gray)
    if segments:
        return segments
    binary = binarize_image(gray)
    return [binary] if np.any(binary) else []


def preprocess_character(segment: np.ndarray) -> np.ndarray:
    centered = center_and_resize(segment.astype(np.uint8), size=(28, 28), margin=4)
    return normalize_array(centered)


def predict_image(model, image: Image.Image) -> str | None:
    segments = segment_image(image_to_grayscale(image))
    if not segments:
        return None

    batch = np.stack([preprocess_character(segment).reshape(-1) for segment in segments])
    predictions = np.asarray(model.predict(batch)).reshape(-1)
    if len(predictions) != len(segments):
        raise ValueError(
            f"Expected one model output per segment ({len(segments)}), "
            f"but received {len(predictions)}."
        )
    return "".join(str(prediction) for prediction in predictions)


def tesseract_ocr(image: Image.Image) -> str | None:
    if pytesseract is None:
        return None
    try:
        return pytesseract.image_to_string(image, config="--psm 6").strip()
    except Exception:
        return None


def list_image_files(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
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
    """Map absolute image paths to expected strings from labels.csv."""
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
                raise ValueError(
                    f"Duplicate filename at CSV row {row_number}: {filename}"
                )
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
    parser = argparse.ArgumentParser(
        description="Test the SVM or Logistic OCR model on an image directory."
    )
    parser.add_argument("--model", help="Model to test: svm or logistic.")
    parser.add_argument(
        "--model-file",
        help="Explicit digit_svm_model.pkl or digit_logistic_model.pkl path.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output result file.")
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
        model = load_selected_model(spec, model_path)
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

    lines: list[str] = []
    correct = 0
    near_match = 0
    failed = 0
    total_edit_distance = 0
    total_expected_characters = 0
    evaluated_images = 0

    for image_path in image_files:
        relative_name = str(image_path.relative_to(images_dir))
        prediction = None
        try:
            with Image.open(image_path) as image:
                prediction = predict_image(model, image)
                if not prediction and not args.no_tesseract:
                    prediction = tesseract_ocr(image)
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

    output_path = pathlib.Path(args.output)
    try:
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote results to {output_path}")
    except Exception as error:
        print(f"Warning: could not write output file: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
