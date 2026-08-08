#!/usr/bin/env python3
"""Evaluate test images and write results in the sample OCR text format.

Usage:
  python scripts/test_ocr.py --dir <images_dir>
  python scripts/test_ocr.py --dir <images_dir> --models-dir src/models
  python scripts/test_ocr.py --dir <images_dir> --output desired_output.txt

The script prefers a supplied model file, otherwise it discovers the first
supported model in --models-dir. If no project model can produce a text result,
it falls back to pytesseract so the output still matches the requested format.
"""

import argparse
import glob
import pathlib
import os
import sys

from PIL import Image
import numpy as np

# Optional imports
try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import joblib
except Exception:
    joblib = None

try:
    import torch
    import torchvision.transforms as T
except Exception:
    torch = None

try:
    import tensorflow as tf
except Exception:
    tf = None

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model import load_alphanumeric_cnn_model, predict_alphanumeric
from src.preprocessing.preprocessing import preprocess_image
from src.segmentation.segmentation import segment_letter_regions

DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "unseen_data" / "Case#1"
DEFAULT_TEST_IMAGES_DIR = DEFAULT_IMAGES_DIR / "Test"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "src" / "models"
DEFAULT_MODEL_FILE = DEFAULT_MODELS_DIR / "alphanumeric_cnn_model.h5"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "desired_output.txt"


def find_model_file(models_dir):
    if not models_dir or not os.path.isdir(models_dir):
        return None
    preferred = pathlib.Path(models_dir) / "alphanumeric_cnn_model.h5"
    if preferred.exists():
        return str(preferred)
    exts = ("*.h5", "*.keras", "*.pkl", "*.joblib", "*.sav", "*.pt", "*.pth", "*.onnx")
    for e in exts:
        files = sorted(glob.glob(os.path.join(models_dir, e)))
        if files:
            return files[0]
    return None

def load_model(model_path):
    if not model_path:
        return None, None
    ext = pathlib.Path(model_path).suffix.lower()
    filename = pathlib.Path(model_path).name.lower()
    try:
        if ext in (".pt", ".pth") and torch:
            model = torch.load(model_path, map_location="cpu")
            # If the saved object is a state_dict, user should customize this loader.
            if isinstance(model, dict) and "state_dict" in model:
                # Can't know architecture — user must integrate; return raw dict.
                return model, "pytorch_state_dict"
            if hasattr(model, "eval"):
                model.eval()
            return model, "pytorch"
        if ext in (".h5", ".keras") and tf:
            if filename == "alphanumeric_cnn_model.h5":
                model = load_alphanumeric_cnn_model(model_path)
                return model, "alphanumeric_cnn"
            model = tf.keras.models.load_model(model_path)
            return model, "keras"
        if ext in (".pkl", ".joblib", ".sav") and joblib:
            model = joblib.load(model_path)
            return model, "sklearn"
        if ext == ".onnx":
            # ONNX requires onnxruntime; user can add support later
            return None, None
    except Exception as e:
        print(f"Warning: failed to load model {model_path}: {e}", file=sys.stderr)
        return None, None
    return None, None

def preprocess_for_pytorch(image, size=(224,224)):
    # basic preprocess: convert to RGB, resize, to tensor, normalize
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(image).unsqueeze(0)  # batch dim

def predict_alphanumeric_text(model, pil_img):
    gray = np.array(pil_img.convert("L"), dtype=np.uint8)
    segments = segment_letter_regions(gray)
    if not segments:
        processed = preprocess_image(
            gray,
            size=(28, 28),
            method="otsu",
            blur_ksize=5,
            adaptive_params=(15, 7),
            thresh=128,
            invert=False,
            normalize=True,
            margin=4,
        )
        return predict_alphanumeric(model, processed)

    predictions = []
    for segment in segments:
        padding = max(2, int(round(max(segment.image.shape) * 0.15)))
        padded = np.pad(segment.image, padding, mode="constant", constant_values=0)
        processed = preprocess_image(
            padded,
            size=(28, 28),
            method="otsu",
            blur_ksize=5,
            adaptive_params=(15, 7),
            thresh=128,
            invert=False,
            normalize=True,
            margin=4,
        )
        predictions.append(predict_alphanumeric(model, processed))

    return "".join(predictions)


def predict_with_model(model_tuple, pil_img, image_name=""):
    model, mtype = model_tuple
    if model is None:
        return None
    try:
        if mtype == "alphanumeric_cnn":
            try:
                return predict_alphanumeric_text(model, pil_img)
            except Exception as error:
                print(f"Warning: alphanumeric prediction failed for {image_name}: {error}", file=sys.stderr)
                return None
        if mtype == "pytorch" and torch:
            try:
                x = preprocess_for_pytorch(pil_img)
                with torch.no_grad():
                    out = model(x)
                # best-effort: if classification logits, pick top-1 and return str
                if hasattr(out, "argmax"):
                    pred_idx = int(out.argmax(dim=1).cpu().numpy()[0])
                    # we don't have label mapping; return index as a string
                    return str(pred_idx)
                # if model returns text directly (rare), convert to str
                return str(out)
            except Exception:
                return None
        if mtype == "keras" and tf:
            try:
                arr = np.array(pil_img.convert("RGB").resize((224,224))).astype("float32")/255.0
                arr = np.expand_dims(arr, 0)
                out = model.predict(arr)
                # best-effort: if a vector, return argmax index
                if isinstance(out, np.ndarray):
                    if out.ndim == 2:
                        pred_idx = int(out.argmax(axis=1)[0])
                        return str(pred_idx)
                    return str(out.tolist())
                return str(out)
            except Exception as error:
                print(f"Warning: keras prediction failed for {image_name}: {error}", file=sys.stderr)
                return None
        if mtype in ("sklearn", "sklearn-like"):
            try:
                arr = np.array(pil_img.convert("L").resize((28, 28))).astype("float32").flatten()[None, :]
                if hasattr(model, "predict"):
                    out = model.predict(arr)
                    if isinstance(out, (list, tuple, np.ndarray)):
                        return str(out[0])
                    return str(out)
            except Exception as error:
                print(f"Warning: sklearn prediction failed for {image_name}: {error}", file=sys.stderr)
                return None
        if mtype == "pytorch_state_dict":
            # can't use without model architecture
            return None
    except Exception as error:
        print(f"Warning: prediction failed for {image_name}: {error}", file=sys.stderr)
        return None
    return None

def tesseract_ocr(pil_img):
    if not pytesseract:
        return None
    try:
        txt = pytesseract.image_to_string(pil_img, config='--psm 6').strip()
        return txt
    except Exception:
        return None

def is_cannot_guess(pred):
    if pred is None:
        return True
    s = str(pred).strip()
    if s == "" or s == "???" or s.lower() == "unknown":
        return True
    return False


def normalize_label(value: str) -> str:
    return str(value).strip().lower()


def get_expected_label(image_path: pathlib.Path, root: pathlib.Path) -> str:
    relative = image_path.relative_to(root)
    if len(relative.parts) < 2:
        return ""
    return normalize_label(relative.parts[-2])

def list_image_files(dirpath):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp", "*.gif")
    root = pathlib.Path(dirpath)
    files = []
    for ext in exts:
        files.extend(root.rglob(ext))
    files = sorted(files, key=lambda p: str(p.relative_to(root)).lower())
    return files

def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR model(s) on a directory of test images.")
    parser.add_argument("--dir", default=str(DEFAULT_IMAGES_DIR), help="Directory containing test images")
    parser.add_argument("--test", action="store_true", help="Force evaluation on Case#1\\Test only")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Directory containing model files")
    parser.add_argument("--model-file", default=str(DEFAULT_MODEL_FILE), help="Specific model file to load (overrides models-dir)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output text file")
    parser.add_argument("--no-tesseract", action="store_true", help="Disable fallback to pytesseract")
    args = parser.parse_args()

    images_dir = DEFAULT_TEST_IMAGES_DIR if args.test else pathlib.Path(args.dir)
    if not images_dir.is_dir():
        print(f"Error: images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(2)

    model_file = args.model_file
    if model_file and not os.path.isfile(model_file) and args.models_dir:
        model_file = find_model_file(args.models_dir)
    elif not model_file and args.models_dir:
        model_file = find_model_file(args.models_dir)

    model, mtype = load_model(model_file) if model_file else (None, None)
    if model_file:
        print(f"Using model: {model_file} (type={mtype})")
    else:
        print("No usable model file found. Will use pytesseract if available (unless --no-tesseract).")

    model_tuple = (model, mtype)

    image_files = list_image_files(images_dir)
    if not image_files:
        print("No images found in directory.", file=sys.stderr)
        sys.exit(3)

    lines = []
    total = 0
    passed = 0
    failed = 0

    for img_path in image_files:
        total += 1
        relative_name = str(img_path.relative_to(images_dir))
        expected_label = get_expected_label(img_path, images_dir)
        try:
            pil_img = Image.open(img_path)
        except Exception as e:
            print(f"Warning: failed to open {relative_name}: {e}", file=sys.stderr)
            pred = ""
            cannot = True
        else:
            pred = None
            # 1) Try model if available
            if model is not None:
                p = predict_with_model(model_tuple, pil_img, relative_name)
                if p is not None:
                    pred = p
            # 2) Fall back to tesseract if enabled
            if (pred is None or str(pred).strip() == "") and (not args.no_tesseract) and pytesseract:
                p2 = tesseract_ocr(pil_img)
                if p2 is not None:
                    pred = p2
            if pred is None:
                pred = ""
            cannot = is_cannot_guess(pred)

        predicted_label = normalize_label(pred)
        if cannot:
            status = "CANNOT GUESS"
            failed += 1
        elif expected_label and predicted_label == expected_label:
            status = "CORRECT"
            passed += 1
        else:
            status = "WRONG"
            failed += 1

        # Format: Testing image_001.png ... Predicted: "Coffee Shop"    --> [CORRECT]
        display_pred = str(pred).replace('\n', ' ').strip()
        # keep empty string shown as "" in output
        line = f'Testing {relative_name} ... Predicted: "{display_pred}"    --> [{status}]'
        print(line)
        lines.append(line)

    # summary block
    success_rate = (passed/total*100) if total > 0 else 0.0
    summary = []
    summary.append("")
    summary.append("="*50)
    summary.append("SUMMARY:")
    summary.append(f"  Total Tested         : {total}")
    summary.append(f"  Passed (Correct)     : {passed}")
    summary.append(f"  Failed (Wrong/Guess) : {failed}")
    # format with one decimal if needed, match sample like 66.7%
    summary.append(f"  Success Rate         : {success_rate:.1f}%")
    summary.append("="*50)

    for s in summary:
        print(s)
        lines.append(s)

    # write to output file
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            for l in lines:
                f.write(l + "\n")
        print(f"\nWrote results to {args.output}")
    except Exception as e:
        print(f"Warning: could not write output file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()