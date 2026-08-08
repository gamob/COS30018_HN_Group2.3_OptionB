#!/usr/bin/env python3
"""
test_ocr.py

Usage:
  python test_ocr.py --dir <images_dir> [--models-dir <models_dir>] [--model-file <model_path>] [--output desired_output.txt] [--no-tesseract]

This script will:
 - Try to load a model from --model-file or the first file discovered in --models-dir.
 - If a model cannot be used, fall back to pytesseract.
 - Run OCR on every image file in the given directory and produce an output file matching the sample format.

Notes:
 - You may need to install dependencies:
     pip install pillow pytesseract numpy opencv-python joblib
   And optionally:
     pip install torch        # if using PyTorch models
     pip install tensorflow   # if using Keras .h5 models
 - If you rely on Tesseract, ensure the tesseract binary is installed and on PATH.
"""

import os
import sys
import argparse
import glob
import pathlib
import math

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

def find_model_file(models_dir):
    if not models_dir or not os.path.isdir(models_dir):
        return None
    # pick the first plausible model file
    exts = ("*.pt", "*.pth", "*.h5", "*.keras", "*.pkl", "*.joblib", "*.sav", "*.onnx")
    for e in exts:
        files = sorted(glob.glob(os.path.join(models_dir, e)))
        if files:
            return files[0]
    return None

def load_model(model_path):
    if not model_path:
        return None, None
    ext = pathlib.Path(model_path).suffix.lower()
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

def predict_with_model(model_tuple, pil_img):
    model, mtype = model_tuple
    if model is None:
        return None
    try:
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
            except Exception:
                return None
        if mtype in ("sklearn", "sklearn-like"):
            try:
                arr = np.array(pil_img.convert("L").resize((128, 128))).astype("float32").flatten()[None, :]
                if hasattr(model, "predict"):
                    out = model.predict(arr)
                    if isinstance(out, (list, tuple, np.ndarray)):
                        return str(out[0])
                    return str(out)
            except Exception:
                return None
        if mtype == "pytorch_state_dict":
            # can't use without model architecture
            return None
    except Exception:
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

def list_image_files(dirpath):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp", "*.gif")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(dirpath, e)))
    files = sorted(files, key=lambda p: os.path.basename(p).lower())
    return files

def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR model(s) on a directory of test images.")
    parser.add_argument("--dir", required=True, help="Directory containing test images (e.g., ./test_samples/)")
    parser.add_argument("--models-dir", default=None, help="Directory containing model files")
    parser.add_argument("--model-file", default=None, help="Specific model file to load (overrides models-dir)")
    parser.add_argument("--output", default="desired_output.txt", help="Output text file (default desired_output.txt)")
    parser.add_argument("--no-tesseract", action="store_true", help="Disable fallback to pytesseract")
    args = parser.parse_args()

    images_dir = args.dir
    if not os.path.isdir(images_dir):
        print(f"Error: images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(2)

    model_file = args.model_file
    if not model_file and args.models_dir:
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
        filename = os.path.basename(img_path)
        try:
            pil_img = Image.open(img_path)
        except Exception as e:
            pred = ""
            note = f"ERROR opening image: {e}"
            cannot = True
        else:
            pred = None
            # 1) Try model if available
            if model is not None:
                p = predict_with_model(model_tuple, pil_img)
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

        status = "CANNOT GUESS" if cannot else "CAN GUESS"
        if cannot:
            failed += 1
        else:
            passed += 1

        # Format: Testing image_001.png ... Predicted: "Coffee Shop"    --> [CAN GUESS]
        display_pred = str(pred).replace('\n', ' ').strip()
        # keep empty string shown as "" in output
        line = f'Testing {filename} ... Predicted: "{display_pred}"    --> [{status}]'
        print(line)
        lines.append(line)

    # summary block
    success_rate = (passed/total*100) if total > 0 else 0.0
    summary = []
    summary.append("")
    summary.append("="*50)
    summary.append("SUMMARY:")
    summary.append(f"  Total Tested         : {total}")
    summary.append(f"  Passed (Can Guess)   : {passed}")
    summary.append(f"  Failed (Cannot Guess): {failed}")
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