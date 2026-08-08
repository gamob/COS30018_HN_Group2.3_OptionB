#!/usr/bin/env python3
"""Batch prediction helper for RBF SVM model.
Outputs CSV: filename,predicted
"""
from pathlib import Path
import sys
# Ensure project root is on sys.path so `src` package can be imported when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv

from src.models.svm_model import load_svm_model, predict_digit
from src.preprocessing.preprocessing import preprocess_image_steps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    p = argparse.ArgumentParser(description="Batch predict with saved SVM model")
    p.add_argument("--dir", required=True, help="Directory of images to predict")
    p.add_argument("--model-path", default="src/models/digit_svm_model.pkl")
    p.add_argument("--output", default="svm_predictions.csv")
    p.add_argument("--method", default="otsu", help="Preprocessing method: otsu/simple/adaptive")
    args = p.parse_args()

    model = load_svm_model(args.model_path)
    pth = Path(args.dir)
    imgs = sorted([f for f in pth.rglob("*") if f.suffix.lower() in IMAGE_EXTS])
    if not imgs:
        print("No images found in", pth)
        return

    with open(args.output, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["filename", "predicted"])
        for f in imgs:
            try:
                steps = preprocess_image_steps(str(f), size=(28,28), method=args.method, normalize=False)
                pred = predict_digit(model, steps["final"])
                writer.writerow([f.name, pred])
            except Exception as e:
                writer.writerow([f.name, f"ERROR: {e}"])
    print(f"Wrote results to {args.output}")


if __name__ == '__main__':
    main()
