#!/usr/bin/env python3
"""Evaluate Digit CNN on a single image or a directory of images.
Outputs CSV with: filename,predicted,confidence_percent
"""
from pathlib import Path
import argparse
import csv
import numpy as np

from src.models.model import load_digit_cnn_model, predict_digit
from src.preprocessing.preprocessing import preprocess_image_steps


def predict_with_cnn(model, img_path: Path, method: str = "otsu"):
    steps = preprocess_image_steps(
        str(img_path), size=(28, 28), method=method, normalize=True
    )
    img = steps["final"]
    # ensure shape (28,28,1)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    # model predict: returns softmax logits/probs
    img_batch = np.expand_dims(img.astype(np.float32), axis=0)
    outputs = model.predict(img_batch, verbose=0)
    probs = outputs[0]
    class_idx = int(np.argmax(probs))
    conf = float(np.max(probs)) * 100.0
    return class_idx, conf


def main():
    p = argparse.ArgumentParser(description="Evaluate Digit CNN on images")
    p.add_argument("--image", help="Single image to predict")
    p.add_argument("--dir", help="Directory of images to predict")
    p.add_argument("--model-path", default="src/models/digit_cnn_model.h5")
    p.add_argument("--output", default="cnn_predictions.csv", help="CSV output file for batch mode")
    p.add_argument("--method", default="otsu", help="Preprocessing method (otsu/simple/adaptive)")
    args = p.parse_args()

    model = load_digit_cnn_model(args.model_path)

    if args.image:
        idx, conf = predict_with_cnn(model, Path(args.image), method=args.method)
        print(f"{Path(args.image).name}, {idx}, {conf:.1f}%")
        return

    if not args.dir:
        print("Please supply --image or --dir")
        return

    pth = Path(args.dir)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    imgs = sorted([f for f in pth.rglob("*") if f.suffix.lower() in exts])
    if not imgs:
        print("No images found in", pth)
        return

    with open(args.output, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["filename", "predicted", "confidence_percent"])
        for f in imgs:
            try:
                idx, conf = predict_with_cnn(model, f, method=args.method)
                writer.writerow([f.name, idx, f"{conf:.1f}"])
            except Exception as e:
                writer.writerow([f.name, "ERROR", str(e)])
    print(f"Wrote results to {args.output}")


if __name__ == '__main__':
    main()
