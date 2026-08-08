#!/usr/bin/env python3

from pathlib import Path
import argparse
from PIL import Image
try:
    from pytesseract import image_to_data, Output
except Exception as e:
    raise SystemExit("pytesseract required. Install: pip install pytesseract pillow opencv-python; and install tesseract binary.") from e

import easyocr

_reader = None

def ocr_predict_and_conf(img_path):
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have CUDA
    results = _reader.readtext(str(img_path), detail=1)  # returns list of (bbox, text, conf)
    if not results:
        return "", -1
    texts = [r[1] for r in results if r[1].strip()]
    confs = [float(r[2]) for r in results if isinstance(r[2], (float,int))]
    text = " ".join(texts).strip()
    best_conf = max(confs) if confs else -1
    # easyocr confidences are 0-1 floats; convert to percent (0-100)
    return text, int(best_conf * 100)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Directory of images to test")
    p.add_argument("--conf", type=int, default=50, help="Minimum confidence to consider 'CAN GUESS' (0-100)")
    p.add_argument("--include-unseen", action="store_true", help="Include unseen images in denominator (do not remove them)")
    args = p.parse_args()

    folder = Path(args.dir)
    imgs = sorted([p for p in folder.rglob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}])
    if not imgs:
        print("No images found.")
        return

    lines = []
    can = 0
    cannot = 0
    removed = 0
    for img in imgs:
        text, conf = ocr_predict_and_conf(img)
        tag = "[CANNOT GUESS]"
        if text and conf >= args.conf:
            tag = "[CAN GUESS]"
            can += 1
        else:
            cannot += 1
        # If removing unseen, we will exclude CANNOT GUESS from the 'tested' count later
        display_text = text if text else "???"
        lines.append((img.name, display_text, conf, tag))

    # Print per-image lines
    for name, text, conf, tag in lines:
        print(f'Testing {name} ... Predicted: "{text}"    --> {tag} (conf={conf})')

    total_images = len(imgs)
    if args.include_unseen:
        tested = total_images
        passed = can
        failed = cannot
    else:
        # remove unseen (CANNOT GUESS) from denominator
        tested = can  # only CAN GUESS are counted as tested after removal
        passed = can
        failed = 0
        removed = cannot

    # Compute success rate: if tested==0 show 0.0%
    success_rate = (passed / tested * 100.0) if tested else 0.0

    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"  Total Found         : {total_images}")
    if not args.include_unseen:
        print(f"  Removed (unseen)    : {removed}")
        print(f"  Total Tested         : {tested}")
        print(f"  Passed (Can Guess)   : {passed}")
        print(f"  Failed (Cannot Guess): {failed}")
    else:
        print(f"  Total Tested         : {tested}")
        print(f"  Passed (Can Guess)   : {passed}")
        print(f"  Failed (Cannot Guess): {failed}")
    print(f"  Success Rate         : {success_rate:.1f}%")
    print("="*50)

if __name__ == "__main__":
    main()