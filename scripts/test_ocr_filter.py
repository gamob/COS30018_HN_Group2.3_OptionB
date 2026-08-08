#!/usr/bin/env python3
"""OCR filter script using EasyOCR (no system Tesseract needed).
- Uses a single EasyOCR Reader instance (singleton) so models / DataLoaders
  are not re-created per image.
- Suppresses repeated pin_memory warnings from torch when no GPU is present.
- Supports --dir, --conf, --move-to and --dry-run.
"""
from pathlib import Path
import argparse
import shutil
import warnings

# Suppress repeated pin_memory warnings from torch (harmless)
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

try:
    import easyocr
except Exception as e:
    raise SystemExit(
        "easyocr is required. Install: pip install easyocr pillow opencv-python\n"
        "See https://www.jaided.ai/easyocr/ for details."
    )

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_reader = None


def get_reader(lang_list=("en",), gpu=False):
    global _reader
    if _reader is None:
        # create once
        _reader = easyocr.Reader(list(lang_list), gpu=gpu)
    return _reader


def ocr_predict_and_conf(img_path: Path):
    """Return (text, conf_percent) where conf_percent is 0-100 or -1 when none."""
    reader = get_reader()
    results = reader.readtext(str(img_path), detail=1)
    if not results:
        return "", -1
    texts = [r[1] for r in results if r[1] and r[1].strip()]
    confs = [float(r[2]) for r in results if isinstance(r[2], (float, int))]
    text = " ".join(texts).strip()
    best_conf = max(confs) if confs else -1.0
    # easyocr confidences are 0..1 -> convert to percent
    try:
        return text, int(best_conf * 100)
    except Exception:
        return text, -1


def main():
    p = argparse.ArgumentParser(description="Filter unreadable images using OCR confidence")
    p.add_argument("--dir", required=True, help="Directory of images to test")
    p.add_argument("--conf", type=int, default=50, help="Minimum confidence (0-100) to keep")
    p.add_argument("--move-to", default=None, help="Folder to move unseen images to (default: <dir>/unseen_removed)")
    p.add_argument("--dry-run", action="store_true", help="Don't move files, only print what would happen")
    p.add_argument("--lang", default="en", help="Language code(s) for EasyOCR (comma-separated), default 'en'")
    p.add_argument("--gpu", action="store_true", help="Enable GPU for EasyOCR if available")
    args = p.parse_args()

    src = Path(args.dir)
    if not src.exists() or not src.is_dir():
        print("Directory not found:", src)
        return

    imgs = sorted([p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not imgs:
        print("No images found.")
        return

    move_to = Path(args.move_to) if args.move_to else src / "unseen_removed"
    if not args.dry_run:
        move_to.mkdir(parents=True, exist_ok=True)

    # initialize reader with requested languages
    langs = [s.strip() for s in args.lang.split(",") if s.strip()]
    get_reader(lang_list=langs or ["en"], gpu=args.gpu)

    results = []
    for img in imgs:
        text, conf = ocr_predict_and_conf(img)
        tag = "[CAN GUESS]" if text and conf >= args.conf else "[CANNOT GUESS]"
        results.append((img, text if text else "???", conf, tag))
        if tag == "[CANNOT GUESS]" and not args.dry_run:
            try:
                shutil.move(str(img), str(move_to / img.name))
            except Exception as ex:
                print(f"Failed to move {img.name}: {ex}")

    # Print per-image
    for img, txt, conf, tag in results:
        print(f'Testing {img.name} ... Predicted: "{txt}"    --> {tag} (conf={conf})')

    total = len(results)
    removed = sum(1 for r in results if r[3] == "[CANNOT GUESS]")
    kept = total - removed
    passed = sum(1 for r in results if r[3] == "[CAN GUESS]")
    failed = 0 if not args.dry_run else removed
    success_rate = (passed / kept * 100.0) if kept else 0.0

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Total Found         : {total}")
    print(f"  Removed (unseen)    : {removed}")
    print(f"  Total Tested         : {kept}")
    print(f"  Passed (Can Guess)   : {passed}")
    print(f"  Failed (Cannot Guess): {failed}")
    print(f"  Success Rate         : {success_rate:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
