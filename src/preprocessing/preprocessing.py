"""Image preprocessing utilities.

Provides functions to load images, convert to grayscale, apply binarization
using several techniques (simple threshold, Otsu, adaptive), and produce a
centered, fixed-size output suitable for ML training (e.g. 28x28 arrays).

Example:
    from preprocessing import preprocess_image
    arr = preprocess_image("photo.jpg", size=(28,28), method="otsu")
"""
from __future__ import annotations

from typing import Tuple, Union, Iterable
import os

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - cv2 may be missing
    cv2 = None


def load_image(path_or_pil: Union[str, Image.Image]) -> Image.Image:
    """Load image from path or accept an existing PIL image.

    Returns an RGB PIL Image.
    """
    if isinstance(path_or_pil, Image.Image):
        return path_or_pil.convert("RGB")
    path = str(path_or_pil)
    img = Image.open(path)
    return img.convert("RGB")


def to_grayscale(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a grayscale uint8 numpy array.

    Shape: (H, W), dtype: uint8
    """
    return np.array(img.convert("L"), dtype=np.uint8)


def _ensure_cv2():
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required for this function")


def binarize_threshold(gray: np.ndarray, thresh: int = 128, invert: bool = False) -> np.ndarray:
    """Simple fixed threshold binarization.

    Returns uint8 array with values 0 or 255.
    """
    arr = (gray > thresh).astype(np.uint8) * 255
    if invert:
        arr = 255 - arr
    return arr


def binarize_otsu(gray: np.ndarray, blur_ksize: int = 5, invert: bool = False) -> np.ndarray:
    """Otsu's thresholding using OpenCV with optional Gaussian blur.

    Returns uint8 array with values 0 or 255.
    """
    _ensure_cv2()
    if blur_ksize and blur_ksize > 1:
        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        blurred = gray
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        th = 255 - th
    return th


def binarize_adaptive(gray: np.ndarray, block_size: int = 15, C: int = 7, invert: bool = False) -> np.ndarray:
    """Adaptive Gaussian thresholding using OpenCV.

    `block_size` must be odd and >=3.
    Returns uint8 array with values 0 or 255.
    """
    _ensure_cv2()
    if block_size % 2 == 0:
        block_size += 1
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block_size, C)
    if invert:
        th = 255 - th
    return th


def center_and_resize(binary: np.ndarray, size: Tuple[int, int] = (28, 28), margin: int = 4) -> np.ndarray:
    """Crop the binary image to the content bbox, resize while keeping aspect
    ratio, then pad to `size` and center the digit.

    Input `binary` should be uint8 with 0/255 values.
    Returns uint8 array with shape `size` and values 0/255.
    """
    h, w = binary.shape
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        # nothing found; return a centered blank image
        return np.zeros(size, dtype=np.uint8)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = binary[y0:y1 + 1, x0:x1 + 1]

    target_w, target_h = size
    inner_w = max(1, target_w - 2 * margin)
    inner_h = max(1, target_h - 2 * margin)

    pil_crop = Image.fromarray(crop)
    # resize preserving aspect ratio
    cw, ch = pil_crop.size
    scale = min(inner_w / cw, inner_h / ch)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resized = pil_crop.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (target_w, target_h), color=0)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return np.array(canvas, dtype=np.uint8)


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Convert uint8 0/255 array to float32 in range [0, 1]."""
    return (arr.astype(np.float32) / 255.0).astype(np.float32)


def preprocess_image(path_or_img: Union[str, Image.Image, np.ndarray], *,
                     size: Tuple[int, int] = (28, 28),
                     method: str = "otsu",
                     blur_ksize: int = 5,
                     adaptive_params: Tuple[int, int] = (15, 7),
                     thresh: int = 128,
                     invert: bool = False,
                     normalize: bool = True,
                     margin: int = 4) -> np.ndarray:
    """High-level preprocessing pipeline.

    Args:
        path_or_img: path to image or PIL Image or grayscale numpy array.
        size: output size (width, height).
        method: one of 'otsu', 'simple', 'adaptive'.
        blur_ksize: gaussian blur kernel for Otsu (odd int).
        adaptive_params: (block_size, C) for adaptive threshold.
        thresh: fixed threshold for 'simple' method.
        invert: invert foreground/background after thresholding.
        normalize: if True, return float32 array in [0,1], else uint8 0/255.
        margin: pixels of margin around resized content.

    Returns:
        numpy array of shape `size`.
    """
    if isinstance(path_or_img, np.ndarray):
        gray = path_or_img
    else:
        pil = load_image(path_or_img) if not isinstance(path_or_img, Image.Image) else path_or_img
        gray = to_grayscale(pil)

    method = method.lower()
    if method == "simple":
        binary = binarize_threshold(gray, thresh=thresh, invert=invert)
    elif method == "otsu":
        binary = binarize_otsu(gray, blur_ksize=blur_ksize, invert=invert)
    elif method == "adaptive":
        block_size, C = adaptive_params
        binary = binarize_adaptive(gray, block_size=block_size, C=C, invert=invert)
    else:
        raise ValueError(f"unknown method: {method}")

    out = center_and_resize(binary, size=size, margin=margin)
    return normalize_array(out) if normalize else out


def save_array_as_image(arr: np.ndarray, out_path: str) -> None:
    """Save a numpy array (float 0..1 or uint8 0/255) as PNG image."""
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)


def _iter_images(input_path: str) -> Iterable[Tuple[str, str]]:
    """Yield (in_path, out_basename) for files under input_path.

    Accepts a single file or a directory. Filters common image extensions.
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if os.path.isfile(input_path):
        yield input_path, os.path.basename(input_path)
        return
    for root, _, files in os.walk(input_path):
        for fn in files:
            if os.path.splitext(fn.lower())[1] in exts:
                yield os.path.join(root, fn), fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess images into fixed-size arrays")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--method", choices=("otsu", "simple", "adaptive"), default="otsu")
    parser.add_argument("--size", type=int, nargs=2, default=(28, 28))
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()

    for in_path, basename in _iter_images(args.input):
        try:
            out_arr = preprocess_image(in_path, size=tuple(args.size), method=args.method, invert=args.invert, normalize=args.normalize)
            out_name = os.path.splitext(basename)[0] + ".png"
            out_path = os.path.join(args.output, out_name)
            save_array_as_image(out_arr, out_path)
        except Exception as e:
            print(f"failed: {in_path} -> {e}")
