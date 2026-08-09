# Preprocessing — Overview

## Purpose

Standardize input images into fixed-size arrays (default 28×28 grayscale) for segmentation and model training. The pipeline produces a centered, optionally normalized numpy array and can save outputs as PNG files.

## Processing flow (function `preprocess_image`)

1. `load_image(path_or_pil)`
   - Open an image from a file path or accept an existing `PIL.Image`; ensure RGB mode.
2. `to_grayscale(img)`
   - Convert the image to a grayscale `uint8` numpy array with shape (H, W).
3. Binarization (choose one method):
   - `binarize_threshold(gray, thresh, invert)` — simple fixed threshold (no OpenCV required).
   - `binarize_otsu(gray, blur_ksize, invert)` — Otsu's thresholding with optional Gaussian blur (requires `opencv-python`).
   - `binarize_adaptive(gray, block_size, C, invert)` — adaptive Gaussian thresholding (requires `opencv-python`).
4. `center_and_resize(binary, size=(28,28), margin)`
   - Crop to the content bounding box, resize while preserving aspect ratio, then pad to the target size and center the content.
5. `normalize_array(arr)` (optional)
   - Convert 0/255 `uint8` data to `float32` in [0, 1] for model input.
6. `save_array_as_image(arr, out_path)`
   - Save the resulting array as a PNG image.

Utility: `_iter_images(input_path)` — iterate over image files in a directory (png/jpg/jpeg/bmp/tif).

## CLI

Usage:

```
python src/preprocessing/preprocessing.py <input> <output> [--method {otsu,simple,adaptive}] [--size W H] [--no-normalize] [--invert]
```

Examples:

```
python src/preprocessing/preprocessing.py data/Word_Level_Training_Set/image tmp/prep_out --method simple --no-normalize
python src/preprocessing/preprocessing.py data/Word_Level_Training_Set/image tmp/prep_out --method otsu
```

## Dependencies

- Required: `numpy`, `Pillow`
- Optional: `opencv-python` (for `otsu` and `adaptive` methods)

## Notes

- Use `--method simple` to avoid the `opencv-python` dependency.
- For large directories or slow filesystems (e.g. OneDrive), test with a single image first.
- If a file fails to process, the script prints: `failed: <path> -> <error>`.
- `preprocess_image` returns a numpy array of shape `size` (dtype `float32` if normalized, or `uint8` if `--no-normalize`).

## Related files

- `preprocessing.py` — contains the implementation of the pipeline and CLI.

---

In short: `preprocess_image` loads and converts an image to grayscale, thresholds it (chosen method), crops + resizes + centers the content, optionally normalizes the array, and returns or saves the result.
