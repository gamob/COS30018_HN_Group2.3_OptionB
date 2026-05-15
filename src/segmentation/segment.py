"""
segment.py — Task 2: Image Segmentation

This module investigates and implements two image segmentation techniques
for partitioning a preprocessed multi-digit image into individual digit
sub-images, ordered left-to-right for correct number construction.

Technique A: Contour Detection (OpenCV findContours)
    - Finds the outlines of connected ink regions using OpenCV.
    - Draws a bounding box around each contour found.
    - Fast and works well when digits are clearly separated.
    - Weakness: if digits touch or overlap, their contours merge into
      one large contour and both digits are returned as one blob.

Technique B: Connected Component Analysis (OpenCV connectedComponentsWithStats)
    - Labels every connected group of white pixels with a unique ID.
    - Gives precise bounding box stats (x, y, w, h, area) per component.
    - Better at filtering noise via minimum area threshold.
    - Weakness: same touching-digit problem as contours; does not split
      merged strokes automatically.

Selected technique: Technique B (Connected Component Analysis)
Reason: Provides cleaner stats per region, easier noise filtering via
area thresholds, and more explicit control over what counts as a digit
vs. a speck. Works better on SVHN-style real-world noisy images.

Integration contract (from src/README.md):
    Input:  clean_image: np.array  — preprocessed float32 image from preprocess.py
                                     shape (H, W), values in [0, 1]
    Output: list[np.array]         — list of individual digit images,
                                     each shape (28, 28), float32, [0, 1],
                                     ordered LEFT TO RIGHT
"""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Output size for each digit (must match ML model input)
DIGIT_SIZE = (28, 28)

# Minimum pixel area to count as a digit (filters out noise specks)
MIN_DIGIT_AREA = 50

# Minimum height ratio relative to image height (filters out tiny noise)
MIN_HEIGHT_RATIO = 0.1

# Padding added around each detected digit bounding box before cropping
BBOX_PADDING = 4


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def _to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert a float32 [0,1] image to uint8 [0,255].
    Accepts already-uint8 images too (passes through unchanged).
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    return image.astype(np.uint8)


def _crop_and_resize(binary: np.ndarray, x: int, y: int,
                     w: int, h: int, padding: int = BBOX_PADDING) -> np.ndarray:
    """
    Crop a digit region from the binary image using its bounding box,
    add padding, resize to DIGIT_SIZE while preserving aspect ratio,
    center on a blank canvas, and normalize to float32 [0,1].

    This matches the center_and_resize logic in preprocess.py so that
    individual digit output format is consistent with MNIST expectations.

    Args:
        binary:  Full binary image (uint8, H x W).
        x, y:    Top-left corner of bounding box.
        w, h:    Width and height of bounding box.
        padding: Extra pixels to include around the bounding box.

    Returns:
        np.ndarray: Shape (28, 28), dtype float32, values in [0, 1].
    """
    img_h, img_w = binary.shape

    # Apply padding, clamped to image boundaries
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    crop = binary[y1:y2, x1:x2]

    if crop.size == 0:
        return np.zeros(DIGIT_SIZE, dtype=np.float32)

    # Resize preserving aspect ratio, then center on 28x28 canvas
    target_w, target_h = DIGIT_SIZE
    margin = 2
    inner_w = target_w - 2 * margin
    inner_h = target_h - 2 * margin

    pil_crop = Image.fromarray(crop)
    cw, ch = pil_crop.size
    scale = min(inner_w / max(cw, 1), inner_h / max(ch, 1))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resized = pil_crop.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (target_w, target_h), color=0)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return np.array(canvas, dtype=np.float32) / 255.0


def _filter_boxes(boxes: List[Tuple[int, int, int, int]],
                  image_height: int) -> List[Tuple[int, int, int, int]]:
    """
    Filter bounding boxes to remove noise specks that are too small
    to be real digits.

    Filters out boxes where:
    - Area (w * h) is below MIN_DIGIT_AREA, OR
    - Height is less than MIN_HEIGHT_RATIO of the full image height

    Args:
        boxes:        List of (x, y, w, h) bounding boxes.
        image_height: Full height of the source image.

    Returns:
        Filtered list of (x, y, w, h) boxes.
    """
    min_h = image_height * MIN_HEIGHT_RATIO
    filtered = []
    for (x, y, w, h) in boxes:
        if w * h >= MIN_DIGIT_AREA and h >= min_h:
            filtered.append((x, y, w, h))
    return filtered


# ─────────────────────────────────────────────
# TECHNIQUE A: Contour Detection
# ─────────────────────────────────────────────

def technique_a_contours(image: np.ndarray) -> List[np.ndarray]:
    """
    Technique A: Digit segmentation via OpenCV Contour Detection.

    Steps:
        1. Convert to uint8 binary image.
        2. Find external contours of all ink regions (RETR_EXTERNAL
           only returns outermost contours, ignoring holes inside digits).
        3. Get bounding box for each contour.
        4. Filter out noise boxes that are too small.
        5. Sort boxes left-to-right by x coordinate.
        6. Crop, resize, and normalize each digit region.

    Strength:  Simple, fast, widely understood.
    Weakness:  Touching digits merge into one contour → missed separation.
               Also sensitive to noise that creates many tiny contours.

    Args:
        image: Preprocessed float32 image, shape (H, W), values [0, 1].

    Returns:
        List of digit arrays, each shape (28, 28), float32, left-to-right.
    """
    binary = _to_uint8(image)
    img_h = binary.shape[0]

    # Find external contours — CHAIN_APPROX_SIMPLE compresses horizontal,
    # vertical, and diagonal segments, saving memory
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding box for each contour
    boxes = [cv2.boundingRect(c) for c in contours]  # (x, y, w, h)

    # Filter noise
    boxes = _filter_boxes(boxes, img_h)

    # Sort left-to-right by x coordinate (critical for correct digit order)
    boxes.sort(key=lambda b: b[0])

    # Crop and resize each digit
    digits = [_crop_and_resize(binary, x, y, w, h) for (x, y, w, h) in boxes]

    return digits


# ─────────────────────────────────────────────
# TECHNIQUE B: Connected Component Analysis [SELECTED]
# ─────────────────────────────────────────────

def technique_b_connected_components(image: np.ndarray) -> List[np.ndarray]:
    """
    Technique B (SELECTED): Digit segmentation via Connected Component Analysis.

    Steps:
        1. Convert to uint8 binary image.
        2. Run connectedComponentsWithStats — labels every connected group
           of white pixels with a unique integer ID and returns stats
           (bounding box + area) for each component.
        3. Skip label 0 (background).
        4. Filter components by area and height to remove noise.
        5. Sort remaining components left-to-right by x position.
        6. Crop, resize, and normalize each digit region.

    Strength:  Gives explicit per-component stats (area, bbox) making
               noise filtering more precise and reliable than contours.
               Less sensitive to contour hierarchy issues.
    Weakness:  Same touching-digit limitation; does not split merged blobs.

    Args:
        image: Preprocessed float32 image, shape (H, W), values [0, 1].

    Returns:
        List of digit arrays, each shape (28, 28), float32, left-to-right.
    """
    binary = _to_uint8(image)
    img_h = binary.shape[0]

    # Connected component analysis
    # Returns: num_labels, label_map, stats, centroids
    # stats columns: CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH,
    #                CC_STAT_HEIGHT, CC_STAT_AREA
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for label in range(1, num_labels):  # Skip label 0 (background)
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, w, h))

    # Filter noise components
    boxes = _filter_boxes(boxes, img_h)

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])

    # Crop and resize each digit
    digits = [_crop_and_resize(binary, x, y, w, h) for (x, y, w, h) in boxes]

    return digits


# ─────────────────────────────────────────────
# MAIN INTERFACE FUNCTION (integration contract)
# ─────────────────────────────────────────────

def segment_digits(clean_image: np.ndarray) -> List[np.ndarray]:
    """
    Main segmentation function — required integration interface.
    (Defined in src/README.md: segment_digits(clean_image: np.array) -> list[np.array])

    Uses the selected technique (Technique B: Connected Component Analysis)
    to partition a preprocessed multi-digit image into individual digit images.

    Args:
        clean_image: Preprocessed image from process_image() in preprocess.py.
                     Shape (H, W), dtype float32, values in [0, 1].
                     Digits should be WHITE (1.0) on BLACK (0.0) background.

    Returns:
        list[np.ndarray]: Individual digit images ordered LEFT TO RIGHT.
                          Each array has shape (28, 28), dtype float32, [0, 1].
                          Returns empty list if no digits found.
    """
    if clean_image is None or clean_image.size == 0:
        return []

    digits = technique_b_connected_components(clean_image)

    return digits


# ─────────────────────────────────────────────
# COMPARISON UTILITY
# ─────────────────────────────────────────────

def compare_techniques(image: np.ndarray, save_path: str = None):
    """
    Visualize Technique A vs Technique B side by side on the same image.
    Shows detected bounding boxes overlaid on the original, plus the
    individual cropped digits each technique produces.

    Used for the report's Image Segmentation comparison section.

    Args:
        image:     Preprocessed float32 image (H, W), values [0, 1].
        save_path: Optional path to save the figure as PNG.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    binary = _to_uint8(image)
    img_h = binary.shape[0]

    # --- Get boxes from each technique ---
    # Technique A boxes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes_a = _filter_boxes([cv2.boundingRect(c) for c in contours], img_h)
    boxes_a.sort(key=lambda b: b[0])

    # Technique B boxes
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes_b = []
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        boxes_b.append((x, y, w, h))
    boxes_b = _filter_boxes(boxes_b, img_h)
    boxes_b.sort(key=lambda b: b[0])

    # --- Get digit crops ---
    digits_a = technique_a_contours(image)
    digits_b = technique_b_connected_components(image)

    # --- Plot ---
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Segmentation Technique Comparison", fontsize=14, fontweight='bold')

    # Row 1: bounding box overlays
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(binary, cmap='gray')
    for (x, y, w, h) in boxes_a:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
    ax1.set_title(f"Technique A: Contour Detection\n({len(boxes_a)} digits found)")
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(binary, cmap='gray')
    for (x, y, w, h) in boxes_b:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax2.add_patch(rect)
    ax2.set_title(f"Technique B: Connected Components ✓ SELECTED\n({len(boxes_b)} digits found)")
    ax2.axis('off')

    # Row 2: individual digit crops
    ax3 = fig.add_subplot(2, 2, 3)
    if digits_a:
        combined_a = np.hstack(digits_a)
        ax3.imshow(combined_a, cmap='gray')
        ax3.set_title(f"Technique A crops ({len(digits_a)} digits)")
    else:
        ax3.set_title("Technique A: No digits found")
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 2, 4)
    if digits_b:
        combined_b = np.hstack(digits_b)
        ax4.imshow(combined_b, cmap='gray')
        ax4.set_title(f"Technique B crops ({len(digits_b)} digits) ✓ SELECTED")
    else:
        ax4.set_title("Technique B: No digits found")
    ax4.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Comparison saved to: {save_path}")

    plt.show()


def save_segments(digits: List[np.ndarray], output_dir: str, prefix: str = "digit"):
    """
    Save a list of digit arrays as PNG files for inspection.

    Args:
        digits:     List of (28, 28) float32 arrays from segment_digits().
        output_dir: Folder to save into (created if it doesn't exist).
        prefix:     Filename prefix (e.g. "digit" → digit_0.png, digit_1.png).
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, digit in enumerate(digits):
        arr_uint8 = (np.clip(digit, 0.0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(arr_uint8, mode="L")
        out_path = os.path.join(output_dir, f"{prefix}_{i}.png")
        img.save(out_path)
        print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────
# STANDALONE TEST (required by src/README.md)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone test — demonstrates this module works independently
    before integration with preprocessing and ML modules.

    Requires preprocess.py to be in src/preprocessing/.

    Usage:
        python segment.py
        python segment.py path/to/image.png
        python segment.py path/to/image.png --save-dir out/segments/
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))

    try:
        from preprocessing import preprocess_image
    except ImportError:
        print("[!] Could not import preprocess.py — make sure it's in src/preprocessing/")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="Segment digits from a preprocessed image")
    parser.add_argument("input", nargs="?", default="data/raw/svhn/1.png",
                        help="Input image path (default: data/raw/svhn/1.png)")
    parser.add_argument("--save-dir", default="out/segments",
                        help="Directory to save cropped digit images")
    parser.add_argument("--method", choices=("otsu", "adaptive", "simple"), default="adaptive",
                        help="Preprocessing method to use before segmenting")
    parser.add_argument("--compare", action="store_true",
                        help="Show side-by-side technique comparison plot")
    args = parser.parse_args()

    print("=" * 55)
    print("  Task 2: Image Segmentation — Standalone Test")
    print("=" * 55)

    if not os.path.exists(args.input):
        print(f"\n[!] Image not found: {args.input}")
        print("    Usage: python segment.py <path_to_image>")
        sys.exit(1)

    print(f"\n[+] Input image  : {args.input}")
    print(f"[+] Preprocess   : {args.method}")

    # Step 1: Preprocess the image (using teammate's preprocess.py)
    clean = preprocess_image(args.input, method=args.method, size=(0, 0), normalize=True)

    # Note: preprocess_image with size=(0,0) returns the full-size image
    # For segmentation we need full size, not 28x28 — load manually if needed
    from PIL import Image as PILImage
    import numpy as _np
    pil = PILImage.open(args.input).convert("L")
    gray = _np.array(pil, dtype=_np.uint8)

    # Apply adaptive threshold manually for full-size output
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clean_full = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 5
    ).astype(_np.float32) / 255.0

    print(f"[+] Clean image shape: {clean_full.shape}")

    # Step 2: Segment with both techniques
    digits_a = technique_a_contours(clean_full)
    digits_b = technique_b_connected_components(clean_full)

    print(f"\n[Technique A] Contour Detection")
    print(f"  Digits found : {len(digits_a)}")

    print(f"\n[Technique B] Connected Component Analysis (SELECTED)")
    print(f"  Digits found : {len(digits_b)}")

    # Step 3: Test main interface function
    result = segment_digits(clean_full)
    print(f"\n[segment_digits()] Integration interface output")
    print(f"  Digits returned : {len(result)}")
    if result:
        print(f"  Each digit shape: {result[0].shape}, dtype: {result[0].dtype}")

    # Step 4: Save individual digit images
    if result:
        print(f"\n[+] Saving digit crops to: {args.save_dir}/")
        save_segments(result, args.save_dir)

    # Step 5: Show comparison if requested
    if args.compare:
        print("\n[+] Displaying comparison plot...")
        compare_techniques(clean_full, save_path="segmentation_comparison.png")

    print(f"\n[✓] Segmentation complete. {len(result)} digit(s) extracted.")