import cv2
import numpy as np
import os
import glob
import shutil
from typing import Tuple, List

def binarize_image(image: np.array) -> np.array:
    """
    Convert an image to a clean inverted binary image with white foreground.
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert so the digits are white on a black background
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh


def crop_to_content(region: np.array) -> np.array:
    """
    Crop a binary region to the minimum bounding box containing ink.
    """
    rows = np.any(region, axis=1)
    cols = np.any(region, axis=0)
    if not np.any(rows) or not np.any(cols):
        return region
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return region[y_min:y_max+1, x_min:x_max+1]


def split_region_horizontally(region: np.array):
    """
    Attempt to split a wide region into separate digits using vertical projection.
    """
    projection = np.sum(region, axis=0)
    max_val = np.max(projection)
    if max_val == 0:
        return [(0, region)]

    threshold = max(5, int(max_val * 0.12))
    segments = []
    in_segment = False
    start = 0

    for col_idx, value in enumerate(projection):
        if value >= threshold and not in_segment:
            start = col_idx
            in_segment = True
        elif value < threshold and in_segment:
            segments.append((start, col_idx - 1))
            in_segment = False

    if in_segment:
        segments.append((start, len(projection) - 1))

    if len(segments) <= 1:
        return [(0, region)]

    split_regions = []
    for start, end in segments:
        sub = region[:, start:end + 1]
        sub = crop_to_content(sub)
        if sub.size and np.any(sub):
            split_regions.append((start, sub))

    return split_regions if split_regions else [(0, region)]


def segment_digits(clean_image: np.array) -> list[np.array]:
    """
    Technique 1: Connected Components with Smart Horizontal Splitting.
    Finds isolated white blobs, and splits them if they are too wide.
    """
    thresh = binarize_image(clean_image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    digit_regions = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 50 or w < 5 or h < 5:
            continue

        region = thresh[y:y+h, x:x+w]
        if w > h * 1.8 or w > 100:
            splits = split_region_horizontally(region)
            for offset, subregion in splits:
                digit_regions.append((x + offset, subregion))
        else:
            digit_regions.append((x, region))

    digit_regions.sort(key=lambda item: item[0])
    digit_images = [crop_to_content(region) for _, region in digit_regions]
    return digit_images


def segment_digits_projection(clean_image: np.array) -> list[np.array]:
    """
    Technique 2: Pure Vertical Projection Segmentation.
    Looks for gaps/valleys of empty space across columns to slice digits apart.
    """
    if clean_image.dtype != np.uint8:
        clean_image = (clean_image * 255).astype(np.uint8)
    if len(clean_image.shape) == 3:
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
    
    # Notice: This assumes dark text on light background; inverting gives white on black
    _, thresh = cv2.threshold(clean_image, 127, 255, cv2.THRESH_BINARY_INV)

    projection = np.sum(thresh, axis=0)
    max_proj = np.max(projection)
    
    # Defensive check: if image is completely blank, avoid bad thresholds
    if max_proj == 0:
        return []

    threshold = max_proj * 0.05
    gap_indices = np.where(projection < threshold)[0]

    if len(gap_indices) == 0:
        digit_images = [thresh]
    else:
        digit_boundaries = []
        start = 0
        for i in range(1, len(gap_indices)):
            if gap_indices[i] != gap_indices[i-1] + 1:
                end = gap_indices[i-1]
                digit_boundaries.append((start, end))
                start = gap_indices[i]
        end = gap_indices[-1]
        digit_boundaries.append((start, end))
        digit_boundaries.append((gap_indices[-1] + 1, thresh.shape[1] - 1))

        digit_images = []
        for start_col, end_col in digit_boundaries:
            digit = thresh[:, start_col:end_col+1]
            col_sums = np.sum(digit, axis=1)
            if np.any(col_sums > 0):
                y_min = np.where(col_sums > 0)[0][0]
                y_max = np.where(col_sums > 0)[0][-1]
                digit = digit[y_min:y_max+1, :]
            if digit.size > 0 and np.any(digit):
                digit_images.append(digit)

    return digit_images


def preprocess_image(digit_img: np.array, size: Tuple[int, int] = (28, 28)) -> np.array:
    """
    Preprocess a single digit image: pad to square, resize to `size`, normalize to float32.

    Uses interpolation heuristics to preserve detail: `INTER_AREA` when downscaling,
    `INTER_LANCZOS4` when upscaling.
    """
    target_w, target_h = size

    if digit_img is None or digit_img.size == 0:
        return np.zeros((target_h, target_w), dtype=np.float32)

    h, w = digit_img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w), dtype=np.float32)

    # Pad to square keeping content centered
    if h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        digit_img = cv2.copyMakeBorder(digit_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        digit_img = cv2.copyMakeBorder(digit_img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # Choose interpolation based on scaling factor
    scale = min(target_w / max(1, digit_img.shape[1]), target_h / max(1, digit_img.shape[0]))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4

    resized = cv2.resize(digit_img, (target_w, target_h), interpolation=interp)
    return resized.astype(np.float32) / 255.0


def segment_and_preprocess(image: np.array, method: str = "connected_components", size: Tuple[int, int] = (28, 28)) -> List[np.array]:
    """
    Unified pipeline execution.
    Args:
        image: Input source image matrix.
        method: Either 'connected_components' or 'projection'.
    """
    if image is None or image.size == 0:
        return []

    if method == "connected_components":
        raw_digits = segment_digits(image)
    elif method == "projection":
        raw_digits = segment_digits_projection(image)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    return [preprocess_image(digit, size=size) for digit in raw_digits]


def print_comparison_note():
    """
    Prints an architectural engineering comparison summary between the techniques.
    """
    note = """
================================================================================
🔬 ARCHITECTURAL COMPARISON: CONNECTED COMPONENTS VS. PROJECTION
================================================================================
Which is better? -> WINNER: Connected Components (Technique 1)

WHY?
1. Noise Immunity: Connected Components filters out small stray marks using an 
   'area threshold' (area < 50 pixels). Pure projection gets confused by random 
   specks because any noise creates a column peak, ruining your slice boundaries.
2. Background Independence: Technique 1 uses an adaptive binarize function that 
   automatically handles both dark-on-light or light-on-dark text. Technique 2 
   hardcodes cv2.THRESH_BINARY_INV, meaning it flips behavior if input styles change.
3. Overlapping Protection: Technique 1 actively looks for extra-wide blobs and 
   proactively runs sub-segmentation logic to cut them apart. 

Conclusion: Technique 1 is vastly more stable and production-ready!
================================================================================
"""
    print(note)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "data", "segmentation"))
    raw_dir = os.path.join(base_data_dir, "raw")
    processed_dir = os.path.join(base_data_dir, "processed")

    os.makedirs(raw_dir, exist_ok=True)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    # Gather any real images if they happen to exist
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(raw_dir, ext)))

    # Create synthetic local samples so the fresher can test immediately without data files!
    samples = {}
    
    # Sample A: 3 Digits
    img_3 = np.zeros((100, 200), dtype=np.uint8)
    cv2.putText(img_3, "456", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    samples["3_digits_sample"] = img_3
    
    # Sample B: 1 Digit
    img_1 = np.zeros((100, 200), dtype=np.uint8)
    cv2.putText(img_1, "7", (80, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    samples["1_digit_sample"] = img_1

    # Sample C: 0 Digits (Completely Empty Canvas!)
    img_0 = np.zeros((100, 200), dtype=np.uint8)
    samples["0_digits_blank_sample"] = img_0

    print(f"✨ Created {len(samples)} synthetic test cases to verify stability.")

    # If real images exist, add them to our execution queue
    for path in image_files:
        base = os.path.splitext(os.path.basename(path))[0]
        real_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if real_img is not None:
            samples[f"real_{base}"] = real_img

    # Execute BOTH methods side-by-side on the exact same dataset images
    for name, img_matrix in samples.items():
        print(f"\nEvaluating profile: '{name}'")
        
        # Test Method 1: Connected Components
        digits_cc = segment_and_preprocess(img_matrix, method="connected_components")
        cc_dir = os.path.join(processed_dir, name, "connected_components")
        os.makedirs(cc_dir, exist_ok=True)
        for idx, d in enumerate(digits_cc, 1):
            cv2.imwrite(os.path.join(cc_dir, f"digit_{idx}.png"), (d * 255).astype(np.uint8))
            
        # Test Method 2: Projection Slicing
        digits_proj = segment_and_preprocess(img_matrix, method="projection")
        proj_dir = os.path.join(processed_dir, name, "projection")
        os.makedirs(proj_dir, exist_ok=True)
        for idx, d in enumerate(digits_proj, 1):
            cv2.imwrite(os.path.join(proj_dir, f"digit_{idx}.png"), (d * 255).astype(np.uint8))

        print(f" -> Connected Components extracted: {len(digits_cc)} digits. (No Crash!)")
        print(f" -> Projection Slicing extracted:    {len(digits_proj)} digits. (No Crash!)")

    # Print out our review notes!
    print_comparison_note()