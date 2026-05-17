import cv2
import numpy as np
import os
import glob
import shutil

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

    Returns a list of (x_offset, subregion) tuples, where x_offset is the
    column offset relative to the original region.
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
    Segments a cleaned image into individual digit images using connected components.

    This function extracts each digit region and then splits wide regions if needed.
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
    Alternative Technique 2: Horizontal Projection for Segmentation.

    This method uses vertical projection to find gaps between digits.
    Assumes digits are separated horizontally and not touching.

    Args:
        clean_image (np.array): A binary image array.

    Returns:
        list[np.array]: List of digit images.
    """
    # Ensure binary
    if clean_image.dtype != np.uint8:
        clean_image = (clean_image * 255).astype(np.uint8)
    if len(clean_image.shape) == 3:
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(clean_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Vertical projection: sum along columns (horizontal projection across image)
    projection = np.sum(thresh, axis=0)

    # Find gaps (where projection is low, indicating space between digits)
    threshold = np.max(projection) * 0.05  # 5% of max as gap threshold
    gap_indices = np.where(projection < threshold)[0]

    # Group consecutive gaps to find digit boundaries
    if len(gap_indices) == 0:
        # No gaps, treat as one digit
        digit_images = [thresh]
    else:
        # Find start and end of each digit
        digit_boundaries = []
        start = 0
        for i in range(1, len(gap_indices)):
            if gap_indices[i] != gap_indices[i-1] + 1:
                end = gap_indices[i-1]
                digit_boundaries.append((start, end))
                start = gap_indices[i]
        end = gap_indices[-1]
        digit_boundaries.append((start, end))
        # Last digit to end of image
        digit_boundaries.append((gap_indices[-1] + 1, thresh.shape[1] - 1))

        digit_images = []
        for start_col, end_col in digit_boundaries:
            digit = thresh[:, start_col:end_col+1]
            # Crop vertically to remove empty rows
            col_sums = np.sum(digit, axis=1)
            if np.any(col_sums > 0):
                y_min = np.where(col_sums > 0)[0][0]
                y_max = np.where(col_sums > 0)[0][-1]
                digit = digit[y_min:y_max+1, :]
            digit_images.append(digit)

    return digit_images

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "data", "segmentation"))
    raw_dir = os.path.join(base_data_dir, "raw")
    processed_dir = os.path.join(base_data_dir, "processed")

    # Ensure raw directory exists
    os.makedirs(raw_dir, exist_ok=True)

    # Rebuild processed directory on each run
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    # Find all image files in data/segmentation/raw
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(raw_dir, ext)))

    if not image_files:
        print(f"No image files found in {raw_dir}. Please add some images to test.")
        # Fallback to test image
        print("Creating test image...")
        img = np.zeros((100, 200), dtype=np.uint8)
        cv2.putText(img, "456", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        digits = segment_digits(img)
        print(f"Test segmented into {len(digits)} digits")
        fallback_dir = os.path.join(processed_dir, "test_image")
        os.makedirs(fallback_dir, exist_ok=True)
        for i, digit in enumerate(digits, start=1):
            output_path = os.path.join(fallback_dir, f"digit_{i}.png")
            cv2.imwrite(output_path, digit)
            print(f"Saved {output_path}")
    else:
        for image_path in image_files:
            print(f"Processing {image_path}...")
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load {image_path}")
                continue

            # Use contour detection as default
            digits = segment_digits(img)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            image_output_dir = os.path.join(processed_dir, base_name)
            os.makedirs(image_output_dir, exist_ok=True)

            for i, digit in enumerate(digits, start=1):
                output_path = os.path.join(image_output_dir, f"digit_{i}.png")
                cv2.imwrite(output_path, digit)
                print(f"Saved {output_path}")

            print(f"Processed {image_path}: {len(digits)} digits extracted into {image_output_dir}.")