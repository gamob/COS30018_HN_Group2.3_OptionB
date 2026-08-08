# Digit Segmentation Workflow

This workflow extracts individual digits from an image using OpenCV image processing techniques.

## Workflow Overview

```text
Input Image
    ↓
Binarization
    ↓
Noise Cleanup
    ↓
Connected Component Detection
    ↓
Wide Region Splitting
    ↓
Crop Individual Digits
    ↓
Save Digit Images
```

---

## 1. Image Binarization

The `binarize_image()` function prepares the image for segmentation.

### Steps

- Convert image to grayscale
- Apply Gaussian blur to reduce noise
- Apply Otsu thresholding
- Invert image if necessary
- Perform morphological cleanup

### Output

Binary image:
- White digits
- Black background

---

## 2. Connected Component Detection

The `segment_digits()` function identifies separate digit regions using:

```python
cv2.connectedComponentsWithStats()
```

Large connected regions are treated as digit bodies. Detached secondary
strokes, such as the base of a handwritten `1`, are attached to the nearest
horizontally overlapping body.

### Filtering

Small noisy regions are removed using:
- minimum area
- minimum width
- minimum height

---

## 3. Wide Region Splitting

Digit segmentation does not split a region solely because its pixel width is
large. This keeps results independent of the input image resolution.

The `split_region_horizontally()` function:
- computes vertical projection
- detects low-density gaps
- splits the region into subregions

This improves segmentation for merged digits.

---

## 4. Cropping Digits

The `crop_to_content()` function removes extra empty space around each digit.

### Result

- tighter bounding box
- cleaner digit samples
- easier downstream OCR/classification

---

## 5. Saving Results

For each processed image:
- a folder is created in `processed/`
- each digit is saved individually

### Example Directory Structure

```text
data/
└── segmentation/
    ├── raw/
    │   └── sample.png
    └── processed/
        └── sample/
            ├── digit_1.png
            ├── digit_2.png
            └── digit_3.png
```

---

## Alternative Method

The script also includes:

```python
segment_digits_projection()
```

This method uses:
- vertical projection analysis
- gap detection between digits

It works best when digits are clearly separated horizontally.

---

## Main Pipeline

```python
Image
 → binarize_image()
 → segment_digits()
 → crop_to_content()
 → save digit images
```

---

## Purpose

This workflow is useful for:
- OCR preprocessing
- handwritten digit extraction
- captcha segmentation
- ML dataset preparation
- computer vision pipelines
