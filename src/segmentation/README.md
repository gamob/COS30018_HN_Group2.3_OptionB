### 📂 File Overview: `segmentation.py`
| Function | Purpose |
| :--- | :--- |
| `binarize_image` | Cleans the image and inverts it for better contour detection. |
| `crop_to_content` | Minimizes the bounding box to remove excess empty space around a digit. |
| `segment_digits` | Uses connected components to isolate digits and handles horizontal splitting. |
| `segment_digits_projection` | An alternative method using projection profiles to find gaps between digits. |
