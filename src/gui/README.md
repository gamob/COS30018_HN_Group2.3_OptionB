# Handwritten Letter and Number Recognition GUI

The Streamlit GUI in `src/gui/app.py` supports both recognition pipelines without sharing class mappings or replacing either CNN:

- **Letter CNN model** loads `src/models/letter_cnn_model.h5`, segments characters, restores word gaps, and can apply conservative English spelling correction.
- **Number CNN model** loads the production checkpoint `src/models/digit_cnn_model.h5`, segments digits, predicts each digit, and joins the results from left to right.
- **Alphanumeric CNN model** loads `src/models/alphanumeric_cnn_model.h5` and recognizes mixed `0-9`/`A-Z` strings with one 36-class CNN.
- **Logistic model** and **RBF SVM model** remain available for digit recognition.
- **Mock model** remains available as a UI fallback.

## Input methods

1. Upload a handwritten letter or number image.
2. Load an image from a directory.
3. Create a multi-digit number by selecting individual digit images from a directory in left-to-right order.

Transparent PNG drawings are flattened onto a white background before preprocessing. The GUI retains controls for Otsu, simple, and adaptive binarization, plus preprocessing previews and segmented character/digit thumbnails.

## Run

From the project root:

```bash
python -m streamlit run src/gui/app.py
```

Then select the model matching the input type. The folder-composition input is intended for Number CNN, Logistic, RBF SVM, or Mock.
