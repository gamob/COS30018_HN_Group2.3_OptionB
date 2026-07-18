# ML Models Command Reference

## Overview
This document contains all commands for training, testing, and predicting across all machine learning models in the handwritten digit recognition project. Each model has a dedicated section for easy navigation and updates.

---

## Table of Contents
1. [CNN Model](#cnn-model)
2. [Logistic Regression Model](#logistic-regression-model)
3. [SVM (Support Vector Machine) Model](#svm-model)
4. [GUI Application](#gui-application)
5. [Preprocessing Commands](#preprocessing-commands)

---

## CNN Model

### Overview
A Convolutional Neural Network trained on MNIST dataset with 5 epochs by default. Architecture includes Conv2D, MaxPooling2D, Flatten, and Dense layers.

### Training
**File:** `src/models/training/train_cnn.py`

```bash
# Train CNN with default settings (5 epochs)
python -m src.models.training.train_cnn

# Train CNN with custom number of epochs
python -m src.models.training.train_cnn --epochs 10

# Train CNN and save model to custom output directory
python -m src.models.training.train_cnn --output-dir ./custom_models --epochs 8
```

### Testing
**File:** `src/models/test/test_model.py`

```bash
# Run CNN model tests
python src/models/test/test_model.py

# Check if trained model exists at default location
# Expected path: src/models/digit_cnn_model.h5
```

### Prediction (Command Line)
**Using main CLI:**

```bash
# Predict a digit using CNN model (requires preprocessed image)
python -m src predict path/to/image.png

# Predict with custom preprocessing method
python -m src predict path/to/image.png --method otsu

# Predict with Gaussian blur for Otsu thresholding
python -m src predict path/to/image.png --method otsu --blur-ksize 7

# Predict with adaptive thresholding
python -m src predict path/to/image.png --method adaptive --adaptive-block-size 21 --adaptive-C 5

# Predict and specify model path explicitly
python -m src predict path/to/image.png --model-path src/models/digit_cnn_model.h5
```

### Prediction (Python API)
```python
from src.models.model import load_digit_cnn_model, predict_digit
import numpy as np

# Load model
model = load_digit_cnn_model("src/models/digit_cnn_model.h5")

# Prepare preprocessed image (28x28, normalized 0-1)
image = np.random.rand(28, 28, 1).astype(np.float32)

# Predict
prediction = predict_digit(model, image)
print(f"Predicted digit: {prediction}")
```

---

## Logistic Regression Model

### Overview
A baseline linear model using scikit-learn's LogisticRegression. Flattens 28x28 images to 784 features. Expected accuracy ~92% on MNIST.

### Training
**File:** `src/models/logistic_model.py`

```bash
# Train Logistic Regression model
python src/models/logistic_model.py

# Train and save to custom directory
python -c "from src.models.logistic_model import train_and_save_logistic_model; from pathlib import Path; train_and_save_logistic_model(Path('./custom_models'))"
```

### Testing
**File:** `src/models/test/test_logistic.py`

```bash
# Run Logistic Regression model tests
python src/models/test/test_logistic.py

# Check if trained model exists at default location
# Expected path: src/models/digit_logistic_model.pkl
```

### Prediction (Command Line)
**Using main CLI:**

```bash
# Predict a digit using Logistic Regression model
python -m src predict path/to/image.png

# Predict with custom preprocessing (no normalization for LR model)
python -m src predict path/to/image.png --method otsu --blur-ksize 5
```

### Prediction (Python API)
```python
from src.models.logistic_model import load_logistic_model, predict_digit
import numpy as np

# Load model
model = load_logistic_model("src/models/digit_logistic_model.pkl")

# Prepare preprocessed image (28x28, uint8 or normalized)
image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

# Predict
prediction = predict_digit(model, image)
print(f"Predicted digit: {prediction}")
```

---

## SVM Model

### Overview
A Support Vector Machine with RBF kernel trained on MNIST. Default uses 15,000 stratified training samples for faster training. Supports custom hyperparameter tuning (C, gamma).

### Training
**File:** `src/models/svm_model.py`

```bash
# Train SVM with default settings (15,000 samples, C=10.0)
python src/models/svm_model.py

# Train SVM on all 60,000 MNIST training samples
python src/models/svm_model.py --max-train-samples 0

# Train SVM with custom hyperparameters
python src/models/svm_model.py --max-train-samples 20000 --C 100.0

# Train SVM with custom gamma parameter
python src/models/svm_model.py --C 10.0 --gamma scale

# Train SVM and save to custom directory
python src/models/svm_model.py --output-dir ./custom_models --max-train-samples 15000

# Full example with all custom parameters
python src/models/svm_model.py --output-dir src/models --max-train-samples 10000 --C 50.0 --gamma 0.001
```

### Testing
**File:** `src/models/test/test_svm.py`

```bash
# Run SVM model tests
python src/models/test/test_svm.py

# Check if trained model and metrics exist
# Expected paths: 
#   - src/models/digit_svm_model.pkl
#   - src/models/digit_svm_metrics.json
```

### Prediction (Command Line)
**Using main CLI:**

```bash
# Predict a digit using SVM model
python -m src predict-svm path/to/image.png

# Predict with custom preprocessing method
python -m src predict-svm path/to/image.png --method simple

# Predict with adaptive thresholding
python -m src predict-svm path/to/image.png --method adaptive --adaptive-block-size 17 --adaptive-C 10

# Predict with model from custom location
python -m src predict-svm path/to/image.png --model-path src/models/digit_svm_model.pkl
```

### Prediction (Python API)
```python
from src.models.svm_model import load_svm_model, predict_digit
import numpy as np

# Load model
model = load_svm_model("src/models/digit_svm_model.pkl")

# Prepare preprocessed image (28x28, uint8 or 0-1 range)
image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

# Predict
prediction = predict_digit(model, image)
print(f"Predicted digit: {prediction}")
```

---

## GUI Application

### Overview
Streamlit-based graphical user interface for interactive image upload, preprocessing preview, and model-based prediction with multiple model options.

### Launch GUI
```bash
# Start the Streamlit GUI application
streamlit run src/gui/app.py

# Specify custom server port
streamlit run src/gui/app.py --server.port 8501

# Run in headless mode (for automation)
streamlit run src/gui/app.py --logger.level=error --client.showErrorDetails=false
```

### GUI Features
- Upload handwritten digit images or load from folder
- Compare multiple preprocessing methods (Otsu, Simple, Adaptive)
- Preview preprocessing pipeline at each stage
- Select model: CNN, Logistic Regression, or SVM
- View segmented digit thumbnails
- Get predicted digits with optional confidence scores

---

## Preprocessing Commands

### Overview
Image preprocessing pipeline for converting raw images to MNIST-compatible 28×28 grayscale format.

### CLI Usage
**File:** `src/preprocessing/preprocessing.py`

```bash
# Preprocess single image with Otsu thresholding
python src/preprocessing/preprocessing.py input.png output_dir --method otsu

# Preprocess directory of images with simple thresholding (no OpenCV needed)
python src/preprocessing/preprocessing.py data/images/ output_dir --method simple

# Preprocess with adaptive thresholding
python src/preprocessing/preprocessing.py data/images/ output_dir --method adaptive

# Preprocess with custom image size
python src/preprocessing/preprocessing.py image.png output_dir --size 32 32

# Preprocess without normalization (keep 0-255 range)
python src/preprocessing/preprocessing.py image.png output_dir --no-normalize

# Preprocess with foreground/background inversion
python src/preprocessing/preprocessing.py image.png output_dir --invert

# Preprocess with custom blur kernel size
python src/preprocessing/preprocessing.py image.png output_dir --method otsu --blur-ksize 7

# Preprocess with custom adaptive thresholding parameters
python src/preprocessing/preprocessing.py image.png output_dir --method adaptive --adaptive-block-size 21 --adaptive-C 10

# Preprocess with custom simple threshold value
python src/preprocessing/preprocessing.py image.png output_dir --method simple --thresh 150

# Preprocess with margin adjustment
python src/preprocessing/preprocessing.py image.png output_dir --margin 2

# Preprocess and save all intermediate steps
python src/preprocessing/preprocessing.py image.png output_dir --save-steps steps

# Full example with all options
python src/preprocessing/preprocessing.py data/input/ data/output --method otsu --size 28 28 --blur-ksize 5 --no-normalize --invert --margin 4
```

### Preprocessing (Python API)
```python
from src.preprocessing.preprocessing import preprocess_image, preprocess_image_steps
import numpy as np

# Preprocess single image
result = preprocess_image(
    "image.png",
    size=(28, 28),
    method="otsu",
    blur_ksize=5,
    normalize=True,
    invert=False,
    margin=4
)

# Get preprocessing pipeline with intermediate steps
steps = preprocess_image_steps(
    "image.png",
    size=(28, 28),
    method="adaptive",
    adaptive_params=(15, 7),
    normalize=True
)

# Access intermediate results
grayscale = steps["grayscale"]       # Converted to grayscale
binary = steps["binary"]             # After binarization
cleaned = steps["cleaned"]           # After morphological cleaning
centered = steps["centered"]         # After centering
final = steps["final"]               # Final 28x28 output
```

---

## Complete Workflow Examples

### Example 1: Train All Models
```bash
# 1. Train CNN
python -m src.models.training.train_cnn --epochs 10

# 2. Train Logistic Regression
python src/models/logistic_model.py

# 3. Train SVM
python src/models/svm_model.py --max-train-samples 15000
```

### Example 2: Test All Models
```bash
# 1. Test CNN
python src/models/test/test_model.py

# 2. Test Logistic Regression
python src/models/test/test_logistic.py

# 3. Test SVM
python src/models/test/test_svm.py
```

### Example 3: Predict Using Different Models
```bash
# Preprocess image first
python src/preprocessing/preprocessing.py sample.png preprocessed_output --method otsu

# Predict with each model
python -m src predict preprocessed_output/sample.png  # CNN
python -m src predict preprocessed_output/sample.png  # Logistic (change in code)
python -m src predict-svm preprocessed_output/sample.png  # SVM
```

### Example 4: Full Pipeline (Preprocess → Segment → Predict)
```bash
# Use GUI for interactive workflow
streamlit run src/gui/app.py

# Or use command line
python src/preprocessing/preprocessing.py raw_image.png prep_output --method otsu
python -m src predict prep_output/raw_image.png
```

---

## Default Model Paths

| Model | Default Save Path | File Format |
|-------|-------------------|-------------|
| CNN | `src/models/digit_cnn_model.h5` | HDF5 |
| Logistic Regression | `src/models/digit_logistic_model.pkl` | Pickle |
| SVM | `src/models/digit_svm_model.pkl` | Pickle |
| SVM Metrics | `src/models/digit_svm_metrics.json` | JSON |

---

## Notes for Maintenance

- **Epochs**: Default CNN training uses 5 epochs; adjust with `--epochs` flag for better accuracy
- **SVM Training**: Default uses 15,000 stratified samples for speed; use `--max-train-samples 0` for full 60,000 training data
- **Image Size**: All models expect 28×28 images; preprocessing ensures this automatically
- **Normalization**: CNN expects normalized images (0-1); Logistic and SVM work with both
- **Model Paths**: Keep default paths consistent or update paths in GUI app (`src/gui/app.py`)
- **OpenCV Dependency**: Only needed for Otsu and Adaptive thresholding; use `--method simple` to avoid it

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'tensorflow'` | Install with `pip install tensorflow` |
| `ModuleNotFoundError: No module named 'cv2'` | Install with `pip install opencv-python` or use `--method simple` |
| Model not found | Check default paths listed above; ensure training completed successfully |
| Prediction inconsistent | Verify preprocessing method matches model training (SVM uses no normalization by default) |
| GUI won't start | Ensure Streamlit is installed: `pip install streamlit` |

---

**Last Updated:** 2026-07-18  
**Repository:** gamob/COS30018_HN_Group2.3_OptionB
