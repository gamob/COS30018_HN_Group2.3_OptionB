# Handwritten Letter and Number Recognition System (HNRS)
Course: COS30018 - Intelligent Systems (Swinburne University of Technology)  
Assignment: Project Assignment - Option B [cite: 2, 3]  
Due Date: 11:59 pm 02/11/2025 (End of Week 12) [cite: 4]  



##  Project Overview
This project implements separate CNN pipelines for handwritten A-Z letters and handwritten digits. The GUI selects the appropriate model, preserves letter word segmentation, and supports multi-digit construction and recognition.[cite: 8, 11]

###  Key Objectives
 Image Acquisition: Loading individual images or auto-creating images from folders[cite: 20].
 Preprocessing: Standardizing input via grayscaling and resizing[cite: 21].
 Segmentation: Partitioning multi-digit numbers into individual digits[cite: 25].
 Classification: Using separate letter and digit CNNs, with Logistic Regression and RBF SVM retained as digit baselines.[cite: 26, 28].
 GUI: Providing a user-friendly interface to control hyper-parameters and visualize results[cite: 17, 18].



##  Repository Structure
We follow a modular structure to allow parallel development as per the project plan[cite: 65].

```text
/
├── .gitignore               # Standard Python ignores
├── README.md                # Project Overview & Integration Rules
├── requirements.txt         # Required libraries (numpy, pandas, torch/tensorflow, etc.) [cite: 14]
├── src/                     # Source Code
│   ├── preprocessing/       # Task 1: Image Preprocessing [cite: 21]
│   ├── segmentation/        # Task 2: Image Segmentation [cite: 25]
│   ├── models/              # Task 3: ML Model Training & Logic [cite: 26]
│   └── gui/                 # GUI Implementation [cite: 17]
├── data/                    # Dataset storage (MNIST) [cite: 15]
├── notebooks/               # Project documentation and experiment planning
├── documents/               # Project reports, assignment brief, and review notes
└── documents/week-reports/ # Weekly progress documentation and report drafts
```



##  Run the Project
Use the package entry points to run preprocessing, training, prediction, and the GUI in a consistent workflow.

- Preprocess images:
  ```bash
  python -m src preprocess data/input_folder output_folder --method otsu --size 28 28
  ```
  
  Preprocessing Task 2: tested on 5+ images from data/sample/part1/image.
  Outputs are 28×28 PNGs (visually clean and centered); pipeline does not crash on blank or very-dark images.
- Train the CNN model on MNIST:
  ```bash
  python -m src train --epochs 5
  ```
- Train the letter CNN from `data/letters`:
  ```bash
  python -m src train-letters --epochs 15
  ```
- Train the SVM baseline on a reproducible, stratified MNIST subset (for model comparison):
  ```bash
  python -m src train-svm --max-train-samples 15000
  ```
  This creates `src/models/digit_svm_model.pkl` and `src/models/digit_svm_metrics.json`. Use
  `--max-train-samples 0` to train on all 60,000 images; this can take substantially longer.
- Predict a single image using the saved model:
  ```bash
  python -m src predict data/sample.png
  ```
- Predict a single letter using the saved letter model:
  ```bash
  python -m src predict-letter data/letters/test/A/000000.png
  ```
- Launch the GUI:
  ```bash
  streamlit run src/gui/app.py
  ```

##  Notebook Documentation
The project documentation notebook is available at:

```text
notebooks/hnrs_project_documentation.ipynb
```

It explains the project workflow, component roles, known limitations, suggested fixes, and recommended experiments for preprocessing, segmentation, model comparison, and end-to-end evaluation.

##  Entry-point modules
The project now supports module-based execution using Python's `-m` switch:
- `python -m src` — unified workflow runner
- `python -m src.preprocessing` — preprocessing CLI
- `python -m src.models.training` — model training
- `python -m src.models.test` — model evaluation/test harness
- `python -m src.gui` — prints the GUI launch command

##  ML Model Commands

### CNN Model (Convolutional Neural Network)

| Task | Command | Details |
|------|---------|---------|
| **Train** | `python -m src train --epochs 5` | Default 5 epochs; saves to `src/models/digit_cnn_model.h5` |
| **Train** | `python -m src train --epochs 10 --output-dir ./models` | Custom epochs and output directory |
| **Test** | `python src/models/test/test_model.py` | Loads model and evaluates on MNIST test set |
| **Predict** | `python -m src predict "data\sample\test\1.png"` | Predicts single digit; auto-preprocesses with Otsu |
| **Predict** | `python -m src predict "data\sample\test\1.png" --method adaptive --blur-ksize 7` | Custom preprocessing before prediction |
| **Train letters** | `python -m src train-letters --epochs 15` | Saves `letter_cnn_model.h5` and its A-Z class mapping |
| **Predict letter** | `python -m src predict-letter "data\letters\test\A\000000.png"` | Predicts one A-Z letter with the letter CNN |

### Logistic Regression Model

| Task | Command | Details |
|------|---------|---------|
| **Train** | `python src/models/logistic_model.py` | Trains on MNIST; flattens 28×28 to 784 features; saves to `src/models/digit_logistic_model.pkl` |
| **Test** | `python src/models/test/test_logistic.py` | Verifies model loads and predicts correctly |
| **Predict** | `python -m src predict "data\sample\test\1.png"` | Same as CNN CLI (select model in code or GUI) |

### SVM Model (Support Vector Machine with RBF kernel)

| Task | Command | Details |
|------|---------|---------|
| **Train** | `python src/models/svm_model.py` | Default: 15,000 stratified samples, C=10.0; saves model and metrics |
| **Train** | `python src/models/svm_model.py --max-train-samples 0` | Train on all 60,000 samples (slower) |
| **Train** | `python src/models/svm_model.py --C 50.0 --gamma 0.001` | Custom hyperparameters |
| **Test** | `python src/models/test/test_svm.py` | Runs stratified sample and prediction tests |
| **Predict** | `python -m src predict-svm "data\sample\test\1.png"` | Predicts using SVM model; auto-preprocesses |
| **Predict** | `python -m src predict-svm "data\sample\test\1.png" --method simple` | Predict with simple thresholding (no OpenCV) |

### Python API Examples

**Digit CNN Prediction:**
```python
from src.models.model import load_digit_cnn_model, predict_digit
model = load_digit_cnn_model("src/models/digit_cnn_model.h5")
prediction = predict_digit(model, preprocessed_image)  # 28×28, normalized 0-1
```

**Letter CNN Prediction:**
```python
from src.models.model import load_letter_cnn_model, predict_letter
model = load_letter_cnn_model("src/models/letter_cnn_model.h5")
prediction = predict_letter(model, preprocessed_image)  # Returns A-Z
```

**Logistic Prediction:**
```python
from src.models.logistic_model import load_logistic_model, predict_digit
model = load_logistic_model("src/models/digit_logistic_model.pkl")
prediction = predict_digit(model, preprocessed_image)  # 28×28, any range
```

**SVM Prediction:**
```python
from src.models.svm_model import load_svm_model, predict_digit
model = load_svm_model("src/models/digit_svm_model.pkl")
prediction = predict_digit(model, preprocessed_image)  # 28×28, any range
```

### Default Model Paths

| Model | Path | Format |
|-------|------|--------|
| Digit CNN | `src/models/digit_cnn_model.h5` | HDF5 |
| Letter CNN | `src/models/letter_cnn_model.h5` | HDF5 |
| Letter class mapping | `src/models/letter_class_mapping.json` | JSON |
| Logistic | `src/models/digit_logistic_model.pkl` | Pickle |
| SVM | `src/models/digit_svm_model.pkl` | Pickle |
| SVM Metrics | `src/models/digit_svm_metrics.json` | JSON |
