# Jupyter Notebook Review and Academic Project Guidance

## Review Scope

This review was requested for Jupyter Notebook (`.ipynb`) files in the project folder. A recursive scan of the repository found no notebook files, and Git does not currently track any `.ipynb` files.

Because notebooks are not present, this document reviews the notebook gap and maps the existing Python project structure to the COS30018 Handwritten Number Recognition System requirements. It also recommends the notebooks that should be added if the project team wants clear experimental evidence for the final report.

## Assignment Requirements Relevant to Notebooks

The assignment expects evidence of research, experimentation, comparison, implementation, integration, and evaluation across:

1. Image preprocessing techniques.
2. Image segmentation techniques.
3. Handwritten digit recognition models trained on MNIST.
4. System integration and performance testing.
5. GUI-based user interaction and image acquisition.

Notebooks are not strictly required by the brief, but they are highly useful as academic evidence because they can show experiments, intermediate images, metrics, plots, and model comparisons in a readable sequence.

## Current Project Role and Structure

The repository is currently structured as a Python application rather than a notebook-led research project.

- `src/preprocessing/preprocessing.py` implements grayscale conversion, thresholding, morphological cleaning, centering, resizing, and normalization.
- `src/segmentation/segmentation.py` implements connected-component segmentation and projection-based segmentation.
- `src/models/training/train_cnn.py` trains a CNN on MNIST and saves a Keras model.
- `src/models/logistic_model.py` trains and saves a Logistic Regression baseline on MNIST.
- `src/models/model.py` loads the CNN model and predicts a single digit.
- `src/gui/app.py` provides a Streamlit interface for image upload, preprocessing preview, model selection, and prediction.
- `scripts/run_pipeline.py` runs a sample image pipeline across acquisition, preprocessing, and segmentation.

This is a reasonable software structure for the final system. The main missing academic artifact is a clear experiment log showing why the chosen methods were selected.

## Recommended Notebook 1: Preprocessing Experiments

### Purpose

This notebook should justify the selected preprocessing technique for Task 1.

### Suggested Workflow

1. Load representative handwritten digit and number images.
2. Display original images.
3. Compare grayscale conversion, simple thresholding, Otsu thresholding, and adaptive thresholding.
4. Apply morphological cleaning.
5. Center and resize outputs to 28x28.
6. Show side-by-side visual comparisons.
7. Record strengths, weaknesses, and the selected method.

### Issues to Address

- The current preprocessing implementation supports multiple techniques, but the project does not yet show experimental evidence comparing them.
- The `invert` setting must be chosen carefully because MNIST-style models usually expect light foreground digits on a dark background after normalization.
- The report should explain why Otsu thresholding is suitable or when adaptive thresholding performs better.

### Contribution to Requirements

This notebook would directly support Task 1 and the report section on data preprocessing.

## Recommended Notebook 2: Segmentation Experiments

### Purpose

This notebook should justify the selected segmentation approach for Task 2.

### Suggested Workflow

1. Load single-digit and multi-digit images.
2. Apply the selected preprocessing step.
3. Run connected-component segmentation.
4. Run projection-based segmentation.
5. Display extracted digit regions in left-to-right order.
6. Compare success and failure cases.
7. Summarize which method is selected for integration.

### Issues to Address

- Connected components are likely more robust to isolated noise and variable spacing.
- Projection segmentation can fail when digits touch, overlap, or contain noisy columns.
- The current pipeline should standardize segmented digit outputs to 28x28 when they are used by MNIST-trained models.

### Contribution to Requirements

This notebook would support Task 2 and provide visual evidence for the project report.

## Recommended Notebook 3: Model Comparison

### Purpose

This notebook should compare machine learning methods for Task 3.

### Suggested Workflow

1. Load MNIST training and test data.
2. Normalize and reshape the data.
3. Train Logistic Regression as a baseline.
4. Train the CNN model.
5. Evaluate both models on the same MNIST test set.
6. Report accuracy, confusion matrix, and sample misclassifications.
7. Select the best model for integration.

### Issues to Address

- The CNN is expected to outperform Logistic Regression on MNIST, but the project should show the actual measured result.
- Training should use a fixed random seed where possible for reproducibility.
- The notebook should save model artifacts consistently to the same path used by the application.
- A confusion matrix is strongly recommended because accuracy alone does not show which digits are confused.

### Contribution to Requirements

This notebook would provide the strongest evidence for Task 3, which has the largest marking weight.

## Recommended Notebook 4: End-to-End System Evaluation

### Purpose

This notebook should test the complete HNRS pipeline from image input to final prediction.

### Suggested Workflow

1. Load image files from the sample dataset.
2. Apply preprocessing.
3. Apply segmentation.
4. Predict each segmented digit.
5. Reconstruct multi-digit numbers from predicted digits.
6. Compare predicted outputs with expected labels where available.
7. Summarize system-level strengths and limitations.

### Issues to Address

- The current project has separate components, but it needs stronger evidence of integrated performance.
- Multi-digit evaluation should check both digit accuracy and full-number accuracy.
- Failure cases should be documented honestly, especially touching digits, poor contrast, or unusual handwriting.

### Contribution to Requirements

This notebook would support Task 4 and the final report section on system functions and performance.

## Specific Code-Level Issues Identified During Review

1. The repository README mentions a `notebook/` directory, but that directory does not exist.
2. `src/gui/app.py` searches for the CNN model at `models/digit_cnn_model.h5`, while the repository stores the model under `src/models/digit_cnn_model.h5`.
3. `scripts/run_pipeline.py` uses a default segmented digit size of 64, but the CNN and Logistic Regression models expect 28x28 input.
4. Some documentation contains mojibake characters caused by text encoding problems, which should be replaced with plain ASCII or valid Unicode.
5. Current tests are closer to manual smoke tests than automated unit tests. They print results but do not consistently assert expected behavior.

## Suggested Fixes

1. Add a `notebooks/` directory, or update the README if the team chooses not to use notebooks.
2. Align all model save and load paths so training, testing, CLI prediction, and GUI prediction use the same location.
3. Use 28x28 as the default digit size for model-facing segmentation output.
4. Add reproducible experiment notebooks with markdown explanations, plots, and evaluation metrics.
5. Add automated tests for preprocessing output shape, segmentation behavior on blank images, model input shape, and GUI model-path loading.

## Academic Recommendation

The current codebase is a good foundation for a working HNRS application, but it needs stronger experimental documentation. For a university project, the final submission should not only show that the system works; it should also explain why each technique was chosen. The recommended notebooks would provide that evidence in a clear and assessable form.
