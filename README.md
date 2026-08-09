# COS30018 Group Project

This repository contains the COS30018 group project for handwritten letter and digit recognition.
It includes image preprocessing, segmentation, model training and evaluation, a GUI component, and utility scripts for dataset management.

## Project structure

- `src/`
  - `preprocessing/` — image loading, grayscale conversion, thresholding, resizing, centering, and normalization.
  - `segmentation/` — character region detection and digit/letter segmentation for OCR input.
  - `models/` — CNN and baseline model code, saved weights, and training utilities.
  - `gui/` — the user interface module for submitting images and showing predictions.
- `data/` — datasets used for training, testing, and evaluation.
  - `data/unseen_data/Case#1/unseen_alphabetical` — the reported unseen dataset folder for evaluation.
  - `data/synthetic-text/` — generated or synthetic text datasets.
  - `data/Trash/` — sample and trash folders used for experiments.
- `scripts/` — command-line scripts for testing OCR models and project utilities.
- `documents/` — project reports, model evaluation notes, and saved prediction results.
- `notebooks/` — experiment notebooks and documentation.
- `output_template/` — example outputs and written result templates.
- `requirements.txt` — Python dependencies needed to run the project.
- `ProjectOverview.md` — existing project overview and workflow notes.
- `COS30018 - Project Assignment - Option B.pdf` — project assignment brief.

## Main files to know

- `desired_output.txt` — saved prediction results from the OCR test run.
- `desired_output_alphanumericCNN_unseenletters.txt` — additional saved OCR evaluation results.
- `scripts/test_ocr_5.py` — current OCR evaluation script with automatic result saving.

## How to use the project

1. Create a virtual environment and install dependencies:
   ```bash
   setup_venv.bat
   ```
   Then activate it in PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. Evaluate the OCR model on a folder:
   ```bash
   python scripts/test_ocr_5.py
   ```
   The script asks for the image directory and writes results under `documents/model-evaluation/Prediction_Results`.

## Notes

- The repository does not currently contain a single integrated application entry point; use the scripts in `scripts/` for evaluation.
- `src/README.md` describes the source code module structure and integration rules.
- `scripts/README.md` explains available scripts and their purpose.
- `data/README.md` documents the dataset layout.

## Recommended reading

- `ProjectOverview.md` for the project goal, objectives, and broader workflow.
- `src/README.md` for source module interfaces and expected pipeline behavior.
- `scripts/README.md` for details about the utility scripts and how to run them.
