# Change Summary

## Overview
This document summarizes recent integration and workflow improvements made in the repository.

## Key changes
- Added Python package markers for `src/`, `src/preprocessing/`, `src/models/`, `src/models/training/`, `src/models/test/`, `src/gui/`, and `src/segmentation/`.
- Added a reusable model loader in `src/models/model.py`.
- Reworked the training script in `src/models/training/train_cnn.py` into a reusable module with CLI support.
- Fixed `src/models/test/test_model.py` to resolve the saved model file path correctly and make it runnable.
- Updated the GUI in `src/gui/app.py` to use the shared Keras model loader instead of an unavailable PyTorch path.
- Cleaned and unified `requirements.txt`, removing merge conflict markers and adding required dependencies.
- Reworked `src/preprocessing/preprocessing.py` CLI flow and added `process_path(...)` for folder-based preprocessing.
- Added workflow entry points:
  - `src/__main__.py`
  - `src/preprocessing/__main__.py`
  - `src/models/training/__main__.py`
  - `src/models/test/__main__.py`
  - `src/gui/__main__.py`
- Updated `README.md` with module-based usage instructions.

## Usage examples
- `python -m src preprocess data/input_folder output_folder --method otsu --size 28 28`
- `python -m src train --epochs 5`
- `python -m src predict data/sample.png`
- `streamlit run src/gui/app.py`

## Notes
- The project now supports a cleaner, package-based workflow.
- The summary is intentionally short and focused on integration changes.
