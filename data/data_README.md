# Data Folder Overview

This folder contains datasets used by the OCR project.

## Key dataset locations

- `data/unseen_data/Case#1/unseen_alphabetical`
  - The main unseen dataset used for evaluation and prediction experiments.
  - Contains subfolders with character images labeled by letter.
- `data/synthetic-text/`
  - Synthetic or generated text images used for training or testing.
- `data/Trash/`
  - Sample and experiment folders that are not part of the main dataset.

## Usage

- Use the `data/unseen_data/Case#1/unseen_alphabetical` path when running evaluation or cleanup scripts.
- Keep the folder structure intact so scripts can resolve relative image paths correctly.

## Notes

- The output of OCR tests is stored in `documents/model-evaluation/Prediction_Results` by default.
- Do not delete or rename image folders unless you are intentionally cleaning the dataset.
