# Scripts Folder Overview

This folder contains command-line scripts for OCR testing and project utilities.

## Important scripts

- `test_ocr.py`
  - Original OCR evaluation script for CNN models.
  - Prompts for the image directory and reports prediction results.
- `test_ocr_5.py`
  - Current OCR evaluation script.
  - Saves results automatically to `documents/model-evaluation/Prediction_Results`.
  - Writes a timestamped version copy for every run.
- `run_pipeline.py`
  - A utility script for running a sequence of project tasks, if present.
- `cherry_pick_part1.py`
  - A helper script for reusing selected dataset or model files during experiments.

## How to run

1. Run OCR evaluation:
   ```bash
   python scripts/test_ocr_5.py
   ```

## Notes

- `scripts/test_ocr_5.py` is the main evaluation entry point for this repository.
- Keep `desired_output.txt` and other result files if you need to re-run evaluation later.
