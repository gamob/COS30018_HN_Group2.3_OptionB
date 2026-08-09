# CNN Training Workflows

The project uses two independent CNN classifiers because digits and letters have different output spaces and datasets.

## Digit CNN

- Dataset: MNIST.
- Output classes: digits `0-9`.
- Model file: `src/models/digit_cnn_model.h5`.
- Command:

```bash
python -m src train --epochs 5
```

The canonical Python functions are `build_digit_model()` and `train_and_save_digit_model()`.
`train_and_save_model()` remains as a backward-compatible alias for digit training.

## Letter CNN

- Dataset: the A-Z directory splits under `data/letters`.
- Output classes: 26 uppercase letters.
- Model file: `src/models/letter_cnn_model.h5`.
- Mapping file: `src/models/letter_class_mapping.json`.
- Command:

```bash
python -m src train-letters --epochs 15
```

The canonical Python functions are `build_model()` and `train_and_save_letter_model()`.
Letter training uses validation checkpoints, early stopping, learning-rate reduction, and augmentation.

Running `python -m src.models.training` directly invokes the standalone letter-training CLI. For an unambiguous workflow, prefer the two `python -m src ...` commands above.

## Other models

Digit Logistic Regression and RBF SVM have separate modules and checkpoint files. The dual-CNN merge does not replace or retrain them.
