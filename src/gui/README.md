# Week 1 Report — GUI Development

## Overview
This week I developed the GUI frontend for the Handwritten Number Recognition System using Streamlit.
The interface supports image input from file upload or from selecting a folder, provides preprocessing controls, and displays a prediction result.

## Implemented Features
- **Image input methods**
  - Upload a handwritten digit image file
  - Load an image from a folder path and choose among available images
- **Preprocessing controls**
  - Binarization method selection: `otsu`, `simple`, `adaptive`
  - Threshold slider for simple binarization
  - Blur kernel size slider for Otsu thresholding
  - Adaptive threshold block size and `C` value controls
  - Invert foreground/background option
- **Preview output**
  - Display original input image
  - Show intermediate preprocessing steps:
    - grayscale
    - binary threshold result
    - centered digit crop
    - final 28x28 normalized image
- **Prediction UX**
  - Mock prediction fallback that displays `I think this is a 5`
  - Optional `SimpleCNN` model path if a trained checkpoint is added later
  - Safe fallback when model files or dependencies are unavailable

## Technical Notes
- The GUI app is implemented in `src/gui/app.py`.
- Renamed the original `src/gui/streamlit.py` file to `src/gui/app.py` to avoid import conflicts with the Streamlit package.
- Used `src/preprocessing/preprocessing.py` from the project to generate preprocessing previews.
- Added support for loading a trained PyTorch model if a `.pth` checkpoint exists in the repository.

## Environment Setup
- Installed required packages in the project virtual environment:
  - `streamlit`
  - `numpy`
  - `Pillow`
  - `torch`
  - `torchvision`
  - `opencv-python`
  - `scikit-learn`
  - `tqdm`
  - `matplotlib`
  - `scipy`
  - `h5py`

## How to Run
From the project root, execute:

```bash
\.venv\Scripts\streamlit.exe run src/gui/app.py
```

Then open the URL shown by Streamlit (typically `http://localhost:8501`).

## Result
The GUI now provides a usable frontend for a non-technical user, including image selection, preprocessing controls, result display, and error-safe behavior.
If a trained PyTorch model checkpoint is added, the interface can switch from mock prediction to a real digit prediction pipeline.
